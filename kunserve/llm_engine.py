import enum
import random
import time
import copy
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, AsyncGenerator
import asyncio
import math
import argparse
import torch
import sys
import numpy as np

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup

from ray.exceptions import GetTimeoutError

from kunserve.flash_backend import FlashBackend
from kunserve.config import (
    ModelConfig,
    ParallelConfig,
    CacheConfig,
    ColocatedSchedConfig,
)
from kunserve.flash_backend import Event

from kunserve.request import (
    SamplingParams,
    Request,
    create_request,
    RequestOutput,
    TokenOutput,
)
from kunserve.tokenizer import get_tokenizer
from kunserve.utils import Counter, EngineState
from kunserve.global_dispatcher import GlobalDispatcher
from kunserve.ray_queue import RayQueueServer, ServerInfo
from kunserve.kunserve_config import *
from kunserve.utils import random_uuid, get_instance_name

from typing import Any, Callable, Type, Union

from kunserve.logger import init_logger
logger = init_logger(__name__)

CHECK_ENGINE_STATE_INTERVAL = 0.1
INSTANCE_LOAD_REPORT_INTERVAL = 0.5
GRACE_TIME = 5 # grace time for stale requests to claim extra blocks

"""
Borrowed from vllm
"""
STOP_ITERATION = Exception() 

class AsyncStream:
    """A stream of RequestOutputs or PoolingRequestOutputs for a request
    that can be iterated over asynchronously via an async generator."""

    def __init__(self, request_id: str, cancel: Callable[[str], None]) -> None:
        self.request_id = request_id
        self._cancel = cancel
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[RequestOutput,
                              Exception]) -> None:
        if not self._finished:
            self._queue.put_nowait(item)

    def finish(
        self,
        exception: Optional[Union[BaseException, Type[BaseException]]] = None,
    ) -> None:
        if not self._finished:
            self._finished = True
            self._queue.put_nowait(
                exception if self._is_raisable(exception) else STOP_ITERATION)

    @property
    def finished(self) -> bool:
        return self._finished
    
    async def generator(
        self
    ) -> AsyncGenerator[RequestOutput, None]:
        try:
            while True:
                result = await self._queue.get()
                if self._is_raisable(result):
                    if result == STOP_ITERATION:
                        return
                    raise result
                yield result
        except GeneratorExit:
            self._cancel(self.request_id)
            raise asyncio.CancelledError from None

    @staticmethod
    def _is_raisable(value: Any):
        return isinstance(value, BaseException) or \
                (isinstance(value, type) and \
                 issubclass(value, BaseException))

class LLMEngine(ABC):
    """
    LLMEngine: An LLMEngine launches the model executor workers and maintains runtime information.

    ## Overview

    This class, LLMEngine, receives requests from upper wrapper class and provides
    interface LLMEngine.generate() that yields the generated tokens for each request.

    It supports the feature of "disaggregate", which basically means to run
    the context stage and the decoding stage on different GPUs to avoid interference.

    ## Implementation

    First let's inspect the automaton of one request:

            After
            context
            stage        |-------------|
    Waiting --------> Decoding <-------| After one decoding stage
                         |
                         |
                         V
                      Finished

    This class is implemented based on queues and event loops. There are three
    queues, two for scheduling and one for communication between event loops:
      - The waiting queue, maintained inside the ContextStageScheduler, which
        contains all the requests that are waiting for processing.
      - The decoding queue, maintained inside the DecodingStageScheduler, which
        contains all the requests that need further decoding.
      - The "bridge" queue, which contains all the requests that have just finished
        the context stage but have not been accepted by the decoding stage.
        (Producer: context stage event loop, Consumer: decoding stage event loop)

    Two event loops are executed concurrently and endlessly:
      - Context stage event loop. This event loop fetches requests from the waiting
        queue, forwards them to the context stage, and then puts them into the
        "bridge" queue.
      - Decoding stage event loop. This event loop accepts requests from the
        "bridge" queue (put them into the decoding queue), and then fetches requests
        from the decoding queue, forwards them to the decoding stage, and then
        informs the caller of the generated tokens.

    Note: Users may not use LLMEngine directly, but use more user-friendly wrapper classes
    OfflineLLM and AsyncLLM instead.
    """

    def __init__(
        self,
        client_id: int,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        dispatch_config: DispatchConfig,
        bench_config: BenchConfig,
    ):
        self.model_config = model_config
        self.cache_config = cache_config
        self.bench_config = bench_config
        self.request_counter = Counter()
        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
        )
        self.sampling_params = SamplingParams()

        # request_id -> list of LifetimeEvent
        # Created when calling self.generate()
        # Cleared by the caller of self.generate() (i.e. the engine does not clear that)
        # TODO: clear this automatically to avoid memory leak
        # self.request_lifetime_events: Dict[int, List[LifetimeEvent]] = {}
        
        self.instances = []

        self.engine_initialized = False
        self.state = EngineState.INITIAL

        # we haven't support redispatch yet
        # self.global_dispatcher = GlobalDispatcher(self.state, dispatch_config.dispatch_strategy)
        # self.dispatch_strategy = dispatch_config.dispatch_strategy

        # request_id -> AsyncStream, used to collect request outputs
        self.request_streams: Dict[int, AsyncStream] = {}
        self.request_output_queue = RayQueueServer()
        self.server_info = ServerInfo(
            client_id,
            self.request_output_queue,
        )
        self.output_tokens = 0
        self.unfinished_requests = 0
        
    def generate(
        self,
        instance_id: int,
        request_id: str,
        arrival_time: float,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[str]],
        max_tokens: int = 1024,
        ignore_eos = True,
    ) -> AsyncGenerator[RequestOutput, None]:
        assert (
            self.engine_initialized
        ), "Engine not initialized. Please call engine.initialize() before generating."

        server_info = copy.deepcopy(self.server_info)
        sampling_params = copy.deepcopy(self.sampling_params)
        sampling_params.max_tokens = max_tokens
        sampling_params.ignore_eos = ignore_eos

        req = create_request(
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            server_info=server_info,
            request_counter=self.request_counter,
            tokenizer=self.tokenizer,
            arrival_time=arrival_time,
            request_id=request_id,
            max_model_len=self.bench_config.max_model_len,
        )
        # we dont need global output collector any more!
        ray.get(self.instances[instance_id].add_request.remote(req), timeout=10)
        results_generator = AsyncStream(
            request_id,
            cancel=self.abort,
        )
        self.request_streams[request_id] = results_generator
        return results_generator.generator()
    
    def batch_generate(
        self,
        instance_id: int,
        request_ids: List[str],
        arrival_time: float,
        batch_prompts: List[str],
        batch_prompt_token_ids: List[List[int]],
        prefix_len: int = 0,
        max_tokens: int = 1024,
        ignore_eos = True,
    ):
        assert (
            self.engine_initialized
        ), "Engine not initialized. Please call engine.initialize() before generating."

        server_info = copy.deepcopy(self.server_info)
        sampling_params = copy.deepcopy(self.sampling_params)
        sampling_params.max_tokens = max_tokens
        sampling_params.ignore_eos = ignore_eos

        requests = []
        generators = []
        for (request_id, prompt, prompt_token_ids) in zip(request_ids, batch_prompts, batch_prompt_token_ids):
            requests.append(create_request(
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                server_info=server_info,
                request_counter=self.request_counter,
                tokenizer=self.tokenizer,
                arrival_time=arrival_time,
                request_id=request_id,
            ))
            results_generator = AsyncStream(
                request_id,
                cancel=self.abort,
            )
            self.request_streams[request_id] = results_generator
            generators.append(results_generator.generator())
        
        # we dont need global output collector any more!
        ray.get(self.instances[instance_id].add_batch_requests.remote(
            requests, prefix_len), timeout=10)
        return generators
    
    def start_get_outputs_loop(self):
        return asyncio.create_task(self.get_request_outputs_loop())

    async def get_request_outputs_loop(self):
        """
        This function collects request outputs from engines and put them into the specific request streams.
        """
        while True:
            request_outputs = await self.request_output_queue.get()
            for request_output in request_outputs:
                request_id = request_output.request_id
                # Request could be dispatched twice when manager is dead, the first request will free the request_streams when finished.
                if request_id not in self.request_streams:
                    continue
                self.request_streams[request_id].put(request_output)
                if request_output.finished:
                    self.request_streams[request_id].finish()
                    del self.request_streams[request_id]

    def execute_instance_func(self, instance_id, func_name, *args):
        ray_func = getattr(self.instances[instance_id], func_name).remote(*args)
        result = ray.get(ray_func, timeout=10)
        return result
    
    async def abort(self, request_id: str):
        # TODO: impl abort logic
        pass

    async def terminate(self):
        pass

    # def _on_new_lifetime_event_callback(self, request_id: int, event: LifetimeEvent):
    #     """
    #     Called by self.context_engine or self.decoding_engine when a new lifetime event
    #     is generated
    #     """
    #     for _event in self.request_lifetime_events[request_id]:
    #         if _event.event_type == event.event_type:
    #             return
    #     self.request_lifetime_events[request_id].append(event)

    @abstractmethod
    async def initialize(self, prefix: str):
        raise NotImplementedError()

class ColocatedEngine(LLMEngine):

    def __init__(
        self,
        server_id: int,
        group_id: int,
        ngroups: int,
        placement_groups: List[PlacementGroup], # each PlacementGroup is prepared for one instance
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: ColocatedSchedConfig,
        balloon_config: BalloonConfig,
        dispatch_config: DispatchConfig,
        bench_config: BenchConfig,
    ):
        super().__init__(
            group_id,
            model_config=model_config,
            cache_config=cache_config,
            dispatch_config=dispatch_config,
            bench_config=bench_config,
        )
        self.group_id = group_id
        self.ngroups = ngroups
        self.parallel_config = parallel_config
        self.balloon_size = parallel_config.replica_size
        self.balloon_mode = balloon_config.balloon_mode
        self.balloon_config = balloon_config
        self.bench_config = bench_config

        self.try_balloon = False

        # instance num shall be equal to balloon size
        self.instance_num = self.balloon_size
        self.engine_load = 0.0
        
        model_config = copy.deepcopy(self.model_config)
        model_config.abandon_hf_config()
        
        for i in range(self.instance_num):
            instance_id = group_id * self.instance_num + i
            parallel_config = copy.deepcopy(parallel_config)
            parallel_config.replica_rank = i
            parallel_config.group_parallel_size = (
                self.instance_num * parallel_config.pipeline_parallel_size
            )
            parallel_config.global_size = self.ngroups * self.instance_num * parallel_config.pipeline_parallel_size

            instance_actor = ray.remote(
                num_cpus=1,
                name=get_instance_name(server_id, instance_id),
                namespace='kunserve',
                max_concurrency=8192,
            )(FlashBackend).options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_groups[i],
                    placement_group_bundle_index=0,
                    placement_group_capture_child_tasks=True,
                ),
            )
            try:
                instance = instance_actor.remote(
                    server_id=server_id,
                    instance_id=instance_id,
                    placement_group=placement_groups[i],
                    model_config=model_config,
                    parallel_config=parallel_config,
                    cache_config=cache_config,
                    sched_config=sched_config,
                    balloon_config=balloon_config,
                    bench_config=bench_config,
                )
                self.instances.append(instance)
            except Exception as e:
                logger.error(f"Failed to create instance {instance_id} due to e: {e}")
                ray.shutdown()
                exit(1)
        for instance in self.instances:
            ray.get(instance.set_instances.remote(self.instances))

        self.engine_in_progress = None
        # every group has a global dispatcher
        # self.global_dispatcher.set_instances(self.instances)

    @DeprecationWarning
    async def _merge_engines(self):
        leader_engine = self.instances[0]

        for i, engine in enumerate(self.instances[1:]):
            try:
                workers, schedulers_data, block_manager_data, unaccepted_requests = await engine.get_instance_properties.remote()
                await leader_engine.merge_engine.remote(
                    i+1, workers, schedulers_data, block_manager_data, unaccepted_requests
                )
            except Exception as e:
                logger.error(f"Failed to merge engine {i+1} due to e: {e}")
                ray.shutdown()
                exit(1)
    
    async def _split_engines(self):
        leader_engine = self.instances[0]
        workers, schedulers_data, block_managers = await leader_engine.split_engine.remote()
        for i, engine in enumerate(self.instances[1:]):
            await engine.set_instance_properties.remote(workers[i], schedulers_data[i], block_managers[i])

    async def initialize(self, global_id):
        inits = []
        gp_id = []
        kvex_id = []
        for i in range(self.parallel_config.tensor_parallel_size):
            gp_id.append(copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())) # used for new pipeline after balloon
            kvex_id.append(copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id()))  # used for kv exchange
        
        for instance in self.instances:
            inits.append(instance.initialize.remote(gp_id, kvex_id, global_id, self.balloon_size))
        try:
            logger.info(f"(group {self.group_id}) Initializing {len(self.instances)} LLM instance...")
            ray.get(inits)
            self.engine_initialized = True
        except Exception as e:
            logger.error(f"LLM instance init failed: {e}")
            ray.shutdown()
            exit(1)
    
    async def start_all_engines(self):
        handlers = []
        for engine in self.instances:
            handlers.append(engine.set_engine_stopped.remote(False))
        await asyncio.gather(*handlers)

    async def terminate(self):
        output_tokens_per_epoch = np.array(self.output_tokens_per_epoch)
        avg_tokens_thpt = output_tokens_per_epoch.mean()
        max_tokens_thpt = output_tokens_per_epoch.max()
        min_tokens_thpt = output_tokens_per_epoch.min()
        logger.info(f"Group({self.group_id}) terminates, avg_tokens_thpt: {avg_tokens_thpt}, max_tokens_thpt: {max_tokens_thpt}, min_tokens_thpt: {min_tokens_thpt}")

        self.engine_initialized = False
        self.state = EngineState.STOP

    async def set_instance_stopped(self, i, stopped: bool):
        await self.instances[i].set_engine_stopped.remote(stopped)

    def get_state(self):
        return self.state

    async def stop_all_engines(self):
        handlers = []
        for engine in self.instances:
            handlers.append(engine.set_engine_stopped.remote(True))
        await asyncio.gather(*handlers)
        
        # wait until all engines are stopped
        logger.info(f"(group {self.group_id}) wait for engine to stop current step.")
        while True:
            try:
                done, _ = ray.wait(
                    self.engine_in_progress,
                    num_returns=len(self.engine_in_progress),
                    timeout=0,
                )
            except Exception as e:
                logger.error(f"Failed to stop all engines: {e}, {self.engine_in_progress=}")
                ray.shutdown()
                exit(1)
            if len(done) == len(self.engine_in_progress):
                break
            await asyncio.sleep(CHECK_ENGINE_STATE_INTERVAL)

    async def stop_followers_and_get_properties(self):
        handlers = []
        for rank, engine in enumerate(self.instances[1:]):
            handlers.append(engine.stop_and_get_properties.remote())
        instance_properties = await asyncio.gather(*handlers)
        return instance_properties

    # async def trigger_balloon(self, from_restore: bool):
    #     start = time.perf_counter()
    #     logger.info(f"(group {self.group_id}) [Balloon Routine] {self.state} -> BALLOON")
    #     logger.info(f"(group {self.group_id}) [Trigger Balloon] step 1: stop all engines")
    #     self.state = EngineState.STOP
    #     await self.stop_all_engines()

    #     logger.info(f"(group {self.group_id}) [Trigger Balloon] step 2: merge engines")
    #     handlers = []
    #     for engine in self.instances[1:]:
    #         handlers.append(engine.get_instance_properties.remote())
    #     instance_properties = await asyncio.gather(*handlers)

    #     logger.info(f"(group {self.group_id}) [Trigger Balloon] step 3: drop parameters and switch to pipeline execution")
    #     free_gpu_blocks = await self.instances[0].trigger_balloon.remote(self.balloon_mode, from_restore, instance_properties)

    #     self.state = EngineState.BALLOON
    #     logger.info(f"Balloon process takes {time.perf_counter() - start} seconds")
    #     return int(free_gpu_blocks * self.balloon_config.max_balloon_memory_ratio)


    async def trigger_balloon(self, from_restore: bool):
        start = time.perf_counter()
        logger.info(
            f"(group {self.group_id}) [Balloon Routine] {self.state} -> BALLOON"
            f"(group {self.group_id}) [Trigger Balloon] step 1: stop all engines"
        )
        self.state = EngineState.STOP

        try:
            instance_properties = await self.stop_followers_and_get_properties()
            logger.info(f"(group {self.group_id}) [Trigger Balloon] step 2: drop parameters and switch to pipeline execution")
            free_gpu_blocks = await self.instances[0].trigger_balloon.remote(self.balloon_mode, from_restore, instance_properties)
        except Exception as e:
            logger.error(f"Failed to trigger balloon: {e}")
            ray.shutdown()
            exit(1)

        self.state = EngineState.BALLOON
        logger.info(f"Balloon process takes {time.perf_counter() - start} seconds")
        return int(free_gpu_blocks * self.balloon_config.max_balloon_memory_ratio)

    async def trigger_restore(self):
        logger.info(f"(group {self.group_id}) [Balloon Routine] {self.state} -> RESTORE")
        leader_engine = self.instances[0]
        logger.info(f"(group {self.group_id}) [Trigger Restore] step 1: reclaim extra blocks")
        await leader_engine.reclaim_extra_blocks.remote()

        logger.info(f"(group {self.group_id}) [Trigger Restore] step 2: load params")
        # `pmm_restore` does two things in C++:
        #   (i)  remaps each dropped weight-layer handle from the kv_data
        #        region back to the weight segment (cuMemMap);
        #   (ii) pulls the actual weight bytes back from the peer in the
        #        same balloon group that kept that layer intact, over
        #        `kv_exchange_comm` via NCCL send/recv.
        # (ii) is the real restore cost (network-bound, dominated by
        # `num_layers * (g-1)/g * layer_size / link_bandwidth`).  The
        # previous simulated `asyncio.sleep(trans_size / NET_BW)` has been
        # replaced by waiting on the actual NCCL stream sync inside
        # `pmm_restore`.
        await leader_engine.pmm_restore.remote()

        logger.info(f"(group {self.group_id}) [Trigger Restore] step 3: stop leader engines")
        self.state = EngineState.STOP
        await self.stop_all_engines()
        
        # restore the blocks to the original format
        logger.info(f"(group {self.group_id}) [Trigger Restore] step 4: restore blocks to original format")
        await leader_engine.restore_instance.remote()
    
        self.state = EngineState.RESTORE
    
    async def trigger_init(self):
        logger.info(f"(group {self.group_id}) [Balloon Routine] {self.state} -> INIT")
        logger.info(f"(group {self.group_id}) [Trigger Init] step 1: stop leader engines")
        self.state = EngineState.STOP
        await self.stop_all_engines()

        logger.info(f"(group {self.group_id}) [Trigger Init] step 2: split engines and restore engines to initial state")
        await self._split_engines()
        
        logger.info(f"(group {self.group_id}) [Trigger Init] step 3: restart engines")
        refs = []
        for engine in self.instances:
            refs.append(engine.trigger_init.remote())
        await asyncio.gather(*refs)

        self.state = EngineState.INITIAL
        # self.global_dispatcher.set_dispather(self.instances, self.state)
    
    async def make_decision_for(self, engines, event):
        refs = []
        for engine in engines:
            refs.append(engine.vote_for.remote(event))
        votes = await asyncio.gather(*refs)
        if event == Event.INIT:
            return all(votes)
        else:
            # watchout, 1 // 2 == 0, so you have to check the votes
            return sum(votes) > 0 and sum(votes) >= len(engines) // 2
    
    async def get_event(self):
        event = Event.KEEP

        if self.state == EngineState.INITIAL:
            decision = await self.make_decision_for(self.instances, Event.BALLOON)
            if decision:
                # start to use pipelined execution
                event = Event.BALLOON
                # await self.trigger_balloon(from_restore=False)
                # await asyncio.sleep(GRACE_TIME)

        elif self.state == EngineState.BALLOON:
            decision = await self.make_decision_for(self.instances[:1], Event.RESTORE)
            if decision:
                event = Event.RESTORE
                # await self.trigger_restore()
                # await asyncio.sleep(GRACE_TIME)
                
        elif self.state == EngineState.RESTORE:
            # if self.try_balloon:
            #     decision = await self.make_decision_for(self.instances[:1], Event.BALLOON)
            #     if decision:
            #         event = Event.BALLOON
            #         # await self.trigger_balloon(from_restore=True)
            #         # await asyncio.sleep(GRACE_TIME)
            #     else:
            #         self.try_balloon = False
            # else:
            #     decision = await self.make_decision_for(self.instances[:1], Event.INIT)
            #     if decision:
            #         event = Event.INIT
            #         # await self.trigger_init()
            #         # await asyncio.sleep(GRACE_TIME)
            #     else:
            #         self.try_balloon = True
            decision = await self.make_decision_for(self.instances[:1], Event.INIT)
            if decision:
                event = Event.INIT

        return self.state, event
                
    '''
    Poll engine state, and restart event loop if necessary
    '''
    async def monitor_engines(self):
        self.engine_in_progress = [
            engine.start_event_loop.remote() for engine in self.instances
        ]
        while True:
            done, _ = ray.wait(
                self.engine_in_progress,
                num_returns=len(self.engine_in_progress),
                timeout=0,
            )
            try:
                for engine in done:
                    i = self.engine_in_progress.index(engine)
                    is_instance_stopped = await self.instances[i].is_instance_stopped.remote()
                    if is_instance_stopped:
                        continue
                    logger.info(f"(group {self.group_id}) restarting engine {i}")
                    self.engine_in_progress[i] = self.instances[i].start_event_loop.remote()
            
            except Exception as e:
                logger.error(f"Engine monitor failed: {e}")
                ray.shutdown()
                exit(1)
            await asyncio.sleep(CHECK_ENGINE_STATE_INTERVAL)
    
    async def get_gpu_blocks_usage(self):
        active_instances = self.instances if self.state == EngineState.INITIAL else self.instances[:1]
        
        refs = []
        for engine in active_instances:
            refs.append(engine.get_gpu_blocks_usage.remote())
        usages = await asyncio.gather(*refs)

        return self.state, usages

    async def start_all_event_loops(self):
        assert (
            self.engine_initialized
        ), "Engine not initialized. Please call engine.initialize() before starting event loops."
        monitor_task = asyncio.create_task(self.monitor_engines())
        output_task = self.start_get_outputs_loop()
        
        await asyncio.gather(monitor_task, output_task)

    '''
    below are for requests transfer
    '''
    async def pop_running_requests_balloon_transfer(self, num_blocks_limit: int):
        await self.stop_all_engines()
        num_blocks_limit //= self.instance_num

        # requests are divided by instance
        refs = []
        for instance in self.instances:
            refs.append(instance.pop_running_requests_transfer.remote(False, num_blocks_limit))
        requests: List[List[Request]] = await asyncio.gather(*refs)

        # get generator object
        request_generators: List[List[AsyncStream]] = []
        for reqs in requests:
            request_generators.append([])
            for req in reqs:
                request_generators[-1].append(self.request_streams[req.request_id])
                del self.request_streams[req.request_id] # avoid memory leak

        # restart the engine
        await self.start_all_engines()

        return {
            "requests": requests,
            "generators": request_generators,
        }
    
    async def append_and_allocate_requests_balloon_transfer(self, requests_map):
        assert self.state == EngineState.BALLOON, f"(group {self.group_id}) engine state is {self.state}"
        for i, reqs in enumerate(requests_map["requests"]):
            # update request info
            for j, req in enumerate(reqs):
                req.fusion_exec = True
                req.server_info = copy.deepcopy(self.server_info)
                self.request_streams[req.request_id] = requests_map["generators"][i][j]
            await self.instances[0].append_and_allocate_requests.remote(i, reqs)

    async def send_requests_balloon_transfer(self, offset: int, dst_group_id: int, requests: List[List[Request]]):
        refs = []
        for i, reqs in enumerate(requests):
            refs.append(self.instances[i].send_requests_balloon_transfer.remote(i, offset, dst_group_id, reqs))
        await asyncio.gather(*refs)

    async def recv_requests_balloon_transfer(self, offset: int, src_group_id: int, requests: List[List[Request]]):
        refs = []
        for i, reqs in enumerate(requests):
            refs.append(self.instances[0].recv_requests_balloon_transfer.remote(i, offset, src_group_id, reqs))
        await asyncio.gather(*refs)

    async def free_blocks_balloon_transfer(self, requests: List[List[Request]]):
        refs = []
        for i, reqs in enumerate(requests):
            refs.append(self.instances[i].free_blocks_balloon_transfer.remote(reqs))
        await asyncio.gather(*refs)

    async def get_avail_restore_blocks(self):
        refs = []
        for engine in self.instances:
            refs.append(engine.get_used_gpu_blocks.remote())
        used_gpu_blocks = await asyncio.gather(*refs)
        max_gpu_blocks = await self.instances[0].get_max_base_gpu_blocks.remote()
        avail_blocks = [
            int(max_gpu_blocks * self.balloon_config.max_restore_memory_ratio) - used_blocks for used_blocks in used_gpu_blocks
        ]

        return sum(avail_blocks)

    async def get_total_used_blocks(self):
        refs = []
        for engine in self.instances:
            refs.append(engine.get_used_gpu_blocks.remote())
        used_gpu_blocks = await asyncio.gather(*refs)
        return sum(used_gpu_blocks)

    async def get_avg_memory_ratio(self):
        if self.state == EngineState.INITIAL:
            refs = []   
            for engine in self.instances:
                refs.append(engine.get_total_demand_ratio.remote())
            memory_ratios = await asyncio.gather(*refs)
            return sum(memory_ratios) / len(memory_ratios)
        else:
            return await self.instances[0].get_total_demand_ratio.remote()

    async def pop_running_requests_restore_transfer(self, num_blocks_limit: int):
        await self.stop_all_engines()

        # requests are divided by instance
        requests: List[List[Request]] = await self.instances[0].pop_running_requests_transfer.remote(True, num_blocks_limit)

        # get generator object
        request_generators: List[List[AsyncStream]] = []
        for reqs in requests:
            request_generators.append([])
            for req in reqs:
                request_generators[-1].append(self.request_streams[req.request_id])
                del self.request_streams[req.request_id] # avoid memory leak

        self.instances[0].set_engine_stopped.remote(False)

        return {
            "requests": requests,
            "generators": request_generators,
        }

    async def append_and_allocate_requests_restore_transfer(self, requests_map):
        await self.stop_all_engines()

        for i, reqs in enumerate(requests_map["requests"]):
            # update request info
            for j, req in enumerate(reqs):
                req.fusion_exec = False
                req.kv_fusion_rank = 0
                req.server_info = copy.deepcopy(self.server_info)
                self.request_streams[req.request_id] = requests_map["generators"][i][j]
            await self.instances[i].append_and_allocate_requests.remote(0, reqs)
    
    async def send_requests_restore_transfer(self, offset: int, dst_group_id: int, requests: List[List[Request]]):
        refs = []
        for i, reqs in enumerate(requests):
            refs.append(self.instances[0].send_requests_restore_transfer.remote(i, offset, dst_group_id, reqs))
        await asyncio.gather(*refs)

    async def recv_requests_restore_transfer(self, offset: int, src_group_id: int, requests: List[List[Request]]):
        refs = []
        for i, reqs in enumerate(requests):
            rank = (i + offset) % self.instance_num
            refs.append(self.instances[rank].recv_requests_restore_transfer.remote(i, offset, src_group_id, reqs))
        await asyncio.gather(*refs)
    
    async def free_blocks_restore_transfer(self, requests: List[List[Request]]):
        await self.instances[0].free_blocks_restore_transfer.remote(requests)

    async def pop_metrics(self):
        if self.state == EngineState.INITIAL:
            ttfts, tbts = [], []
            for engine in self.instances:
                ttft, tbt = await engine.pop_metrics.remote()
                ttfts.extend(ttft)
                tbts.extend(tbt)
            return ttfts, tbts
        else:
            return await self.instances[0].pop_metrics.remote()