import asyncio
from collections import deque
import random
import sys
import threading
import time
from typing import Callable, Deque, Dict, List, Tuple, Optional
import copy

import ray
from ray.util.queue import Queue
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch

from kunserve.block_manager import BlockLocation, BlockManager, BlockManagerData
from kunserve.colocated_scheduler import ColocatedScheduler, ColocatedScheduler, get_colocated_scheduler, ColocatedSchedulerData
from kunserve.config import ParallelConfig, ModelConfig, CacheConfig, ColocatedSchedConfig, DisaggParallelConfig
from kunserve.request import BatchedRequests, Request, TokenOutput, RequestOutput, MigratingRequest
from kunserve.tokenizer import get_tokenizer
from kunserve.utils import Stage, get_instance_name
from kunserve.kunserve_config import *
from enum import Enum

from collections import deque, defaultdict
import threading
import queue

from kunserve.worker import ParaWorker
from kunserve.ray_queue import PutQueue

from kunserve.logger import init_logger
logger = init_logger(__name__)

# Sleep for this many seconds when there is no request in ContextStageLLMEngine.step()
# We need to sleep for a while because the whole program is a asyncio-based,
# event driven, single thread program. We save some CPU time for other coroutines.
SLEEP_WHEN_CONTEXT_NO_REQUEST = 0.003

# Sleep for this many seconds when there is no request in DecodingStageLLMEngine.step()
SLEEP_WHEN_DECODING_NO_REQUEST = 0.003

# Sleep for this many seconds in each event loop, useful for debugging
SLEEP_IN_EACH_EVENT_LOOP = 0

SLEEP_AFTER_POLLING_ALL_BRIDGES = 0.003

# Print engine status every this many seconds
PRINT_STATUS_INTERVAL = 1

# Global dispatcher will try to redispatch requests every this many seconds
REDISPATCH_INTERVAL = 0.1

# Global dispatcher will redispatch requests if load imbalance exceeds this threshold
REDISPATCH_THRESHOLD = 0.05

# Global dispatcher will sleep for this many seconds when there are too many unaccepted requests
SLEEP_WHEN_TOO_MANY_UNACCEPTED = 0.1

LIVE_RESHARD_INTERVAL = 0.25

def restore_block_manager_from_data(block_manager: BlockManager, data: BlockManagerData):
    block_manager.free_gpu_blocks_list = data.free_gpu_blocks_list
    block_manager.free_cpu_blocks_list = data.free_cpu_blocks_list
    block_manager.reserved_gpu_blocks_list = data.reserved_gpu_blocks_list
    block_manager.reserved_gpu_blocks_set = data.reserved_gpu_blocks_set
    block_manager.free_fusion_gpu_blocks_list = data.free_fusion_gpu_blocks_list
    block_manager.free_extend_gpu_blocks_list = data.free_extend_gpu_blocks_list
    block_manager.is_fusion_state = data.is_fusion_state
    block_manager.mock_block = data.mock_block
    block_manager.backup_gpu_blocks = data.backup_gpu_blocks
    block_manager.backup_free_gpu_blocks_list = data.backup_free_gpu_blocks_list
    block_manager.swapping_gpu_blocks_list = data.swapping_gpu_blocks_list
    block_manager.swapping_cpu_blocks_list = data.swapping_cpu_blocks_list
    block_manager.block_table = data.block_table
    block_manager.request_location = data.request_location
    block_manager.num_base_gpu_blocks = data.num_base_gpu_blocks
    block_manager.max_num_fusion_gpu_blocks = data.max_num_fusion_gpu_blocks
    block_manager.fusion_size = data.fusion_size
    block_manager.max_num_reserved_gpu_blocks = data.max_num_reserved_gpu_blocks
    block_manager.max_num_extend_gpu_blocks = data.max_num_extend_gpu_blocks
    return block_manager


class Event(Enum):
    """Voting mechanism in BalloonLLM"""
    INIT = 0
    PREPARE = 1
    BALLOON = 2
    RESTORE = 3
    KEEP = 4

class FlashBackend:

    def _get_scheduler(self) -> ColocatedScheduler:
        return get_colocated_scheduler(
            self.sched_config,
            self.parallel_config,
            self.model_config,
            self.block_managers,
            self._remote_call_all_workers_async,
        )
        
    def __init__(
        self,
        server_id: int,
        instance_id: int,
        placement_group: PlacementGroup,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: ColocatedSchedConfig,
        balloon_config: BalloonConfig,
        bench_config: BenchConfig,
        # reschedule_request_callback: Callable[[Request], None],
    ):
        self.stage = Stage.COLOCATED
        self.server_id = server_id
        self.instance_id = instance_id

        self.model_config = model_config
        self.model_config.load_hf_config()
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.sched_config = sched_config
        logger.info(f"use_tensor_cores: {self.sched_config.use_tensor_cores}")
        self.balloon_config = balloon_config
        logger.info(f"enable_balloon: {self.balloon_config.enable_balloon}")

        self.bench_config = bench_config
        # self.reschedule_request_callback = reschedule_request_callback
        
        self.scheduler = []
        self.scheduler_assigned_chunk = []
        self.scheduler_launch_events = []
        self.scheduler_running = []

        self.chunk_size_before_balloon = self.sched_config.chunked_prefill_budget
        self.chunk_enabled_before_balloon = self.sched_config.enable_chunked_prefill

        self.prefill_only_chunk = False
        self.balloon_start = None
        
        # self.to_accept_prefill = False
        # self.prefill_budget = 0
        # self.decode_budget = 0

        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
        )
        self.placement_group = placement_group

        # workers[i][j] is the j-th tensor-parallel worker in pipeline stage i
        self.workers = []
        self.block_managers: List[BlockManager] = []
        self.epoch = 0

        self.unaccepted_queue: Deque[Request] = deque()
        self.decode_queue = deque()
        self.num_instances = 1

        # metrics, consider packing them up
        self.finished_tokens = 0
        self.finished_requests = 0
        self.last_epoch_finished_tokens = 0
        self.last_epoch_finished_requests = 0
        
        # allocations from different coroutines may race
        # llumnix uses lock to prevent data race too (RX: we dont need it if we do not compare llumnix!)
        self.allocate_blocks_lock = threading.Lock()

        # RX: can we merge the following states?
        # engine will stop after finishing current step
        # when stopped is True
        self.is_fusion_state = False
        self.stopped = False
        self.state = "initial"
        self.is_resharding = [False] * self.parallel_config.pipeline_parallel_size
        # self.finish_resharding = True
        self.is_restoring = False
        self.is_reclaiming_ex_blocks = False
        
        window_size = 30
        self.sliding_window = deque([0]*window_size, maxlen=window_size)
        self.rate_window = deque([0]*window_size, maxlen=window_size)
        
        '''only for debug, we shall not use this!'''
        self.vote_epoch = 0

        self.put_queue_args_queue = queue.Queue()
        self.put_queue_loop_thread = threading.Thread(
            target=self._start_put_queue_loop, args=(), daemon=True, name="put_queue_loop"
        )
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=0,
            placement_group_capture_child_tasks=True,
        )
        self.put_queue_actor = ray.remote(
            num_cpus=1,
            scheduling_strategy=scheduling_strategy,
        )(PutQueue).remote(self.server_id, self.instance_id)
        self.put_queue_loop_thread.start()
        self.actor_name = get_instance_name(server_id, instance_id)
        self.verbose = True
        
        self.next_shed = 0

        '''for profile'''
        pp_size = self.parallel_config.pipeline_parallel_size
        self.latency_per_epoch = [[] for _ in range(pp_size)]
        self.num_tokens_per_epoch = [[] for _ in range(pp_size)]
        self.latency_diff = []
        self.num_tokens_diff = []
        self.ttfts = []
        self.tbts = []

        self.num_swap = 0
        self.is_migrating_out = False
    
    """Initialize workers, load models and initialize k/v cache

        We seperate this function from __init__ because we want to run it in an async way
        to enable parallel initialization between Engines.
    """
    def _start_put_queue_loop(self):
        """
        This mission work in a separate thread to minimize performance
        overhead on the generation thread.
        """
        while True:
            args = self.put_queue_args_queue.get()
            request_outputs = args
            # our put queue actor will return outputs to frontend
            self._put_requests_outputs_to_server(request_outputs)

    def _put_requests_outputs_to_server(self, request_outputs: List[RequestOutput]) -> None:
        server_request_outputs = defaultdict(list)
        server_info_dict = {}

        # Reorganize data in orther to put request output to queue in batch at one time.
        for request_output in request_outputs:
            server_info = request_output.server_info
            request_output.server_info = None # unset request_output.server_info as it is no longer required
            server_id = server_info.server_id
            server_request_outputs[server_id].append(request_output)
            if server_id not in server_info_dict:
                server_info_dict[server_id] = server_info
        # TODO: llumnix team is trying to optimize cross-actor overhead here, need to follow up
        self.put_queue_actor.put_nowait_to_servers.remote(server_request_outputs, server_info_dict)
      
    async def start_event_loop(self):
        async def step_event_loop():
            virtual_engine_num = len(self.scheduler)
            requests_in_progress = [
                asyncio.create_task(self._step(virtual_engine))
                for virtual_engine in range(virtual_engine_num)
            ]
                        
            while True:
                done, _ = await asyncio.wait(
                    requests_in_progress, return_when=asyncio.FIRST_COMPLETED
                )
                # relaunch event handler for done engines
                if not self.stopped:
                    for task in done:
                        virtual_engine = requests_in_progress.index(task)
                        requests_in_progress[virtual_engine] = asyncio.create_task(
                            self._step(virtual_engine)
                        )
                elif all(task.done() for task in requests_in_progress):
                    break
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)

        await asyncio.create_task(step_event_loop())
    
    def ready(self):
        return True
    
    def is_migrating(self):
        return self.is_migrating_out

    def set_instances(self, instances):
        self.instances = instances

    def abort_request(self, request_ids: List[str]):
        # TODO: not impl yet
        pass

    async def initialize(self, gp_id, kvex_id, global_id, balloon_size):
        try:
            await self._init_workers(gp_id, kvex_id, global_id, balloon_size)
            self.num_gpu_blocks, self.num_cpu_blocks = await self._init_model()
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            ray.shutdown()
            exit(1)
        
        self.prefix = "" # TODO: don we need to add prefix?
        self.block_managers.append(BlockManager(
            self.stage,
            self.prefix,
            self.num_gpu_blocks,
            self.num_cpu_blocks,
            self.model_config,
            self.parallel_config,
            self.cache_config,
            self._remote_call_all_workers_async,
        ))
        
        num_virtual_engine = self.parallel_config.pipeline_parallel_size
        self.scheduler = [
            self._get_scheduler()
            for _ in range(num_virtual_engine)
        ]
        self.scheduler_running = [
            False for _ in range(num_virtual_engine)
        ]
        logger.info(f"Instance {self.instance_id} initialized.")
    
    async def _init_workers(self, gp_id, kvex_id, global_id, balloon_size):
        """
        for each pipeline stage, create tensor_parallel_size workers
        each worker will be assigned a GPU
        the worker will be placed in the corresponding placement group
        """
        self.num_instance_workers = self.parallel_config.world_size
        pp_per_group = self.parallel_config.pipeline_parallel_size * balloon_size
        pp_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())

        init_handlers = []
        group_rank_start = (
            self.parallel_config.replica_rank
            * self.parallel_config.pipeline_parallel_size
        )

        for i in range(self.parallel_config.pipeline_parallel_size):
            workers = []
            tp_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())

            for j in range(self.parallel_config.tensor_parallel_size):
                parallel_config = copy.deepcopy(self.parallel_config)
                parallel_config.pipeline_parallel_rank = i
                parallel_config.tensor_parallel_rank = j
                parallel_config.group_parallel_rank = (group_rank_start + i) % pp_per_group # rank inside group
                parallel_config.group_parallel_size = pp_per_group                          # size of a group
                parallel_config.global_rank = self.instance_id * self.parallel_config.pipeline_parallel_size + i

                model_config = copy.deepcopy(self.model_config)
                model_config.abandon_hf_config()
                
                worker = ParaWorker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.placement_group,
                    ),
                ).remote(
                    worker_id=(i * self.parallel_config.tensor_parallel_size + j),
                    stage=self.stage,
                    model_config=model_config,
                    cache_config=self.cache_config,
                    parallel_config=parallel_config,
                    sched_config=self.sched_config,
                    pipeline_parallel_id=pp_id,
                    tensor_parallel_id=tp_id,
                    group_parallel_id=gp_id[j],
                    kv_exchange_id=kvex_id[j],
                    global_id=global_id[j],
                    disagg_parallel_config=DisaggParallelConfig(),
                )
                workers.append(worker)
                init_handlers.append(asyncio.wrap_future(worker.ready.remote().future()))
            self.workers.append(workers)

        await asyncio.wait(init_handlers)

    async def _init_model(self):
        """
        init model by call init_model() on all workers
        """
        num_gpu_blocks, num_cpu_blocks, block_size_in_bytes = await self.workers[0][
            0
        ]._profile_num_available_blocks.remote(
            self.cache_config.block_size,
            self.cache_config.gpu_memory_utilization,
            self.cache_config.cpu_swap_space,
            self.cache_config.kv_cache_ratio,
        )
        if self.bench_config.num_gpu_blocks is not None:
            num_gpu_blocks = self.bench_config.num_gpu_blocks
        self.block_size_in_bytes = block_size_in_bytes
        logger.info(
            f"Profiling result: num_gpu_blocks: {num_gpu_blocks}, num_cpu_blocks: {num_cpu_blocks}"
        )
        
        handlers = self._remote_call_all_workers_async(
            "init_model",
            num_gpu_blocks,
        )
        await asyncio.wait(handlers)
        
        return num_gpu_blocks, num_cpu_blocks
    
    def set_engine_stopped(self, stopped: bool):
        self.stopped = stopped
        
    async def stop_and_get_properties(self):
        self.stopped = True
        while any(self.scheduler_running):
            await asyncio.sleep(0)
        """safe to trigger balloon/restore from now on"""
        return self.get_instance_properties()

    def is_instance_stopped(self):
        return self.stopped

    def get_instance_properties(self):
        def get_schedulers_data():
            return [_scheduler.get_data() for _scheduler in self.scheduler]
        
        def get_block_manager_data():
            # At the beginning, there is only one block manager
            return self.block_managers[0].get_block_manager_data()

        workers = self.workers
        schedulers_data = get_schedulers_data()
        block_manager_data = get_block_manager_data()
        unaccepted_requests = list(self.unaccepted_queue)
        self.unaccepted_queue.clear()
        return workers, schedulers_data, block_manager_data, unaccepted_requests

    def set_instance_properties(self, workers, schedulers_data, block_manager_data):
        def set_block_manager(data):
            self.block_managers = [
                restore_block_manager_from_data(
                    BlockManager(
                        self.stage,
                        self.prefix,
                        self.num_gpu_blocks,
                        self.num_cpu_blocks,
                        self.model_config,
                        self.parallel_config,
                        self.cache_config,
                        self._remote_call_all_workers_async,
                    ),
                    data,
                )]

        def set_schedulers(schedulers_data: List[ColocatedSchedulerData]):
            for ve, data in enumerate(schedulers_data):
                for request in data.running_queue.requests:
                    request.kv_fusion_rank = 0
                for request in data.swapped_queue:
                    request.kv_fusion_rank = 0
                
                self.scheduler[ve].unaccepted_queue = data.unaccepted_queue
                self.scheduler[ve].running_queue = data.running_queue
                self.scheduler[ve].swapped_queue = data.swapped_queue
                self.scheduler[ve].block_managers = self.block_managers
                self.scheduler[ve].is_fusion_state = False
        
        self.workers = workers
        set_block_manager(block_manager_data)
        set_schedulers(schedulers_data)
    
    def merge_engine(self, rank, workers, schedulers_data, block_manager_data, unaccepted_requests):
        def merge_workers(workers: List[List[ParaWorker]]):
            self.workers.extend(workers)
            self.num_instances += 1
        
        def merge_block_manager(block_manager_data):
            self.block_managers.append(restore_block_manager_from_data(
                    BlockManager(
                        self.stage,
                        self.prefix,
                        self.num_gpu_blocks,
                        self.num_cpu_blocks,
                        self.model_config,
                        self.parallel_config,
                        self.cache_config,
                        self._remote_call_all_workers_async,
                    ),
                    block_manager_data,
                )
            )
        
        def merge_schedulers(rank, follower_schedulers_data: List[ColocatedSchedulerData]):
            # merge old instances' running requests to leader engine's queue
            for data in follower_schedulers_data:
                for request in data.running_queue.requests:
                    request.kv_fusion_rank = rank
                for request in data.swapped_queue:
                    request.kv_fusion_rank = rank

                self.scheduler.append(self._get_scheduler())
                self.scheduler_running.append(False)
                self.scheduler_assigned_chunk.append([])
                self.scheduler_launch_events.append(asyncio.Event())
                self.scheduler[-1].unaccepted_queue.extend(data.unaccepted_queue)
                self.scheduler[-1].running_queue.add_requests(data.running_queue.requests)
                self.scheduler[-1].swapped_queue.extend(data.swapped_queue)
                self.scheduler[-1].block_managers = self.block_managers
        
        merge_workers(workers)
        merge_block_manager(block_manager_data)
        merge_schedulers(rank, schedulers_data)
        self.unaccepted_queue.extend(unaccepted_requests)

    def split_engine(self):
        def split_workers():
            pp_size = self.parallel_config.pipeline_parallel_size
            splitted_workers = []
            for i in range(1, self.num_instances):
                splitted_workers.append(self.workers[i*pp_size: (i+1)*pp_size])
            self.workers = self.workers[:pp_size]
            return splitted_workers
        
        def split_schedulers():
            pp_size = self.parallel_config.pipeline_parallel_size
            splitted_schedulers = []
            for i in range(1, self.num_instances):
                splitted_schedulers.append([
                    self.scheduler[i*pp_size+j].get_data()
                    for j in range(pp_size)
                ])
            self.scheduler = self.scheduler[:pp_size]
            self.scheduler_running = self.scheduler_running[:pp_size]
            self.scheduler_assigned_chunk = self.scheduler_assigned_chunk[:pp_size]
            self.scheduler_launch_events = self.scheduler_launch_events[:pp_size]
            return splitted_schedulers
    
        def split_block_manager():
            for block_manager in self.block_managers:
                block_manager.restore_from_fusion()
            block_managers_data = [self.block_managers[i].get_block_manager_data() for i in range(1, self.num_instances)]
            self.block_managers = [self.block_managers[0]]
            return block_managers_data

        workers = split_workers()
        schedulers_data = split_schedulers()
        block_managers_data = split_block_manager()

        # at split time, the unaccepted queue is not important
        return workers, schedulers_data, block_managers_data

    def set_fusion_state(self, value: bool):
        self.is_fusion_state = value
        for block_manager in self.block_managers:
            block_manager.is_fusion_state = value
        for _scheduler in self.scheduler:
            _scheduler.is_fusion_state = value

    def get_unfinished_requests(self):
        unfinished_requests = len(self.unaccepted_queue)
        for scheduler in self.scheduler:
            unfinished_requests += len(scheduler.running_queue.requests)
            unfinished_requests += len(scheduler.waiting_queue)
            unfinished_requests += len(scheduler.swapped_queue)
        return unfinished_requests
    
    def get_running_requests(self):
        running_requests = 0
        for scheduler in self.scheduler:
            running_requests += len(scheduler.running_queue)
        return running_requests

    def get_used_gpu_blocks(self):
        # total_used_blocks = sum(self._get_block_needed(req.get_num_tokens()) for req in self.unaccepted_queue)
        total_used_blocks = 0
        for scheduler in self.scheduler:
            total_used_blocks += (
                # sum(self._get_block_needed(req.get_num_tokens()) for req in scheduler.unaccepted_queue)
                0
                + sum(self._get_block_needed(req.get_num_tokens()) for req in scheduler.waiting_queue)
                + sum(self._get_block_needed(req.get_num_tokens()) for req in scheduler.swapped_queue)
                + sum(self._get_block_needed(req.get_num_tokens()) for req in scheduler.running_queue.requests)
            )
        return total_used_blocks

    def get_free_gpu_blocks(self):
        if self.is_fusion_state:
            # in fusion state, reserved blocks refer to those replica blocks
            return min([
                bm.get_num_free_gpu_blocks() for bm in self.block_managers
            ])
        else:
            # in non-fusion state, reserved blocks refer to those pipeline blocks
            return sum([
                len(bm.free_gpu_blocks_list) for bm in self.block_managers
            ])
    
    def get_max_base_gpu_blocks(self):
        if self.is_fusion_state:
            return sum([
                bm.get_max_num_base_gpu_blocks() for bm in self.block_managers
            ]) // self.num_instances + sum([bm.get_max_num_reserved_gpu_blocks() for bm in self.block_managers])
        else:
            return sum([
                bm.get_max_num_base_gpu_blocks() for bm in self.block_managers
            ]) + sum(
                [bm.get_max_num_reserved_gpu_blocks() for bm in self.block_managers]
            ) // self.num_instances # pipeline blocks shall be seen as one
    
    def get_num_preemption(self):
        total_num_preemption = 0
        for _scheduler in self.scheduler:
            total_num_preemption += _scheduler.num_preemption
        return total_num_preemption
    
    def vote_for(self, event: Event):
        def oom_check(warn_usage):
            pending_blocks = 0
            pending_blocks += sum([self._get_block_needed(req.get_num_tokens()) for req in self.unaccepted_queue])
            for scheduler in self.scheduler:
                pending_blocks += sum([self._get_block_needed(req.get_num_tokens()) for req in scheduler.unaccepted_queue])
            
            free_blocks = self.get_free_gpu_blocks()
            current_usage = self.sliding_window[-1]
            
            memory_over_warn = current_usage >= warn_usage

            # recommend pending_blocks > free_blocks for BurstGPT dataset
            free_blocks_overflow = pending_blocks >= free_blocks
            # if pending_blocks > 0:
            #     logger.info(f"{free_blocks_overflow=}, {pending_blocks=}, {free_blocks=}, {current_usage=}")

            if memory_over_warn and free_blocks_overflow:
                logger.info(f"Instance {self.instance_id} decide for balloon, {current_usage=}, {free_blocks=}, {pending_blocks=}")
            to_balloon = memory_over_warn and free_blocks_overflow
            
            if to_balloon:
                self.stopped = True
            return to_balloon
        
        def valley_check(warn_usage):
            if not self.bench_config.enable_restore:
                return False

            if self.check_pending_num(): 
                return False

            # warn_usage = 60
            current_usage = self.sliding_window[-1]
            
            memory_decreasing = sum([rate for rate in list(self.rate_window)]) < 0
            memory_safe = current_usage <= warn_usage
            
            to_restore = memory_decreasing and memory_safe
            if to_restore:
                logger.info(f"Instance {self.instance_id} decide for restore, {current_usage=}, {memory_safe=}, {memory_decreasing=}")
            return to_restore
                
            # restore_vote = not self.is_resharding and self.can_reclaim_extend_blocks() and current_usage < warn_usage
            # return memory_increase_rate < 0 and restore_vote
        
        self.vote_epoch += 1
        oom_rate = 95
        valley_rate = 50

        if event == Event.BALLOON:
            # return self.instance_id < 2 and self.vote_epoch >= 50 and self.vote_epoch < 60
            return oom_check(oom_rate)
        elif event == Event.RESTORE:
            # return self.instance_id < 2 and self.vote_epoch >= 100 and self.vote_epoch < 110
            # if the usage of base region < watermark, it means that the requests of current burst is terminating
            return valley_check(valley_rate)
        elif event == Event.INIT:
            num_fusion_requests = 0
            for _scheduler in self.scheduler:
                num_fusion_requests += _scheduler.get_num_fusion_requests()
            return num_fusion_requests == 0
    
    async def trigger_balloon(self, balloon_mode: str, from_restore: bool, instance_properties: list):
        logger.info(f"before balloon, total gpu blocks: {self.block_managers[0].num_base_gpu_blocks}")
        """
        Step 0. merge other instances' properties
        """
        self.stopped = True
        logger.info(f"Leader engine {self.instance_id} merge properties of {len(instance_properties)} instances.")
        for rank, properties in enumerate(instance_properties):
            self.merge_engine(rank + 1, *properties)
        
        """
        Fusion the model on leader and followers' engines.
        Currently, we only support merge continuous instances.
        """
        stale_requests: List[Request] = []
        for _scheduler in self.scheduler:
            stale_requests.extend([req for req in _scheduler.running_queue.requests if not req.fusion_exec])
        logger.info(f"Balloon need to handle {len(stale_requests)} stale requests")
        
        '''
        step 1: switch system execution plan to pipeline
        '''
        self.set_fusion_state(True)
        
        '''
        step 2: convert free blocks into pipeline format
        '''
        # logger.info(f"before balloon, number of free blocks are {self.block_managers[0].max_num_gpu_blocks}")
        if from_restore:
            """RESTORE->BALLOON"""
            for bm in self.block_managers:
                # RESTORE state sees pipeline blocks as reserved blocks
                bm.convert_to_pipeline_format_w_reserve(self.num_instances)
        else:
            """INITIAL->BALLOON"""
            for bm in self.block_managers:
                bm.convert_to_pipeline_format(self.num_instances)
        
        '''
        step 3: drop parameters
        '''
        # TODO make it parallel with each other
        for rank in range(self.num_instances):
            handlers = self._remote_call_partial_workers_async(
                rank * self.parallel_config.pipeline_parallel_size,
                (rank + 1) * self.parallel_config.pipeline_parallel_size,
                "pmm_drop",
                rank,
                self.num_instances,
            )
            free_gpu_blocks = await handlers[-1]
            self.block_managers[rank].add_ex_fusion_blocks(free_gpu_blocks)
            # logger.info(f"add {free_gpu_blocks} ex blocks to instance {rank}")
                
        '''
        step 4: handle stale requests
        '''
        self.is_resharding = [False] * len(self.scheduler)

        if balloon_mode == "recompute":
            for ve in self.scheduler:
                ve.trigger_recompute_for_stale_requests()
        
        elif balloon_mode == "reshard":
            requests: List[Request] = []
            for ve in self.scheduler:
                stale_requests = ve.trigger_reshard_for_stale_requests(live_reshard=False)
                requests.extend(stale_requests)

            # mark all stale requests as new requests
            for request in requests:
                request.fusion_exec = True
                request.once_fusion_exec = True

            start_time = time.perf_counter()
            await self.reshard_blocks(requests)
            end_time = time.perf_counter()

            kv_cache_size = 0
            for request in requests:
                kv_cache_size += len(self.block_managers[request.kv_fusion_rank].get_single_block_table(request.request_id)) * self.block_size_in_bytes
            
            kv_cache_size = kv_cache_size * self.parallel_config.tensor_parallel_size * (self.num_instances - 1) / self.num_instances / 1024 / 1024 / 1024
            logger.info(f"Resharded {len(requests)} requests, total {kv_cache_size:.2f} GB data, cost {end_time - start_time:.2f} seconds")

        elif balloon_mode == "livereshard":
            '''live reshard will be handled during forward'''
            self.is_resharding = [True] * len(self.scheduler)
            # stale_requests = []
            # for sche in self.scheduler:
            #     stale_requests.extend(sche.trigger_reshard_for_stale_requests(live_reshard=True))
            #     for req in stale_requests:
            #         req.kvcache_ready = False
            #     sche.waiting_queue.extend(stale_requests)
            # self.finish_resharding = False

        # if balloon_mode != "livereshard":
        for bm in self.block_managers:
            bm.convert_all_reserved_blocks_into_fusion()
        
        # free_base_blocks = self.block_managers[0].max_num_fusion_gpu_blocks
        # free_extend_blocks = self.block_managers[0].max_num_extend_gpu_blocks
        # logger.info(f"after balloon, number of free blocks are {free_base_blocks=} + {free_extend_blocks=} = {free_base_blocks + free_extend_blocks}")
        '''All stale requests are handled before this line'''
        
        self.state = "balloon"

        engine_loads = [
            len(scheduler.running_queue)
            for scheduler in self.scheduler
        ]
        max_load_ve = engine_loads.index(max(engine_loads))
        min_load_ve = engine_loads.index(min(engine_loads))
        num_request_to_move = (engine_loads[max_load_ve] - engine_loads[min_load_ve]) // 2
            
        requests_to_move = self.scheduler[max_load_ve].running_queue.requests[:num_request_to_move]
        self.scheduler[max_load_ve].running_queue.requests =\
            self.scheduler[max_load_ve].running_queue.requests[num_request_to_move:]
            
        self.scheduler[min_load_ve].running_queue.requests.extend(requests_to_move)        
        
        
        # add new requests immediately to avoid unnecessary waiting
        # decide each request's destination
        # self.balloon_start = time.time()

        logger.info(f"after balloon, total gpu blocks: {self.block_managers[0].num_base_gpu_blocks + self.block_managers[0].num_extend_gpu_blocks}")

        self.stopped = False
        self.sched_config.enable_chunked_prefill = True
        for ve in range(len(self.scheduler)):
            await self._get_next_batch(ve)
        return self.get_free_gpu_blocks()
    
    def trigger_init(self):
        # assert self.get_num_fusion_requests() == 0, "init shall be trigger after all reserved blocks are reclaimed"
        # for block_manager in self.block_managers:
        #     block_manager.restore_from_fusion()

        # self.sched_config.enable_chunked_prefill = self.chunk_enabled_before_balloon
        # self.sched_config.chunked_prefill_budget = self.chunk_size_before_balloon
        
        self.num_instances = 1
        self.state = "initial"
        self.stopped = False
    

    def prepare_reshard_blocks(self, requests: List[Request]) -> Tuple[List[Request], List[Request]]:
        def free_reshard_blocks(request: Request):
            with self.allocate_blocks_lock:
                for i, block_manager in enumerate(self.block_managers):
                    if i == request.kv_fusion_rank:
                        continue
                    try:
                        block_manager.free_blocks(request, only_new_blocks=True)
                    except:
                        pass
        
        def allocate_reshard_blocks(request: Request) -> bool:
            try:
                with self.allocate_blocks_lock:
                    for i, block_manager in enumerate(self.block_managers):
                        if i == request.kv_fusion_rank:
                            continue
                        block_manager.allocate_blocks(request, only_new_blocks=True, live_reshard=False)
            except:
                try:
                    free_reshard_blocks(request)
                except:
                    assert False, "failed to free reshard blocks"
                return False
            else:
                return True
        '''allocate new blocks on all block managers to reshard stale requests'''
        allocated_requests = []
        not_allocated_requests = []
        for request in requests:
            if allocate_reshard_blocks(request):
                allocated_requests.append(request)
            else:
                not_allocated_requests.append(request)
        return allocated_requests, not_allocated_requests
    
    async def _reshard_blocks_batched(self, requests: List[Request]):
        """
        Coordinated KVCache exchange (KunServe paper §4.2) used when N DP
        instances are fused into one PP group.

        Layout contract (see `FlashTransformer/src/csrc/model/sota/llama.cc`
        `kv_exchange` for the GPU side):
          - Each (instance i, local pp_rank) worker participates in a single
            NCCL `kv_exchange_comm` of size `num_instances * pp_size`.
            The rank of a worker in that comm is `i * pp_size + pp_rank`.
          - For one call to this method, every peer pair exchanges its stale
            requests' KVCache.  The C++ side groups all send/recvs of one
            post-drop layer into a single `ncclGroupStart/ncclGroupEnd` so the
            2*(g-1) pairwise transfers progress concurrently.
          - `block_table_map[j]` lists the pages belonging to instance j:
              * when j == i, they are pre-fusion local block IDs (the source
                pages to pack and send);
              * when j != i, they are fused IDs allocated on instance i's
                block manager (where received pages must be scattered).
          - We sort `block_table_map[j]` by peer j's pre-fusion local IDs so
            the sender's Phase-1 reads hit contiguous runs in `kv_data` that
            the C++ side coalesces into a single cudaMemcpyAsync.  The SAME
            permutation is applied on every instance so NCCL send/recv pairs
            line up (ith element on every instance refers to the same logical
            page of the same stale request).

        Concurrency note:
          `_reshard_blocks_batched` drives `kv_exchange_comm`, which is shared
          among all workers of a fused group.  Ray serializes RPCs per actor,
          so two concurrent callers (e.g. different virtual engines during
          live-reshard) will be queued at each worker and their NCCL groups
          will run back-to-back rather than interleaved.  The one thing that
          MUST stay true is that every worker observes the same ORDERED
          sequence of `kv_exchange` calls.  Because this method is an
          `async def` driven from a single leader engine, callers awaiting
          it from the same event loop are naturally serialized; DO NOT
          dispatch `kv_exchange` RPCs from multiple leader engines in
          parallel, or the sender/receiver side NCCL pairings will diverge.
        """
        requests_map = []
        pp_size = self.parallel_config.pipeline_parallel_size

        for _ in range(self.num_instances):
            requests_map.append([])
        for request in requests:
            requests_map[request.kv_fusion_rank].append(request)

        if self.num_instances <= 1 or len(requests) == 0:
            for instance in range(self.num_instances):
                self.block_managers[instance].free_blocks_into_fusion_except_batched(
                    requests_map[instance], self.num_instances, instance
                )
            for request in requests:
                request.kvcache_ready = True
            return

        # --- B3 (Python side): compute one sort permutation per sender j,
        # keyed on j's own pre-fusion local IDs.  Applying the same permutation
        # on every instance keeps the logical page-i <-> page-i mapping between
        # peers while letting the sender see its own pages in monotonically
        # increasing order.  With no duplicates (each stale page is owned by
        # one request only), sorting is enough -- no dedup is needed.
        per_peer_perm: List[List[int]] = []
        for j in range(self.num_instances):
            bm_j = self.block_managers[j]
            raw_j: List[int] = []
            for request in requests_map[j]:
                raw_j.extend(bm_j.get_single_block_table(request.request_id))
            # Stable sort on (value, original_index) so ties (shouldn't occur
            # in practice) are broken deterministically.
            perm = sorted(range(len(raw_j)), key=lambda k: (raw_j[k], k))
            per_peer_perm.append(perm)

        # Build (block_table_map, num_base_blocks, num_extend_blocks) per
        # instance, applying `per_peer_perm[j]` to every block_table_map[j].
        per_instance_args = []
        for i in range(self.num_instances):
            bm = self.block_managers[i]
            block_table_map: List[List[int]] = [[] for _ in range(self.num_instances)]
            for j in range(self.num_instances):
                raw: List[int] = []
                for request in requests_map[j]:
                    raw.extend(bm.get_single_block_table(request.request_id))
                perm = per_peer_perm[j]
                block_table_map[j] = [raw[k] for k in perm]
            num_base_blocks = bm.num_base_gpu_blocks // self.num_instances
            num_extend_blocks = bm.num_extend_gpu_blocks
            per_instance_args.append((block_table_map, num_base_blocks, num_extend_blocks))

        # Total payload (logging only).
        kv_cache_size = 0
        for request in requests:
            kv_cache_size += (
                len(self.block_managers[request.kv_fusion_rank]
                    .get_single_block_table(request.request_id))
                * self.block_size_in_bytes
            )
        kv_cache_size = kv_cache_size * (self.num_instances - 1) / self.num_instances

        start_time = time.perf_counter()

        # Dispatch the NCCL-based exchange to every (instance, pp_rank, tp_rank)
        # worker.  Each pp stage runs its own kv_exchange on io_stream; stages
        # are independent because NCCL pairs within `kv_exchange_comm` only
        # communicate with the same pp_rank on other instances
        # (`dst = peer * pp_size + pp_rank`).
        #
        # Batching note: the C++ side issues 2*(g-1) `ncclSend/ncclRecv` per
        # layer inside ONE `ncclGroupStart/ncclGroupEnd`.  Per the paper, the
        # layer granularity is intentionally kept (each chunk is roughly one
        # pipeline stage long) so activation transfers can pre-empt the
        # exchange between layers; we do NOT merge NCCL groups across layers.
        handlers = []
        for i in range(self.num_instances):
            block_table_map, num_base_blocks, num_extend_blocks = per_instance_args[i]
            for pp_rank in range(pp_size):
                global_pp_stage = i * pp_size + pp_rank
                handlers.extend(self._remote_call_partial_workers_async(
                    global_pp_stage,
                    global_pp_stage + 1,
                    "kv_exchange",
                    block_table_map,        # pages_of_instances
                    pp_rank,                # pp_rank (local within instance)
                    pp_size,                # pp_size (local within instance)
                    i,                      # group_rank (instance rank in fused group)
                    self.num_instances,     # group_size (# instances fused)
                    num_base_blocks,        # P  = pre-fusion per-instance
                    num_extend_blocks,      # E  = per-post-drop-layer extend
                ))

        # A4: gather with cooperative cancellation.  If one worker's kv_exchange
        # throws (NCCL error, OOM, etc.), the peer workers are blocked in
        # ncclGroupEnd forever -- `asyncio.gather` by itself does NOT cancel
        # its children on first exception, so we must do it ourselves and
        # then fail loud rather than hang the whole group.
        if len(handlers) > 0:
            try:
                await asyncio.gather(*handlers)
            except Exception as first_exc:
                for h in handlers:
                    if not h.done():
                        h.cancel()
                # Drain the cancellations so we don't leak tasks.
                await asyncio.gather(*handlers, return_exceptions=True)
                logger.error(
                    f"[kv_exchange] a worker failed ({type(first_exc).__name__}: "
                    f"{first_exc}); aborting this reshard batch. The NCCL comm "
                    f"state may now be inconsistent; the engine will stop."
                )
                ray.shutdown()
                raise

        trans_time = time.perf_counter() - start_time
        kv_cache_size_gb = kv_cache_size / 1024 / 1024 / 1024
        logger.info(
            f"[kv_exchange] Resharded {len(requests)} requests, "
            f"total {kv_cache_size_gb:.2f} GB across {self.num_instances} instances, "
            f"NCCL took {trans_time:.2f}s."
        )

        for instance in range(self.num_instances):
            self.block_managers[instance].free_blocks_into_fusion_except_batched(
                requests_map[instance], self.num_instances, instance
            )
        for request in requests:
            request.kvcache_ready = True
    
    async def reshard_blocks(self, requests: List[Request]):
        """
        Drive the initial balloon-time KV reshard.  Each iteration:
          1. carve up to `exchange_batch_size` requests off the queue (so a
             single `_reshard_blocks_batched` call never tries to allocate
             `BIG_SCRATCH = num_layers * N_r * stride_page` bigger than the
             per-instance peer-free budget);
          2. let `prepare_reshard_blocks` allocate peer slots for that chunk,
             whatever can't be placed right now is returned to the queue;
          3. run NCCL exchange for what got placed.
        Without the `exchange_batch_size` cap, `N_r` can approach `P` for a
        fully loaded instance and the C++ `big_scratch` cudaMalloc can OOM
        (see B2 in the audit).  `exchange_batch_size` is already used by
        live-reshard / live-restore; we now honour it here too.
        """
        remaining = list(requests)
        batch_cap = self.balloon_config.exchange_batch_size or 0
        while remaining:
            if batch_cap > 0 and len(remaining) > batch_cap:
                head, tail = remaining[:batch_cap], remaining[batch_cap:]
            else:
                head, tail = remaining, []
            allocated_requests, not_allocated_requests = self.prepare_reshard_blocks(head)
            # Anything that couldn't get peer blocks this round goes back to
            # the front of the queue to retry after we've freed up slots by
            # actually running the exchange.
            remaining = not_allocated_requests + tail
            if len(allocated_requests) > 0:
                await self._reshard_blocks_batched(allocated_requests)
            elif not_allocated_requests:
                # No progress made this iteration -- avoid busy-looping if the
                # peer free-block budget is permanently insufficient.
                logger.warning(
                    f"[reshard_blocks] none of {len(not_allocated_requests)} "
                    f"requests could be placed on peer instances; giving up "
                    f"this batch to avoid an infinite loop"
                )
                break

    async def reclaim_extra_blocks(self):
        # ask scheduler to copy ex blocks to base region if possible
        self.is_reclaiming_ex_blocks = True
        handlers = []
        for block_manager in self.block_managers:
            handlers.append(asyncio.create_task(block_manager.wait_for_ex_blocks_to_return()))
        await asyncio.wait(handlers)
        self.is_reclaiming_ex_blocks = False

    async def pack_blocks_into_base_region(self, requests: List[Request]):
        for block_manager in self.block_managers:
            block_manager.pack_blocks_into_base_region(requests)
    '''
        move ex blocks to base region initiatively
    '''
    # async def relocate_extra_blocks(self, requests: List[Request]):
    #     ex_blocks_maps = [dict() for i in range(len(self.block_managers))]
    #     base_blocks_maps = [dict() for i in range(len(self.block_managers))]

    #     def relocate_extra_blocks_inner(request: Request, instance: int, block_manager: BlockManager):
    #         blocks = copy.deepcopy(block_manager.get_partial_block_table([request.request_id])[0])
    #         ex_blocks = [block for block in blocks if block >= block_manager.num_base_gpu_blocks]
    #         if len(ex_blocks) == 0:
    #             return
    #         try:
    #             base_blocks = block_manager.get_base_region_blocks(request, len(ex_blocks))
    #         except:
    #             return
    #         else:
    #             assert len(base_blocks) == len(ex_blocks), f"num base blocks: {len(base_blocks)} != num ex blocks: {len(ex_blocks)}"
    #             for i, block in enumerate(base_blocks):
    #                 assert block < block_manager.num_base_gpu_blocks, f"base_blocks[{i}]: {block} >= {block_manager.num_base_gpu_blocks}"
    #             ex_blocks_maps[instance][request.request_id] = ex_blocks
    #             base_blocks_maps[instance][request.request_id] = base_blocks

    #     for request in requests:
    #         if request.fusion_exec:
    #             for instance, block_manager in enumerate(self.block_managers):
    #                 relocate_extra_blocks_inner(request, instance, block_manager)

    #     for instance, block_manager in enumerate(self.block_managers):
    #         all_ex_blocks = []
    #         for ex_blocks in ex_blocks_maps[instance].values():
    #             all_ex_blocks.extend(ex_blocks)

    #         all_base_blocks = []
    #         for base_blocks in base_blocks_maps[instance].values():
    #             all_base_blocks.extend(base_blocks)

    #         assert len(all_base_blocks) == len(all_ex_blocks), f"num all base blocks: {len(all_base_blocks)} != num all ex blocks: {len(all_ex_blocks)}"
    #         if len(all_ex_blocks) <= 0:
    #             continue

    #         remote_calls = self._remote_call_partial_workers_async(
    #             instance * self.parallel_config.pipeline_parallel_size,
    #             (instance + 1) * self.parallel_config.pipeline_parallel_size,
    #             "local_copy_blocks",
    #             block_manager.num_base_gpu_blocks + block_manager.num_extend_gpu_blocks,
    #             all_ex_blocks,
    #             all_base_blocks,
    #         )

    #         await asyncio.wait(remote_calls)
    #         # logger.info(f"reclaimed {len(all_ex_blocks)} ex blocks")

    #         for request_id in ex_blocks_maps[instance]:
    #             block_manager.relocate_extra_blocks(request_id, ex_blocks_maps[instance][request_id], base_blocks_maps[instance][request_id])

    async def pmm_restore(self):
        split_calls = self._remote_call_all_workers_async(
            "pmm_restore",
            self.num_instances,
        )

        for handle in split_calls:
            await handle
    
    def restore_instance(self):
        self.set_fusion_state(False)
        
        for block_manager in self.block_managers:
            block_manager.restore_kvblocks()
        
        self.state = "restore"
        self.stopped = False
    
    '''
    below are for request transfer
    '''
    def get_total_demand_ratio(self):
        total_used_blocks = self.get_used_gpu_blocks()
        max_gpu_blocks = self.get_max_base_gpu_blocks()        
        return total_used_blocks / max_gpu_blocks

    def pop_running_requests_transfer(self, is_pp: bool, num_blocks_limit: int):
        num_blocks_limit //= len(self.scheduler)
        requests = []
        for _scheduler in self.scheduler:
            if is_pp:
                requests.append(_scheduler.pop_running_requests_transfer(is_pp, num_blocks_limit))
            else:
                requests.extend(_scheduler.pop_running_requests_transfer(is_pp, num_blocks_limit))
        return requests

    # FIXME: not condider multi pp case
    async def send_requests_balloon_transfer(self, src_group_rank: int, offset: int, dst_group_id: int, requests: List[Request]):
        group_size = self.parallel_config.replica_size

        dst_group_rank = (src_group_rank + offset) % group_size # rank inside group
        dst_rank = dst_group_id * group_size + dst_group_rank   # rank inside world

        layer_per_instance = self.model_config.get_num_layers(self.parallel_config) // group_size
        layer_start = dst_group_rank * layer_per_instance
        layer_end = (dst_group_rank + 1) * layer_per_instance

        # logger.info(f"[send_requests_balloon_transfer] {src_group_rank=} {dst_group_rank=} {dst_rank=} {len(requests)=}")
        block_table = self.block_managers[0].get_partial_block_table(requests)

        handlers = self._remote_call_all_workers_async(
            "send_blocks",
            dst_rank,
            layer_start,
            layer_end,
            self.block_managers[0].num_base_gpu_blocks,
            block_table,
        )
        await asyncio.gather(*handlers)

    # FIXME: not condider multi pp case
    async def recv_requests_balloon_transfer(self, src_group_rank: int, offset: int, src_group_id: int, requests: List[Request]):
        group_size = self.parallel_config.replica_size
        pp_size = self.parallel_config.pipeline_parallel_size

        dst_group_rank = (src_group_rank + offset) % group_size # rank inside group
        src_rank = src_group_id * group_size + src_group_rank   # rank inside world

        # logger.info(f"[recv_requests_balloon_transfer] {src_group_rank=} {dst_group_rank=} {src_rank=} {len(requests)=} {len(self.workers)=}")
        layer_per_instance = self.model_config.get_num_layers(self.parallel_config) // group_size
        block_table = self.block_managers[dst_group_rank].get_partial_block_table(requests)

        handlers = self._remote_call_partial_workers_async(
            dst_group_rank * pp_size,
            (dst_group_rank + 1) * pp_size,
            "recv_blocks",
            src_rank,
            0,
            layer_per_instance,
            self.block_managers[0].num_base_gpu_blocks + self.block_managers[0].num_extend_gpu_blocks,
            block_table,
        )
        await asyncio.gather(*handlers)

    def free_blocks_balloon_transfer(self, requests: List[Request]):
        for request in requests:
            request.fusion_exec = False
            self.block_managers[0].free_blocks(request)
            request.fusion_exec = True

    # FIXME: not condider multi pp case
    async def send_requests_restore_transfer(self, src_group_rank: int, offset: int, dst_group_id: int, requests: List[Request]):
        group_size = self.parallel_config.replica_size
        pp_size = self.parallel_config.pipeline_parallel_size

        dst_group_rank = (src_group_rank + offset) % group_size # rank inside group
        dst_rank = dst_group_id * group_size + dst_group_rank   # rank inside world

        layer_per_instance = self.model_config.get_num_layers(self.parallel_config) // group_size

        block_table = self.block_managers[src_group_rank].get_partial_block_table(requests)

        handlers = self._remote_call_partial_workers_async(
            src_group_rank * pp_size,
            (src_group_rank + 1) * pp_size,
            "send_blocks",
            dst_rank,
            0,
            layer_per_instance,
            self.block_managers[src_group_rank].num_base_gpu_blocks,
            block_table,
        )
        await asyncio.gather(*handlers)

    # FIXME: not condider multi pp case
    async def recv_requests_restore_transfer(self, src_group_rank: int, offset: int, src_group_id: int, requests: List[Request]):
        group_size = self.parallel_config.replica_size

        src_rank = src_group_id * group_size + src_group_rank   # rank inside world

        layer_per_instance = self.model_config.get_num_layers(self.parallel_config) // group_size
        layer_start = src_group_rank * layer_per_instance
        layer_end = (src_group_rank + 1) * layer_per_instance

        block_table = self.block_managers[0].get_partial_block_table(requests)

        handlers = self._remote_call_all_workers_async(
            "recv_blocks",
            src_rank,
            layer_start,
            layer_end,
            self.block_managers[0].num_base_gpu_blocks,
            block_table,
        )
        await asyncio.gather(*handlers)

    def free_blocks_restore_transfer(self, requests: List[List[Request]]):
        for reqs in requests:
            for req in reqs:
                req.fusion_exec = True
                for bm in self.block_managers:
                    bm.free_blocks(req)
                req.fusion_exec = False

    """
    call func_name asynchronously on all workers, return the futures immediately
    """
    def _remote_call_all_workers_async(self, func_name: str, *args):
        handlers = []
        for stage in self.workers:
            for worker in stage:
                ray_func = getattr(worker, func_name).remote(*args)
                handlers.append(asyncio.wrap_future(ray_func.future()))
        return handlers

    def _remote_call_partial_workers_async(
        self, start_worker_id, end_worker_id, func_name: str, *args
    ):
        handlers = []
        # logger.info(f"access worker from {start_worker_id} to {end_worker_id}")
        for stage in self.workers[start_worker_id:end_worker_id]:
            for worker in stage:
                ray_func = getattr(worker, func_name).remote(*args)
                handlers.append(asyncio.wrap_future(ray_func.future()))
        return handlers
    
    def _allocate_blocks(self, request: Request):
        if request.fusion_exec:
            for block_manager in self.block_managers:
                block_manager.allocate_blocks(request)
        else:
            self.block_managers[request.kv_fusion_rank].allocate_blocks(request)
    
    def _free_blocks(self, request: Request):
        if request.fusion_exec:
            for block_manager in self.block_managers:
                block_manager.free_blocks(request)
        else:
            self.block_managers[request.kv_fusion_rank].free_blocks(request)
    
    # FIXME: not condider multi pp case
    def append_and_allocate_requests(self, index: int, requests: List[Request]):
        self.scheduler[index].running_queue.add_requests(requests)
        for request in requests:
            self._allocate_blocks(request)

    def _free_request_resources(self, request: Request) -> None:
        self._free_blocks(request)
        self._remote_call_all_workers_async("clear_request_resource", request.request_id)
    
    def _get_block_needed(self, length: int):
        block_size = self.block_managers[0].cache_config.block_size
        return (length + block_size - 1) // block_size
    
    def get_min_cost_scheduler(self):
        '''add request to the scheduler with least unfinished requests'''
        costs = [
            scheduler.get_num_unfinished_requests()
            for scheduler in self.scheduler
        ]
        index = costs.index(min(costs))
        return self.scheduler[index]
    
    def add_request(self, request: Request) -> None:
        block_size = self.block_managers[0].cache_config.block_size
        input_tokens = request.get_input_len()
        request.output_capacity = (
            self._get_block_needed(input_tokens) * block_size - input_tokens
        )

        if self.bench_config.enable_look_ahead:
            # we will add request via our balance_prefill function
            self.unaccepted_queue.append(request)
        else:
            scheduler = self.get_min_cost_scheduler()
            scheduler.unaccepted_queue.append(request)
    
    def add_requests(self, requests: List[Request]) -> None:
        block_size = self.block_managers[0].cache_config.block_size
        for request in requests:
            input_tokens = request.get_input_len()
            request.output_capacity = (
                self._get_block_needed(input_tokens) * block_size - input_tokens
            )
        self.unaccepted_queue.extend(requests)

    def add_batch_requests(self, requests: List[Request], prefix_len=0) -> None:
        for i, request in enumerate(requests):
            self.add_request(request)

            if len(requests) == 1:
                request.prompt_token_ids += [random.randint(0, 16384)] * prefix_len
                request.prefill_begin_index = prefix_len

    def check_pending_num(self):
        pending_in_scheduler = sum([
            len(scheduler.unaccepted_queue) + len(scheduler.long_unaccepted_queue)
            for scheduler in self.scheduler
        ])
        return len(self.unaccepted_queue) + pending_in_scheduler
    
    def pop_unaccepted_queue(self, virtual_engine: int) -> Request:
        if len(self.scheduler[virtual_engine].unaccepted_queue):
            return self.scheduler[virtual_engine].unaccepted_queue.popleft()

        if len(self.unaccepted_queue) == 0:
            return None
        return self.unaccepted_queue.popleft()

    def fetch_unaccepted_queue(self, virtual_engine: int, schedule_global_queue: bool) -> Request:
        if len(self.scheduler[virtual_engine].unaccepted_queue):
            return self.scheduler[virtual_engine].unaccepted_queue[0]
        
        if not schedule_global_queue:
            return None

        if len(self.unaccepted_queue) == 0:
            return None
        return self.unaccepted_queue[0]
    
    def pop_long_unaccepted_queue(self, virtual_engine: int) -> Request:
        if len(self.scheduler[virtual_engine].long_unaccepted_queue) == 0:
            return None
        return self.scheduler[virtual_engine].long_unaccepted_queue.popleft()
    
    def fetch_long_unaccepted_queue(self, virtual_engine: int) -> Request:
        if len(self.scheduler[virtual_engine].long_unaccepted_queue) == 0:
            return None
        return self.scheduler[virtual_engine].long_unaccepted_queue[0]
    
    def pop_decode_queue(self) -> List[Request]:
        if len(self.decode_queue) == 0:
            return None
        return self.decode_queue.popleft()
    
    def fetch_decode_queue(self) -> List[Request]:
        if len(self.decode_queue) == 0:
            return None
        return self.decode_queue[0]

    def free_blocks_batched(self, requests: List[Request]):
        if len(requests) == 0:
            return
        for request in requests:
            self._free_blocks(request)
    
    '''
    add request to the current batch, and allocate blocks for it
    '''
    def _try_add_to_cur_batch(self, virtual_engine: int, request: Request, is_long_req: bool=False) -> bool:
        # try allocate blocks immediately! should not fail in this function
        self._allocate_blocks(request)
        # logger.info(f"request {request.request_id} is poped out from unaccepted queue by _try_add_to_cur_batch, {request.prefill_begin_index=}, {request.prefill_end_index=}, tokens: {len(request.prompt_token_ids)}")
        if is_long_req:
            self.pop_long_unaccepted_queue(virtual_engine)
        else:
            self.pop_unaccepted_queue(virtual_engine)
        self.scheduler[virtual_engine].add_to_cur_batch(request)
        return True
    
    def _add_to_scheduler_chunk_queue(self, virtual_engine: int, request: Request):
        self._allocate_blocks(request)
        # logger.info(f"request {request.request_id} is poped out from unaccepted queue by _add_to_scheduler_chunk_queue")
        self.pop_unaccepted_queue(virtual_engine)
        self.scheduler_assigned_chunk[virtual_engine].append(request)
        return True
        
    '''
    check whether exceed the limit of the batch size, the number of tokens and free blocks
    '''
    def _check_add_to_cur_batch(self, virtual_engine: int, request: Request, is_waiting: bool = False) -> Tuple[int, int, int]:
        num_blocks_needed = self._get_block_needed(request.get_num_tokens())
        kv_rank = virtual_engine // self.parallel_config.pipeline_parallel_size # index of instance
        
        request.fusion_exec = self.is_fusion_state
        request.kv_fusion_rank = kv_rank
        
        '''1. check whether exceed the limit of the number of free blocks'''
        if self.is_fusion_state:
            num_overflow_blocks = max(
                num_blocks_needed - bm.get_num_free_gpu_blocks()
                for bm in self.block_managers
            )
        else:
            num_overflow_blocks = num_blocks_needed - self.block_managers[kv_rank].get_num_free_gpu_blocks()
        
        '''2. check whether exceed the limit of the batch size'''
        num_batch_requests = 0
        num_batch_tokens = 0
        scheduler = self.scheduler[virtual_engine]
        
        if request.is_context_stage() and (not self.sched_config.enable_chunked_prefill or self.prefill_only_chunk):
            # vllm use prefill first scheduling, which will only schedule prefill requests when there is new prefill requests
            cur_batch_requests = scheduler.running_queue.get_context_requests()
        else:
            cur_batch_requests = scheduler.running_queue.requests
        
        num_batch_requests = len(cur_batch_requests)
        num_batch_tokens = sum([req.get_num_input_tokens() for req in cur_batch_requests])
        
        # the batch limit shall targeted at only prefill requests when scheduling new requests
        num_overflow_requests = (
            num_batch_requests + 1 - self.sched_config.max_batch_size
        )
        
        '''3. check whether exceed the limit of the number of tokens'''
        # we will not limit token numbers when there is only one requests
        token_limit = self.sched_config.chunked_prefill_budget if self.sched_config.enable_chunked_prefill \
            else self.sched_config.max_tokens_per_batch
        
        num_overflow_tokens = (
            num_batch_tokens + request.get_num_input_tokens() - token_limit
        )

        if num_overflow_blocks < 0 or is_waiting:
            num_overflow_blocks = 0
        if num_overflow_requests < 0:
            num_overflow_requests = 0
        if num_overflow_tokens < 0:
            num_overflow_tokens = 0

        return (num_overflow_blocks, num_overflow_requests, num_overflow_tokens)
    
    def balance_decode(self, virtual_engine: int):
        """
        Lookahead scheduling -- decode-side cross-ve rebalancing.

        Simplified policy: only when "this ve == the currently most-loaded ve",
        move (max - min) / 2 decode requests from the busiest ve to the least
        busy one. Each round performs a single pairwise rebalance; global
        convergence is achieved across multiple rounds. This is sufficient for
        PP=2; for PP>2 convergence still happens but takes more rounds.

        WARNING -- KV migration caveat (must read):
        The current implementation only moves the Request object from one
        ve.running_queue to another; it does NOT update request.kv_fusion_rank
        and it does NOT migrate KV blocks. Consequently:
          - Fusion KV (the two ves share a single KV replica)   : correct.
          - Non-fusion PP (each ve owns its own KV replica)     : INCORRECT;
            decode would look up KV on the wrong block_manager. Do not enable.
        This function should only be called when self.is_fusion_state == True,
        or the KV-migration logic must be extended before generalising.
        """
        # running_prefills = sum([
        #     len(scheduler.running_queue.get_context_requests())
        #     for scheduler in self.scheduler
        # ])
        # if running_prefills > 0:
        #     return

        decode_requests = [
            scheduler.running_queue.get_decode_requests()
            for scheduler in self.scheduler
        ]
        
        engine_decode_loads = [
            len(decodes) for decodes in decode_requests
        ]

        max_load_ve = engine_decode_loads.index(max(engine_decode_loads))
        min_load_ve = engine_decode_loads.index(min(engine_decode_loads))
        if max_load_ve == virtual_engine:
            prefills = self.scheduler[virtual_engine].running_queue.get_context_requests()
            decodes = decode_requests[virtual_engine]
            num_request_to_move = (engine_decode_loads[max_load_ve] - engine_decode_loads[min_load_ve]) // 2
            requests_to_move = decodes[0:num_request_to_move]
            
            self.scheduler[virtual_engine].running_queue.requests = (
                prefills + decodes[num_request_to_move:]
            )
            self.scheduler[min_load_ve].running_queue.requests.extend(requests_to_move)

    def balance_prefill(self, virtual_engine):
        """
        ==========================================================================
        Lookahead scheduling -- Stage 1: cross-ve prefill placement (balance)
        ==========================================================================

        Motivation
        ----------
        With pipeline parallelism there are N virtual engines (ve), one per
        micro-batch. The micro-batches flow back-to-back through the pipeline
        stages. If their per-step execution times diverge, the pipeline
        develops bubbles. The kunserve paper solves this with a precise cost
        model plus recursive divide-and-conquer that requires hardware
        profiling. This file implements a much simpler two-stage approximation
        that does NOT require profiling:

            Stage 1: balance_prefill()         <-- this function
                Route newly-arrived prefill requests across the N ves so that
                each ve has a similar total number of pending prefill tokens
                and long requests are spread out rather than piled onto one ve.

            Stage 2: schedule_packed_prefill() <-- see its docstring below
                When actually assembling one batch on a ve, chunk the long
                requests so their per-step token count resembles that of a
                typical short request in the same batch, keeping the step
                latencies of the two ves close.

        Algorithm
        ---------
        1. Drain self.unaccepted_queue (the global, not-yet-routed queue) and
           sort by input_len in descending order (Longest-Job-First). Placing
           the large items first is the standard greedy intuition for multiway
           partitioning and balances better than FCFS.
        2. For each request, push it onto the ve whose pending prefill token
           count is currently smallest.
        3. If enable_packing is True, use the absolute threshold
               long_threshold_abs = chunked_prefill_budget * long_threshold
           to split long/short: long requests go to the ve's
           long_unaccepted_queue (to be chunked in Stage 2); short requests
           go to the normal unaccepted_queue (dispatched whole in Stage 2).
        4. Optional reorder:
             - enable_reorder=False -> order by arrival_time (FCFS).
             - enable_reorder=True  -> if the ve was empty before this round,
               order by input_len (SJF). Restricting SJF to the "empty queue
               + fresh batch" case reduces average TTFT without starving long
               jobs.
        5. If enable_decode_balance is True, additionally call balance_decode()
           to move half the imbalance of decode requests from the busiest ve
           to the least busy one. See balance_decode() for the KV-migration
           caveat.

        Parameters
        ----------
        enable_look_ahead : bool  (BenchConfig)
            Master switch. When False, requests bind to a ve at add_request()
            time via get_min_cost_scheduler() and no cross-ve rebalance is
            performed; this function is not invoked.

        enable_packing : bool  (BenchConfig)
            Whether to enable long/short partitioning plus Stage 2's
            "only chunk longs" policy.
              - False: go through schedule_prefill() (Sarathi-style chunked
                       prefill, which chunks purely by token budget without
                       distinguishing long from short).
              - True : go through schedule_packed_prefill() (recommended for
                       mixed long/short workloads).

        long_threshold (alpha) : float in (0, 1]  (BenchConfig)
            Relative threshold for long/short partitioning. A request with
            input_len > chunked_prefill_budget * alpha is treated as long.
            See "long_threshold (alpha) derivation and tuning" below.

        chunked_prefill_budget : int  (SchedConfig)
            The per-step token budget of one micro-batch, also the denominator
            of alpha. Pick according to the GPU:
              - H100: 4096 ~ 8192
              - A100: 1024 ~ 4096
            Too small -> kernel-launch overhead dominates. Too large -> TBT of
            concurrent decodes suffers because one large prefill stalls a step.

        enable_decode_balance : bool  (BenchConfig)
            Whether balance_prefill additionally triggers cross-ve migration
            of decode requests.
              - Long-output / chat workloads: turn on.
              - Pure short-prefill stress tests (BurstGPT): does not matter.
            WARNING: balance_decode() currently only moves the Request object
            and does NOT migrate KV blocks. It is correct only under fusion KV
            (both ves share one KV replica). Non-fusion PP requires extending
            the KV migration logic first.

        enable_reorder : bool  (BenchConfig)
            If True, the "empty before this round" ve is reordered into SJF
            to reduce average TTFT; otherwise FCFS is kept. This conservative,
            localised SJF does not starve long jobs.

        =========================================================================
        long_threshold (alpha) derivation and tuning
        =========================================================================
        This is the main knob to tune. The derivation below explains why.

        --- (1) Lower bound: alpha must be large enough that a chunked long
                request can reach the share of "one typical short request" ---

        A hard upper bound on chunk_size follows directly from Stage 2's code
        (see schedule_packed_prefill: the `mean_input_len` computation and
        `num_avail_tokens = min(free_budget, mean_input_len)`):

            chunk_size  <=  mean_short  <=  alpha * budget                     (*)

        The first inequality holds because `chunk_size` is taken from
        min(..., mean_short); the second holds because `mean_short` is the
        mean of values each bounded by `alpha * budget`. This is a purely
        algebraic consequence of the code and is independent of the workload
        distribution.

        To make the chunked long request "look like a typical short" so that
        the two micro-batches have similar per-step token counts, we want

            chunk_size  ~=  budget / E[B]

        where E[B] is the steady-state expected number of prefill requests
        per batch. E[B] is a system-level quantity that can be measured
        directly as an EMA of `len(running_queue.get_context_requests())`;
        it does not require any knowledge of the input-length distribution.

        Combining (*) with the target gives the hard lower bound on alpha:

            alpha * budget  >=  budget / E[B]   =>   alpha  >=  1 / E[B]

        In practice one leaves a multiplicative slack k above 1/E[B] because
        mean_short is typically strictly less than alpha * budget (unless the
        distribution is concentrated near the threshold). Picking k:

            k = 1     distribution hugs the threshold
                      (mean_short ~ alpha * budget)             -> alpha = 1 / E[B]
            k = 2     distribution roughly uniform on
                      [0, alpha * budget]                       -> alpha = 2 / E[B]
            k >> 2    heavy long-tail, shorts pile near 0       -> enlarge alpha

        Being off by a factor of two in k is usually harmless because the
        clip in (2)(3) below will dominate the final value. Start with
        k = 2 (alpha = 2 / E[B]); bump to 3 ~ 4 / E[B] for heavily long-
        tailed workloads (e.g. BurstGPT).

        --- (2) Upper bound: alpha <= 0.5 ---

        If alpha > 0.5, the threshold exceeds half the budget and two "short"
        requests can no longer fit together in the same batch (e.g.
        1600 + 1600 > 2048). Long/short partitioning effectively degenerates
        and behaves as if packing were disabled. So alpha must not exceed 0.5.

        --- (3) Hard lower bound: alpha >= FLOOR_CHUNK / budget ---

        See the FLOOR_CHUNK section below. If alpha * budget is smaller than
        the kernel's efficiency knee, Stage 2 will chunk long requests below
        that knee and kernel-launch overhead will eat all the balance wins.

        --- (4) Online tuning loop ---

            bs_ema     = EMA over ~100 steps of
                         len(running_queue.get_context_requests())
            alpha_want = k / bs_ema                 # k=2 by default;
                                                    # 3~4 for long-tail workloads
            alpha      = clip(alpha_want,
                              FLOOR_CHUNK / chunked_prefill_budget,   # lower
                              0.5)                                    # upper

            if per-ve step latency gap > 15%     ->  lower alpha one notch.
            if long-request TTFT regression > 2x ->  raise alpha one notch.

        This procedure depends only on system-internal observables
        (running_queue length, per-ve latency) -- no offline distribution
        scan is required.

        =========================================================================
        FLOOR_CHUNK: the smallest sensible chunk size
        =========================================================================
        FLOOR_CHUNK in the alpha formula above is the "chunk too small and
        throughput starts dropping" knee.

        The current implementation does NOT use FLOOR_CHUNK explicitly (Stage 2
        uses mean_short directly, with no floor). FLOOR_CHUNK appears here
        only as the clip lower bound on alpha, to avoid the degenerate path
        "alpha tiny -> mean_short tiny -> long requests chunked too finely".

        Empirical values below come from profiling FlashAttention / FlashInfer
        kernels under dense attention + chunked prefill on this project's
        flash_backend:

            GPU            FLOOR_CHUNK
            A100 80GB      ~256
            H100           ~512

        These are empirical values that depend on both hardware and kernel
        implementation, not theoretical optima. When moving to a different
        GPU (e.g. H200 / B200 / newer devices which we have not profiled) or
        switching the underlying attention kernel (e.g. hand-written Triton,
        a different page_size, a different KV block_size), re-profile before
        trusting these numbers. Using stale values will either (a) pin the
        lower-bound clip too high, forcing alpha large and hurting balance,
        or (b) allow chunks that are too small and drop kernel utilisation
        by 50% or more.

        --- How to profile FLOOR_CHUNK yourself ---

        Fix a representative prefill request (e.g. input_len=4096, no
        prefix). Run chunked prefill sweeping
        chunk_size in {64, 128, 256, 512, 1024, 2048}. Record tokens/sec:

            tokens/sec(chunk_size) = chunk_size / avg_step_latency(chunk_size)

        A typical curve rises roughly linearly per doubling until it
        saturates; after the knee the marginal gain tapers off. The
        chunk_size at that knee IS FLOOR_CHUNK.

        --- What FLOOR_CHUNK affects ---

        It affects exactly one thing: the worst admissible point of the
        "long-request TTFT vs PP balance" trade-off.

          - Larger FLOOR_CHUNK -> alpha's lower bound is larger -> long
            requests are chunked more coarsely -> shorter TTFT but higher
            imbalance risk.
          - Smaller FLOOR_CHUNK -> alpha can go lower -> finer chunking ->
            better balance but longer TTFT for long requests, possibly
            falling into the kernel's inefficient region.

        Conservative advice: take the profiled knee and add 20% ~ 50% before
        using it as FLOOR_CHUNK, to avoid sitting just below the knee in the
        partially-inefficient region.

        =========================================================================
        Per-workload tuning cookbook
        =========================================================================
        * Homogeneous short requests (RAG-like, 1~2k):
            enable_look_ahead=True, enable_packing=False, enable_reorder=False
            (Nothing is "long", so packing has nothing to chunk.)

        * Mixed long/short (ShareGPT, 100~32k):
            enable_look_ahead=True, enable_packing=True, enable_reorder=True
            Start alpha at 2 / E[B] clipped to [FLOOR_CHUNK/budget, 0.5].
            Iterate based on the observed per-ve latency gap.

        * Heavy long-tail (BurstGPT, P99 >> P90):
            Same as above, but take k = 3 ~ 4 (shorts concentrate near 0, so
            mean_short is well below alpha * budget and a larger alpha is
            needed to land chunk_size near the target). Enable
            enable_decode_balance alongside.

        * Decode-dominated (long-output chat):
            enable_decode_balance=True is the most important switch; packing
            rarely matters because prefills are short.

        =========================================================================
        Gap versus the paper's lookahead algorithm
        =========================================================================
        - Paper: recursive divide-and-conquer with cost-optimal splits at
          every level; requires an accurate per-chunk cost model and hardware
          profiling.
        - This implementation: one-shot greedy two-way partition (Stage 1)
          plus a fixed chunk-size rule (Stage 2). On perfectly homogeneous
          batches (e.g. all ~2k requests) it degenerates into the Sarathi
          baseline and therefore forgoes the paper's advertised gains on
          that regime; on mixed long/short workloads the two are close in
          practice.
        """
        enable_look_ahead = self.bench_config.enable_look_ahead
        long_threshold = self.sched_config.chunked_prefill_budget * self.bench_config.long_threshold

        assert enable_look_ahead, "enable_look_ahead must be True when you want to use balance_prefill"

        # plan the prefill load to ensure balance (see docstring above for the two-stage design)
        pending_prefills = []
        while len(self.unaccepted_queue):
            request = self.unaccepted_queue.popleft()
            pending_prefills.append(request)

        # sort in long job first to minimize diff
        if enable_look_ahead:
            prefills = deque(sorted(pending_prefills, key=lambda x: x.get_input_len(), reverse=True))
        else:
            prefills = deque(pending_prefills)

        # decode balance is not necessary for short prefill workloads like burstgpt
        if self.bench_config.enable_decode_balance:
            self.balance_decode(virtual_engine)

        if len(prefills) == 0:
            return
        
        tokens_in_queue = [
            ve.num_pending_prefill_tokens + ve.running_queue.get_num_input_tokens()
            for ve in self.scheduler
        ]
        is_empty_before_sched = all([
            len(scheduler.unaccepted_queue) == 0
            for scheduler in self.scheduler
        ])
        
        while len(prefills):
            request = prefills.popleft()
            min_load_ve = tokens_in_queue.index(min(tokens_in_queue))
            tokens_in_queue[min_load_ve] += request.get_input_len()
            # add the request to the head, so that it is a natural a SJF

            if self.bench_config.enable_packing:
                # judge whether request is a long one
                if request.get_input_len() > long_threshold:
                    self.scheduler[min_load_ve].long_unaccepted_queue.append(request)
                else:
                    self.scheduler[min_load_ve].unaccepted_queue.append(request)
            else:
                self.scheduler[min_load_ve].unaccepted_queue.appendleft(request)
        
        if enable_look_ahead:
            if not self.bench_config.enable_reorder:
                # first come first serve scheduling
                for ve in self.scheduler:
                    ve.unaccepted_queue = deque(sorted(ve.unaccepted_queue, key=lambda x: x.arrival_time))
            else:
                if is_empty_before_sched:
                    # short job first scheduling, but avoid infinite preemption of short jobs
                    logger.info(f"reorder unaccepted queue into SJF, since no previous requests exist.")
                    for ve in self.scheduler:
                        ve.unaccepted_queue = deque(sorted(ve.unaccepted_queue, key=lambda x: x.get_input_len()))
       

    def schedule_prefill(self, virtual_engine: int, enable_zigzag_prefill: bool = False, schedule_global_queue: bool = True):
        chunk_size = 0
        curr_batch = self.scheduler[virtual_engine].running_queue
        num_prefill_chunks = 1
        if enable_zigzag_prefill:
            pending_requests = len(self.scheduler[virtual_engine].unaccepted_queue)
            running_decodes = len(curr_batch.get_decode_requests())
            running_prefills = len(curr_batch) - running_decodes
            
            num_prefill_chunks = min(pending_requests + running_prefills, self.bench_config.zigzag_chunk_limit)
            if num_prefill_chunks:
                chunk_size = (self.sched_config.chunked_prefill_budget - running_decodes) // num_prefill_chunks

        """
        set next chunk for requests if required
        """
        free_token_budget = 0
        enable_chunked_prefill = self.sched_config.enable_chunked_prefill
        if enable_chunked_prefill:
            free_token_budget = (
                self.sched_config.chunked_prefill_budget - running_decodes - num_prefill_chunks * chunk_size
            ) if chunk_size \
                else (self.sched_config.chunked_prefill_budget - curr_batch.get_num_input_tokens())
            for request in curr_batch.requests:
                if request.have_follow_up_chunk():
                    if chunk_size:
                        # the unused tokens will be allocated automatically to the next request
                        free_token_budget = request.set_next_chunk(chunk_size + free_token_budget)
                    else:
                        free_token_budget = request.allocate_next_chunk(free_token_budget)

        
        blocks_overflow = False
        while (request := self.fetch_unaccepted_queue(virtual_engine, schedule_global_queue)) is not None:

            (num_overflow_blocks, num_overflow_requests, num_overflow_tokens) = \
                self._check_add_to_cur_batch(virtual_engine, request)

            # if (num_overflow_blocks, num_overflow_requests, num_overflow_tokens) != (0, 0, 0):
            #     logger.info(f"queuing happens due to {num_overflow_blocks=}, {num_overflow_requests=}, {num_overflow_tokens=}")

            if (num_overflow_blocks, num_overflow_requests, num_overflow_tokens) == (0, 0, 0):

                # reserved for zigzag chunked-prefill
                if chunk_size:
                    free_token_budget = request.set_chunk(chunk_size + free_token_budget)
                    
                if not self._try_add_to_cur_batch(virtual_engine, request):
                    # give up if allocate blocks failed
                    break
            
            elif (enable_chunked_prefill
                and num_overflow_requests == 0 
                and num_overflow_blocks == 0
            ): # decode-first, but will schedule prefill through request chunking
                num_avail_tokens = request.get_num_input_tokens() - num_overflow_tokens
                
                # not even one token is available, just give up
                if num_avail_tokens == 0:
                    break
                    
                if chunk_size:
                    if num_avail_tokens < chunk_size:
                        break
                    free_token_budget = request.set_chunk(chunk_size + free_token_budget)
                else:
                    logger.info(f"set chunksize of {num_avail_tokens=} for request {request.request_id}")
                    request.set_chunk(num_avail_tokens)

                self._try_add_to_cur_batch(virtual_engine, request)
                if request.have_follow_up_chunk(): # it means current request has used up all token budget
                    break
            
            else: # just give up scheduling when there are too many requests in the batch
                if num_overflow_blocks > 0:
                    blocks_overflow = True
                # logger.info(f"[queueing] overflow in blocks: {num_overflow_blocks}, overflow in requests: {num_overflow_requests}, overflow in tokens: {num_overflow_tokens}")
                break
        ve = self.scheduler[virtual_engine]
        ve.num_pending_prefill_tokens = sum([req.get_num_tokens() for req in ve.unaccepted_queue]) + ve.running_queue.get_unfinished_prefill_tokens()
        return blocks_overflow
    
    def schedule_packed_prefill(self, virtual_engine: int, schedule_global_queue=True):
        """
        ==========================================================================
        Lookahead scheduling -- Stage 2: intra-batch chunk sizing ("only chunk longs")
        ==========================================================================

        Preconditions
        -------------
        Stage 1 (balance_prefill) has already routed prefills into this ve's
        two queues:
            ve.unaccepted_queue       -- short requests
                                         (input_len <= long_threshold_abs)
            ve.long_unaccepted_queue  -- long requests
                                         (input_len >  long_threshold_abs)
        Stage 1 has also ensured the per-ve token totals are roughly balanced.

        Algorithm (called once per ve per step)
        ---------------------------------------
        1. Compute mean_input_len over the short requests currently in play:
           the ongoing prefills plus the queued shorts in ve.unaccepted_queue.
           This approximates the typical short-request size in this round.
        2. Pop (up to) one long request from long_unaccepted_queue and
           set_chunk it to
               chunk_size = min(free_token_budget, mean_input_len)
           so that it "looks like" another typical short request sharing the
           batch.
        3. Continue packing short requests from unaccepted_queue whole. Only
           the last one may be chunked, and only when it overflows the token
           budget (Sarathi behaviour: we never proactively chunk a short
           request for the sake of balance).

        Why chunk_size = mean(short)?
        -----------------------------
        The per-step time of a PP micro-batch is approximately linear in the
        total token count plus a quadratic attention term. To match the two
        ves' step times we need both (i) similar total tokens (ensured by
        Stage 1) and (ii) similar per-request chunk sizes (the quadratic
        term is sensitive to the largest chunk in a batch). Chunking the
        long request to mean(short) keeps every prefill in the batch at the
        same order of magnitude, so the two ves' per-step latencies do not
        easily diverge.

        Alternatives considered:
          - chunk_size = max(short)
              Pro: no short is ever collaterally chunked.
              Con: inflates total batch tokens, breaking Stage 1's balance.
          - chunk_size = fixed (budget / K)
              Simple but workload-oblivious; cuts too coarsely or too finely
              in long-tail regimes.
          - The paper's exact cost-balance
              Requires hardware profiling plus a cost model; high engineering
              overhead.

        Known limitations and possible extensions (read this when tuning)
        -----------------------------------------------------------------
        These three issues are not handled in the current implementation; if
        you observe abnormal TTFT amplification, check here first.

        (a) No floor. If short requests happen to be tiny (e.g.
            mean_short=32), a 16k-token long request gets sliced ~500 ways
            and its TTFT is amplified by orders of magnitude -- exactly the
            "over-chunking for the sake of balance" pitfall. The current
            implementation constrains chunk_size indirectly, from upstream,
            via the clip `alpha >= FLOOR_CHUNK / budget` described in
            balance_prefill's docstring (because chunk_size <= alpha * budget);
            this is an indirect constraint, not an in-function floor.
            Direct hardening (not enabled):
                FLOOR_CHUNK = 256 (A100) / 512 (H100)
                    # see the FLOOR_CHUNK section in balance_prefill's docstring
                chunk_size = max(chunk_size, FLOOR_CHUNK)

        (b) No "long request is almost done" short-circuit. If the long
            request has only 200 prefill tokens remaining, it is pointless to
            chunk to mean_short=64 and take 4 more steps.
            Extension:
                chunk_size = min(chunk_size, long_req.remaining_prefill_tokens)

        (c) Only one long per step. If long_unaccepted_queue backs up, K > 1
            simultaneous longs are possible but require solving
                chunk_size * K + sum(fit_shorts) <= chunk_budget
            jointly. The single-long design is simpler and sufficient for
            K = 1 micro-batches in practice.

        Putting it all together, the recommended full chunk_size formula
        (not enabled; shown for reference):
            short_lens = [r.get_num_input_tokens() for r in shorts_this_step]
            target     = mean(short_lens) if short_lens else long_threshold_abs
            lo         = FLOOR_CHUNK
            hi         = free_token_budget    # tokens remaining in the batch
            chunk_size = max(lo, min(target, hi))
            chunk_size = min(chunk_size, long_req.remaining_prefill_tokens)

        Fallbacks
        ---------
        * No short requests on this ve: mean_input_len falls back to
          long_threshold_abs (the long/short cutoff itself), avoiding division
          by zero and giving a sensible chunk magnitude.
        * long_unaccepted_queue is empty: skip the Stage 2 chunking branch
          and fall straight through to short-request packing.
        * free_token_budget <= 0: add nothing and return; the current batch
          continues at the next step.

        See balance_prefill's docstring for the parameter list; the key
        knobs are long_threshold and chunked_prefill_budget.
        """
        ve = self.scheduler[virtual_engine]
        curr_batch = ve.running_queue
        chunk_budget = self.sched_config.chunked_prefill_budget
        long_threshold = chunk_budget * self.bench_config.long_threshold
        enable_chunked_prefill = self.sched_config.enable_chunked_prefill

        ongoing_prefills = curr_batch.get_context_requests()
        short_prefill_chunks = 0
        n_short_prefill_chunks = 0
        for req in (ongoing_prefills + list(ve.unaccepted_queue)):
            input_tokens = req.get_num_input_tokens()
            if input_tokens <= long_threshold:
                short_prefill_chunks += input_tokens
                n_short_prefill_chunks += 1
        mean_input_len = (
            short_prefill_chunks // n_short_prefill_chunks 
            if n_short_prefill_chunks > 0 else int(long_threshold)
        )
        
        """
        set next chunk for requests if required
        """
        free_token_budget = chunk_budget - curr_batch.get_num_input_tokens()
        for request in curr_batch.requests:
            if request.have_follow_up_chunk():
                free_token_budget = request.allocate_next_chunk(free_token_budget)

        blocks_overflow = False
        if (request := self.fetch_long_unaccepted_queue(virtual_engine)) is not None:
            (num_overflow_blocks, num_overflow_requests, num_overflow_tokens) = \
                self._check_add_to_cur_batch(virtual_engine, request)

            if (num_overflow_requests == 0 
                and num_overflow_blocks == 0):
                num_avail_tokens = min(free_token_budget, mean_input_len)
                if num_avail_tokens > 0:
                    logger.info(f"set chunksize of {num_avail_tokens=} for request {request.request_id}")
                    request.set_chunk(num_avail_tokens)
                    self._try_add_to_cur_batch(virtual_engine, request, is_long_req=True)
            elif num_overflow_blocks:
                blocks_overflow = True


        while (request := self.fetch_unaccepted_queue(virtual_engine, schedule_global_queue)) is not None:

            (num_overflow_blocks, num_overflow_requests, num_overflow_tokens) = \
                self._check_add_to_cur_batch(virtual_engine, request)

            if (num_overflow_blocks, num_overflow_requests, num_overflow_tokens) == (0, 0, 0):

                # reserved for zigzag chunked-prefill
                    
                if not self._try_add_to_cur_batch(virtual_engine, request):
                    # give up if allocate blocks failed
                    break
            
            elif (enable_chunked_prefill
                and num_overflow_requests == 0 
                and num_overflow_blocks == 0
            ): # decode-first, but will schedule prefill through request chunking
                num_avail_tokens = request.get_num_input_tokens() - num_overflow_tokens
                
                # not even one token is available, just give up
                if num_avail_tokens == 0:
                    break
                    
                logger.info(f"set chunksize of {num_avail_tokens=} for request {request.request_id}")
                request.set_chunk(num_avail_tokens)

                self._try_add_to_cur_batch(virtual_engine, request)
                if request.have_follow_up_chunk(): # it means current request has used up all token budget
                    break
            
            else: # just give up scheduling when there are too many requests in the batch
                if num_overflow_blocks > 0:
                    blocks_overflow = True
                # logger.info(f"[queueing] overflow in blocks: {num_overflow_blocks}, overflow in requests: {num_overflow_requests}, overflow in tokens: {num_overflow_tokens}")
                break
        ve = self.scheduler[virtual_engine]
        ve.num_pending_prefill_tokens = sum([req.get_num_tokens() for req in ve.unaccepted_queue]) + ve.running_queue.get_unfinished_prefill_tokens()
        return blocks_overflow
        

    @DeprecationWarning
    async def schedule_chunked_pipe_prefill(self, virtual_engine: int):
        # the first virtual engine will schedule all micro-batches
        if virtual_engine > 0:
            return

        while (request := self.fetch_decode_queue()) is not None:
            self.get_min_cost_scheduler().running_queue.add_request(request)
            self.pop_decode_queue()
        
        micro_batch = 0
        unfinished_chunked_prefill = []
        while (request := self.fetch_unaccepted_queue()) is not None:
            (num_overflow_blocks, num_overflow_requests, num_overflow_tokens) = \
                self._check_add_to_cur_batch(micro_batch, request)

            # we will ensure one micro-batch is saturated before switching to the next engine
            if (num_overflow_blocks, num_overflow_requests, num_overflow_tokens) == (0, 0, 0):
                # request.is_last_chunk = True
                self._try_add_to_cur_batch(self.scheduler[micro_batch], request)
            
            elif num_overflow_requests == 0 and num_overflow_blocks == 0:
                
                num_avail_tokens = request.get_num_input_tokens() - num_overflow_tokens
                if num_avail_tokens == 0:
                    break

                self._try_add_to_cur_batch(self.scheduler[micro_batch], request)

                if request.have_follow_up_chunk():
                    request = copy.deepcopy(request)
                    request.prepare_for_sequence_parallel_chunk()
                    unfinished_chunked_prefill.append(request)
                
                micro_batch = (micro_batch + 1) % len(self.scheduler)
                if micro_batch == 0:
                    break
                
                """ Sequence pipeline parallelism: nearly useless when pipeline stage number=2 """
                if micro_batch == 1: 
                    # if current request is chunked, we will fill the following pipeline stages with its remaining tokens
                    
                    while request.have_follow_up_chunk():
                        request = copy.deepcopy(request)
                        request.prepare_for_sequence_parallel_chunk()

                        (num_overflow_blocks, num_overflow_requests, num_overflow_tokens) = self._check_add_to_cur_batch(micro_batch, request)
                        
                        num_avail_tokens = request.get_num_input_tokens() - num_overflow_tokens
                        if num_avail_tokens != 0:
                            request.set_chunk(num_avail_tokens)
                            request.is_last_chunk = not request.have_follow_up_chunk()
                            self.scheduler_assigned_chunk[micro_batch].append(request)
                        
                        # let next pipeline stage try
                        micro_batch = (micro_batch + 1) % len(self.scheduler)
                        if micro_batch == 0: # it means all engine has been filled, just give up
                            if not request.is_last_chunk:
                                if num_avail_tokens != 0: # request has been chunked again
                                    request = copy.deepcopy(request)
                                    request.prepare_for_sequence_parallel_chunk()
                                unfinished_chunked_prefill.append(request)
                            break
                    
                    # no need to try, all engine has been filled
                    if micro_batch == 0:
                        break
            else:
                # no space left for current micro-batch, move on
                micro_batch = (micro_batch + 1) % len(self.scheduler)
                if micro_batch == 0:
                    break
        
        # put unfinished chunked prefill requests back to the unaccepted queue
        for request in reversed(unfinished_chunked_prefill):
            self.unaccepted_queue.appendleft(request)
        if len(self.unaccepted_queue):
            logger.info(f"number of unaccepted requests after scheduling: {len(self.unaccepted_queue)}")

    async def _get_next_batch(self, virtual_engine: int, force_accept: bool = False) -> BatchedRequests:
        ve = self.scheduler[virtual_engine]
        # note that swap will only be used in replica serving
        use_swap = (self.sched_config.preempt_method == "swap")

        if self.stopped:
            return BatchedRequests([])
        
        finished_reqs = ve.running_queue.pop_finished_requests()
        self.free_blocks_batched(finished_reqs)
        if self.sched_config.preempt_method == "swap":
            self._remote_call_all_workers_async(
                "clear_request_resource_batched", finished_reqs
            )
        '''
        step 1: allocate blocks for the current batch
        '''
        curr_batch = ve.running_queue
        curr_batch_allocated = False
        latest_kv_rank = -1

        try:
            while not curr_batch_allocated:
                try:
                    for request in curr_batch.requests:
                        if request.fusion_exec:
                            latest_kv_rank = -1
                        else:
                            latest_kv_rank = request.kv_fusion_rank
                        self._allocate_blocks(request)
                    curr_batch_allocated = True
                except:
                    logger.error(f"failed to allocate blocks for request {request.request_id}, {latest_kv_rank=}")
                    is_fusion = (latest_kv_rank == -1)
                    # not enough blocks for the current batch, trigger swap or recompute
                    if use_swap:
                        exit
                        victim = await ve.trigger_swap(is_fusion, latest_kv_rank)
                        if victim is None:
                            return BatchedRequests([])
                    else:
                        victim = ve.trigger_recompute(is_fusion, latest_kv_rank, resched_local=True)
                        if victim is None:
                            logger.info(f"no victim found for request {request.request_id}, exit")
                            return BatchedRequests([])
                        else:
                            logger.info(f"recompute victim {victim.request_id} for request {request.request_id}")
        except Exception as e:
            logger.info(f"failed to allocate blocks for the current batch: {e}")
            ray.shutdown()
            exit(0)

        '''
        Step 2. add requests in waiting queue back to the current batch
        '''

        while (request := ve.fetch_waiting_queue()) is not None:
            # logger.info(f"#{self.server_id} {virtual_engine=} try to fetch request {request.request_id} from waiting queue")
            (num_overflow_blocks, num_overflow_requests, num_overflow_tokens) \
                = self._check_add_to_cur_batch(virtual_engine, request, is_waiting=True)
            if (num_overflow_blocks, num_overflow_requests, num_overflow_tokens) == (0, 0, 0):
                self.scheduler[virtual_engine].swap_in_from_waiting_queue(request)
            else:
                # logger.info(f"#{self.server_id} {virtual_engine=} failed to fetch request {request.request_id} from waiting queue")
                break
        '''
        step 3: schedule pending requests in the swapped queue
        '''
        if use_swap:
            while (request := ve.fetch_swapped_queue()) is not None:
                (num_overflow_blocks, num_overflow_requests, num_overflow_tokens) = self._check_add_to_cur_batch(virtual_engine, request)
                if (num_overflow_blocks, num_overflow_requests, num_overflow_tokens) == (0, 0, 0):
                    await ve.swap_in_request()
                    self._allocate_blocks(request)
                else:
                    break
           
        '''
        step 4: schedule new requests from the unaccepted queue in FCFS manner
        '''
        if virtual_engine == 0 and self.bench_config.enable_migration:
            pending_blocks = sum([self._get_block_needed(req.get_num_tokens()) for req in self.unaccepted_queue])
            free_blocks = self.get_free_gpu_blocks()

            # queuing happening, try to migrate
            if pending_blocks >= free_blocks:
                local_load = self.get_used_gpu_blocks()
                min_load = None
                min_load_index = None
                for i, instance in enumerate(self.instances):
                    if i == self.parallel_config.replica_rank:
                        continue
                    load = await instance.get_used_gpu_blocks.remote()
                    if min_load is None or load < min_load:
                        min_load = load
                        min_load_index = i
                is_migrating = await self.instances[min_load_index].is_migrating.remote()
                logger.info(f"#{self.server_id} instance {self.parallel_config.replica_rank}: queueing happens, try to migrate, {local_load=}, {min_load=}")
                if min_load < local_load and not is_migrating:
                    # start migration
                    self.is_migrating_out = True
                    max_migrate_blocks = (local_load - min_load) // 2
                    sorted_unaccepted_queue = sorted(
                        self.unaccepted_queue, key=lambda x: x.get_num_tokens()
                    )
                    migrated_blocks = 0
                    migrated_requests = []
                    for request in sorted_unaccepted_queue:
                        num_blocks_needed = self._get_block_needed(request.get_num_tokens())
                        if num_blocks_needed + migrated_blocks > max_migrate_blocks:
                            break
                        migrated_blocks += num_blocks_needed
                        self.unaccepted_queue.remove(request)
                        migrated_requests.append(request)
                    if len(migrated_requests) > 0:
                        await self.instances[min_load_index].add_requests.remote(migrated_requests)
                        logger.info(f"#{self.server_id} instance {self.parallel_config.replica_rank}: migrated {len(migrated_requests)} requests to instance {min_load_index} ({max_migrate_blocks=})")
                        local_load = self.get_used_gpu_blocks()
                        min_load = await self.instances[min_load_index].get_used_gpu_blocks.remote()
                        logger.info(f"#{self.server_id} instance {self.parallel_config.replica_rank}: after migration, {local_load=}, {min_load=}")
                    self.is_migrating_out = False

        if True:
            if len(self.scheduler) > 1 and self.state != "restore":
                # Lookahead scheduling dispatch (see detailed docstrings on
                # balance_prefill / schedule_packed_prefill / balance_decode for
                # the full algorithm and how to tune long_threshold,
                # chunked_prefill_budget, enable_packing, enable_reorder,
                # enable_decode_balance).
                #
                #   enable_look_ahead  -> call balance_prefill (Stage 1: cross-ve placement)
                #   enable_packing     -> call schedule_packed_prefill (Stage 2: only-chunk-longs)
                #                         otherwise: schedule_prefill (Sarathi-style chunked prefill)
                #
                # Note on schedule_global_queue: ablation showed it should only be
                # enabled on BurstGPT-like workloads; other datasets should disable it.

                # enable_zigzag_prefill = self.bench_config.zigzag_chunk_limit > 0
                enable_zigzag_prefill = False

                if self.bench_config.enable_look_ahead:
                    self.balance_prefill(virtual_engine)

                if self.bench_config.enable_packing:
                    self.schedule_packed_prefill(virtual_engine)
                else:
                    self.schedule_prefill(
                        virtual_engine, 
                        enable_zigzag_prefill,
                        # schedule_global_queue=False, # only enable for burstGPT!
                    )
            else:
                self.schedule_prefill(virtual_engine)
     
    # NOTE: `reverse_kv_exchange` (sleep-simulated bulk restore) was removed.
    # The live `trigger_restore` path (see llm_engine.py) DOES NOT call it;
    # fusion_exec requests coming out of balloon are drained by the live
    # restore path inside `step_inner` (see the `num_restore_requests`
    # handling there).  Re-introducing a bulk restore would need a real
    # NCCL routine mirroring `kv_exchange` in the opposite direction
    # (Phase 1: pre-read post-drop data; NCCL route it back to each
    # request's home instance; scatter into pre-drop layer slots at new
    # pre-fusion local IDs).

    
    async def _step(self, virtual_engine: int):
        
        async def step_inner(batched_requests: BatchedRequests, fusion_exec: bool):
            if self.stopped:
                return

            if len(batched_requests) == 0:
                return

            batch_size = len(batched_requests)
            kv_rank = virtual_engine // self.parallel_config.pipeline_parallel_size
            layers_per_instance = self.model_config.get_num_layers(self.parallel_config) // self.num_instances
            batched_requests.step_method = "step"

            '''
            live kv restore: convert pipeline requests to replica requests
                             by transferring blocks layer by layer
            ''' 
            new_block_table = []
            num_restore_requests = 0
            num_restore_blocks = 0
            # if self.state == "restore" and fusion_exec and not self.is_restoring:
            if self.state == "restore" and fusion_exec:
                self.is_restoring = True
                for req in batched_requests.requests:
                    bm = self.block_managers[kv_rank]
                    new_blocks = bm.get_free_gpu_blocks(len(bm.get_single_block_table(req.request_id)))
                    if new_blocks == []:
                        break
                    new_block_table.append(new_blocks)
                    num_restore_blocks += len(new_blocks)
                    num_restore_requests += 1
                    if num_restore_requests >= self.balloon_config.exchange_batch_size:
                        break
            
            if num_restore_requests == 0:
                self.is_restoring = False

            start_time = time.time()
            batched_requests.start_one_iteration(start_time)
            forward_futures = []
            for instance in range(self.num_instances):
                '''
                if we are in balloon state, we shall schedule requests at all instances no matter pipeline or rattn
                otherwise, we shall only schedule replica requests at the instance of kv_rank
                '''
                if (not self.is_fusion_state) and (not fusion_exec) and (instance != kv_rank):
                    continue

                bm = self.block_managers[instance]
                block_table = bm.get_block_table(batched_requests.requests)
                if not self.is_fusion_state and fusion_exec:
                    # pipeline request in restore state
                    max_num_pages = bm.num_base_gpu_blocks * self.num_instances
                else:
                    # other cases
                    max_num_pages = bm.num_base_gpu_blocks + bm.num_extend_gpu_blocks

                local_layer_start = instance * layers_per_instance if fusion_exec else 0
                local_layer_end = (instance + 1) * layers_per_instance if fusion_exec else self.model_config.get_num_layers(self.parallel_config)

                full_block_table = bm.get_partial_block_table(batched_requests.get_request_ids())

                # workers are of shape (pp_size, tp_size), but the returned remote_calls are of shape (pp_size * tp_size)
                remote_calls = self._remote_call_partial_workers_async(
                    instance * pp_size,
                    (instance + 1) * pp_size,
                    batched_requests.step_method,
                    batched_requests.get_request_ids(),             # request_ids
                    batch_size,                                     # num_requests
                    batched_requests.get_num_input_tokens(),        # num_tokens
                    max_num_pages,                                  # max_num_pages
                    local_layer_start,                              # local_layer_start
                    local_layer_end,                                # local_layer_end
                    batched_requests.get_input_tokens_batched(),    # input_tokens_batched
                    batched_requests.get_first_token_indexes(),     # first_token_indexes
                    batched_requests.get_is_context_stage(
                        use_tensor_cores=self.sched_config.use_tensor_cores
                    ),                                              # is_prefill_requests
                    block_table,                                    # pages_of_reqs
                    # below are parameters for live kv restore
                    0, # num_restore_requests,                      # num_restore_requests
                    full_block_table,                               # pages_of_pp
                    new_block_table,                                # pages_of_dp
                    kv_rank,                                        # receiver_rank
                )
                forward_futures.append(remote_calls[(pp_size - 1) * tp_size])

                # live kv exchange, overlap with pipeline execution
                if fusion_exec and num_resharding_requests > 0:
                    requests: List[Request] = []
                    for ve in range(len(resharding_requests)):
                        # divide into `num_instances` equal parts
                        for i, request in enumerate(resharding_requests[ve]):
                            if i % self.num_instances == instance:
                                requests.append(request)

                    # reshard blocks after pipeline transfer finished
                    if instance > 0:
                        await forward_futures[instance - 1]

                        # WARN: shall sync with YX to ensure bug-free
                        # if virtual_engine + 1 < len(self.scheduler_launch_events):
                        #     self.scheduler_launch_events[virtual_engine + 1].set()

                    start_reshard_time = time.perf_counter()
                    # `scale` used to halve the analytic-bandwidth sleep when
                    # the NCCL was overlapping with activation transfer; now
                    # that the transfer runs for real on NCCL, the parameter
                    # is meaningless and has been removed from the callee.
                    await self._reshard_blocks_batched(requests)
                    end_reshard_time = time.perf_counter()
                    logger.info(f"#{self.server_id} Live-resharded (iofused) {len(requests)} requests, cost {end_reshard_time-start_reshard_time:.3f} seconds")
                # else:
                #     # balloon pipeline execution and normal pipeline execution
                #     if instance > 0 or pp_size > 1:
                #         await remote_calls[0]
                #         if virtual_engine + 1 < len(self.scheduler_launch_events):
                #             self.scheduler_launch_events[virtual_engine + 1].set()

            try:
                # if fusion_exec and num_resharding_requests > 0:
                #     logger.info(f"before await forward future")
                future_result = await forward_futures[-1]
                # if fusion_exec and num_resharding_requests > 0:
                #     logger.info(f"after await forward future")
            except Exception as e:
                logger.error(f"Error in forward_future: {e}")
                ray.shutdown()
                exit(1)
            end_time = time.time()

            '''
            post live kv restore: replace the original blocks with the new blocks
            '''

            if num_restore_requests > 0:
                # NOTE: there used to be an `asyncio.sleep(trans_time - step_time)`
                # here that padded the step to the analytic transfer time when
                # the real transfer was shorter than the forward.  That was a
                # bandwidth-model simulation; the actual KV transfer now runs
                # inside `_reshard_blocks_batched` via NCCL and is already
                # synchronized on io_stream before we reach this point, so the
                # padding would only add spurious latency.  Leave the logging
                # below if someone wants to cross-check analytic vs measured.
                trans_size = num_restore_blocks * self.block_size_in_bytes * (self.num_instances - 1) / self.num_instances
                trans_size_gb = trans_size / 1024 / 1024 / 1024
                step_time = end_time - start_time
                logger.info(f"[live restore] trans {trans_size_gb:.3f}GB, forward step {step_time:.3f}s")

                for i in range(num_restore_requests):
                    req = batched_requests.requests[i]
                    for bm in self.block_managers:
                        bm.free_blocks(req)
                        
                    # add new blocks to the block table
                    bm = self.block_managers[kv_rank]
                    bm.request_location[req.request_id] = BlockLocation.GPU
                    bm.block_table[req.request_id] = new_block_table[i]
                    # update request info
                    req.kv_fusion_rank = kv_rank
                    req.fusion_exec = False
                self.is_restoring = False
            
            # the element after batch_size is reserved for other infomation
            generated_tokens_ids = future_result[:batch_size]
            generated_tokens = []
            for gen_token_id in generated_tokens_ids:
                try:
                    token = self.tokenizer.decode(gen_token_id)
                except Exception as e:
                    logger.warn(f"Cannot decode token with id {gen_token_id}. Error: {e}")
                    token = ""
                generated_tokens.append(token)
                
            batched_requests.finish_one_iteration(
                generated_tokens, generated_tokens_ids, end_time
            )
            requests_outputs = []
            for request in batched_requests.requests:
                # request has no generate tokens, which means its chunked prefill has not finished
                if request.is_context_stage():
                    continue
                self.finished_tokens += 1
                if request.is_finished:
                    self.finished_requests += 1
                
                if request.need_to_responsed():
                    requests_outputs.append(
                        RequestOutput(
                            request.request_id,
                            request.prompt_token_ids,
                            request.outputs,
                            start_time=request.start_time,
                            server_info=request.server_info,
                            finished=request.is_finished,
                        )
                    )

                if len(request.outputs) == 1:
                    self.ttfts.append(request.outputs[0].new_token_time - request.arrival_time)
                elif len(request.outputs) >= min(20, request.sampling_params.max_tokens):
                    # we set a limit of 20 tokens to tolerant occasional spikes
                    self.tbts.append(
                        (request.outputs[-1].new_token_time - request.outputs[0].new_token_time) / (len(request.outputs) - 1)
                    )
            
            if len(requests_outputs):
                self.put_queue_args_queue.put_nowait(requests_outputs)
            
        async def post_step(requests: List[Request]):
            # reclaim extra blocks initiatively
            if self.is_reclaiming_ex_blocks:
                await self.pack_blocks_into_base_region(requests)
                # await self.relocate_extra_blocks(self.scheduler[virtual_engine].schedule()[0])
        
        if self.stopped:
            return
        
        self.scheduler_running[virtual_engine] = True
        pp_size = self.parallel_config.pipeline_parallel_size
        tp_size = self.parallel_config.tensor_parallel_size
        
        resharding_requests: List[List[Request]] = []
        num_resharding_requests = 0
        if self.is_resharding[virtual_engine]:
            for i, sche in enumerate(self.scheduler):
                if i != virtual_engine: continue
                stale_requests = sche.trigger_reshard_for_stale_requests(
                    live_reshard=True,
                    live_reshard_batch_size=self.balloon_config.exchange_batch_size)

                # stale_requests = []
                # num_stale_requests = 0
                # for req in sche.waiting_queue:
                #     if not req.kvcache_ready and num_stale_requests < self.balloon_config.exchange_batch_size // len(self.scheduler):
                #         stale_requests.append(req)
                #         num_stale_requests += 1

                # allocate blocks for the requests
                allocated_requests, not_allocated_requests = self.prepare_reshard_blocks(stale_requests)
                

                sche.running_queue.add_requests(not_allocated_requests)
                sche.waiting_queue.extend(allocated_requests)

                resharding_requests.append(allocated_requests)
                num_resharding_requests += len(allocated_requests)
                for req in allocated_requests:
                    # mark all stale requests as new requests
                    req.fusion_exec = True
                    req.once_fusion_exec = True
                    req.kvcache_ready = False
            if len(stale_requests) == 0:
                self.is_resharding[virtual_engine] = False
                logger.info(f"{self.server_id=} {virtual_engine=} finish live resharding") 
            # else:
            #     logger.info(f"{self.server_id=} {virtual_engine=} start live resharding, {num_resharding_requests=}") 
            
        batched_requests, _ = self.scheduler[virtual_engine].schedule(prefill_only_chunk=True)
        batched_requests.set_sched_time()
        ve = self.scheduler[virtual_engine]
        try:
            pipeline_batched_requests, replica_batched_requests = batched_requests.get_fusion_requests()
            sche = self.scheduler[virtual_engine]
            # if self.state == "balloon" and not any([val==True for val in self.is_resharding]):
            #     logger.info(f"{self.server_id=} {virtual_engine=} start step {len(pipeline_batched_requests)=} {len(replica_batched_requests)=} {len(sche.running_queue)=} {len(sche.waiting_queue)=}")
            if len(pipeline_batched_requests) > 0:
                await step_inner(BatchedRequests(pipeline_batched_requests), fusion_exec=True)
            elif num_resharding_requests > 0:
                for reqs in resharding_requests:
                    await self._reshard_blocks_batched(reqs)
            if not self.is_fusion_state:
                await step_inner(BatchedRequests(replica_batched_requests), fusion_exec=False)

            await post_step(pipeline_batched_requests)

        except Exception as e:
            logger.error(f"Error in _step({virtual_engine=}): {e}")
            ray.shutdown()
            exit(1)

        # get next batch (move here so that the log can be printed)
        try:
            with self.allocate_blocks_lock:
                await self._get_next_batch(virtual_engine)
        except Exception as e:
            logger.error(f"Error in _get_next_batch({virtual_engine=}): {e}")
            ray.shutdown()
            exit(1)
        self.scheduler_running[virtual_engine] = False

    def get_pp_diff(self):
        return self.latency_diff, self.num_tokens_diff

    def print_engine_status(self):
        # for i, bm in enumerate(self.block_managers):
            # bm.print_block_usage(self.instance_id, i)
        unaccepted = len(self.unaccepted_queue)
        running = sum([len(_scheduler.running_queue.requests) for _scheduler in self.scheduler])
        waiting = sum([len(_scheduler.waiting_queue) for _scheduler in self.scheduler])
        swapped = sum([len(_scheduler.swapped_queue) for _scheduler in self.scheduler])
        # logger.info(f"(instance {self.instance_id}) {unaccepted=}, {running=}, {waiting=}, {swapped=}, {self.finished_requests=}.")
    
    """
    The func return the local gpu_blocks_usage in normal state, 
    and return the gpu_blocks_usage of the whole system in the `balloon` state.
    Specifically, it returns a tuple with four values
        1. the ratio of block demands of all requests
        2. the ratio of block demands of the unaccepted queue
        3. the ratio of free blocks
        4. the increasing rate of upcoming requests
    """
    def get_gpu_blocks_usage(self):
        total_used_blocks = self.get_used_gpu_blocks()
        free_gpu_blocks = self.get_free_gpu_blocks()
        max_gpu_blocks = self.get_max_base_gpu_blocks()

        # assert total_used_blocks + free_gpu_blocks >= max_gpu_blocks, f"block number is inconsistent! {total_used_blocks=}, {free_gpu_blocks=}, {max_gpu_blocks=}"
        
        total_demand_ratio = total_used_blocks / max_gpu_blocks * 100
        # if total_demand_ratio > 100.0:
        #     logger.warn(f"Queueing happens in instance({self.instance_id}), current memory demand is {total_demand_ratio:.2f}%.")
        # head_demand_ratio = head_blocks_need / max_gpu_blocks * 100
        # free_ratio = free_gpu_blocks / max_gpu_blocks * 100
        
        self.sliding_window.append(total_demand_ratio)
        memory_increase_rate = (total_demand_ratio - self.sliding_window[0]) / len(self.sliding_window)
        self.rate_window.append(memory_increase_rate)

        output_tokens = self.finished_tokens - self.last_epoch_finished_tokens
        finished_requests = self.finished_requests - self.last_epoch_finished_requests
        self.last_epoch_finished_tokens = self.finished_tokens
        self.last_epoch_finished_requests = self.finished_requests
        
        return (total_demand_ratio, 
                self.get_unfinished_requests(), 
                -free_gpu_blocks, # remaining memory
                output_tokens, 
                finished_requests)
    
    def pop_metrics(self):
        ttfts = self.ttfts
        self.ttfts = []

        tbts = self.tbts
        self.tbts = []
        
        return ttfts, tbts