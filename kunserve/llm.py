import copy
import time
from typing import List, Union, Optional, AsyncGenerator

import asyncio
import torch
from tqdm import tqdm
import argparse

import numpy as np

from kunserve.config import (
    ModelConfig,
    ParallelConfig,
    CacheConfig,
    DisaggParallelConfig,
    ContextStageSchedConfig,
    DecodingStageSchedConfig, DisaggSchedConfig, ColocatedSchedConfig
)

from kunserve.coordinator import Coordinator
from kunserve.llm_engine import LLMEngine, ColocatedEngine, INSTANCE_LOAD_REPORT_INTERVAL
from kunserve.logger import init_logger
from kunserve.request import Request, SamplingParams, RequestOutput
from kunserve.metrics import calculate_metrics
from kunserve.kunserve_config import *
from kunserve.utils import random_uuid, get_log_name

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import pandas as pd



logger = init_logger(__name__)

ENGINE_REPORT_INTERVAL = 1

class AsyncLLM:
    """A Large Language Model (LLM) for online inference."""

    def __init__(
        self,
        server_id: int,
        ngroups: int,
        placement_groups: List[PlacementGroup],
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: ColocatedSchedConfig,
        balloon_config: BalloonConfig,
        dispatch_config: DispatchConfig,
        bench_config: BenchConfig,
    ):
        self.server_id = server_id
        dp = parallel_config.replica_size
        self.parallel_config = parallel_config
        self.engines = [
            ColocatedEngine(
                server_id,
                group_id,
                ngroups,
                placement_groups[group_id * dp:(group_id+1) * dp],
                model_config=model_config,
                parallel_config=parallel_config,
                cache_config=cache_config,
                sched_config=sched_config,
                balloon_config=balloon_config,
                dispatch_config=dispatch_config,
                bench_config=bench_config,
            )
            for group_id in range(ngroups)
        ]
        self.dispatch_strategy = dispatch_config.dispatch_strategy
        self.next_engine = 0

        self.enable_balloon = balloon_config.enable_balloon

        self.coordinator = Coordinator(
            dispatch_config.dispatch_strategy,
            parallel_config.replica_size,
            balloon_config,
            self.engines)

        self.log_path = get_log_name(
            bench_config.log_path, 
            bench_config.qps, 
            bench_config.cv, 
            bench_config.dist,
            bench_config.dataset_scale_factor,
        )
        self.epoch_time = []
        self.reveived_requests = 0
        self.finished_requests = 0
        self.request_id = 0

        self.server_thpt_per_epoch = []
        self.recv_reqs_per_epoch = []
        self.finish_reqs_per_epoch = []
        self.mean_ttft_per_epoch = []
        self.mean_tbt_per_epoch = []
        self.mem_demand_per_epoch = []
        self.state_per_epoch = []
    
    @classmethod
    def from_engine_args(
        cls,
        server_id: int,
        ngroups: int,
        placement_groups: List[PlacementGroup],
        kunserve_config: KunServeConfig,
    ):
        model_config = ModelConfig(
            model=kunserve_config.bench_config.model,
            tokenizer=kunserve_config.bench_config.model,
            trust_remote_code=True,
            seed=kunserve_config.bench_config.seed,
            use_dummy_weights=False,
            dtype=kunserve_config.engine_config.dtype,
        )
        parallel_config = ParallelConfig(
            tensor_parallel_size=kunserve_config.engine_config.tp,
            pipeline_parallel_size=kunserve_config.engine_config.pp,
            replica_size=kunserve_config.engine_config.group_size,
        )
        cache_config = CacheConfig(
            block_size=kunserve_config.engine_config.block_size,
            gpu_memory_utilization=kunserve_config.engine_config.gpu_util,
        )
        colocated_sched_config = ColocatedSchedConfig(
            policy=kunserve_config.engine_config.schedule_policy,
            preempt_method=kunserve_config.engine_config.preempt_method,
            max_batch_size=kunserve_config.engine_config.max_batch_size,
            max_tokens_per_batch=kunserve_config.engine_config.max_batch_tokens,
            enable_chunked_prefill=kunserve_config.engine_config.enable_chunked_prefill,
            chunked_prefill_size=kunserve_config.engine_config.chunked_prefill_size,
            chunked_prefill_budget=kunserve_config.engine_config.chunked_prefill_budget,
            use_tensor_cores=kunserve_config.engine_config.use_tensor_cores,
        )
        async_llm_actor = ray.remote(
            num_cpus=1,
            max_concurrency=8192,
        )(cls).options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_groups[0],
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True)
        )
        return async_llm_actor.remote(
            server_id,
            ngroups,
            placement_groups,
            model_config,
            parallel_config,
            cache_config,
            colocated_sched_config,
            balloon_config=kunserve_config.balloon_config,
            dispatch_config=kunserve_config.dispatch_config,
            bench_config=kunserve_config.bench_config,
        )

    async def initialize(self):
        global_id = []
        for i in range(self.parallel_config.tensor_parallel_size):
            global_id.append(copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id()))

        tasks = [engine.initialize(global_id) for engine in self.engines]
        await asyncio.gather(*tasks)

        self.server_routines = [
            asyncio.create_task(engine.start_all_event_loops())
            for engine in self.engines
        ]

        self.monitor_task = asyncio.create_task(self.coordinator.start_all_event_loops(self.enable_balloon))

        # self.report_task = asyncio.create_task(self.collect_engine_outputs())
        logger.info(f"AsyncLLM is ready for serving.")

    def report_server_metrics(self):
        output_tokens = sum([
            self.coordinator.report_instance_outputs(i)
            for i in range(len(self.engines))
        ])
        avg_load = np.mean(self.coordinator.avg_loads)
        return output_tokens, self.reveived_requests, self.finished_requests, avg_load, self.coordinator.state

    async def pop_metrics(self):
        ttfts, tbts = [], []
        for engine in self.engines:
            ttft, tbt = await engine.pop_metrics()
            ttfts.extend(ttft)
            tbts.extend(tbt)
        return ttfts, tbts

    async def collect_engine_outputs(self):
        prev_recv_requests = 0
        prev_finish_requests = 0
        engine_outputs = [0] * len(self.engines)

        epoch_start = time.perf_counter()
        global_start = epoch_start
        while True:
            start_time = time.perf_counter()
            outputs_in_cur_epoch = 0
            
            engine_outputs_in_cur_epoch = [self.coordinator.report_instance_outputs(i) for i in range(len(self.engines))]

            ttfts, tbts = [], []
            for engine in self.engines:
                ttft, tbt = await engine.pop_metrics()
                ttfts.extend(ttft)
                tbts.extend(tbt)

            epoch_end = time.perf_counter()
            
            for i, output_tokens in enumerate(engine_outputs_in_cur_epoch):
                outputs_in_cur_epoch += output_tokens - engine_outputs[i]
                engine_outputs[i] = output_tokens
            
            self.epoch_time.append(epoch_end - global_start)
            self.server_thpt_per_epoch.append(outputs_in_cur_epoch / (epoch_end - epoch_start))
        
            recv_reqs_cur_epoch = self.reveived_requests - prev_recv_requests
            self.recv_reqs_per_epoch.append(recv_reqs_cur_epoch)
            prev_recv_requests = self.reveived_requests

            finish_reqs_cur_epoch = self.finished_requests - prev_finish_requests
            self.finish_reqs_per_epoch.append(finish_reqs_cur_epoch)
            prev_finish_requests = self.finished_requests

            mean_ttft_in_cur_epoch = np.mean(ttfts) if len(ttfts) > 0 else 0
            self.mean_ttft_per_epoch.append(mean_ttft_in_cur_epoch)

            mean_tbt_in_cur_epoch = np.mean(tbts) if len(tbts) > 0 else 0
            self.mean_tbt_per_epoch.append(mean_tbt_in_cur_epoch)

            mem_demand_cur_epoch = np.mean(self.coordinator.avg_loads)
            self.mem_demand_per_epoch.append(mem_demand_cur_epoch)

            self.state_per_epoch.append(self.coordinator.state)

            logger.info(f"[Server {self.server_id}] outputs: {outputs_in_cur_epoch}, received: {recv_reqs_cur_epoch}, finished: {finish_reqs_cur_epoch}, mean ttft: {mean_ttft_in_cur_epoch:.3f}s, avg load: {mem_demand_cur_epoch:.2f}, state: {self.state_per_epoch[-1]}")

            epoch_start = epoch_end

            elapsed_time = time.perf_counter() - start_time
            await asyncio.sleep(max(0, ENGINE_REPORT_INTERVAL - elapsed_time))

    async def generate(
        self,
        request_id: int,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[str]] = None,
        max_tokens: int = 1024,
        ignore_eos = True,
    ):
        """Generate outputs for a single request.

        This method is a coroutine. It adds the request into the engine, and
        yields the StepOutput objects from the LLMEngine for the request.

        Args:
            request_id: The unique id of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            sampling_params: The sampling parameters of the request.

        Yields:
            The output `StepOutput` objects from the LLMEngine for the
            request.
        """
        if prompt is None and prompt_token_ids is None:
            raise ValueError("prompt or prompt_token_ids must be provided")

        # request_id = random_uuid()
        self.reveived_requests += 1

        arrival_time = time.time()
        try:
            group_id, instance_id = await self.coordinator.get_next_assign()
            # logger.info(f"assign request {request_id} to engine {group_id}-{instance_id}")
            results_generator = self.engines[group_id].generate(
                instance_id,
                request_id,
                arrival_time,
                prompt,
                prompt_token_ids,
                max_tokens,
                ignore_eos,
            )
        except Exception as e:
            logger.error(f"Failed to generate request {request_id}: {e}")
            ray.shutdown()
            exit(1)

        final_outputs: RequestOutput = None
        async for output in results_generator:
            final_outputs = output
        # logger.info(f"request {request_id} is finished with {len(final_outputs.outputs)} output tokens.")
        self.finished_requests += 1
        return {
            "request_id": request_id,
            "prompt": prompt,
            "prompt_len": len(final_outputs.prompt_token_ids),
            "outputs": final_outputs.outputs,
            "metrics": calculate_metrics(final_outputs, arrival_time),
        }
    
    async def batch_generate(
        self,
        batch_prompts: List[str],
        batch_prompt_token_ids: List[List[int]],
        prefix_len: int = 0,
        max_tokens: int = 1024,
        ignore_eos = True,
    ):
        group_id, instance_id = await self.coordinator.get_next_assign()
        arrival_time = time.time()
        request_ids = []
        for i in range(len(batch_prompts)):
            request_id = self.request_id
            self.request_id += 1
            request_ids.append(request_id)

        results_generators = self.engines[group_id].batch_generate(
            instance_id,
            request_ids,
            arrival_time,
            batch_prompts,
            batch_prompt_token_ids,
            prefix_len,
            max_tokens,
            ignore_eos,
        )
        
        batch_results = []
        final_outputs: RequestOutput = None
        for i, results_generator in enumerate(results_generators):
            async for output in results_generator:
                final_outputs = output
            
            self.finished_requests += 1
            batch_results.append({
                "request_id": request_id,
                "prompt": batch_prompts[i],
                "prompt_len": len(final_outputs.prompt_token_ids),
                "outputs": final_outputs.outputs,
                "metrics": calculate_metrics(final_outputs, arrival_time),
            })
        return batch_results

    def get_unfinised_num(self):
        return self.reveived_requests - self.finished_requests
    
    def check_pending_num(self, group_id: int, instance_id: int):
        check_num = self.engines[group_id].execute_instance_func(
            instance_id, "check_pending_num"
        )
        return check_num
        

    async def terminate(self):
        return
        # for engine in self.engines:
        #     await engine.terminate()
        data = {
            "time": self.epoch_time,
            "thpt": self.server_thpt_per_epoch,
            "received requests": self.recv_reqs_per_epoch,
            "finished requests": self.finish_reqs_per_epoch,
            "memory demand": self.mem_demand_per_epoch,
            "mean ttft": self.mean_ttft_per_epoch,
            "mean tbt": self.mean_tbt_per_epoch,
            "state": self.state_per_epoch,
        }
        df = pd.DataFrame(data)

        mem_demand = df["memory demand"]
        avg_mem = mem_demand.mean()
        logger.info(f"[LLM Server {self.server_id}] average memory demand is {avg_mem:.2f}%.")

        df.to_csv(self.log_path + f'_{self.server_id}.csv', index=False)

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """

        logger.info(f"Aborted request {request_id}.")
        await self.engines.abort_request(request_id)
