"""
Adapted from https://github.com/vllm/worker/worker.py
"""

import copy
import time
from typing import List, Tuple, Optional
import socket
import threading, asyncio
import concurrent.futures

import ray
import torch


from kunserve.config import ColocatedSchedConfig, ContextStageSchedConfig, DecodingStageSchedConfig, ModelConfig, CacheConfig, ParallelConfig, DisaggParallelConfig
from kunserve.request import Request, BatchedRequests
from kunserve.utils import set_random_seed, cudaMemoryIpcHandle, Stage
from kunserve.models import get_model_op
from kunserve.utils import (
    get_gpu_memory, get_gpu_memory_usage, set_random_seed, 
    GB, MB
)
from kunserve.logger import init_logger
from kunserve.downloader import download_and_convert_weights

logger = init_logger(__name__)


@ray.remote(num_cpus=0, num_gpus=1)
class ParaWorker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache, the KV swap and executing the model on the GPU.
    In case of distributed inference, each worker is assigned a partition of
    the model.

    """

    def __init__(
        self,
        worker_id: int,
        stage: Stage,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        sched_config: (
            ContextStageSchedConfig | DecodingStageSchedConfig | ColocatedSchedConfig
        ),
        disagg_parallel_config: DisaggParallelConfig,
        parallel_config: ParallelConfig = ParallelConfig(),
        tensor_parallel_id: List[int] = None,  # Although the type is list[int], it is actually a NCCL unique ID
        pipeline_parallel_id: List[int] = None,  # Same as above
        group_parallel_id: List[int] = None,
        kv_exchange_id: List[int] = None,
        global_id: List[int] = None,
    ) -> None:
        self.worker_id = worker_id
        self.stage = stage
        self.model = None
        self.model_config = model_config
        self.model_config.load_hf_config()
        
        self.sched_config = sched_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.tensor_parallel_id = tensor_parallel_id
        self.pipeline_parallel_id = pipeline_parallel_id
        self.group_parallel_id = group_parallel_id
        self.kv_exchange_id = kv_exchange_id
        self.global_id = global_id

        self.gpu_id = ray.get_gpu_ids()[0]
        self.forward_executor = concurrent.futures.ThreadPoolExecutor()
        self.is_context_worker = 1 if self.stage == Stage.CONTEXT or self.stage == Stage.COLOCATED else 0
        self.disagg_parallel_config = disagg_parallel_config
        
        self.device = torch.device(f"cuda:0")
        torch.cuda.set_device(self.device)

        # K/V & X cache on GPU
        self.k_cache = None
        self.v_cache = None
        
        # K/V & X swap on CPU
        self.k_swap = None
        self.v_swap = None
        
        # warning, x cache is deprecated now
        self.x_cache = None
        self.x_swap = None
        # CUDA streams for swapping in and out
        self.swap_in_stream = torch.cuda.Stream()
        self.swap_out_stream = torch.cuda.Stream()
        # The swap_event_table, refer to block_manager.py for more details
        self.swap_event_table = {}
        # The latest swap event in each stream
        # Used when we need to wait for all swap events to finish
        self.latest_swap_in_event = None
        self.latest_swap_out_event = None
        # Statistics
        self.execution_time = 0.0
        self.blocked_swapping_time = 0.0
        self.split_time = 0.0

    def ready(self):
        """
        Ray functions queue inside one single actor to be executed in order.
        If ready is called, the actor is ready.
        """
        logger.info(
            f"Worker {self.stage}.#{self.worker_id} created on host {socket.gethostname()} and gpu #{self.gpu_id}"
        )
        pass

    def init_model(self, num_gpu_blocks):
        # Initialize the model.
        set_random_seed(self.model_config.seed)
        self.model = get_model_op(
            self.model_config,
            self.parallel_config,
            self.cache_config,
            self.sched_config,
            num_gpu_blocks,
        )

        # if self.is_context_worker: # context worker or a colocated worker
        #     context_parallel_config = self.parallel_config
        #     decoding_parallel_config = self.disagg_parallel_config.decoding
        # else: # decoding worker
        #     context_parallel_config = self.disagg_parallel_config.context
        #     decoding_parallel_config = self.parallel_config

        self.model.init_nccl_comm(self.tensor_parallel_id,
                                     self.pipeline_parallel_id,
                                     self.group_parallel_id,
                                     self.kv_exchange_id,
                                     self.global_id,
                                     self.parallel_config.tensor_parallel_rank,
                                     self.parallel_config.tensor_parallel_size,
                                     self.parallel_config.pipeline_parallel_rank,
                                     self.parallel_config.pipeline_parallel_size,
                                     self.parallel_config.group_parallel_rank,
                                     self.parallel_config.group_parallel_size,
                                     self.parallel_config.global_rank,
                                     self.parallel_config.global_size)

        torch.cuda.synchronize()
        if self.model_config.use_dummy_weights:
            self.model.init_model("")
        else:
            self.model.init_model(self.model_config.model)
        torch.cuda.synchronize()
        logger.info(
            f"(Worker #{self.worker_id}) model {self.model_config.model} loaded, "
            f"current cuda memory usage: {get_gpu_memory_usage()/1000:.3f}%."
        )


    def init_kvcache_and_swap(
        self, num_gpu_blocks, num_cpu_blocks
    ) -> Tuple[cudaMemoryIpcHandle, cudaMemoryIpcHandle]:
        """
        Allocate the K/V cache and swap.

        Return K/V cache's memory handle
        """
        # kv shape is [num_gpu_blocks, num_layers, num_local_heads, block_size, head_dim]
        # profile the GPU to get num_gpu_blocks
        self.kv_cache_shape = (
            num_gpu_blocks,
            self.model_config.get_num_layers(self.parallel_config),
            self.model_config.get_num_heads(self.parallel_config),
            self.cache_config.block_size,
            self.model_config.get_head_size(),
        )
        logger.info(f"kv_cache_shape: {self.kv_cache_shape}")
        self.k_cache = torch.empty(
            self.kv_cache_shape, dtype=self.model_config.get_torch_dtype(), device="cuda"
        )
        self.v_cache = torch.empty(
            self.kv_cache_shape, dtype=self.model_config.get_torch_dtype(), device="cuda"
        )
        # kv swap is [num_cpu_blocks, num_layers, num_local_heads, block_size, head_dim]
        # We pin memory here in order to leverage cudaMemcpyAsync when swapping
        kv_swap_shape = (num_cpu_blocks,) + self.kv_cache_shape[1:]
        self.k_swap = torch.empty(
            kv_swap_shape,
            dtype=self.model_config.get_torch_dtype(),
            device="cpu",
            pin_memory=True,
        )
        self.v_swap = torch.empty(
            kv_swap_shape,
            dtype=self.model_config.get_torch_dtype(),
            device="cpu",
            pin_memory=True,
        )
        torch.cuda.synchronize()

        return torch.ops.block_migration_ops.get_ipc_mem_handle(
            self.k_cache
        ), torch.ops.block_migration_ops.get_ipc_mem_handle(self.v_cache)

    def _get_block_size_in_bytes(
        self,
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        # the shape of one slot in k/v cache is [num_layers, num_local_heads, block_size, head_dim]
        num_layers = model_config.get_num_layers(parallel_config)
        num_heads = model_config.get_num_heads(parallel_config)
        head_dim = model_config.get_head_size()
        key_cache_size = num_layers * num_heads * block_size * head_dim
        total = key_cache_size * 2
        dtype_size = model_config.get_dtype_size()
        return total * dtype_size

    @torch.inference_mode()
    def _profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
        kv_cache_ratio: float,
    ) -> Tuple[int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # GPU and CPU blocks that can be allocated with the remaining free memory.

        # Profile memory usage with max_batch_size requests and the total
        # number of tokens equal to max_tokens_per_batch.
        total_gpu_memory = get_gpu_memory()
        logger.info(f"{total_gpu_memory=}")

        model_bytes = self.model_config.get_model_size_in_bytes(parallel_config=self.parallel_config)
        peak_runtime_memory = (
            total_gpu_memory * 0.01 + model_bytes
        )
        logger.info(f"Model runtime peak memory: {peak_runtime_memory / GB:.3f} GB")
        block_size_in_bytes = self._get_block_size_in_bytes(
            block_size, self.model_config, self.parallel_config
        )
        cache_gpu_memory = (
            total_gpu_memory * gpu_memory_utilization - peak_runtime_memory
        )
        logger.info(
            f"KV cache size for one token: {block_size_in_bytes / block_size / MB:.5f} MB, "
            f"total KV cache memory: {cache_gpu_memory / GB:.3f} GB.")
        kv_cache_gpu_memory = cache_gpu_memory * kv_cache_ratio
        num_gpu_blocks = int(kv_cache_gpu_memory // block_size_in_bytes)
        kv_cache_swap_space = cpu_swap_space * kv_cache_ratio
        num_cpu_blocks = int(kv_cache_swap_space // block_size_in_bytes)

        num_gpu_blocks = max(num_gpu_blocks, 0)
        logger.info(f"{block_size=}, each block has {block_size_in_bytes} bytes")
        logger.info(f"num_gpu_blocks: {num_gpu_blocks}")
        num_cpu_blocks = max(num_cpu_blocks, 0)
        logger.info(f"num_cpu_blocks: {num_cpu_blocks}")

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        return (
            num_gpu_blocks,
            num_cpu_blocks,
            block_size_in_bytes
        )

    def step(
        self,
        request_ids: List[int],
        num_requests: int,
        num_tokens: int,
        max_num_pages: int,
        local_layer_start: int,
        local_layer_end: int,
        input_tokens_batched: List[List[int]],
        first_token_indexes: List[int],
        is_prefill_requests: List[int],
        pages_of_reqs: List[List[int]],
        num_restore_requests: int,
        pages_of_pp: List[List[int]],
        pages_of_dp: List[List[int]],
        receiver_rank: int,
    ) -> List[int]:
        """Run one step of inference on the batch of requests."""

        # Check whether synchronization is necessary
        for request_id in request_ids:
            if request_id in self.swap_event_table:
                # We let the current stream wait for the swap event
                # This is non-blocking (It just stop the current stream instead
                # of chocking the CPU)
                self.swap_event_table[request_id].wait(torch.cuda.current_stream())
                self.swap_event_table.pop(request_id, None)
        
        generated_tokens_ids = self.model.pforward(
            num_requests,
            num_tokens,
            max_num_pages,
            local_layer_start,
            local_layer_end,
            input_tokens_batched,
            first_token_indexes,
            is_prefill_requests,
            pages_of_reqs,
            num_restore_requests,
            pages_of_pp,
            pages_of_dp,
            receiver_rank,
        )

        return generated_tokens_ids

    def rattn_step(
        self,
        num_requests: int,
        num_tokens: int,
        max_num_pages: int,
        num_extend_pages: int,
        input_tokens_batched: List[List[int]],
        first_token_indexes: List[int],
        pages_of_reqs: List[List[int]],
        num_requests_at_ranks: List[int],
    ) -> List[int]:
        """Run one step of inference on the batch of requests."""
        generated_tokens_ids = self.model.rattn_forward(
            num_requests,
            num_tokens,
            max_num_pages,
            num_extend_pages,
            input_tokens_batched,
            first_token_indexes,
            pages_of_reqs,
            num_requests_at_ranks,
        )

        return generated_tokens_ids

    def reshard_step(
        self,
        request_ids: List[int],
        is_context_stage,
        input_tokens_batched,
        first_token_indexes,
        block_table,
        cpu_block_table,
        gpu_memory_utilization: float,
        is_sender: bool,
        sender_rank: int,
        local_layer_offset: int,
        local_layer_num: int
    ) -> Tuple[List[int], float]:
        """Run one step of inference on the batch of requests."""

        # start = time.time()
        # Check whether synchronization is necessary
        for request_id in request_ids:
            if request_id in self.swap_event_table:
                # We let the current stream wait for the swap event
                # This is non-blocking (It just stop the current stream instead
                # of chocking the CPU)
                self.swap_event_table[request_id].wait(torch.cuda.current_stream())
                self.swap_event_table.pop(request_id, None)
        # self.blocked_swapping_time += time.time() - start

        # start = time.time()
        # run forward
        max_gpu_memory_per_batch = int(get_gpu_memory() * (1 - gpu_memory_utilization) * 0.8)

        generated_tokens_ids = self.model.reshard_forward(
            is_context_stage,
            input_tokens_batched,
            first_token_indexes,
            self.k_cache,
            self.v_cache,
            block_table,
            cpu_block_table,
            max_gpu_memory_per_batch,
            is_sender,
            sender_rank,
            local_layer_offset,
            local_layer_num,
        )
        # forward_time = time.time() - start
        # self.execution_time += forward_time

        return generated_tokens_ids

    def fusion_step(
        self,
        request_ids: List[int],
        is_context_stage,
        input_tokens_batched,
        first_token_indexes,
        block_table,
        cpu_block_table,
        gpu_memory_utilization: float,
        fusion_start_rank,
        local_rank_in_fusion,   # stage rank of current worker
        kv_rank_in_fusion,      # the rank of worker that hold kv cache required by current worker
        fusion_para_size,
        fusion_exec,
        local_layer_offset,
        local_layer_num,
    ) -> List[int]:
        # start = time.time()
        # Check whether synchronization is necessary
        for request_id in request_ids:
            if request_id in self.swap_event_table:
                # We let the current stream wait for the swap event
                # This is non-blocking (It just stop the current stream instead
                # of chocking the CPU)
                self.swap_event_table[request_id].wait(torch.cuda.current_stream())
                self.swap_event_table.pop(request_id, None)
        
        # self.blocked_swapping_time += time.time() - start

        # start = time.time()
        # run forward
        # logger.info("start fusion_forward")
        """ fusion_forward has two execution mode:
            1. fusion execution(True): the worker will execute the inference request in the new fusion pipeline
            2. remote execution(False): the worker will execute non-attention ops in the new pipeline, but attention
            op in the old way
        """

        max_gpu_memory_per_batch = int(get_gpu_memory() * (1 - gpu_memory_utilization) * 0.8)

        generated_tokens_ids = self.model.fusion_forward(
            is_context_stage,
            input_tokens_batched,
            first_token_indexes,
            self.k_cache,
            self.v_cache,
            block_table,
            cpu_block_table,
            max_gpu_memory_per_batch,
            fusion_start_rank,
            local_rank_in_fusion,
            kv_rank_in_fusion,
            fusion_para_size,
            fusion_exec,                   # whether to use `fusion_exec`
            local_layer_offset,
            local_layer_num,
        )
        # self.execution_time += time.time() - start
        # print(f"Worker {self.stage}.#{self.worker_id} Step end")
        # logger.info("finish fusion_forward")

        return generated_tokens_ids

    def pmm_drop(self, rank_in_group, num_group_instances):
        """ @return: num of released gpu blocks """
        num_gpu_blocks = self.model.pmm_drop(
            rank_in_group,
            num_group_instances,
        )
        
        # calculate new pipeline/replica size/rank
        new_pp_size = self.parallel_config.pipeline_parallel_size * num_group_instances
        new_rp_size = self.parallel_config.replica_size // num_group_instances
        tp_group_rank = self.parallel_config.pipeline_parallel_size * self.parallel_config.replica_rank + self.parallel_config.pipeline_parallel_rank
        new_pp_rank = tp_group_rank % new_pp_size
        new_rp_rank = tp_group_rank // new_pp_size

        # modify parallel_config
        self.parallel_config.pipeline_parallel_size = new_pp_size
        self.parallel_config.replica_size = new_rp_size
        self.parallel_config.pipeline_parallel_rank = new_pp_rank
        self.parallel_config.replica_rank = new_rp_rank

        # modify disagg_parallel_config
        if self.stage == Stage.DECODING:
            self.disagg_parallel_config.decoding.pipeline_parallel_size = new_pp_size
            self.disagg_parallel_config.decoding.replica_size = new_rp_size
        else:
            self.disagg_parallel_config.context.pipeline_parallel_size = new_pp_size
            self.disagg_parallel_config.context.replica_size = new_rp_size

        return num_gpu_blocks
    
    def pmm_restore(self, num_instances):
        """
        split is performed in an async process
        """
        start = time.time()
        self.model.pmm_restore()

        # calculate new pipeline/replica size/rank
        new_pp_size = self.parallel_config.pipeline_parallel_size // num_instances
        new_rp_size = self.parallel_config.replica_size * num_instances
        tp_group_rank = self.parallel_config.pipeline_parallel_size * self.parallel_config.replica_rank + self.parallel_config.pipeline_parallel_rank
        new_pp_rank = tp_group_rank % new_pp_size
        new_rp_rank = tp_group_rank // new_pp_size

        # modify parallel_config
        self.parallel_config.pipeline_parallel_size = new_pp_size
        self.parallel_config.replica_size = new_rp_size
        self.parallel_config.pipeline_parallel_rank = new_pp_rank
        self.parallel_config.replica_rank = new_rp_rank

        # modify disagg_parallel_config
        if self.stage == Stage.DECODING:
            self.disagg_parallel_config.decoding.pipeline_parallel_size = new_pp_size
            self.disagg_parallel_config.decoding.replica_size = new_rp_size
        else:
            self.disagg_parallel_config.context.pipeline_parallel_size = new_pp_size
            self.disagg_parallel_config.context.replica_size = new_rp_size

        self.split_time += time.time() - start
    
    def kv_exchange(
        self,
        pages_of_instances: List[List[int]],
        pp_rank: int,
        pp_size: int,
        group_rank: int,
        group_size: int,
        num_base_blocks: int,
        num_extend_blocks: int,
    ):
        self.model.kv_exchange(
            pages_of_instances,
            pp_rank,
            pp_size,
            group_rank,
            group_size,
            num_base_blocks,
            num_extend_blocks,
        )

    def balloon_down(self, balloon_size: int):
        assert self.stage == Stage.CONTEXT, "balloon_down is only supported for context engines"
        new_pp_size = self.disagg_parallel_config.decoding.pipeline_parallel_size * balloon_size
        new_rp_size = self.disagg_parallel_config.decoding.replica_size // balloon_size
        self.disagg_parallel_config.decoding.pipeline_parallel_size = new_pp_size
        self.disagg_parallel_config.decoding.replica_size = new_rp_size

    def balloon_up(self, balloon_size: int):
        assert self.stage == Stage.CONTEXT, "balloon_up is only supported for context engines"
        new_pp_size = self.disagg_parallel_config.decoding.pipeline_parallel_size // balloon_size
        new_rp_size = self.disagg_parallel_config.decoding.replica_size * balloon_size
        self.disagg_parallel_config.decoding.pipeline_parallel_size = new_pp_size
        self.disagg_parallel_config.decoding.replica_size = new_rp_size

    def register_cache_mem_handles(
        self,
        context_parallel_config: ParallelConfig,
        kvcache_ipc_mem_handles: List[
            List[Tuple[cudaMemoryIpcHandle, cudaMemoryIpcHandle]]
        ],
    ):
        for pp_rank, stage_workers in enumerate(kvcache_ipc_mem_handles):
            for tp_rank, mem_handle in enumerate(stage_workers):
                tmp_parallel_config = copy.deepcopy(context_parallel_config)
                tmp_parallel_config.pipeline_parallel_rank = pp_rank
                tmp_parallel_config.tensor_parallel_rank = tp_rank
                torch.ops.block_migration_ops.register_ipc_mem_handle(
                    kvcache_ipc_mem_handles[pp_rank][tp_rank][0],
                    kvcache_ipc_mem_handles[pp_rank][tp_rank][1],
                    self.model_config.get_num_layers(),
                    self.model_config.get_num_heads(),
                    tmp_parallel_config.to_list(),
                    self.parallel_config.to_list(),
                )

        torch.cuda.synchronize()

    def migrate_blocks(
        self,
        context_block_indexes: List[int],
        context_parallel_config: ParallelConfig,
        decoding_block_indexes: List[int],
    ):
        torch.ops.block_migration_ops.migrate_blocks(
            context_parallel_config.pipeline_parallel_size,
            context_parallel_config.tensor_parallel_size,
            context_block_indexes,
            self.parallel_config.pipeline_parallel_size,
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_rank,
            self.parallel_config.tensor_parallel_rank,
            decoding_block_indexes,
            self.k_cache,
            self.v_cache,
        )

    def migrate_blocks_in_one_layer(
        self,
        context_block_indexes: List[int],
        context_parallel_config: ParallelConfig,
        decoding_block_indexes: List[int],
        context_pp_stage: int,
        layer_id: int
    ):
        torch.ops.block_migration_ops.migrate_blocks_in_one_layer(
            context_parallel_config.pipeline_parallel_size,
            context_parallel_config.tensor_parallel_size,
            context_block_indexes,
            self.parallel_config.pipeline_parallel_size,
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_rank,
            self.parallel_config.tensor_parallel_rank,
            decoding_block_indexes,
            self.k_cache,
            self.v_cache,
            context_pp_stage,
            layer_id
        )
        
    def migrate_blocks_in_fusion(
        self,
        context_block_indexes: List[int],
        context_parallel_config: ParallelConfig,
        decoding_block_indexes: List[int],
        fusion_rank: int,
        fusion_size: int,
    ):
        self.model.migrate_blocks_in_fusion(
            context_parallel_config.pipeline_parallel_size,
            context_parallel_config.tensor_parallel_size,
            context_block_indexes,
            fusion_size,
            self.parallel_config.tensor_parallel_size,
            fusion_rank,
            self.parallel_config.tensor_parallel_rank,
            decoding_block_indexes,
            self.k_cache,
            self.v_cache,
            
            self.kv_cache_shape[1] // fusion_size,  # layers per worker
            self.kv_cache_shape[2],                 # heads per worker
            self.kv_cache_shape[3],                 # block_size
            self.kv_cache_shape[4]                  # head_dim
        )

    def swap_blocks(
        self,
        request_ids: List[int],
        source_block_ids: List[int],
        target_block_ids: List[int],
        is_swap_in: bool,
    ):
        """Swap some blocks between CPU and GPU
        If is_swap_in, then move blocks from CPU to GPU, i.e. CPU block
        #source_block_ids[0] will be copied to GPU block #target_block_ids[0]
        and so on. Similar for is_swap_in = False
        """

        # print(f"Swap {source_block_ids} ({'CPU' if is_swap_in else 'GPU'}) to {target_block_ids} ({'GPU' if is_swap_in else 'CPU'})")
        stream = self.swap_in_stream if is_swap_in else self.swap_out_stream

        # Record event
        event = torch.cuda.Event()
        event.record(stream)

        # Save that event
        for request_id in request_ids:
            if request_id in self.swap_event_table:
                # If we've issued another swapping operation before, we shall wait it
                # Pay attention to the difference between wait() and synchronize()
                self.swap_event_table[request_id].wait(stream)
            self.swap_event_table[request_id] = event
        if is_swap_in:
            self.latest_swap_in_event = event
        else:
            self.latest_swap_out_event = event

        # Swap
        with torch.cuda.stream(stream):
            torch.ops.swapping_ops.swap(
                source_block_ids,
                target_block_ids,
                is_swap_in,
                self.k_cache,
                self.v_cache,
                self.k_swap,
                self.v_swap,
            )

    def swap_x_cache_blocks(
        self,
        request_ids: List[int],
        source_block_ids: List[int],
        target_block_ids: List[int],
        is_swap_in: bool,
    ):
        """Swap some blocks between CPU and GPU
        If is_swap_in, then move blocks from CPU to GPU, i.e. CPU block
        #source_block_ids[0] will be copied to GPU block #target_block_ids[0]
        and so on. Similar for is_swap_in = False
        """

        # print(f"Swap {source_block_ids} ({'CPU' if is_swap_in else 'GPU'}) to {target_block_ids} ({'GPU' if is_swap_in else 'CPU'})")
        stream = self.swap_in_stream if is_swap_in else self.swap_out_stream

        # Record event
        event = torch.cuda.Event()
        event.record(stream)

        # Save that event
        for request_id in request_ids:
            if request_id in self.swap_event_table:
                # If we've issued another swapping operation before, we shall wait it
                # Pay attention to the difference between wait() and synchronize()
                self.swap_event_table[request_id].wait(stream)
            self.swap_event_table[request_id] = event
        if is_swap_in:
            self.latest_swap_in_event = event
        else:
            self.latest_swap_out_event = event

        # Swap
        with torch.cuda.stream(stream):
            torch.ops.swapping_ops.swap_x_cache(
                source_block_ids,
                target_block_ids,
                is_swap_in,
                self.x_cache,
                self.x_swap,
            )

    def clear_request_resource(self, request_id: int):
        """Clear the resources associated with the request."""
        """This is called by LLMEngine when a request is finished or aborted"""
        # Clear the swap event table
        self.swap_event_table.pop(request_id, None)

    def clear_request_resource_batched(self, requests: List[Request]):
        """Clear the resources associated with the requests."""
        for request in requests:
            self.clear_request_resource(request.request_id)

    def wait_for_all_swap_in(self):
        """Wait for all swap in to finish"""
        if self.latest_swap_in_event is not None:
            self.latest_swap_in_event.synchronize()
            self.latest_swap_in_event = None

    def wait_for_all_swap_out(self):
        """Wait for all swap out to finish"""
        if self.latest_swap_out_event is not None:
            self.latest_swap_out_event.synchronize()
            self.latest_swap_out_event = None

    def send_blocks(self, is_fusion_exec, block_table, start_layer, end_layer, target_is_context_worker, target_rp_rank):
        self.model.send_blocks(
            self.k_cache,
            self.v_cache,
            is_fusion_exec,
            block_table,

            start_layer,
            end_layer,

            self.disagg_parallel_config.context.pipeline_parallel_size,     # context_pp_size
            self.disagg_parallel_config.context.replica_size,               # context_rp_size
            self.disagg_parallel_config.decoding.pipeline_parallel_size,    # decoding_pp_size
            
            0 if self.stage == Stage.DECODING else 1,       # source_is_context_worker
            self.parallel_config.pipeline_parallel_rank,    # source_pp_rank

            target_is_context_worker,
            target_rp_rank,
        )


    def receive_blocks(self, is_fusion_exec, block_table, start_layer, end_layer, source_is_context_worker, source_rp_rank):
        self.model.receive_blocks(
            self.k_cache,
            self.v_cache,
            is_fusion_exec,
            block_table,

            start_layer,
            end_layer,

            self.disagg_parallel_config.context.pipeline_parallel_size,     # context_pp_size
            self.disagg_parallel_config.context.replica_size,               # context_rp_size
            self.disagg_parallel_config.decoding.pipeline_parallel_size,    # decoding_pp_size
            
            source_is_context_worker,
            source_rp_rank,

            0 if self.stage == Stage.DECODING else 1,       # target_is_context_worker
            self.parallel_config.pipeline_parallel_rank,    # target_pp_rank
        )

    def reshard_blocks(self, block_table, local_start_layer, local_end_layer, is_sender, source_pp_rank, target_pp_rank, balloon_size):
        if is_sender:
            assert source_pp_rank == self.parallel_config.pipeline_parallel_rank, f"mismatched {source_pp_rank} and {self.parallel_config.pipeline_parallel_rank}"
        else:
            assert target_pp_rank == self.parallel_config.pipeline_parallel_rank, f"mismatched {target_pp_rank} and {self.parallel_config.pipeline_parallel_rank}"

        self.model.reshard_blocks(
            self.k_cache,
            self.v_cache,
            block_table,

            local_start_layer,
            local_end_layer,

            self.disagg_parallel_config.context.pipeline_parallel_size,     # context_pp_size
            self.disagg_parallel_config.context.replica_size,               # context_rp_size
            self.disagg_parallel_config.decoding.pipeline_parallel_size,    # decoding_pp_size

            is_sender,
            0 if self.stage == Stage.DECODING else 1,                       # is_context_worker
            self.parallel_config.replica_rank,                              # rp_rank

            source_pp_rank,
            target_pp_rank,
            balloon_size,
        )

    def local_copy_blocks(self, max_num_pages: int, all_ex_blocks: List[int], all_base_blocks: List[int]):
        self.model.local_copy_blocks(
            max_num_pages,
            all_ex_blocks,
            all_base_blocks,
        )

    def send_blocks(
        self,
        dst_rank: int,
        layer_start: int,
        layer_end: int,
        max_num_pages: int,
        block_table: List[List[int]],
    ):
        self.model.send_blocks(
            dst_rank,
            layer_start,
            layer_end,
            max_num_pages,
            block_table,
        )

    def recv_blocks(
        self,
        src_rank: int,
        layer_start: int,
        layer_end: int,
        max_num_pages: int,
        block_table: List[List[int]],
    ):
        self.model.recv_blocks(
            src_rank,
            layer_start,
            layer_end,
            max_num_pages,
            block_table,
        )