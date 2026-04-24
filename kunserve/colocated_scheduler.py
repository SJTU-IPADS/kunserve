import copy
from abc import ABC, abstractmethod
import json
import time
from typing import List, Callable, Optional, Deque, Tuple
import warnings
from collections import deque

from kunserve.block_manager import BlockManager
from kunserve.config import ColocatedSchedConfig, ModelConfig, ParallelConfig
from kunserve.logger import init_logger
from kunserve.request import Request, BatchedRequests

logger = init_logger(__name__)



class ColocatedScheduler():
    def __init__(
        self,
        sched_config: ColocatedSchedConfig,
        parallel_config: ParallelConfig,
        model_config: ModelConfig,
        block_managers: List[BlockManager],
        _remote_call_all_workers_async: Callable,
    ):
        self.sched_config = sched_config
        # If the request has not been accepted, then it will be put into the unaccepted queue.
        self.unaccepted_queue: Deque[Request] = deque()
        self.long_unaccepted_queue: Deque[Request] = deque()
        # If the current batch is full, the requests will be put into the waiting queue.
        # FIXME: reserve this field to avoid error when merging / spliting instance
        self.waiting_queue: Deque[Request] = deque()
        # If one request was in running queue before, but swapped out, it will be put into the swapped queue.
        self.swapped_queue: Deque[Request] = deque()

        # dynamic batch balance
        self.num_pending_prefill_tokens = 0
        
        # Processing requests
        self.running_queue: BatchedRequests = BatchedRequests([])
        self.parallel_config = copy.deepcopy(parallel_config)
        self.model_config = copy.deepcopy(model_config)
        self.block_managers = block_managers
        self._remote_call_all_workers_async = _remote_call_all_workers_async

        self.num_preemption = 0
        self.is_fusion_state = False

    @classmethod
    def from_dict(cls, dicts):
        scheduler = cls.__new__(cls)
        scheduler.__dict__.update(dicts)
        return scheduler

    def get_data(self):
        return ColocatedSchedulerData(
            unaccepted_queue=self.unaccepted_queue,
            waiting_queue=self.waiting_queue,
            running_queue=self.running_queue,
            swapped_queue=self.swapped_queue,
        )

    def get_num_fusion_requests(self):
        return len([
            request
            for request in (self.running_queue.requests + list(self.waiting_queue) + list(self.swapped_queue) + list(self.unaccepted_queue))
            if request.fusion_exec
        ])

    def _get_block_needed(self, length: int):
        block_size = self.block_managers[0].cache_config.block_size
        return (length + block_size - 1) // block_size

    def add_request(self, request: Request) -> None:
        # We take a simple approach here: Accept any request that comes in.
        block_size = self.block_managers[0].cache_config.block_size
        input_tokens = request.get_input_len()
        request.output_capacity = (
            self._get_block_needed(input_tokens) * block_size - input_tokens
        )
        self.unaccepted_queue.append(request)

    def abort_request(self, request_id: int) -> None:
        # scan the current batch
        for _, request in enumerate(self.running_queue.requests):
            if request.request_id == request_id:
                # This request may be under processed by the model currently,
                # so it is not safe to delete it from current batch directly.
                # Mark it as finished will release the resources it holds finally.
                request.is_finished = True
                return

        # scan the waiting queue
        for i, request in enumerate(self.waiting_queue):
            if request.request_id == request_id:
                del self.waiting_queue[i]
                return
        
        # scan the swapped queue
        for i, request in enumerate(self.swapped_queue):
            if request.request_id == request_id:
                del self.swapped_queue[i]
                return
    
    def get_latest_running_request(self, is_fusion: bool, rank: int = 0) -> Request:
        if is_fusion:
            for request in reversed(self.running_queue.requests):
                # do not swap / recompute migrating or resharding requests
                if (request.fusion_exec 
                    and not request.is_migrating_out 
                    and not request.is_resharding 
                    and not request.is_context_stage()):
                    self.running_queue.requests.remove(request)
                    return request
        else:
            for request in reversed(self.running_queue.requests):
                # do not swap / recompute migrating or resharding requests
                # logger.info(f"check request {request.request_id}, {request.fusion_exec=} at {request.kv_fusion_rank} rank, {request.is_migrating_out=}, {request.is_resharding=}, {request.is_context_stage()=}")
                if (not request.fusion_exec
                    and request.kv_fusion_rank == rank 
                    and not request.is_migrating_out 
                    and not request.is_resharding 
                    and not request.is_context_stage()
                ):
                    self.running_queue.requests.remove(request)
                    return request
        return None

    async def trigger_swap(self, is_fusion: bool = False, rank: int = 0) -> bool:
        if len(self.running_queue) == 0:
            logger.info(f"error, current running queue is empty!")
            return None

        request = self.get_latest_running_request(is_fusion, rank)
        if request is None:
            logger.info(f"fail to find a victim!")
            return request

        if self.is_fusion_state:
            # FIXME: lack of update for a long time
            if request.fusion_exec:
                idx = 0
                is_all_swapped = True
                for block_manager in self.block_managers:
                    swap_result = await block_manager.swap_out_one_request(request)
                    if not swap_result:
                        is_all_swapped = False
                        break
                    idx += 1
                if is_all_swapped:
                    self.swapped_queue.append(request)
                    self.preempt_request(request, False) # Set false because the request has already been removed from the running queue
                    return request
                # rollback
                for i in range(idx):
                    await self.block_managers[i].swap_in_one_request(request)
                self.running_queue.requests.append(request)
                return None
            else:
                # old request
                swap_result = await self.block_managers[request.kv_fusion_rank].swap_out_one_request(request)
                if swap_result:
                    self.swapped_queue.append(request)
                    self.preempt_request(request, False)
                    return request
                self.running_queue.requests.append(request)
                return None
        else:
            swap_result = await self.block_managers[rank].swap_out_one_request(request)
            if swap_result:
                self.swapped_queue.append(request)
                self.preempt_request(request, False)
                return request
            else:
                logger.error(f"fail to swap request with {request.get_num_tokens()} tokens")
            self.running_queue.requests.append(request)
            return None
        
    def trigger_recompute_for_stale_requests(self):
        removed_requests = []
        for request in reversed(self.running_queue.requests):
            if not request.fusion_exec:
                request.once_fusion_exec = True
                removed_requests.append(request)
        self.recompute_requests(removed_requests, in_running_queue=True, resched_local=True)
        
    def trigger_reshard_for_stale_requests(self, live_reshard: bool, live_reshard_batch_size: int=-1):
        valid_requests, stale_requests = self.running_queue.get_fusion_requests()

        if live_reshard:
            if live_reshard_batch_size > 0:
                valid_requests.extend(stale_requests[live_reshard_batch_size:])
                stale_requests = stale_requests[0:live_reshard_batch_size]

            self.running_queue = BatchedRequests(valid_requests)

        for request in stale_requests:
            # reshard is also seen as a badput as it has generation stall
            request.preempted_times += 1
            self.num_preemption += 1

        return stale_requests
    
    def trigger_recompute(self, is_fusion: bool, rank: int = 0, resched_local: bool = False, abort: bool = False):
        if len(self.running_queue) == 0:
            return None

        # requests that have not been allocated yet
        request = self.get_latest_running_request(is_fusion, rank)
        if request is None:
            return request       
        if self.recompute_requests([request], in_running_queue=False, resched_local=resched_local, abort=abort):
            return request
        
        # fail to preempt the request, add it back
        self.running_queue.requests.append(request)
        return None
    
    def trigger_abort(self, budget: int = 1):
        aborted_requests = []
        
        curr_req_idx = 0
        curr_released_blocks = 0
        
        while curr_released_blocks < budget and curr_req_idx < len(self.running_queue.requests):
            request = self.running_queue.requests[curr_req_idx]
            aborted_requests.append(request)
            curr_released_blocks += self._get_block_needed(request.get_num_tokens())
            curr_req_idx += 1
        self.running_queue.requests = self.running_queue.requests[curr_req_idx:]
        self.recompute_requests(aborted_requests, in_running_queue=False, abort=True)
        return aborted_requests

    async def swap_out_requests(self, requests: List[Request]) -> bool:
        def _swap_out_one_request(request: Request):
            if len(self.block_managers) > 0:
                if request.fusion_exec:
                    idx = 0
                    is_all_swapped = True
                    for block_manager in self.block_managers:
                        if not block_manager.swap_out_one_request(request):
                            is_all_swapped = False
                            return False
                        idx += 1
                    if is_all_swapped:
                        return True
                    # rollback
                    for i in range(idx):
                        self.block_managers[i].swap_in_one_request(request)
                    return False
                else:
                    if self.block_managers[request.kv_fusion_rank].swap_out_one_request(request):
                        return True
                    return False
            else:
                if self.block_managers[0].swap_out_one_request(request):
                    return True
                return False
                    
        def _swap_in_one_request(request: Request):
            if len(self.block_managers) > 0:
                if request.fusion_exec:
                    for block_manager in self.block_managers:
                        block_manager.swap_in_one_request(request)
                else:
                    self.block_managers[request.kv_fusion_rank].swap_in_one_request(request)
            else:
                self.block_managers[0].swap_in_one_request(request)

        # logger.info(f"start swapping {len(requests)} requests, number of block managers: {len(self.block_managers)}")

        if len(self.block_managers) > 1:
            idx = 0
            is_all_requests_swapped = True
            for request in requests:
                if not _swap_out_one_request(request):
                    is_all_requests_swapped = False
                    break
                idx += 1
            if is_all_requests_swapped:
                self.swapped_queue.extend(requests)
                self.preempt_requests(requests, True)
                return True
            # rollback
            for i in range(idx):
                _swap_in_one_request(requests[i])
            return False
        else:
            swap_result = self.block_managers[0].swap_out_requests(requests)
            # logger.info(f"finish swapping out {len(requests)} requests, swap result is {swap_result}, request location is {self.block_managers[0].get_location(requests[0].request_id)}")
            if swap_result:
                self.swapped_queue.extend(requests)
                # Remove the swapped out requests from the running queue
                self.preempt_requests(requests, True)
                return True
            return False
    
    def recompute_requests(self, requests: List[Request], in_running_queue: bool = True, resched_local: bool = False, abort: bool = False) -> bool:
        # Free KVCache of the requests; remove them from the running queue
        # first (if still there) so the block-manager release is consistent.
        self.preempt_requests(requests, in_running_queue)
        for request in requests:
            if request.fusion_exec:
                for i, block_manager in enumerate(self.block_managers):
                    block_manager.free_blocks(request)
            else:
                self.block_managers[request.kv_fusion_rank].free_blocks(request)

        handlers = self._remote_call_all_workers_async(
            "clear_request_resource_batched", requests
        )
        
        # add current generated tokens to the prompt
        for request in requests:
            request.prepare_for_recompute(self.is_fusion_state)
            
        # recompute the requests
        if not abort:
            for request in requests:
                self.unaccepted_queue.appendleft(request)
        return True
    
    def preempt_request(self, request: Request, in_running_queue: bool):
        if in_running_queue:
            self.running_queue.requests.remove(request)
        request.preempted_times += 1
        request.preempting = False # reset the preempt flag
        self.num_preemption += 1
    
    def preempt_requests(self, requests: List[Request], in_running_queue: bool):
        for request in requests:
            self.preempt_request(request, in_running_queue)

    def swap_out_requests_to_waiting_queue(self, requests: List[Request]) -> bool:
        self.waiting_queue.extend(requests)
        # Remove the swapped out requests from the running queue
        for request in requests:
            self.running_queue.requests.remove(request)
        return True

    async def swap_in_request(self) -> None:
        assert len(self.swapped_queue) > 0, "No request to swap in."
        request = self.swapped_queue.popleft()
        if self.is_fusion_state:
            # FIXME: lack of update for a long time
            if request.fusion_exec:
                for block_manager in self.block_managers:
                    await block_manager.swap_in_one_request(request)
            else:
                await self.block_managers[request.kv_fusion_rank].swap_in_one_request(request)
        else:
            await self.block_managers[request.kv_fusion_rank].swap_in_one_request(request)
        self.running_queue.add_request(request)

    def swap_in_from_waiting_queue(self, request: Request) -> None:
        assert len(self.waiting_queue) > 0, "No request to swap in."
        self.running_queue.add_request(request)
        self.waiting_queue.remove(request)

    def _is_allocated(self, request: Request) -> bool:
        if len(self.block_managers) > 0:
            if request.fusion_exec:
                return all([block_manager.is_allocated(request.request_id) for block_manager in self.block_managers])
            else:
                # FIXME: can cause bug in pipeline parallelism?
                # logger.info(f"check allocation for request at {request.kv_fusion_rank} rank, with {len(self.block_managers)} block managers")
                return self.block_managers[request.kv_fusion_rank].is_allocated(request.request_id)
        else:
            return self.block_managers[0].is_allocated(request.request_id)
        
    def _is_valid_for_preempt(self, request: Request, is_fusion: bool = None) -> bool:
        # we only preempted requests that have generated at least one token
        if self.is_fusion_state:
            return request.fusion_exec and not request.is_context_stage() and self._is_allocated(request)
        return (is_fusion == None or request.fusion_exec == is_fusion) and not request.is_context_stage() and self._is_allocated(request)

    # FIXME: found victim for different type of requests
    def try_get_victim_request(self, preempt_limit: int, is_fusion: bool = None) -> Request:
        # preempt_limit: the maximum number of times a request can be preempted, <0 means no limit
        
        old_requests = [request for request in self.running_queue.requests if self._is_valid_for_preempt(request, is_fusion)]
        if len(old_requests) == 0:
            return None
        
        # FCFS preemption, guarantee that newer requests will not be preempted
        victim_requests = sorted(old_requests, key=lambda r: r.arrival_time, reverse=True)

        for victim in victim_requests:
            # logger.info(f"{preempt_limit=}, req preempt times: {victim.preempted_times}")
            if not victim.preempting and (preempt_limit < 0 or victim.preempted_times < preempt_limit):
                victim.preempting = True
                return victim
        return None

    def fetch_unaccepted_queue(self) -> Request:
        if len(self.unaccepted_queue) == 0:
            return None
        return self.unaccepted_queue[0]

    def fetch_swapped_queue(self) -> Request:
        if len(self.swapped_queue) == 0:
            return None
        return self.swapped_queue[0]
    
    def fetch_waiting_queue(self) -> Request:
        if len(self.waiting_queue) == 0:
            return None
        for request in self.waiting_queue:
            if request.kvcache_ready:
                return request
        return None

    def pop_unaccepted_queue(self) -> Request:
        if len(self.unaccepted_queue) == 0:
            return None
        return self.unaccepted_queue.popleft()
    
    def pop_tail_unaccepted_queue(self) -> Request:
        if len(self.unaccepted_queue) == 0:
            return None
        return self.unaccepted_queue.pop()
    
    def pop_swapped_queue(self) -> Request:
        if len(self.swapped_queue) == 0:
            return None
        return self.swapped_queue.popleft()

    def add_to_cur_batch(self, request: Request) -> None:
        self.running_queue.add_request(request)

    def schedule(self, prefill_only_chunk: bool = False) -> Tuple[BatchedRequests, bool]:
        prefill_requests, decode_requests = self.running_queue.split_requests()
        # we use prefill-first scheduling by default
        if self.sched_config.enable_chunked_prefill and not prefill_only_chunk:
            # our chunked require prefill to be ahead of decode
            return BatchedRequests(prefill_requests + decode_requests), True
        else:
            if len(prefill_requests):
                return BatchedRequests(prefill_requests), True
            return BatchedRequests(decode_requests), False

    def pop_finished_requests(self) -> List[Request]:
        return self.running_queue.pop_finished_requests()
    
    def pop_finished_remote_attn_requests(self) -> List[Request]:
        return self.running_queue.pop_finished_remote_attn_requests()
    
    def pop_unassigned_requests(self) -> List[Request]:
        """This method can only be used in fusion pipeline. It pops the chunked requests from current running queue, and schedule it to other virtual engine."""
        return self.running_queue.pop_unassigned_requests()

    def get_num_unfinished_requests(self) -> int:
        return len(self.unaccepted_queue) +  len(self.swapped_queue) + len(self.running_queue) + len(self.waiting_queue)
    
    def get_num_decode_requests(self) -> int:
        num_decodes = 0
        for request in self.running_queue.requests:
            if not request.is_context_stage():
                num_decodes += 1
        return num_decodes

    def pop_running_requests_transfer(self, is_pp: bool, num_blocks_limit: int) -> List[Request]:
        total_blocks = 0
        poped_requests, remained_requests = [], []
        for request in self.running_queue.requests:
            blocks = self._get_block_needed(request.get_num_tokens())
            if request.fusion_exec == is_pp and total_blocks + blocks <= num_blocks_limit:
                poped_requests.append(request)
                total_blocks += blocks
            else:
                remained_requests.append(request)
        self.running_queue.requests = remained_requests
        return poped_requests

    def get_total_gpu_blocks(self, is_fusion: bool, rank: int):
        total_gpu_blocks = 0
        if is_fusion:
            for req in self.running_queue.requests + list(self.waiting_queue):
                if req.fusion_exec:
                    total_gpu_blocks += self._get_block_needed(req.get_num_tokens())
        else:
            for req in self.running_queue.requests + list(self.waiting_queue):
                if not req.fusion_exec and req.kv_fusion_rank == rank:
                    total_gpu_blocks += self._get_block_needed(req.get_num_tokens())
        return total_gpu_blocks

    def get_total_gpu_blocks_needed(self):
        total_gpu_blocks = 0
        num_instances = len(self.block_managers)
        for req in self.running_queue.requests + list(self.waiting_queue):
            total_gpu_blocks += self._get_block_needed(req.get_num_tokens())
        for req in self.unaccepted_queue:
            total_gpu_blocks += self._get_block_needed(req.get_num_tokens())
        return total_gpu_blocks

    def print_status(self) -> None:
        logger.info(
            f"(colocated) {len(self.unaccepted_queue)} unaccepted, {self.get_processing_num_requests()} processing"
        )

    # Getter functions
    def get_total_num_requests(self) -> int:
        return self.get_processing_num_requests() + self.get_waiting_num_requests()

    def get_processing_num_requests(self) -> int:
        return len(self.running_queue)

    def get_waiting_num_requests(self) -> int:
        return len(self.waiting_queue)

    def __repr__(self) -> str:
        return (
            f"PColocatedScheduler(max_batch_size={self.sched_config.max_batch_size}, "
            f"max_tokens_per_batch={self.sched_config.max_tokens_per_batch})"
        )

class ColocatedSchedulerData:
    def __init__(
            self, 
            unaccepted_queue: Optional[List[Request]] = None,
            waiting_queue: Optional[List[Request]] = None,
            running_queue: Optional[BatchedRequests] = None,
            swapped_queue: Optional[List[Request]] = None,
    ):
        if unaccepted_queue is None:
            unaccepted_queue = deque()
        if waiting_queue is None:
            waiting_queue = deque()
        if running_queue is None:
            running_queue = BatchedRequests()
        if swapped_queue is None:
            swapped_queue = deque()

        self.unaccepted_queue = unaccepted_queue
        self.waiting_queue = waiting_queue
        self.running_queue = running_queue
        self.swapped_queue = swapped_queue


def get_colocated_scheduler(
    sched_config: ColocatedSchedConfig,
    parallel_config: ParallelConfig,
    model_config: ModelConfig,
    block_managers: List[BlockManager],
    _remote_call_all_workers_async: Callable,
) -> ColocatedScheduler:
    return ColocatedScheduler(
        sched_config, parallel_config, model_config, block_managers, _remote_call_all_workers_async
    )
