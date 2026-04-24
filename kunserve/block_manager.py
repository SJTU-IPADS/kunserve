import time
from typing import Dict, List, Callable
from enum import Enum
from dataclasses import dataclass
import asyncio

from kunserve.config import ModelConfig, ParallelConfig, CacheConfig
from kunserve.request import Request, BatchedRequests
from kunserve.logger import init_logger
from kunserve.utils import Stage

logger = init_logger(__name__)


class BlockLocation(Enum):
    """The location of a block"""

    GPU = "gpu"
    MOCK = "mock"
    CPU = "cpu"

class BlockManagerData:
    def __init__(
        self,
        free_gpu_blocks_list,
        free_cpu_blocks_list,
        reserved_gpu_blocks_list,
        reserved_gpu_blocks_set,
        free_fusion_gpu_blocks_list,
        free_extend_gpu_blocks_list,
        is_fusion_state,
        mock_block,
        backup_gpu_blocks,
        backup_free_gpu_blocks_list,
        swapping_gpu_blocks_list,
        swapping_cpu_blocks_list,
        block_table,
        request_location,
        num_base_gpu_blocks,
        max_num_fusion_gpu_blocks,
        fusion_size,
        max_num_reserved_gpu_blocks,
        max_num_extend_gpu_blocks,
    ):
        self.free_gpu_blocks_list = free_gpu_blocks_list
        self.free_cpu_blocks_list = free_cpu_blocks_list

        self.reserved_gpu_blocks_list = reserved_gpu_blocks_list
        self.reserved_gpu_blocks_set = reserved_gpu_blocks_set
        self.free_fusion_gpu_blocks_list = free_fusion_gpu_blocks_list
        self.free_extend_gpu_blocks_list = free_extend_gpu_blocks_list
        self.is_fusion_state = is_fusion_state
        self.mock_block = mock_block

        # activated when some model parameters are kicked out, the id start from max_num_gpu_blocks to
        self.backup_gpu_blocks = backup_gpu_blocks
        self.backup_free_gpu_blocks_list = backup_free_gpu_blocks_list

        self.swapping_gpu_blocks_list = swapping_gpu_blocks_list
        self.swapping_cpu_blocks_list = swapping_cpu_blocks_list

        # request_id => [block0_id, block1_id, ...]
        # If the blocks of the request are on GPU, then block0_id, block1_id are GPU block
        # ids, and vice versa
        self.block_table = block_table

        # request_id => BlockLocation
        self.request_location = request_location
        self.num_base_gpu_blocks = num_base_gpu_blocks
        self.max_num_fusion_gpu_blocks = max_num_fusion_gpu_blocks
        self.fusion_size = fusion_size
        self.max_num_reserved_gpu_blocks = max_num_reserved_gpu_blocks
        self.max_num_extend_gpu_blocks = max_num_extend_gpu_blocks


class BlockManager:
    """A Block Manager that maintains the key-value cache in block-level"""

    """For subroutines and algorithms related to swapping, please refer to
    the big comment block above swap_requests()"""

    def __init__(
        self,
        stage: Stage,
        prefix: str,
        max_num_gpu_blocks: int,
        max_num_cpu_blocks: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        engine_remote_call_all_workers_async: Callable,
    ):
        self.stage = stage
        self.prefix = prefix
        self.max_num_gpu_blocks = max_num_gpu_blocks
        self.max_num_cpu_blocks = max_num_cpu_blocks
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.engine_remote_call_all_workers_async = engine_remote_call_all_workers_async

        self.free_gpu_blocks_list = list(range(max_num_gpu_blocks))
        self.free_cpu_blocks_list = list(range(max_num_cpu_blocks))

        self.reserved_gpu_blocks_list = []
        self.reserved_gpu_blocks_set = set()
        self.free_fusion_gpu_blocks_list = []
        self.free_extend_gpu_blocks_list = []
        self.is_fusion_state = False
        self.mock_block = 0

        # activated when some model parameters are kicked out, the id start from max_num_gpu_blocks to
        self.backup_gpu_blocks = 0
        self.backup_free_gpu_blocks_list = []

        self.swapping_gpu_blocks_list = []
        self.swapping_cpu_blocks_list = []

        # request_id => [block0_id, block1_id, ...]
        # If the blocks of the request are on GPU, then block0_id, block1_id are GPU block
        # ids, and vice versa
        self.block_table = {}

        # request_id => BlockLocation
        self.request_location = {}
        
        self.num_base_gpu_blocks = self.max_num_gpu_blocks
        self.num_extend_gpu_blocks = 0

        self.max_num_fusion_gpu_blocks = self.max_num_gpu_blocks
        
        self.fusion_size = 1
        self.max_num_reserved_gpu_blocks = 0
        self.max_num_extend_gpu_blocks = 0
        # NOTE: distinguish from `max_num_reserved_gpu_blocks`
        # the number of free reserved blocks will be kept less than or equal to `num_reserved_blocks`
        self.num_reserved_blocks = 0

        # record the allocation time for each extend block
        self.allocation_times: Dict[int, int] = {}
        self.max_allocation_times = cache_config.max_allocation_times

    @classmethod
    def from_dict(cls, dicts):
        bm = cls.__new__(cls)
        bm.__dict__.update(dicts)
        return bm

    def get_block_manager_data(self):
        return BlockManagerData(
            self.free_gpu_blocks_list,
            self.free_cpu_blocks_list,
            self.reserved_gpu_blocks_list,
            self.reserved_gpu_blocks_set,
            self.free_fusion_gpu_blocks_list,
            self.free_extend_gpu_blocks_list,
            self.is_fusion_state,
            self.mock_block,
            self.backup_gpu_blocks,
            self.backup_free_gpu_blocks_list,
            self.swapping_gpu_blocks_list,
            self.swapping_cpu_blocks_list,
            self.block_table,
            self.request_location,
            self.num_base_gpu_blocks,
            self.max_num_fusion_gpu_blocks,
            self.fusion_size,
            self.max_num_reserved_gpu_blocks,
            self.max_num_extend_gpu_blocks,
        )

    def set_block_manager_data(self, data: BlockManagerData):
        self.free_gpu_blocks_list = data.free_gpu_blocks_list
        self.free_cpu_blocks_list = data.free_cpu_blocks_list
        self.reserved_gpu_blocks_list = data.reserved_gpu_blocks_list
        self.reserved_gpu_blocks_set = data.reserved_gpu_blocks_set
        self.free_fusion_gpu_blocks_list = data.free_fusion_gpu_blocks_list
        self.free_extend_gpu_blocks_list = data.free_extend_gpu_blocks_list
        self.is_fusion_state = data.is_fusion_state
        self.mock_block = data.mock_block
        self.backup_gpu_blocks = data.backup_gpu_blocks
        self.backup_free_gpu_blocks_list = data.backup_free_gpu_blocks_list
        self.swapping_gpu_blocks_list = data.swapping_gpu_blocks_list
        self.swapping_cpu_blocks_list = data.swapping_cpu_blocks_list
        self.block_table = data.block_table
        self.request_location = data.request_location
        self.num_base_gpu_blocks = data.num_base_gpu_blocks
        self.max_num_fusion_gpu_blocks = data.max_num_fusion_gpu_blocks
        self.fusion_size = data.fusion_size
        self.max_num_reserved_gpu_blocks = data.max_num_reserved_gpu_blocks
        self.max_num_extend_gpu_blocks = data.max_num_extend_gpu_blocks
        
    def convert_to_pipeline_format_w_reserve(self, num_pipeline_stages: int):
        """The only difference from `convert_to_pipeline_format` is that the reserved blocks are translated directly into fusion blocks."""
        
        
        """In restore->balloon, the meaning of `reserved blocks` changes from pipeline blocks to normal blocks"""
        max_gpu_blocks = self.max_num_gpu_blocks + self.max_num_reserved_gpu_blocks // num_pipeline_stages
        self.free_fusion_gpu_blocks_list = self.reserved_gpu_blocks_list
        self.reserved_gpu_blocks_list = []
        self.num_reserved_blocks = max_gpu_blocks
        
        """The blocks owned by normal requests are reserved (only for live-reshard)."""
        self.max_num_reserved_gpu_blocks = self.max_num_gpu_blocks - len(self.free_gpu_blocks_list)
        self.max_num_gpu_blocks = 0

        if self.num_reserved_blocks <= len(self.free_gpu_blocks_list):
            self.reserved_gpu_blocks_list = self.free_gpu_blocks_list[:self.num_reserved_blocks]
            self.free_gpu_blocks_list = self.free_gpu_blocks_list[self.num_reserved_blocks:]
        else:
            self.reserved_gpu_blocks_list = self.free_gpu_blocks_list
            self.free_gpu_blocks_list = []
        
        self.reserved_gpu_blocks_set = set(self.reserved_gpu_blocks_list)
        
        """reserved blocks = blocks own by stale requests + blocks reserved for their future generation"""
        self.max_num_reserved_gpu_blocks += len(self.reserved_gpu_blocks_list)
        # convert all remaining free blocks to pipeline blocks
        for block in self.free_gpu_blocks_list:
            self.free_fusion_gpu_blocks_list += [
                i * self.num_base_gpu_blocks + block for i in range(num_pipeline_stages)
            ]
        self.free_gpu_blocks_list = []
        self.num_base_gpu_blocks = max_gpu_blocks * num_pipeline_stages

        # the reserved blocks will be converted into fusion blocks gradually
        self.max_num_fusion_gpu_blocks = (max_gpu_blocks - self.max_num_reserved_gpu_blocks) * num_pipeline_stages
        self.is_fusion_state = True
        self.fusion_size = num_pipeline_stages
        

    def convert_to_pipeline_format(self, num_pipeline_stages: int):
        self.num_reserved_blocks = self.max_num_gpu_blocks
        self.max_num_reserved_gpu_blocks = self.max_num_gpu_blocks - len(self.free_gpu_blocks_list)

        self.num_base_gpu_blocks = self.max_num_gpu_blocks
        self.max_num_gpu_blocks = 0
        
        if self.num_reserved_blocks <= len(self.free_gpu_blocks_list):
            self.reserved_gpu_blocks_list = self.free_gpu_blocks_list[:self.num_reserved_blocks]
            self.free_gpu_blocks_list = self.free_gpu_blocks_list[self.num_reserved_blocks:]
        else:
            self.reserved_gpu_blocks_list = self.free_gpu_blocks_list
            self.free_gpu_blocks_list = []
        
        self.reserved_gpu_blocks_set = set(self.reserved_gpu_blocks_list)        
        self.max_num_reserved_gpu_blocks += len(self.reserved_gpu_blocks_list)

        # convert all remaining free blocks to pipeline blocks
        for block in self.free_gpu_blocks_list:
            self.free_fusion_gpu_blocks_list += [
                self.num_base_gpu_blocks * i + block for i in range(num_pipeline_stages)
            ]
        self.free_gpu_blocks_list = []

        # the reserved blocks will be converted into fusion blocks gradually
        self.num_base_gpu_blocks *= num_pipeline_stages
        self.max_num_fusion_gpu_blocks = len(self.free_fusion_gpu_blocks_list)
        self.is_fusion_state = True
        self.fusion_size = num_pipeline_stages
            
    def add_ex_fusion_blocks(self, ex_blocks: int):
        for ex_block in range(ex_blocks):
            self.free_extend_gpu_blocks_list.append(
                self.num_base_gpu_blocks + ex_block
            )
        self.max_num_extend_gpu_blocks = len(self.free_extend_gpu_blocks_list)
        # add this value because max_num_extend_gpu_blocks will be changed when waiting for ex blocks to return
        self.num_extend_gpu_blocks = self.max_num_extend_gpu_blocks

    def convert_all_reserved_blocks_into_fusion(self):
        """
        This fusion is called after `convert_info_fusion`, where `reserved_gpu_blocks_list` represents stale blocks and 
        `free_fusion_gpu_blocks_list` inherits all pipeline blocks from PREPARE state (see reserve_for_pipeline).
        """
        for block in self.reserved_gpu_blocks_list:
            for i in range(self.fusion_size):
                self.free_fusion_gpu_blocks_list.append(self.num_base_gpu_blocks // self.fusion_size * i + block)
        self.reserved_gpu_blocks_list = []
        self.max_num_fusion_gpu_blocks += self.max_num_reserved_gpu_blocks * self.fusion_size
        self.max_num_reserved_gpu_blocks = 0
        
    async def wait_for_ex_blocks_to_return(self):
        while self.max_num_extend_gpu_blocks > 0:
            # prevent returned_ex_blocks being allocated again
            free_extend_blocks = len(self.free_extend_gpu_blocks_list)
            self.max_num_extend_gpu_blocks -= free_extend_blocks
            if free_extend_blocks:
                logger.info(f"reclaim {free_extend_blocks} free blocks in the extended region")
            self.free_extend_gpu_blocks_list = []
            await asyncio.sleep(0)
        self.num_extend_gpu_blocks = 0

    def restore_kvblocks(self, is_reserved: bool = True):
        self.num_base_gpu_blocks = self.num_base_gpu_blocks // self.fusion_size
        logger.info(f"[restore_kvblocks] num base blocks is {self.num_base_gpu_blocks}, reserved gpu blocks: {self.max_num_reserved_gpu_blocks}")

        self.free_gpu_blocks_list = self.reserved_gpu_blocks_list
        self.max_num_gpu_blocks = self.max_num_reserved_gpu_blocks
        
        self.num_reserved_blocks = int(self.num_base_gpu_blocks * self.fusion_size * self.cache_config.reserved_blocks_ratio) if is_reserved else 0
        self.free_fusion_gpu_blocks_list.sort(key=lambda x: x % self.num_base_gpu_blocks)
        
        if self.num_reserved_blocks <= len(self.free_fusion_gpu_blocks_list):
            self.max_num_reserved_gpu_blocks = self.max_num_fusion_gpu_blocks - (len(self.free_fusion_gpu_blocks_list) - self.num_reserved_blocks)
            self.reserved_gpu_blocks_list = self.free_fusion_gpu_blocks_list[:self.num_reserved_blocks]
            self.free_fusion_gpu_blocks_list = self.free_fusion_gpu_blocks_list[self.num_reserved_blocks:]
        else:
            logger.info(f"[restore_kvblocks] all free blocks are reserved (demand: {self.num_reserved_blocks}) for old requests, {len(self.free_fusion_gpu_blocks_list)}")
            self.max_num_reserved_gpu_blocks = self.max_num_fusion_gpu_blocks
            self.reserved_gpu_blocks_list = self.free_fusion_gpu_blocks_list
            self.free_fusion_gpu_blocks_list = []
            
        # convert the rest of the fusion blocks into base blocks
        base_blocks = []
        for i, block in enumerate(self.free_fusion_gpu_blocks_list):
            if (
                block < self.num_base_gpu_blocks
                and (i + self.fusion_size - 1) < len(self.free_fusion_gpu_blocks_list)
                and block == (self.free_fusion_gpu_blocks_list[i + self.fusion_size - 1] % self.num_base_gpu_blocks)
            ):
                base_blocks.append(block)
        for block in base_blocks:
            self.free_gpu_blocks_list.append(block)
            for i in range(self.fusion_size):
                self.free_fusion_gpu_blocks_list.remove(block + i * self.num_base_gpu_blocks)
        
        self.max_num_gpu_blocks += len(base_blocks)
        self.reserved_gpu_blocks_list.extend(self.free_fusion_gpu_blocks_list)
        self.reserved_gpu_blocks_set = set()
        self.max_num_reserved_gpu_blocks += len(self.free_fusion_gpu_blocks_list)
        
        logger.info(f"[restore_kvblocks] gpu blocks: {len(self.free_gpu_blocks_list)}/{self.max_num_gpu_blocks}, reserved gpu blocks: {len(self.reserved_gpu_blocks_list)}/{self.max_num_reserved_gpu_blocks}")
        self.free_fusion_gpu_blocks_list = []
        self.max_num_fusion_gpu_blocks = 0

    def restore_from_fusion(self):
        self.max_num_gpu_blocks += self.max_num_reserved_gpu_blocks // self.fusion_size
        self.max_num_reserved_gpu_blocks = 0
        assert self.max_num_gpu_blocks == self.num_base_gpu_blocks, f"{self.max_num_gpu_blocks=}, {self.num_base_gpu_blocks=}"
        
        for i, block in enumerate(self.reserved_gpu_blocks_list):
            if block < self.num_base_gpu_blocks:
                self.free_gpu_blocks_list.append(block)
        self.reserved_gpu_blocks_list = []
        
        self.is_fusion_state = False
        self.fusion_size = 1
        self.num_base_gpu_blocks = self.max_num_gpu_blocks

    def _fetch_an_origin_block(self, index, fusion_size):
        for i in range(1, fusion_size):
            # find contiguous blocks
            if (
                index + i >= len(self.free_fusion_gpu_blocks_list)
                or self.free_gpu_blocks_list[index + i]
                != self.free_gpu_blocks_list[index] + i
            ):
                return index + i
        return index + fusion_size

    def get_num_avail_gpu_blocks(self) -> int:
        """Get the number of available GPU blocks"""
        return len(self.free_gpu_blocks_list) + len(self.swapping_gpu_blocks_list)

    def get_num_avail_fusion_blocks(self) -> int:
        return len(self.free_fusion_gpu_blocks_list)

    def get_num_avail_extend_blocks(self) -> int:
        return len(self.free_extend_gpu_blocks_list)
    
    def get_num_used_extend_blocks(self) -> int:
        return self.max_num_extend_gpu_blocks - self.get_num_avail_extend_blocks()

    def get_num_avail_reserved_blocks(self) -> int:
        return len(self.reserved_gpu_blocks_list)

    def get_num_avail_cpu_blocks(self) -> int:
        """Get the number of available CPU blocks"""
        return len(self.free_cpu_blocks_list) + len(self.swapping_cpu_blocks_list)

    def _get_free_blocks(self, num_blocks: int, location: BlockLocation) -> List[int]:
        """Get free blocks from the free block pool indicated by `location`"""
        """When `location` is GPU, the returned blocks are on GPU, and vice versa"""
        assert location in [BlockLocation.GPU, BlockLocation.CPU]
        if location == BlockLocation.GPU:
            num_avail_blocks = self.get_num_avail_gpu_blocks()
            assert (
                num_avail_blocks >= num_blocks
            ), f"not enough free blocks on GPU, requested {num_blocks}, available {num_avail_blocks}"
            if len(self.free_gpu_blocks_list) < num_blocks:
                # Need to "flush" self.swapping_gpu_blocks_list, i.e. make sure all
                # swapping-out operations have finished, thus blocks in self.swapping_gpu_blocks_list
                # can be moved to self.free_gpu_blocks_list
                self.engine_remote_call_all_workers_async("wait_for_all_swap_out")
                self.free_gpu_blocks_list += self.swapping_gpu_blocks_list
                self.swapping_gpu_blocks_list = []
            blocks = self.free_gpu_blocks_list[:num_blocks]
            self.free_gpu_blocks_list = self.free_gpu_blocks_list[num_blocks:]
        else:
            num_avail_blocks = self.get_num_avail_cpu_blocks()
            # If CPU is full, reject the swap request
            if num_avail_blocks < num_blocks:
                return None
            if len(self.free_cpu_blocks_list) < num_blocks:
                # Need to "flush" self.swapping_cpu_blocks_list, i.e. make sure all
                # swapping-in operations have finished, thus blocks in self.swapping_cpu_blocks_list
                # can be moved to self.free_cpu_blocks_list
                self.engine_remote_call_all_workers_async("wait_for_all_swap_in")
                self.free_cpu_blocks_list += self.swapping_cpu_blocks_list
                self.swapping_cpu_blocks_list = []
            blocks = self.free_cpu_blocks_list[:num_blocks]
            self.free_cpu_blocks_list = self.free_cpu_blocks_list[num_blocks:]
        return blocks

    def get_free_gpu_blocks(self, num_blocks: int) -> List[int]:
        try:
            blocks = self._get_free_blocks(num_blocks, BlockLocation.GPU)
            return blocks
        except:
            return []

    def _get_free_fusion_blocks(
        self, num_blocks: int, location: BlockLocation
    ) -> List[int]:
        assert location in [BlockLocation.GPU, BlockLocation.CPU]
        if location == BlockLocation.GPU:
            num_avail_blocks = self.get_num_avail_fusion_blocks()
            assert (
                num_avail_blocks >= num_blocks
            ), f"not enough free blocks on GPU, requested {num_blocks}, available {num_avail_blocks}"
            if len(self.free_gpu_blocks_list) < num_blocks:
                # Flush swapping_gpu_blocks_list before reusing freed GPU blocks.
                self.engine_remote_call_all_workers_async("wait_for_all_swap_out")
                self.free_gpu_blocks_list += self.swapping_gpu_blocks_list
                self.swapping_gpu_blocks_list = []
            blocks = self.free_fusion_gpu_blocks_list[:num_blocks]
            self.free_fusion_gpu_blocks_list = self.free_fusion_gpu_blocks_list[
                num_blocks:
            ]
        else:
            assert False, "do not support free CPU blocks to fusion yet"
        return blocks

    def _get_free_extend_blocks(
        self, num_blocks: int, location: BlockLocation
    ) -> List[int]:
        assert location in [BlockLocation.GPU, BlockLocation.CPU]
        if location == BlockLocation.GPU:
            num_avail_blocks = self.get_num_avail_extend_blocks()
            assert (
                num_avail_blocks >= num_blocks
            ), f"not enough free blocks on GPU, requested {num_blocks}, available {num_avail_blocks}"
            if len(self.free_gpu_blocks_list) < num_blocks:
                # Flush swapping_gpu_blocks_list before reusing freed GPU blocks.
                self.engine_remote_call_all_workers_async("wait_for_all_swap_out")
                self.free_gpu_blocks_list += self.swapping_gpu_blocks_list
                self.swapping_gpu_blocks_list = []
            blocks = self.free_extend_gpu_blocks_list[:num_blocks]
            self.free_extend_gpu_blocks_list = self.free_extend_gpu_blocks_list[
                num_blocks:
            ]
        else:
            assert False, "do not support free CPU blocks to fusion yet"
        return blocks

    def _get_free_reserved_blocks(
        self, num_blocks: int, location: BlockLocation
    ) -> List[int]:
        assert location in [BlockLocation.GPU, BlockLocation.CPU]
        if location == BlockLocation.GPU:
            num_avail_blocks = self.get_num_avail_reserved_blocks()
            assert (
                num_avail_blocks >= num_blocks
            ), f"not enough free blocks on GPU, requested {num_blocks}, available {num_avail_blocks}"
            blocks = self.reserved_gpu_blocks_list[:num_blocks]
            self.reserved_gpu_blocks_list = self.reserved_gpu_blocks_list[num_blocks:]
        else:
            assert False, "do not support free CPU blocks to fusion yet"
        return blocks

    def get_allocated_num_blocks(self, request_id: int) -> int:
        """Get the number of allocated blocks for a request"""
        return len(self.block_table.get(request_id, []))

    def is_allocated(self, request_id):
        return request_id in self.block_table
    
    def get_location(self, request_id: int) -> BlockLocation:
        """Get the kvcache blocks location of a request"""
        return self.request_location.get(request_id, None)

    def get_num_blocks_needed(self, request: Request):
        """Get the number of blocks needed for a request"""
        num_blocks_needed = (
            request.get_num_tokens()
            + self.cache_config.block_size
            - 1
        ) // self.cache_config.block_size
        return num_blocks_needed

    def get_num_append_blocks_needed(self, request: Request) -> int:
        """Get the number of blocks needed for a request already in GPU"""
        assert (
            self.request_location[request.request_id] == BlockLocation.GPU
        ), f"request {request.request_id} is not on GPU when calling get_num_append_blocks_needed"
        num_blocks_cur = len(self.block_table[request.request_id])
        num_blocks_needed = self.get_num_blocks_needed(request)
        return num_blocks_needed - num_blocks_cur

    def _allocate_fusion_blocks(self, request: Request, live_reshard: bool = False):
        assert (
            request.request_id not in self.block_table
            or self.request_location.get(request.request_id, None) == BlockLocation.GPU
        ), f"request {request.request_id} is currently on CPU. Please migrate it to GPU before allocating ore blocks"

        num_blocks_needed = self.get_num_blocks_needed(request)
        if live_reshard: # allocate one more block for live-resharding requests
            num_blocks_needed += 1
        
        if request.request_id in self.block_table:
            num_blocks_cur = len(self.block_table[request.request_id])
            if num_blocks_cur < num_blocks_needed:
                num_blocks_needed -= num_blocks_cur
            else:
                return
            
        if self.get_num_avail_fusion_blocks() >= num_blocks_needed:
            blocks = self._get_free_fusion_blocks(num_blocks_needed, BlockLocation.GPU)
        elif self.get_num_avail_fusion_blocks() + self.get_num_avail_extend_blocks() >= num_blocks_needed:
            num_avail_fusion_blocks = self.get_num_avail_fusion_blocks()
            blocks = (
                self._get_free_fusion_blocks(num_avail_fusion_blocks, BlockLocation.GPU)
                + self._get_free_extend_blocks(num_blocks_needed - num_avail_fusion_blocks, BlockLocation.GPU)
            )
        else:
            assert False, f"not enough free blocks on GPU for request {request.request_id}, {request.kv_fusion_rank}, {request.is_context_stage()}"
        
        if request.request_id not in self.block_table:
            self.block_table[request.request_id] = blocks
        else:
            self.block_table[request.request_id] += blocks
        self.request_location[request.request_id] = BlockLocation.GPU
                
    def _allocate_reserved_blocks(self, request: Request):
        assert (
            request.request_id not in self.block_table
            or self.request_location.get(request.request_id, None) == BlockLocation.GPU
        ), f"request {request.request_id} is currently on CPU. Please migrate it to GPU before allocating ore blocks"

        num_blocks_needed = self.get_num_blocks_needed(request)
        
        if request.request_id not in self.block_table:
            # This request has not been allocated before
            self.block_table[request.request_id] = self._get_free_reserved_blocks(
                num_blocks_needed, BlockLocation.GPU
            )
            self.request_location[request.request_id] = BlockLocation.GPU
        else:
            assert self.request_location[request.request_id] == BlockLocation.GPU
            num_blocks_cur = len(self.block_table[request.request_id])
            if num_blocks_cur < num_blocks_needed:
                self.block_table[request.request_id] += self._get_free_reserved_blocks(
                    num_blocks_needed - num_blocks_cur, BlockLocation.GPU
                )

    def check_blocks(self, request: Request):
        num_blocks_needed = self.get_num_blocks_needed(request)
        if request.request_id not in self.block_table:
            # This request has not been allocated before
            return num_blocks_needed
        else:
            num_blocks_cur = len(self.block_table[request.request_id])
            return num_blocks_needed - num_blocks_cur
    
    def check_blocks_batched(self, batch_requests: BatchedRequests):
        num_blocks_needed = 0
        for request in batch_requests.requests:
            num_blocks_needed += self.check_blocks(request)
        
        return num_blocks_needed
    
    def can_allocate(self, request: Request):
        if request.request_id not in self.block_table:
            num_blocks_needed = self.get_num_blocks_needed(request)
        else:
            num_blocks_cur = len(self.block_table[request.request_id])
            num_blocks_needed = self.get_num_blocks_needed(request) - num_blocks_cur
        
        if self.is_fusion_state:
            return len(self.free_fusion_gpu_blocks_list) >= num_blocks_needed
        return len(self.free_gpu_blocks_list) >= num_blocks_needed

    def allocate_blocks(self, request: Request, only_new_blocks: bool = False, live_reshard: bool = False):
        if self.is_fusion_state:
            if only_new_blocks or request.fusion_exec:
                self._allocate_fusion_blocks(request, live_reshard)
            else:
                self._allocate_reserved_blocks(request)
            return

        if request.fusion_exec:
            self._allocate_reserved_blocks(request)
            return

        """Allocate blocks for a request"""
        assert (
            request.request_id not in self.block_table
            or self.request_location.get(request.request_id, None) == BlockLocation.GPU
        ), f"request {request.request_id} is currently on CPU. Please migrate it to GPU before allocating ore blocks"

        num_blocks_needed = self.get_num_blocks_needed(request)
        if request.request_id not in self.block_table:
            # This request has not been allocated before
            self.block_table[request.request_id] = self._get_free_blocks(
                num_blocks_needed, BlockLocation.GPU
            )
            self.request_location[request.request_id] = BlockLocation.GPU
        else:
            assert self.request_location[request.request_id] == BlockLocation.GPU
            num_blocks_cur = len(self.block_table[request.request_id])
            if num_blocks_cur < num_blocks_needed:
                self.block_table[request.request_id] += self._get_free_blocks(
                    num_blocks_needed - num_blocks_cur, BlockLocation.GPU
                )

    def get_base_region_blocks(self, request: Request, num_blocks: int, only_new_blocks: bool = False) -> List[int]:
        if self.is_fusion_state:
            if only_new_blocks or request.fusion_exec:
                return self._get_free_fusion_blocks(num_blocks, BlockLocation.GPU)
            else:
                return self._get_free_reserved_blocks(num_blocks, BlockLocation.GPU)

        if request.fusion_exec:
            return self._get_free_reserved_blocks(num_blocks, BlockLocation.GPU)

        """Allocate blocks for a request"""
        # Make sure the request is not already allocated or its blocks are on GPU
        assert (
            request.request_id not in self.block_table
            or self.request_location.get(request.request_id, None) == BlockLocation.GPU
        ), f"request {request.request_id} is currently on CPU. Please migrate it to GPU before allocating ore blocks"

        return self._get_free_blocks(num_blocks, BlockLocation.GPU)

    def relocate_extra_blocks(self, request_id: int, ex_blocks: List[int], base_blocks: List[int]):
        num_reclaim_blocks = len(ex_blocks)
        reclaim_block_id = 0

        for block_id, block in enumerate(self.block_table[request_id]):
            # Check if the current block matches the block to reclaim
            if reclaim_block_id < num_reclaim_blocks and block == ex_blocks[reclaim_block_id]:
                # Replace the ex_block with the corresponding base_block
                self.block_table[request_id][block_id] = base_blocks[reclaim_block_id]
                reclaim_block_id += 1
                
                # Stop if all blocks to reclaim have been replaced
                if reclaim_block_id == num_reclaim_blocks:
                    break

        # reclaim ex_blocks
        self.max_num_extend_gpu_blocks -= num_reclaim_blocks

    def allocate_blocks_batched(self, batch_requests: BatchedRequests):
        """Allocate blocks for a batch of requests"""
        for request in batch_requests.requests:
            self.allocate_blocks(request)

    def free_blocks(self, request: Request, only_new_blocks: bool = False):
        request_id = request.request_id
        """Free blocks for a request"""
        assert request_id in self.block_table, f"[free_blocks] request {request_id} not allocated"
        location = self.request_location[request_id]
        if location == BlockLocation.GPU:
            blocks_list = self.block_table.pop(request_id)
            if self.is_fusion_state:
                if only_new_blocks or request.fusion_exec:
                    for block in blocks_list:
                        if block < self.num_base_gpu_blocks:
                            self.free_fusion_gpu_blocks_list.append(block)
                        else:
                            self.free_extend_gpu_blocks_list.append(block)
                else:
                    # this is reserved blocks for normal requests, convert them into fusion blocks
                    for block in blocks_list:
                        if block in self.reserved_gpu_blocks_set:
                            self.reserved_gpu_blocks_list.append(block)
                        else:
                            if len(self.reserved_gpu_blocks_set) < self.num_reserved_blocks:
                                self.reserved_gpu_blocks_list.append(block)
                                self.reserved_gpu_blocks_set.add(block)
                            else:
                                self.free_fusion_gpu_blocks_list += [
                                    self.num_base_gpu_blocks // self.fusion_size * i + block for i in range(self.fusion_size)
                                ]
                                self.max_num_reserved_gpu_blocks -= 1
                                self.max_num_fusion_gpu_blocks += self.fusion_size
            else:
                if request.fusion_exec:
                    self.reserved_gpu_blocks_list += blocks_list
                    # restore redundant fusion blocks
                    self.reserved_gpu_blocks_list.sort(key=lambda x: x % self.num_base_gpu_blocks)

                    while len(self.reserved_gpu_blocks_list) > self.num_reserved_blocks:
                        # find the first block continous blocks
                        removed_block_id = -1
                        for i, block in enumerate(self.reserved_gpu_blocks_list):
                            if i + self.fusion_size - 1 >= len(self.reserved_gpu_blocks_list):
                                break
                            if block >= self.num_base_gpu_blocks:
                                continue
                            if block == self.reserved_gpu_blocks_list[i + self.fusion_size - 1] % self.num_base_gpu_blocks:
                                removed_block_id = block
                                break
                        if removed_block_id == -1:
                            break
                        for i in range(self.fusion_size):
                            self.reserved_gpu_blocks_list.remove(removed_block_id + i * self.num_base_gpu_blocks)
                        self.free_gpu_blocks_list.append(removed_block_id)
                        self.max_num_gpu_blocks += 1
                        self.max_num_reserved_gpu_blocks -= self.fusion_size
                else:
                    self.free_gpu_blocks_list += blocks_list
        elif location == BlockLocation.CPU:
            self.free_cpu_blocks_list += blocks_list
        self.request_location.pop(request_id)

    def free_blocks_batched(self, requests: List[Request]):
        """Free blocks for a batch of requests"""
        for i, request in enumerate(requests):
            self.free_blocks(request)

    def free_blocks_into_fusion(self, request_id: int, fusion_size: int):
        """Old blocks are returned to the fusion blocks"""
        assert request_id in self.block_table, f"[free_blocks_into_fusion] request {request_id} not allocated"
        if self.request_location[request_id] == BlockLocation.GPU:
            old_blocks = self.block_table.pop(request_id)
            for block in old_blocks:
                self.free_fusion_gpu_blocks_list += [
                    block * fusion_size + i for i in range(fusion_size)
                ]
            self.max_num_reserved_gpu_blocks -= len(old_blocks)
            self.max_num_fusion_gpu_blocks += len(old_blocks) * fusion_size
        else:
            assert False, "do not support free CPU blocks to fusion yet"
        self.request_location.pop(request_id)

    def free_blocks_into_fusion_except(self, request_id: int, fusion_size: int, fusion_rank: int):
        """Old blocks are returned to the fusion blocks"""
        assert request_id in self.block_table, f"[free_blocks_into_fusion_except] request {request_id} not allocated"
        if self.request_location[request_id] == BlockLocation.GPU:
            old_blocks = self.block_table[request_id]
            for block in old_blocks:
                self.free_fusion_gpu_blocks_list += [
                    self.num_base_gpu_blocks // fusion_size * i + block for i in range(fusion_size) if i != fusion_rank
                ]
            self.max_num_reserved_gpu_blocks -= len(old_blocks)
            self.max_num_fusion_gpu_blocks += len(old_blocks) * fusion_size

            new_blocks = [self.num_base_gpu_blocks // fusion_size * fusion_rank + block for block in old_blocks]
            self.block_table[request_id] = new_blocks
        else:
            assert False, "do not support free CPU blocks to fusion yet"

    def free_blocks_into_fusion_batched(self, requests: List[Request], fusion_size: int):
        for request in requests:
            self.free_blocks_into_fusion(request.request_id, fusion_size)

    def free_blocks_into_fusion_except_batched(self, requests: List[Request], fusion_size: int, fusion_rank: int):
        for request in requests:
            self.free_blocks_into_fusion_except(request.request_id, fusion_size, fusion_rank)

    def get_single_block_table(self, request_id: int) -> List[int]:
        """Get the block table for a request"""
        assert request_id in self.block_table, f"[get_block_table] request {request_id} not allocated"
        return self.block_table[request_id]

    def get_partial_block_table(self, request_ids: List[int]) -> List[List[int]]:
        """Get the block table for a batch of requests"""
        block_table = []
        for request_id in request_ids:
            block_ids = self.block_table.get(request_id, [])
            block_table.append(block_ids)
        return block_table
    
    def get_block_table(self, requests: List[Request]) -> List[List[int]]:
        block_table = []
        for request in requests:
            block_ids = self.block_table.get(request.request_id, [])
            if request.is_context_stage():
                cur_seq_len = request.prefill_end_index
                cur_required_blocks = (cur_seq_len + self.cache_config.block_size - 1) // self.cache_config.block_size
                block_ids = block_ids[:cur_required_blocks]
            block_table.append(block_ids)
        return block_table
    
    def pack_blocks_into_base_region(self, requests: List[Request]) -> bool:
        all_requests_packed = True
        for request in requests:
            block_table = sorted(self.block_table[request.request_id])

            # find first block table id >= self.num_base_gpu_blocks
            ex_block_id = 0
            for block in block_table:
                if block >= self.num_base_gpu_blocks:
                    break
                ex_block_id += 1
            num_ex_blocks = len(block_table) - ex_block_id
            
            if num_ex_blocks == 0:
                continue

            if num_ex_blocks < len(self.free_fusion_gpu_blocks_list):
                # copy blocks from extend region to base region
                extend_blocks = block_table[ex_block_id:]
                block_table = block_table[:ex_block_id] + self.free_fusion_gpu_blocks_list[:num_ex_blocks]
                self.free_fusion_gpu_blocks_list = self.free_fusion_gpu_blocks_list[num_ex_blocks:]
                self.block_table[request.request_id] = block_table
                self.free_extend_gpu_blocks_list += extend_blocks
            else: # no free blocks in base region, give up
                all_requests_packed = False
                break
        return all_requests_packed

    def __repr__(self) -> str:
        return (
            f"BlockManager(max_num_gpu_blocks={self.max_num_gpu_blocks}, "
            f"max_num_cpu_blocks={self.max_num_cpu_blocks}, "
            f"blocksize={self.cache_config.block_size})"
        )

    def get_max_num_base_gpu_blocks(self):
        return self.max_num_fusion_gpu_blocks if self.is_fusion_state else self.max_num_gpu_blocks

    def get_max_num_gpu_blocks(self):
        return (self.max_num_fusion_gpu_blocks + self.max_num_extend_gpu_blocks) if self.is_fusion_state else self.max_num_gpu_blocks
    
    def get_max_num_extend_gpu_blocks(self):
        return self.max_num_extend_gpu_blocks

    def get_max_num_reserved_gpu_blocks(self):
        return self.max_num_reserved_gpu_blocks
    
    def check_reserved_blocks(self):
        return len(self.reserved_gpu_blocks_list) == self.max_num_reserved_gpu_blocks
    
    def get_num_free_fusion_gpu_blocks(self):
        return (
            len(self.free_gpu_blocks_list) if not self.is_fusion_state else len(self.free_fusion_gpu_blocks_list)
            # add a limit for ex block buffer to avoid over-sized decode batch
            + len(self.free_extend_gpu_blocks_list)
        )
    
    def get_num_free_gpu_blocks(self):
        free_gpu_blocks = (
            (
                len(self.free_gpu_blocks_list)
                if not self.is_fusion_state
                else len(self.free_fusion_gpu_blocks_list)
            )
            + len(self.free_extend_gpu_blocks_list)
        )
        return free_gpu_blocks
    
    def out_of_memory(self, normal_block_demand, pipeline_block_demand):
        if self.is_fusion_state:
            return pipeline_block_demand > self.get_max_num_gpu_blocks() or normal_block_demand > self.get_max_num_reserved_gpu_blocks()
        else:
            return normal_block_demand > self.get_max_num_gpu_blocks() or pipeline_block_demand > self.get_max_num_reserved_gpu_blocks()
    
    def print_block_usage(self, instance_id: int, fusion_rank: int):
        total_gpu_blocks = self.get_max_num_gpu_blocks()
        if self.is_fusion_state:
            free_gpu_blocks = (
                len(self.free_fusion_gpu_blocks_list)
                + len(self.free_extend_gpu_blocks_list)
                + len(self.reserved_gpu_blocks_list) * self.fusion_size
            )
        else:
            free_gpu_blocks = (
                len(self.free_gpu_blocks_list)
                + len(self.free_extend_gpu_blocks_list)
                + len(self.reserved_gpu_blocks_list) // self.fusion_size
            )
        num_gpu_blocks_used = (
            total_gpu_blocks
            - free_gpu_blocks
            - len(self.swapping_gpu_blocks_list)
        )
        assert (
            0 <= num_gpu_blocks_used <= total_gpu_blocks
        ), (
            f"GPU block accounting inconsistent: used={num_gpu_blocks_used}, "
            f"total={total_gpu_blocks}, free={free_gpu_blocks}, "
            f"swapping={len(self.swapping_gpu_blocks_list)}"
        )

        logger.info(
            f"({self.stage} {instance_id}, #{fusion_rank}) GPU blocks: {num_gpu_blocks_used} / {total_gpu_blocks} "
            f"({num_gpu_blocks_used / total_gpu_blocks * 100:.2f}%) used"
        )

    """The following methods are used for swapping
    We use the term "swap in" to mean moving blocks from CPU to GPU, and
    "swap out" to mean moving blocks from GPU to CPU.

    The followings explain our logic and code layout for swapping:

    # Swapping

    ## Overview

    We use swap_in as the example here. Swapping-out is similar.

    The scheduler calls LLMEngine.swap_in_request, which is a thin wrapper
    around BlockManager.swap_in_requests. The latter allocate blocks (i.e. GPU
    blocks when swapping in and CPU blocks when swapping out) for the requests
    and then call ParaWorker.swap_blocks on every worker.

    scheduler -> LLMEngine.swap_in_requests ->
        BlockManager.swap_in_requests -> ParaWorker.swap_blocks
    
    ## Synchonization

    We need to deal with two types of synchronization:

    Requirement 1: When calling step(), we need to ensure that all blocks that
        are related to the request are on GPU, and have finished swapping if
        we've issued a swap-in operation on them before.

    Requirement 2: When swapping in/out, we need to ensure that the target blocks
        (i.e. blocks that we are copying to) are free. If there were another
        swapping operation which use them as source, we must wait for that operation
        to finish.
    
    Requirement 3: When swapping in/out, the source blocks must be ready, i.e.
        if we have issued swapping-in operation on request #0 before and now I
        want to swap it out, then the swap-in operation must have finished.

    After every swapping in/out operation, we push a CUDA event into the corresponding
    CUDA stream. We maintain a dict called `swap_event_table`, which maps request_ids,
    to the latest cuda event related to that request.

    When calling step(), we iterate through all requests in the batch, retrieve
    the CUDA event from `swap_event_table`, and wait for it to finish. This ensures
    Requirement 1.

    When swapping in/out, we call event.wait() on the corresponding swap out/in
    event and use the corresponding stream (if we are swapping-in then it is the
    swap-in stream, vice versa) as the argument. Wait() makes all future work
    submitted to the given stream wait for this event, so the swapping operation
    that I'm going to issue will wait for the previous swapping operation to finish.
    This ensures Requirement 3.

    For Requirement 2, a possible solution is to mark the CUDA event of every block
    and wait for them to finish. However, this is not efficient. Instead, for
    each device (CPU/GPU), we maintain two lists: `free_blocks_list` and
    `swapping_blocks_list`. The former contains blocks that are we are sure to
    be free (it does not contain any useful data), and the latter contains blocks
    which we have issued a swapping operation from but we are not sure if the operation
    has finished.
    
    When we want to allocate a block, we first check if there are enough free blocks
    in `free_blocks_list`. If unfortunately this is not the case, we let all
    workers to finish all operations in `swapping_blocks_list` and then move
    all blocks in `swapping_blocks_list` to `free_blocks_list`. This ensures
    Requirement 2 and the overhead is small.

    ## Further Optimization

    To optimize more aggressively, consider the following idea: if we mark every
    CUDA event in the same stream with an increasing id, then if "The id associated
    with the most synced CUDA event" is larger than "The id associated with the
    CUDA event that we are waiting for", then we can skip the waiting.

    Currently we do not implement this idea because we want to make sure the
    correctness of the code first. We will implement this idea in the future.
    """

    def swap_requests(self, requests: List[Request], is_swap_in: bool) -> bool:
        """Swap blocks for a batch of requests
        If `is_swap_in` is True, then swap in blocks from CPU to GPU, and vice versa
        """
        cur_location = BlockLocation.CPU if is_swap_in else BlockLocation.GPU
        target_location = BlockLocation.GPU if is_swap_in else BlockLocation.CPU
        source_block_ids = []  # block ids on cur_location
        target_block_ids = []  # block ids on target_location
        old_block_ids = []
        new_block_ids = []
        for i, request in enumerate(requests):
            assert (
                request.request_id in self.block_table
            ), f"request {request.request_id} not allocated"
            assert (
                self.request_location[request.request_id] == cur_location
            ), f"request {request.request_id} is on {target_location} now"
            old_block_ids.append(self.block_table[request.request_id])
            new_block_ids.append(self._get_free_blocks(len(old_block_ids[-1]), target_location))
            # If CPU is full, reject the swap request
            if target_location == BlockLocation.CPU and new_block_ids[-1] == None:
                return False
            source_block_ids += old_block_ids[-1]
            target_block_ids += new_block_ids[-1]
        
        for i, request in enumerate(requests):
            self.block_table[request.request_id] = new_block_ids[i]
            self.request_location[request.request_id] = target_location
            if cur_location == BlockLocation.CPU:
                self.swapping_cpu_blocks_list += old_block_ids[i]
            else:
                self.swapping_gpu_blocks_list += old_block_ids[i]
        self.engine_remote_call_all_workers_async(
            "swap_blocks", requests, source_block_ids, target_block_ids, is_swap_in
        )
        return True

    def swap_in_requests(self, requests: List[Request]) -> bool:
        """Swap in blocks for a batch of requests"""
        return self.swap_requests(requests, is_swap_in=True)

    def swap_out_requests(self, requests: List[Request]) -> bool:
        """Swap out blocks for a batch of requests"""
        return self.swap_requests(requests, is_swap_in=False)

    async def swap_one_request(self, request: Request, is_swap_in: bool) -> bool:
        """Swap blocks for a batch of requests
        If `is_swap_in` is True, then swap in blocks from CPU to GPU, and vice versa
        """
        cur_location = BlockLocation.CPU if is_swap_in else BlockLocation.GPU
        target_location = BlockLocation.GPU if is_swap_in else BlockLocation.CPU
        source_block_ids = []  # block ids on cur_location
        target_block_ids = []  # block ids on target_location
        assert (
            request.request_id in self.block_table
        ), f"request {request.request_id} not allocated"
        assert (
            self.request_location[request.request_id] == cur_location
        ), f"request {request.request_id} is on {target_location} now"
        old_block_ids = self.block_table[request.request_id]
        if self.is_fusion_state and target_location == BlockLocation.GPU:
            new_block_ids = self._get_free_fusion_blocks(len(old_block_ids), target_location)
        else:
            new_block_ids = self._get_free_blocks(len(old_block_ids), target_location)
        # If CPU is full, reject the swap request
        if target_location == BlockLocation.CPU and new_block_ids == None:
            return False
        source_block_ids += old_block_ids
        target_block_ids += new_block_ids
        self.block_table[request.request_id] = new_block_ids
        self.request_location[request.request_id] = target_location
        if cur_location == BlockLocation.CPU:
            self.swapping_cpu_blocks_list += old_block_ids
        else:
            self.swapping_gpu_blocks_list += old_block_ids
        
        await asyncio.wait(
            self.engine_remote_call_all_workers_async(
                "swap_blocks", [request], source_block_ids, target_block_ids, is_swap_in
        ))
        
        if is_swap_in:
            await asyncio.wait(self.engine_remote_call_all_workers_async("wait_for_all_swap_in"))
            self.free_cpu_blocks_list += self.swapping_cpu_blocks_list
            self.swapping_cpu_blocks_list = []
        else:
            await asyncio.wait(self.engine_remote_call_all_workers_async("wait_for_all_swap_out"))
            self.free_gpu_blocks_list += self.swapping_gpu_blocks_list
            self.swapping_gpu_blocks_list = []
        return True

    async def swap_in_one_request(self, request: Request) -> bool:
        """Swap in blocks for a batch of requests"""
        swap_result = await self.swap_one_request(request, is_swap_in=True)
        return swap_result

    async def swap_out_one_request(self, request: Request) -> bool:
        """Swap out blocks for a batch of requests"""
        swap_result = await self.swap_one_request(request, is_swap_in=False)
        return swap_result

    def is_all_requests_on_gpu(self, requests: BatchedRequests):
        """Check if all requests in a batch are on GPU"""
        for request in requests.requests:
            if self.request_location[request.request_id] == BlockLocation.CPU:
                return False
        return True


class InfiniBlockManager(BlockManager):
    def __init__(
        self,
        stage: Stage,
        prefix: str,
        max_num_gpu_blocks: int,
        max_num_cpu_blocks: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        engine_remote_call_all_workers_async: Callable):
        super().__init__(
            stage, 
            prefix, 
            max_num_gpu_blocks, 
            max_num_cpu_blocks, 
            model_config, 
            parallel_config, 
            cache_config, 
            engine_remote_call_all_workers_async
        )
        self.curr_index = 0
    
    def _get_free_blocks(self, num_blocks: int, location: BlockLocation) -> List[int]:
        assert location in [BlockLocation.GPU]
        blocks = self.free_gpu_blocks_list[self.curr_index:self.curr_index + num_blocks]
        self.curr_index = (self.curr_index + num_blocks) % len(self.free_gpu_blocks_list)
        return blocks
    
    
    