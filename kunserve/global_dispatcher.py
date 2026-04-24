from kunserve.logger import init_logger
from kunserve.request import Request, MigratingRequest
from kunserve.flash_backend import (
    SLEEP_WHEN_CONTEXT_NO_REQUEST, 
    REDISPATCH_INTERVAL, 
    SLEEP_WHEN_TOO_MANY_UNACCEPTED,
    REDISPATCH_THRESHOLD,
)

from typing import List, Tuple
from heapq import heappush, heappop
from collections import deque

from kunserve.utils import EngineState
import ray
import asyncio
from enum import Enum
import numpy as np

logger = init_logger(__name__)

@DeprecationWarning
class GlobalDispatcher():
    def __init__(self, state: EngineState, dispatch_strategy: str, redispatch_policy: str = "none"):
        self.instances = []
        self.unaccepted_queue: List[Request] = []
        self.state = state
        self.next_instance = 0
        self.dispatch_strategy = dispatch_strategy
        self.redispatch_policy = redispatch_policy

    def add_request(self, request: Request):
        if self.dispatch_strategy == "round-robin":
            assign = self.next_instance
            self.next_instance = (self.next_instance + 1) % len(self.instances)
        elif self.dispatch_strategy == "memory" or self.dispatch_strategy == "load":
            # TODO: support load balance scheduling
            assign = self.instance_loads.index(min(self.instance_loads))
            self.instance_loads[assign] += 1 # avoid stale scheduling
        else:
            assert False, f"Invalid dispatch strategy: {self.dispatch_strategy}."

        try:
            ray.get(self.instances[assign].add_request.remote(request), timeout=10)
        except Exception as e:
            logger.error(f"Failed to dispatch request {request.request_id} to instance {assign} due to e: {e}")
            exit(1)

    def get_loads(self, instance_reports):
        memory_utils, output_tokens = None, None
        if self.dispatch_strategy == "memory":
            # prioritize instance with maximal free memory
            self.instance_loads = [report[2] for report in instance_reports]
        else:
            # prioritize instance with minimal number of requests
            self.instance_loads = [report[1] for report in instance_reports]
        # default load is represented by memory load
        memory_utils = [report[0] for report in instance_reports]
        unfinished_requests = sum([report[1] for report in instance_reports])
        output_tokens = sum([report[3] for report in instance_reports])
        return memory_utils, output_tokens, sum(self.instance_loads)
    
    async def get_instance_loads(self):
        loads = []
        for instance in self.instances:
            # the usage info is reported as (gpu_util, unfinished_requests, -free_blocks)
            ray_func = instance.get_gpu_blocks_usage.remote()
            loads.append(asyncio.wrap_future(ray_func.future()))
        try:
            memory_utils, output_tokens, group_load = self.get_loads(await asyncio.gather(*loads))
            avg_load = sum(memory_utils) / len(memory_utils)
            max_load = max(memory_utils)
            min_load = min(memory_utils)
        except Exception as e:
            logger.error(f"Failed to get instance loads due to e: {e}")
            ray.shutdown()
            exit(1)
        return avg_load, max_load, min_load, output_tokens, group_load
    
    @DeprecationWarning
    async def dispatch(self):
        assert len(self.instances) > 0, "No instances available"
        
        request = self.unaccepted_queue.pop(0)
        
        # TODO: update it
        # best throughput, best memory usage
        if self.dispatch_strategy == "round-robin":
            assign = self.next_instance
            self.next_instance = (self.next_instance + 1) % len(self.instances)
        elif self.dispatch_strategy == "load-balance":
            refs = []
            for engine in self.instances:
                refs.append(engine.get_gpu_blocks_usage.remote())
            gpu_block_usage_list = await asyncio.gather(*refs)
            assign, max_free_gpu_blocks = max(enumerate(gpu_block_usage_list), key=lambda x: x[1][2])
        elif self.dispatch_strategy == "memory-balance":
            refs = []
            for engine in self.instances:
                refs.append(engine.get_gpu_running_load.remote())
            running_load_list = await asyncio.gather(*refs)
            assign, min_running_load = min(enumerate(running_load_list), key=lambda x: x[1])
        else:
            assert False, f"Invalid dispatch strategy: {self.dispatch_strategy}."
        
        # logger.info(f"dispatched request {request.request_id} to engine {assign}")
        await self.instances[assign].add_request.remote(request)

    @DeprecationWarning
    async def redispatch(self):
        assert len(self.decoding_instances) > 0, "No instances available"

        if self.state != EngineState.STOP:

            # redispatch unaccepted requests
            if self.redispatch_policy == "unaccepted":
                # check if there is significant imbalance between instances
                refs = []
                for engine in self.instances:
                    refs.append(engine.get_available_ratio.remote())
                free_gpu_ratios = await asyncio.gather(*refs)
                
                max_idx, max_free_ratio = max(enumerate(free_gpu_ratios), key=lambda x: x[1])
                min_idx, min_free_ratio = min(enumerate(free_gpu_ratios), key=lambda x: x[1])
                if max_free_ratio - min_free_ratio < REDISPATCH_THRESHOLD:
                    # if there is no significant imbalance, do not redispatch
                    return

                refs = []
                for engine in self.decoding_instances:
                    refs.append(engine.fetch_instance_load.remote(True))
                loads = await asyncio.gather(*refs)
                loads_with_idx = [(i, load) for i, load in enumerate(loads)]
                
                sorted_loads = deque(sorted(loads_with_idx, key=lambda x: x[1]))
                lower_load_instances = []
                higher_load_instances = []
                gap_between_pairs = []
                
                while len(sorted_loads) > 1:
                    low = sorted_loads.popleft()
                    high = sorted_loads.pop()
                    gap = (high[1] - low[1]) // 2
                    if gap > 0:
                        # logger.info(f"load of instance {high[0]}: {high[1]}, load of instance {low[0]}: {low[1]}, gap: {gap}")
                        lower_load_instances.append(low[0])
                        higher_load_instances.append(high[0])
                        gap_between_pairs.append(gap)

            # redispatch running requests (llumnix)
            elif self.redispatch_policy == "running":
                def get_load_from_result(result):
                    num_available_gpu_blocks, num_requests, num_gpu_blocks_of_shortest_running = result
                    if num_requests == 0:
                        return -np.inf
                    return (num_available_gpu_blocks / num_requests)*(-1)
                
                def get_load_after_migration(result, num_block_to_migrate):
                    num_available_gpu_blocks, num_requests, num_gpu_blocks_of_shortest_running = result
                    num_available_gpu_blocks += num_block_to_migrate
                    num_requests += 1
                    return (num_available_gpu_blocks / num_requests)*(-1)

                refs = []
                for engine in self.decoding_instances:
                    refs.append(engine.fetch_instance_load.remote(False))
                results = await asyncio.gather(*refs) # [(num_available_gpu_blocks, num_requests, num_gpu_blocks_of_shortest_running)]
                loads = [get_load_from_result(result) for result in results]
                loads_with_idx = [(i, load) for i, load in enumerate(loads)]

                # instances with load < migrate_out_threshold are lower load instances
                # instances with load >= migrate_out_threshold are higher load instances
                migrate_out_threshold = -3.0

                sorted_loads = deque(sorted(loads_with_idx, key=lambda x: x[1]))
                lower_load_instances = []
                higher_load_instances = []
                gap_between_pairs = []

                while len(sorted_loads) > 1:
                    low = sorted_loads.popleft()
                    high = sorted_loads.pop()
                    low_id, low_load = low
                    high_id, high_load = high
                    if low_load < migrate_out_threshold and high_load >= migrate_out_threshold:
                        low_load_after_migration = get_load_after_migration(results[low_id], results[high_id][2])
                        if low_load_after_migration < migrate_out_threshold:
                            gap = (high_load - low_load) / 2
                            lower_load_instances.append(low_id)
                            higher_load_instances.append(high_id)
                            gap_between_pairs.append(gap)
                            
            else:
                raise NotImplementedError(f"redispatch policy {self.redispatch_policy} not supported")

            # send the rebalance plan to instances
            # assert len(lower_load_instances) > 0, "An fail rebalance plan"
            
            refs = []
            for i in range(len(higher_load_instances)):
                if self.redispatch_policy == "unaccepted":
                    refs.append(
                        # move at most `gap` blocks from higher load instance to lower load instance
                        self.decoding_instances[higher_load_instances[i]].rebalance_instance_load.remote(
                            lower_load_instances[i], gap_between_pairs[i])
                    )
                elif self.redispatch_policy == "running":
                    refs.append(
                        self.decoding_instances[higher_load_instances[i]].migrate_running_request.remote(
                            lower_load_instances[i])
                    )
                else:
                    raise NotImplementedError(f"redispatch policy {self.redispatch_policy} not supported")
            await asyncio.gather(*refs)

    def set_dispather(self, instances, state):
        self.set_instances(instances)
        self.state = state

    def set_instances(self, instances: List):
        self.instances = instances
        self.instance_loads = [0] * len(instances)
        self.next_instance = 0

    def set_state(self, state: EngineState):
        self.state = state

    @DeprecationWarning
    async def start_event_loop(self):
        """
        We dont need an async loop to dispatch requests again.
        """

        async def dispatch_event_loop():
            while True:
                if len(self.unaccepted_queue) > 0 and self.state != EngineState.STOP:
                    await self.dispatch()
                else:
                    await asyncio.sleep(SLEEP_WHEN_CONTEXT_NO_REQUEST)

        async def redispatch_event_loop():
            while True:
                if self.state != EngineState.STOP:
                    await self.redispatch()
                await asyncio.sleep(REDISPATCH_INTERVAL)

        if self.redispatch_policy != "none":
            await asyncio.gather(dispatch_event_loop(), redispatch_event_loop())
        else:
            await asyncio.gather(dispatch_event_loop())
