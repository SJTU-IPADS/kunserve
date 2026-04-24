import asyncio
import time
from typing import List, Tuple

import ray
from kunserve.kunserve_config import BalloonConfig
from kunserve.llm_engine import AsyncStream, ColocatedEngine, LLMEngine
from kunserve.flash_backend import Event
from kunserve.logger import init_logger
from kunserve.request import Request
from kunserve.utils import EngineState


logger = init_logger(__name__)

CHECK_EVENTS_INTERVAL = 0.1
INSTANCE_LOAD_REPORT_INTERVAL = 0.5
GRACE_PERIOD = 5


class Coordinator:
    def __init__(self, dispatch_strategy:str, group_size: int, balloon_config: BalloonConfig, engines: List[ColocatedEngine]):
        self.dispatch_strategy = dispatch_strategy
        self.group_size = group_size
        self.balloon_config = balloon_config
        self.engines = engines

        self.output_tokens = [0] * len(self.engines)
        self.instance_loads = [[0] * self.group_size for _ in range(len(self.engines))]

        # for round-robin dispatch
        self.next_instance = 0
        self.num_instances = len(self.engines) * group_size
        self.avg_loads = [0] * len(self.engines)
        self.state = "initial"

    '''belows are for engine state transition'''

    async def transfer_requests_balloon(self, src_engines: List[ColocatedEngine], dst_engine: ColocatedEngine, num_blocks: int):
        for src_engine in src_engines:
            logger.info(f"[BalloonCoordinator] transfer requests group {src_engine.group_id} -> {dst_engine.group_id}")
            
            # step 1: pop requests from src engines
            requests_map = await src_engine.pop_running_requests_balloon_transfer(num_blocks)
        
            # step 2: allocate blocks for the requests on the dst engine
            await dst_engine.append_and_allocate_requests_balloon_transfer(requests_map)
        
            # step 3: transfer requests to the dst engine
            for offset in range(self.group_size):
                refs = []
                refs.append(src_engine.send_requests_balloon_transfer(offset, dst_engine.group_id, requests_map["requests"]))
                refs.append(dst_engine.recv_requests_balloon_transfer(offset, src_engine.group_id, requests_map["requests"]))
                await asyncio.gather(*refs)
    
            # step 4: clear resources on src engine
            await src_engine.free_blocks_balloon_transfer(requests_map["requests"])
        
        # restart ballon engine
        await dst_engine.set_instance_stopped(0, False)
        
    async def transfer_requests_restore(self, events):
        start_time = time.perf_counter()
        # target: make engine load balance

        pp_engines = [self.engines[i] for i, event in enumerate(events) if event[0] == EngineState.BALLOON]
        dp_engines = [self.engines[i] for i, event in enumerate(events) if event[0] == EngineState.INITIAL]

        if len(pp_engines) == 0 or len(dp_engines) == 0:
            return

        # get the balloon engine with the highest load
        max_pp_ratio = 0
        src_engine = None
        for engine in pp_engines:
            ratio = await engine.get_avg_memory_ratio()
            if ratio > max_pp_ratio:
                max_pp_ratio = ratio
                src_engine = engine

        # get the loads of the initial engines
        refs = []
        for engine in dp_engines:
            refs.append(engine.get_avg_memory_ratio())
        dp_ratios = await asyncio.gather(*refs)


        # check whether start transfer
        threshold = self.balloon_config.restore_memory_threshold
        if any(ratio >= threshold for ratio in dp_ratios) or max_pp_ratio < threshold or any(max_pp_ratio < ratio for ratio in dp_ratios):
            return

        dst_engines = dp_engines

        # step 1: collect engine load
        refs = []
        for dst_engine in dst_engines:
            refs.append(dst_engine.get_avail_restore_blocks())
        avail_blocks = await asyncio.gather(*refs)
        total_avail_blocks = sum(avail_blocks)

        refs = []
        used_blocks = [await src_engine.get_total_used_blocks()]
        for dst_engine in dst_engines:
            refs.append(dst_engine.get_total_used_blocks())
        used_blocks.extend(await asyncio.gather(*refs))
        avg_used_blocks = used_blocks[0] - sum(used_blocks) // len(used_blocks)

        transfer_blocks = min(total_avail_blocks, avg_used_blocks)

        if transfer_blocks <= 0:
            return

        # step 2: pop stale requests from restored engine
        requests_map = await src_engine.pop_running_requests_restore_transfer(transfer_blocks)

        for rank, dst_engine in enumerate(dst_engines):
            logger.info(f"[BalloonCoordinator] transfer requests group {src_engine.group_id} -> {dst_engine.group_id}")

            # divide requests to len(dst_engines) parts
            cur_requests: List[List[Request]] = []
            cur_generators: List[List[AsyncStream]] = []
            for i, reqs in enumerate(requests_map["requests"]):
                cur_requests.append([])
                cur_generators.append([])
                for j, req in enumerate(reqs):
                    if j % len(dst_engines) == rank:
                        cur_requests[-1].append(req)
                        cur_generators[-1].append(requests_map["generators"][i][j])
            cur_requests_map = {"requests": cur_requests, "generators": cur_generators}

            # step 3: allocate blocks for the requests on the dst engine
            await dst_engine.append_and_allocate_requests_restore_transfer(cur_requests_map)

            # step 4: transfer requests to the dst engine
            for offset in range(self.group_size):
                refs = []
                refs.append(src_engine.send_requests_restore_transfer(offset, dst_engine.group_id, requests_map["requests"]))
                refs.append(dst_engine.recv_requests_restore_transfer(offset, src_engine.group_id, requests_map["requests"]))
                await asyncio.gather(*refs)

            # step 5: clear resources on src engine
            await dst_engine.start_all_engines()
            await src_engine.free_blocks_restore_transfer(requests_map["requests"])
        

        logger.info(f"[BalloonCoordinator] transfer requests at restore cost {time.perf_counter() - start_time:.2f}s")

    async def trigger_balloon(self, events):
        # choose one engine to balloon
        refs = []
        for i, event in enumerate(events):
            # if event[1] == Event.BALLOON:
                balloon_engine = i
                # break
        
                # trigger balloon for the chosen engine
                logger.info(f"[BalloonCoordinator] (group {balloon_engine}) {events[balloon_engine][0]} => {EngineState.BALLOON}")
                from_restore = (events[balloon_engine][0]==EngineState.RESTORE)
                refs.append(self.engines[balloon_engine].trigger_balloon(from_restore))

                # transfer requests to the ballooned engine
                # INITIAL instead of RESTORE
                if False:
                    dp_engines = [self.engines[i] for i, event in enumerate(events) if event[0] == EngineState.INITIAL and i != balloon_engine]
                    if len(dp_engines) == 0:
                        await self.engines[balloon_engine].set_instance_stopped(0, False)
                        return
                    
                    available_blocks = available_blocks // len(dp_engines)
                    start_time = time.perf_counter()
                    await self.transfer_requests_balloon(dp_engines, self.engines[balloon_engine], available_blocks)
                    logger.info(f"[BalloonCoordinator] transfer requests at balloon cost {time.perf_counter() - start_time:.2f}s")
        await asyncio.gather(*refs)
        self.state = "balloon"

    async def trigger_restore(self, events):
        # choose one engine to restore
        refs = []
        for i, event in enumerate(events):
            # if event[1] == Event.RESTORE:
                restore_engine = i
        
                # trigger restore for the chosen engine
                logger.info(f"[BalloonCoordinator] (group {restore_engine}) {events[restore_engine][0]} => {EngineState.RESTORE}")
                refs.append(self.engines[restore_engine].trigger_restore())
        await asyncio.gather(*refs)
        self.state = "restore"

    async def trigger_init(self, events):
        refs = []
        for i, event in enumerate(events):
            if event[1] == Event.INIT:
                refs.append(self.engines[i].trigger_init())
        await asyncio.gather(*refs)
        self.state = "initial"

    async def handle_events(self, events):
        # trigger init immediately, avoid unnecessary waiting
        if any(event[1] == Event.INIT for event in events):
            await self.trigger_init(events)

        # case 1: there exists an engine requesting balloon
        # if sum(event[1] == Event.BALLOON for event in events) >= len(self.engines) // 2:
        if any(event[1] == Event.BALLOON for event in events):
            await self.trigger_balloon(events)
            # await asyncio.sleep(GRACE_PERIOD)
        # case 2: no engine requesting balloon, but there exists an engine requesting restore or init
        # elif sum(event[1] == Event.RESTORE for event in events) >= len(self.engines) // 2:
        elif any(event[1] == Event.RESTORE for event in events):
            await self.trigger_restore(events)
            # await asyncio.sleep(GRACE_PERIOD)
        # case 3: try to transfer requests from balloon engine to initial engine aggresively
        # else:
        #     await self.transfer_requests_restore(events)

        # note that there should not exist the case that some engines request balloon while some engines request restore
        # because in this case, we should schedule the requests to the ballooned engine with priority
        
    async def monitor_engine_events(self):
        while True:
            refs = []
            for engine in self.engines:
                refs.append(engine.get_event())
            events = await asyncio.gather(*refs)
            
            await self.handle_events(events)

            await asyncio.sleep(CHECK_EVENTS_INTERVAL)

    '''belows are for instance load collection'''

    async def get_next_assign(self):
        async def check_instance_valid(group, instance):
            state = await self.engines[group].get_state()
            if state == EngineState.STOP:
                return False
            if state != EngineState.INITIAL and instance > 0:
                return False
            return True

        if self.dispatch_strategy == "round-robin":
            group = self.next_instance // self.group_size
            instance = self.next_instance % self.group_size
            # while (await check_instance_valid(group, instance)) == False:
            #     self.next_instance = (self.next_instance + 1) % self.num_instances
            #     group = self.next_instance // self.group_size
            #     instance = self.next_instance % self.group_size
            self.next_instance = (self.next_instance + 1) % self.num_instances
            return group, instance
        elif self.dispatch_strategy == "memory" or self.dispatch_strategy == "load":
            group = 0
            instance = 0
            min_load = self.instance_loads[0][0]
            for i, loads in enumerate(self.instance_loads):
                for j, load in enumerate(loads):
                    if load < min_load:
                        min_load = load
                        group = i
                        instance = j
            self.instance_loads[group][instance] += 1 # avoid assigning the same instance
            return group, instance
        else:
            assert False, f"Invalid dispatch strategy: {self.dispatch_strategy}."

    def report_instance_outputs(self, i):
        return self.output_tokens[i]

    def get_loads(self, instance_reports):
        memory_utils, output_tokens = None, None
        if self.dispatch_strategy == "memory":
            # prioritize instance with maximal free memory
            instance_loads = [report[2] for report in instance_reports]
        else:
            # prioritize instance with minimal number of requests
            instance_loads = [report[1] for report in instance_reports]
        # default load is represented by memory load
        memory_utils = [report[0] for report in instance_reports]
        unfinished_requests = sum([report[1] for report in instance_reports])
        output_tokens = sum([report[3] for report in instance_reports])
        return memory_utils, output_tokens, instance_loads
    
    async def get_instance_loads(self, loads):
        try:
            memory_utils, output_tokens, instance_loads = self.get_loads(loads)
            avg_load = sum(memory_utils) / len(memory_utils)
            max_load = max(memory_utils)
            min_load = min(memory_utils)
        except Exception as e:
            logger.error(f"Failed to get instance loads due to: {e} {loads=}")
            ray.shutdown()
            exit(1)
        return avg_load, max_load, min_load, output_tokens, instance_loads

    async def collect_instance_loads(self):
        epoch = 0
        while True:
            start_time = time.perf_counter()

            refs = []
            for engine in self.engines:
                refs.append(engine.get_gpu_blocks_usage())
            usages = await asyncio.gather(*refs)

            self.avg_loads = []
            for i, (state, usage) in enumerate(usages):
                avg_load, max_load, min_load, output_tokens, instance_loads = \
                    await self.get_instance_loads(usage)
                logger.info(
                    f"[Group({i}) ({state}) reports] avg: {avg_load:.3f}%, max: {max_load:.3f}%, min: {min_load:.3f}%, "
                    f"instance loads: {instance_loads}, outputs: {output_tokens}."
                )
                self.output_tokens[i] += output_tokens
                self.instance_loads[i] = instance_loads
                self.avg_loads.append(avg_load)
            
            epoch += 1

            elapsed_time = time.perf_counter() - start_time
            await asyncio.sleep(max(INSTANCE_LOAD_REPORT_INTERVAL - elapsed_time, 0))

    async def start_all_event_loops(self, enable_balloon: bool):
        tasks = []

        tasks.append(asyncio.create_task(self.collect_instance_loads()))
        if enable_balloon:
            tasks.append(asyncio.create_task(self.monitor_engine_events()))
        
        await asyncio.gather(*tasks)
