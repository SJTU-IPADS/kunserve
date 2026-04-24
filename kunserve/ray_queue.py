from collections.abc import Iterable
from typing import Any, List, Dict
import time
import asyncio
import ray
from ray.util.queue import Queue as RayQueue
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from kunserve.utils import get_instance_name
from kunserve.logger import init_logger
logger = init_logger(__name__)

class RayQueueServer:
    def __init__(self) -> None:
        self.queue = RayQueue(
            actor_options={
                "scheduling_strategy":
                    NodeAffinitySchedulingStrategy(
                        node_id=ray.get_runtime_context().get_node_id(),
                        soft=False
                    )
            }
        )

    async def get(self):
        item = await self.queue.actor.get.remote()
        if isinstance(item, Iterable):
            for request_output in item:
                if hasattr(request_output, 'request_timestamps'):
                    request_output.request_timestamps.queue_server_receive_timestamp = time.time()
        return item

    async def get_nowait_batch(self):
        qsize = await self.queue.actor.qsize.remote()
        items = await self.queue.actor.get_nowait_batch.remote(qsize)
        for request_output in items:
            if hasattr(request_output, 'request_timestamps'):
                request_output.request_timestamps.queue_server_receive_timestamp = time.time()
        return items

    async def run_server_loop(self):
        pass

    def cleanup(self):
        try:
            ray.kill(self.queue)
        # pylint: disable=broad-except, unused-variable
        except Exception as e:
            pass


class ServerInfo:
    """
    Here server means "APIServer", each api server is bind to a specific GraphGenClient.
    """
    def __init__(self,
                 server_id: str,
                 request_output_queue: RayQueueServer) -> None:
        self.server_id = server_id
        self.request_output_queue = request_output_queue.queue
        
        """ 
        ServerInfo also includes an optional request_timestamps member, 
        it will be create only when `log_request_timestamps` is enabled.
        """

class RayQueueClient:
    async def put_nowait(self, item: Any, server_info: ServerInfo):
        output_queue = server_info.request_output_queue
        # if isinstance(item, Iterable):
        #     for request_output in item:
        #         if hasattr(request_output, 'request_timestamps'):
        #             request_output.request_timestamps.queue_client_send_timestamp = time.time()
        return await output_queue.actor.put_nowait.remote(item)

    async def put_nowait_batch(self, items: Iterable, server_info: ServerInfo):
        output_queue = server_info.request_output_queue
        # for request_output in items:
        #     if hasattr(request_output, 'request_timestamps'):
        #         request_output.request_timestamps.queue_client_send_timestamp = time.time()
        return await output_queue.actor.put_nowait_batch.remote(items)
    
class PutQueue:
    def __init__(self, server_id: int, instance_id: str):
        self.job_id = ray.get_runtime_context().get_job_id()
        self.worker_id = ray.get_runtime_context().get_worker_id()
        self.actor_id = ray.get_runtime_context().get_actor_id()
        self.node_id = ray.get_runtime_context().get_node_id()
        self.server_id = server_id
        self.instance_id = instance_id
        logger.info("PutQueue(job_id={}, worker_id={}, actor_id={}, node_id={}, server_id={}, instance_id={})".format(
                        self.job_id, self.worker_id, self.actor_id, self.node_id, self.server_id, self.instance_id))
        self.request_output_queue_client = RayQueueClient()
        self.engine_actor_handle = None

    def __repr__(self):
        return f"{self.__class__.__name__}(iid={self.instance_id})"

    async def put_nowait_to_servers(self,
                                    server_request_outputs: Dict[str, List],
                                    server_info_dict: Dict[str, ServerInfo]) -> None:
        if self.engine_actor_handle is None:
            self.engine_actor_handle = ray.get_actor(get_instance_name(self.server_id, self.instance_id), namespace="kunserve")
        tasks = []
        for server_id, req_outputs in server_request_outputs.items():
            server_info = server_info_dict[server_id]
            # for req_output in req_outputs:
            #     if hasattr(req_output, 'request_timestamps'):
            #         req_output.request_timestamps.engine_actor_put_queue_timestamp = time.time()
            tasks.append(asyncio.create_task(self.request_output_queue_client.put_nowait(req_outputs, server_info)))
        rets = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, ret in enumerate(rets):
            if isinstance(ret, Exception):
                server_id = list(server_request_outputs.keys())[idx]
                server_info = server_info_dict[server_id]
                logger.warning("Server {} is dead.".format(server_id))
                req_outputs = list(server_request_outputs.values())[idx]
                request_ids = [req_output.request_id for req_output in req_outputs]
                self.engine_actor_handle.abort_request.remote(request_ids)