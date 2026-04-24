import asyncio
import ray

from kunserve.llm import AsyncLLM
from kunserve.kunserve_config import KunServeConfig

from kunserve.logger import init_logger
logger = init_logger(__name__)

def connect_to_ray_cluster(head_node_ip: str = None,
                           port: int = None,
                           namespace: str ="kunserve",
                           log_to_driver: bool=True) -> None:
    if head_node_ip is not None and port is not None:
        ray.init(
            address=f"{head_node_ip}:{port}", 
            ignore_reinit_error=True, 
            namespace=namespace, 
            log_to_driver=log_to_driver,
            runtime_env={
                "env_vars": {
                    "PYTHONWARNINGS": "ignore::UserWarning:numpy._core.getlimits"
                }
            }
        )
    else:
        ray.init(
            ignore_reinit_error=True, 
            namespace=namespace, 
            log_to_driver=log_to_driver,
            runtime_env={
                "env_vars": {
                    "PYTHONWARNINGS": "ignore::UserWarning:numpy._core.getlimits"
                }
            }
        )

def setup_kunserve(kunserve_config: KunServeConfig):

    nservers = kunserve_config.engine_config.nservers
    ngroups = kunserve_config.engine_config.ngroups
    dp = kunserve_config.engine_config.group_size
    pp = kunserve_config.engine_config.pp
    tp = kunserve_config.engine_config.tp

    # [dp, pp * tp]
    placement_groups = []
    world_size = pp * tp
    for _ in range(nservers * ngroups * dp):
        placement_group = ray.util.placement_group(
            [{"CPU": 4}] + [{"GPU": 1}] * world_size,
            strategy="PACK",
        )
        ray.get(placement_group.ready(), timeout=1000)
        placement_groups.append(placement_group)

    servers = [
        AsyncLLM.from_engine_args(
            server_id,
            ngroups, 
            placement_groups[server_id * ngroups * dp: (server_id + 1) * ngroups * dp],
            kunserve_config,
        )
        for server_id in range(nservers)
    ]

    refs = []
    for server in servers:
        refs.append(server.initialize.remote())
    ray.get(refs)
    
    logger.info("KunServe is ready for serving.")
    return servers