from __future__ import annotations

import logging
import os
import socket
import time
from pathlib import Path
from typing import Optional

from sglang.srt.kunserve_manager.controller import KunServeController

logger = logging.getLogger(__name__)


def _get_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _get_int_env(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _get_str_list_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]



def _resolve_advertise_host(host: str) -> str:
    if host and host not in ("0.0.0.0", "::"):
        return host
    return socket.gethostbyname(socket.gethostname())


def register_current_replica_from_env(*, host: str, port: int) -> None:
    discovery_file = os.getenv("SGLANG_KUNSERVE_DISCOVERY_FILE", "").strip()
    if not discovery_file:
        return
    address = f"{_resolve_advertise_host(host)}:{int(port)}"
    path = Path(discovery_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if path.exists():
        existing = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    if address not in existing:
        with path.open("a", encoding="utf-8") as f:
            f.write(address + "\n")
    logger.info("[KunServeManager] registered replica address %s in %s", address, path)


def _discover_replica_addresses() -> list[str]:
    addresses = _get_str_list_env("SGLANG_KUNSERVE_REPLICA_ADDRESSES")
    if len(addresses) >= 2:
        return addresses[:2]

    discovery_file = os.getenv("SGLANG_KUNSERVE_DISCOVERY_FILE", "").strip()
    if not discovery_file:
        return addresses

    deadline = time.time() + _get_float_env("SGLANG_KUNSERVE_DISCOVERY_TIMEOUT", 300.0)
    path = Path(discovery_file)
    while time.time() < deadline:
        if path.exists():
            discovered = []
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and line not in discovered:
                    discovered.append(line)
            if len(discovered) >= 2:
                return discovered[:2]
        time.sleep(0.5)
    return addresses

def should_start_kunserve_manager() -> bool:
    return _get_bool_env("SGLANG_KUNSERVE_MANAGER_ENABLE", False)



def _win_manager_election() -> bool:
    discovery_file = os.getenv("SGLANG_KUNSERVE_DISCOVERY_FILE", "").strip()
    if not discovery_file:
        return True
    lock_path = Path(discovery_file + ".manager.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        logger.info("[KunServeManager] another sglang process won manager election: %s", lock_path)
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()}\n")
    logger.info("[KunServeManager] won manager election: %s", lock_path)
    return True

def build_kunserve_controller_from_env(*, model_path: str) -> Optional[KunServeController]:
    """Build the in-sglang KunServe controller from env config.

    Expected env on exactly one sglang HTTP process (the coordinator):
      SGLANG_KUNSERVE_MANAGER_ENABLE=1
      SGLANG_KUNSERVE_REPLICA_ADDRESSES=host0:port0,host1:port1
    Optional env mirrors the old verl-side knobs.
    """

    if not should_start_kunserve_manager():
        return None
    if not _win_manager_election():
        return None

    addresses = _discover_replica_addresses()
    if len(addresses) != 2:
        raise ValueError(
            "SGLANG_KUNSERVE_MANAGER_ENABLE=1 requires "
            "SGLANG_KUNSERVE_REPLICA_ADDRESSES with exactly two host:port entries"
        )

    controller = KunServeController.from_server_addresses(
        addresses,
        model_path=os.getenv("SGLANG_KUNSERVE_MODEL_PATH", model_path),
        poll_interval=_get_float_env("SGLANG_KUNSERVE_POLL_INTERVAL", 2.0),
        min_running_requests_per_replica=int(
            _get_int_env("SGLANG_KUNSERVE_MIN_RUNNING_REQUESTS_PER_REPLICA", 1)
        ),
        offload_local_experts=_get_int_env("SGLANG_KUNSERVE_OFFLOAD_LOCAL_EXPERTS", None),
        group_name=os.getenv("SGLANG_KUNSERVE_GROUP_NAME", "kunserve_global_ep"),
        backend=os.getenv("SGLANG_KUNSERVE_BACKEND", "nccl"),
        enable_restore=_get_bool_env("SGLANG_KUNSERVE_ENABLE_RESTORE", False),
        eager_warmup=_get_bool_env("SGLANG_KUNSERVE_EAGER_WARMUP", True),
    )
    logger.info("[KunServeManager] built controller for replicas=%s", addresses)
    return controller
