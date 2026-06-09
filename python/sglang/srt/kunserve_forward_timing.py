from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator, Optional

_FALSE_VALUES = {"0", "false", "False", "no", "off", ""}
_TRUE_VALUES = {"1", "true", "True", "yes", "on"}
_PID = os.getpid()


def _nvtx_enabled() -> bool:
    return (
        os.environ.get("KUNSERVE_NVTX_STAGE_PROFILE", "0") in _TRUE_VALUES
        or os.environ.get("KUNSERVE_STAGE_NVTX", "0") in _TRUE_VALUES
        or os.environ.get("KUNSERVE_NSYS_STAGE_PROFILE", "0") in _TRUE_VALUES
    )


def kunserve_timing_enabled() -> bool:
    return _nvtx_enabled()


def kunserve_detailed_timing_enabled() -> bool:
    return _nvtx_enabled()


def kunserve_stage_profile_enabled() -> bool:
    return False


def kunserve_scheduler_gap_timing_enabled() -> bool:
    return False


def kunserve_graph_internal_timing_enabled() -> bool:
    return False


def kunserve_graph_internal_timing_interval() -> int:
    return 0


def kunserve_timing_log(event: str, **fields: Any) -> None:
    return None


def _field_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float, str)):
        return str(value).replace("|", "/").replace("=", ":").replace(" ", "_")
    return str(value).replace("|", "/").replace("=", ":").replace(" ", "_")


def _nvtx_message(event: str, fields: dict[str, Any]) -> str:
    base = {
        "event": event,
        "pid": _PID,
        "rank": os.environ.get("RANK", ""),
        "local_rank": os.environ.get("LOCAL_RANK", ""),
        "replica_rank": os.environ.get("SGLANG_REPLICA_RANK", ""),
    }
    base.update({str(k): v for k, v in fields.items()})
    return "ks_stage|" + "|".join(
        f"{k}={_field_value(v)}" for k, v in base.items()
    )


@contextlib.contextmanager
def kunserve_timing_scope(event: str, **fields: Any) -> Iterator[None]:
    if not _nvtx_enabled():
        yield
        return
    try:
        import torch

        torch.cuda.nvtx.range_push(_nvtx_message(event, fields))
        pushed = True
    except Exception:
        pushed = False
    try:
        yield
    finally:
        if pushed:
            try:
                import torch

                torch.cuda.nvtx.range_pop()
            except Exception:
                pass


@contextlib.contextmanager
def kunserve_cuda_graph_timing_capture(graph_key: str) -> Iterator[None]:
    yield


def kunserve_log_cuda_graph_stage_events(
    graph_id: str,
    *,
    max_events: Optional[int] = None,
    **fields: Any,
) -> int:
    return 0


def kunserve_accumulate_cuda_graph_stage_events(
    graph_id: str,
    *,
    variant: str,
    replay_total_ms: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> int:
    return 0
