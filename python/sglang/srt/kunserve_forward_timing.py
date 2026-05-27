from __future__ import annotations

import contextlib
import json
import os
import time
from typing import Any, Dict, Iterator, Optional

import torch


_FALSE_VALUES = {"0", "false", "False", "no", "off", ""}
_PID = os.getpid()


def _enabled_path() -> Optional[str]:
    if os.environ.get("KUNSERVE_FORWARD_TIMING", "1") in _FALSE_VALUES:
        return None

    path = os.environ.get("KUNSERVE_FORWARD_TIMING_LOG", "").strip()
    if path:
        return path

    out_dir = os.environ.get("SGLANG_KUNSERVE_OUTPUT_DIR", "").strip()
    if out_dir:
        return os.path.join(out_dir, "kunserve_forward_timing.jsonl")
    return None


def kunserve_timing_enabled() -> bool:
    return bool(_enabled_path())


def kunserve_detailed_timing_enabled() -> bool:
    return kunserve_timing_enabled() and (
        os.environ.get("KUNSERVE_FORWARD_TIMING_DETAIL", "0") not in _FALSE_VALUES
    )


def kunserve_scheduler_gap_timing_enabled() -> bool:
    override = os.environ.get("KUNSERVE_SCHEDULER_GAP_TIMING")
    if override is not None:
        return kunserve_timing_enabled() and override not in _FALSE_VALUES
    return kunserve_detailed_timing_enabled()


def kunserve_graph_internal_timing_enabled() -> bool:
    override = os.environ.get("KUNSERVE_GRAPH_INTERNAL_TIMING")
    if override is not None:
        return kunserve_timing_enabled() and override not in _FALSE_VALUES
    return kunserve_detailed_timing_enabled()


def kunserve_graph_internal_timing_interval() -> int:
    try:
        return max(1, int(os.environ.get("KUNSERVE_GRAPH_INTERNAL_TIMING_INTERVAL", "128")))
    except Exception:
        return 128


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def kunserve_timing_log(event: str, **fields: Any) -> None:
    path = _enabled_path()
    if not path:
        return

    record: Dict[str, Any] = {
        "ts": time.time(),
        "perf_ns": time.perf_counter_ns(),
        "pid": _PID,
        "replica_rank": os.environ.get("SGLANG_REPLICA_RANK", ""),
        "rank": os.environ.get("RANK", ""),
        "local_rank": os.environ.get("LOCAL_RANK", ""),
        "event": event,
    }
    record.update({str(k): _json_safe(v) for k, v in fields.items()})

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


_CUDA_GRAPH_TIMING_CAPTURE_KEY: Optional[str] = None
_CUDA_GRAPH_STAGE_EVENTS: Dict[str, list[tuple[str, Dict[str, Any], Any, Any]]] = {}


@contextlib.contextmanager
def kunserve_cuda_graph_timing_capture(graph_key: str) -> Iterator[None]:
    global _CUDA_GRAPH_TIMING_CAPTURE_KEY
    if not kunserve_graph_internal_timing_enabled():
        yield
        return

    prev_key = _CUDA_GRAPH_TIMING_CAPTURE_KEY
    key = str(graph_key)
    _CUDA_GRAPH_TIMING_CAPTURE_KEY = key
    _CUDA_GRAPH_STAGE_EVENTS[key] = []
    try:
        yield
    finally:
        _CUDA_GRAPH_TIMING_CAPTURE_KEY = prev_key


def _record_cuda_graph_stage_begin(
    event: str, fields: Dict[str, Any]
) -> Optional[tuple[str, Dict[str, Any], Any, Any]]:
    key = _CUDA_GRAPH_TIMING_CAPTURE_KEY
    if key is None or not kunserve_graph_internal_timing_enabled():
        return None
    try:
        if not torch.cuda.is_available():
            return None
        try:
            start_event = torch.cuda.Event(enable_timing=True, external=True)
            end_event = torch.cuda.Event(enable_timing=True, external=True)
        except TypeError:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return event, dict(fields), start_event, end_event
    except Exception:
        return None


def _record_cuda_graph_stage_end(
    record: Optional[tuple[str, Dict[str, Any], Any, Any]]
) -> None:
    if record is None:
        return
    key = _CUDA_GRAPH_TIMING_CAPTURE_KEY
    if key is None:
        return
    try:
        record[3].record()
        _CUDA_GRAPH_STAGE_EVENTS.setdefault(key, []).append(record)
    except Exception:
        pass


def kunserve_log_cuda_graph_stage_events(
    graph_id: str,
    *,
    max_events: Optional[int] = None,
    **fields: Any,
) -> int:
    if not kunserve_timing_enabled():
        return 0
    events = _CUDA_GRAPH_STAGE_EVENTS.get(str(graph_id), [])
    if not events:
        return 0
    if max_events is not None:
        events = events[: max(0, int(max_events))]

    logged = 0
    for stage, stage_fields, start_event, end_event in events:
        try:
            elapsed_ms = float(start_event.elapsed_time(end_event))
        except Exception:
            continue
        payload = dict(fields)
        payload.update(stage_fields)
        kunserve_timing_log(
            f"graph_{stage}_end",
            elapsed_ms=round(elapsed_ms, 3),
            **payload,
        )
        logged += 1
    return logged


@contextlib.contextmanager
def kunserve_timing_scope(event: str, **fields: Any) -> Iterator[None]:
    if not kunserve_timing_enabled():
        yield
        return

    start_ns = time.perf_counter_ns()
    cuda_stage_record = _record_cuda_graph_stage_begin(event, fields)
    emit_host_log = (
        _CUDA_GRAPH_TIMING_CAPTURE_KEY is None
        or os.environ.get("KUNSERVE_GRAPH_CAPTURE_HOST_TIMING", "0")
        not in _FALSE_VALUES
    )
    if emit_host_log:
        kunserve_timing_log(f"{event}_begin", **fields)
    try:
        yield
    finally:
        _record_cuda_graph_stage_end(cuda_stage_record)
        if emit_host_log:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            kunserve_timing_log(
                f"{event}_end",
                elapsed_ms=round(elapsed_ms, 3),
                **fields,
            )
