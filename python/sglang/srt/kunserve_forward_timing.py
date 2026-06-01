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


def _raw_detail_flag() -> bool:
    """The explicit KUNSERVE_FORWARD_TIMING_DETAIL env, independent of profile."""
    return os.environ.get("KUNSERVE_FORWARD_TIMING_DETAIL", "0") not in _FALSE_VALUES


def kunserve_stage_profile_enabled() -> bool:
    """Stage-profile mode: one-flag clean per-stage GPU breakdown.

    ``KUNSERVE_STAGE_PROFILE=1`` turns the existing ``kunserve_timing_scope``
    brackets (attention / dispatch / expert / combine / TP all-reduce ...) into
    graph-recorded CUDA events, but routes the *readout* through the
    accumulate-and-summarize path (``kunserve_accumulate_cuda_graph_stage_events``)
    instead of the per-event-per-step logger that polluted the gap (+20ms/step)
    and distorted earlier analyses.  It also suppresses all per-step host logging
    so the only output is a compact periodic summary line.
    """
    if os.environ.get("KUNSERVE_STAGE_PROFILE", "0") in _FALSE_VALUES:
        return False
    return kunserve_timing_enabled()


def kunserve_detailed_timing_enabled() -> bool:
    # stage-profile activates the scopes (but with host logging suppressed,
    # see kunserve_timing_scope) so it can record the graph CUDA events.
    if kunserve_stage_profile_enabled():
        return True
    return kunserve_timing_enabled() and _raw_detail_flag()


def kunserve_scheduler_gap_timing_enabled() -> bool:
    override = os.environ.get("KUNSERVE_SCHEDULER_GAP_TIMING")
    if override is not None:
        return kunserve_timing_enabled() and override not in _FALSE_VALUES
    # NOT triggered by stage-profile: gap sub-stage host logging is exactly the
    # per-step pollution we want to avoid during a clean stage profile.
    return kunserve_timing_enabled() and _raw_detail_flag()


def kunserve_graph_internal_timing_enabled() -> bool:
    override = os.environ.get("KUNSERVE_GRAPH_INTERNAL_TIMING")
    if override is not None:
        return kunserve_timing_enabled() and override not in _FALSE_VALUES
    if kunserve_stage_profile_enabled():
        return True
    return kunserve_detailed_timing_enabled()


def kunserve_graph_internal_timing_interval() -> int:
    # In stage-profile mode the readout is the cheap accumulator, so sample
    # every replay (interval=1) for dense, stable per-stage means.  The old
    # heavy per-event logger needed the sparse 128 default to stay affordable.
    if kunserve_stage_profile_enabled() and "KUNSERVE_GRAPH_INTERNAL_TIMING_INTERVAL" not in os.environ:
        return 1
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


# ----------------------------------------------------------------------
# Stage-profile accumulator (clean, low-overhead readout)
# ----------------------------------------------------------------------
# variant -> {"n": int, "total_ms": float, "stages": {stage: cumulative_ms},
#             "last_bs": int}
_STAGE_PROFILE_ACCUM: Dict[str, Dict[str, Any]] = {}


def _stage_profile_flush_every() -> int:
    try:
        return max(1, int(os.environ.get("KUNSERVE_STAGE_PROFILE_FLUSH", "500")))
    except Exception:
        return 500


def _stage_profile_sample_every() -> int:
    try:
        return max(1, int(os.environ.get("KUNSERVE_STAGE_PROFILE_SAMPLE", "1")))
    except Exception:
        return 1


def kunserve_accumulate_cuda_graph_stage_events(
    graph_id: str,
    *,
    variant: str,
    replay_total_ms: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> int:
    """Accumulate per-stage graph CUDA-event times and periodically summarize.

    This is the clean replacement for ``kunserve_log_cuda_graph_stage_events`` in
    the hot path.  Instead of writing one JSON line per event per step (which
    cost ~20ms/step and polluted ``gap_ms``), it:

      1. Reads ``elapsed_time`` for every stage event of this replay and sums
         per-stage across all 48 layers into a *per-step* total.
      2. Adds those per-step totals into a cumulative running mean keyed by
         ``variant`` (``local`` / ``global`` / ``baseline``).
      3. Every ``KUNSERVE_STAGE_PROFILE_FLUSH`` steps emits ONE compact summary
         line (``event="kunserve_stage_profile"``) with the running mean ms per
         stage, the replay total, and step count.

    The events themselves use ``external=True`` so they do not serialize the
    captured graph; the only added cost is reading ``elapsed_time`` (a few
    microseconds each, sampled every ``KUNSERVE_STAGE_PROFILE_SAMPLE`` steps).
    """
    if not kunserve_timing_enabled():
        return 0
    events = _CUDA_GRAPH_STAGE_EVENTS.get(str(graph_id), [])
    if not events:
        return 0

    bucket = _STAGE_PROFILE_ACCUM.setdefault(
        variant, {"n": 0, "total_ms": 0.0, "stages": {}}
    )
    # n counts every step seen (for replay_total mean); sampled steps also add
    # the per-stage detail.  Keep them consistent by only counting sampled steps.
    sample_every = _stage_profile_sample_every()
    seen = int(bucket.get("_seen", 0)) + 1
    bucket["_seen"] = seen
    if (seen % sample_every) != 0:
        return 0

    per_step: Dict[str, float] = {}
    for stage, _stage_fields, start_event, end_event in events:
        try:
            elapsed_ms = float(start_event.elapsed_time(end_event))
        except Exception:
            continue
        per_step[stage] = per_step.get(stage, 0.0) + elapsed_ms

    bucket["n"] = int(bucket["n"]) + 1
    if replay_total_ms is not None:
        bucket["total_ms"] = float(bucket["total_ms"]) + float(replay_total_ms)
    if batch_size is not None:
        bucket["last_bs"] = int(batch_size)
    stages = bucket["stages"]
    for stage, ms in per_step.items():
        stages[stage] = float(stages.get(stage, 0.0)) + ms

    n = int(bucket["n"])
    if n % _stage_profile_flush_every() == 0:
        mean_stages = {k: round(v / n, 4) for k, v in stages.items()}
        kunserve_timing_log(
            "kunserve_stage_profile",
            variant=str(variant),
            n_steps=n,
            replay_total_ms=round(float(bucket["total_ms"]) / n, 4)
            if bucket["total_ms"]
            else None,
            last_bs=bucket.get("last_bs"),
            stages=mean_stages,
        )
    return len(per_step)


@contextlib.contextmanager
def kunserve_timing_scope(event: str, **fields: Any) -> Iterator[None]:
    if not kunserve_timing_enabled():
        yield
        return

    start_ns = time.perf_counter_ns()
    cuda_stage_record = _record_cuda_graph_stage_begin(event, fields)
    emit_host_log = (
        not kunserve_stage_profile_enabled()
        and (
            _CUDA_GRAPH_TIMING_CAPTURE_KEY is None
            or os.environ.get("KUNSERVE_GRAPH_CAPTURE_HOST_TIMING", "0")
            not in _FALSE_VALUES
        )
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
