from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from kunserve_manager.client import (
    KunServeHttpReplicaClient,
    KunServeReplicaClient,
    _commit_response_succeeded,
    _parse_server_address,
    _response_error,
    _response_succeeded,
    _unwrap_output_list,
    _unwrap_status_response,
)
from kunserve_manager.layout import (
    KunServeLayoutPlan,
    build_layout_plan_from_statuses,
)
from kunserve_manager.net import get_free_port
from kunserve_manager.runtime_config import KunServeRuntimeBackendConfig

logger = logging.getLogger(__name__)


class KunServeController:
    def __init__(
        self,
        replicas: Sequence[KunServeReplicaClient],
        *,
        poll_interval: float = 2.0,
        min_running_requests_per_replica: int = 1,
        offload_local_experts: Optional[int] = None,
        group_name: str = "kunserve_global_ep",
        backend: str = "nccl",
        comm_backend: str = "sglang",
        capture_policy: str = "auto",
        enable_restore: bool = False,
        pg_init_max_attempts: int = 8,
        pg_init_retry_delay: float = 1.0,
        eager_warmup: bool = False,
        output_dir: Optional[str] = None,
        write_bw_log: Optional[bool] = None,
    ):
        if len(replicas) != 2:
            raise ValueError(
                f"KunServeController currently supports exactly 2 replicas, got {len(replicas)}"
            )
        self._replicas = list(replicas)
        self.poll_interval = float(poll_interval)
        self.min_running_requests_per_replica = int(min_running_requests_per_replica)
        self.offload_local_experts = (
            None if offload_local_experts is None else int(offload_local_experts)
        )
        # group_name is the *desired* base name; the actually-used name (with a
        # uniqueness suffix) is recorded back into self.group_name after a
        # successful PG bring-up so downstream RPCs reference the live group.
        self._base_group_name = group_name
        self.group_name = group_name
        # Phase F lane subgroup names, populated by _ensure_lane_subgroups
        # after the global group succeeds.  Format:
        #   {0: "kunserve_lane0_v...", 1: "kunserve_lane1_v..."}
        # Empty until lane init runs successfully.  Used when building the
        # warmup_balloon payload so each replica can resolve its lane
        # process group by name.
        self.lane_group_names: dict[int, str] = {}
        self.runtime_backend = KunServeRuntimeBackendConfig(
            comm_backend=comm_backend,
            capture_policy=capture_policy,
        )
        requested_backend = str(backend or "nccl")
        if (
            self.runtime_backend.comm_backend == "sglang"
            and requested_backend.lower() in ("nccl", "")
        ):
            self.backend = "kunserve_pynccl"
        else:
            self.backend = requested_backend
        # Disabling RESTORE keeps the control surface minimal for the first
        # end-to-end bring-up. The restore code paths are preserved unchanged
        # so they can be re-enabled later as a follow-up optimization.
        self.enable_restore = bool(enable_restore)
        self._pg_init_max_attempts = int(pg_init_max_attempts)
        self._pg_init_retry_delay = float(pg_init_retry_delay)
        # When True (default), call /kunserve/warmup_balloon on every replica
        # at startup so the GLOBAL CUDA graph is captured before the first
        # retract_decode arrives. This trades ~30s of extra startup time for
        # near-zero BALLOON entry latency. Set False to fall back to the old
        # behaviour where prepare_balloon does the capture lazily.
        self.eager_warmup = bool(eager_warmup)
        self._warmup_done = False
        self.output_dir = Path(
            output_dir
            or os.environ.get("KUNSERVE_MANAGER_OUTPUT_DIR")
            or os.environ.get("SGLANG_KUNSERVE_OUTPUT_DIR")
            or "/workspace/sglang/output"
        )
        self._bw_log_path = self.output_dir / "bw_throughput.jsonl"
        if write_bw_log is None:
            write_bw_log = os.environ.get(
                "KUNSERVE_MANAGER_WRITE_BW_LOG", "1"
            ).lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        self.write_bw_log = bool(write_bw_log)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._thread_started = threading.Event()
        self._balloon_active = False
        self._process_group_initialized = False
        self._layout_plan: Optional[KunServeLayoutPlan] = None
        self._last_asymmetric_balloon_signature: Optional[
            tuple[tuple[int, int], ...]
        ] = None
        self._tick_count = 0
        self._last_decision: Optional[str] = None
        self._stale_decode_warn_seconds = float(
            os.environ.get("KUNSERVE_STALE_DECODE_WARN_SECONDS", "30")
        )
        self._decode_progress_state: dict[int, tuple[int, float, float]] = {}

    def _emit(self, message: str) -> None:
        print(f"[KunServeController] {message}", flush=True)
        logger.warning("[KunServeController] %s", message)

    def _summarize_statuses(self, statuses: Sequence[dict[str, Any]]) -> str:
        parts = []
        for idx, status in enumerate(statuses):
            parts.append(
                "r%d(state=%s variant=%s expand=%s running=%s waiting=%s "
                "offloaded=%s slots=%s fwd=%s cur=%s/%s last=%s/%s)"
                % (
                    idx,
                    status.get("state"),
                    status.get("runtime_variant"),
                    status.get("expand_requested"),
                    status.get("num_running_requests"),
                    status.get("num_waiting_requests"),
                    status.get("offloaded_local_experts"),
                    status.get("added_kv_slots"),
                    status.get("scheduler_forward_ct"),
                    status.get("scheduler_cur_batch_mode"),
                    status.get("scheduler_cur_batch_size"),
                    status.get("scheduler_last_batch_mode"),
                    status.get("scheduler_last_batch_size"),
                )
            )
        return " ".join(parts)

    @classmethod
    def from_server_addresses(
        cls,
        server_addresses: Sequence[str],
        *,
        model_path: str,
        timeout: float = 60.0,
        max_attempts: int = 3,
        retry_delay: float = 2.0,
        max_start_wait_time: float = 300.0,
        max_connections: int = 64,
        **kwargs,
    ) -> "KunServeController":
        replicas = []
        for idx, server_address in enumerate(server_addresses):
            host, port = _parse_server_address(server_address)
            replicas.append(
                KunServeHttpReplicaClient(
                    name=f"replica_{idx}",
                    host=host,
                    port=port,
                    model_path=model_path,
                    timeout=timeout,
                    max_attempts=max_attempts,
                    retry_delay=retry_delay,
                    max_start_wait_time=max_start_wait_time,
                    max_connections=max_connections,
                )
            )
        return cls(replicas, **kwargs)

    async def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._emit(
            "starting: replicas=%d poll_interval=%.2fs min_running=%d group=%s "
            "backend=%s comm_backend=%s capture_policy=%s enable_restore=%s"
            % (
                len(self._replicas),
                self.poll_interval,
                self.min_running_requests_per_replica,
                self.group_name,
                self.backend,
                self.runtime_backend.comm_backend,
                self.runtime_backend.capture_policy,
                self.enable_restore,
            )
        )

        # The manager can be elected by whichever replica HTTP process reaches
        # lifespan startup first.  At that moment the discovery file may already
        # contain both host:port entries, but the peer HTTP server can still be
        # finishing model/cuda-graph warmup and may not answer /kunserve/status
        # yet.  Treat this as normal bootstrap skew and wait here instead of
        # letting a single early status probe permanently kill the manager.
        initial_statuses = await self._wait_for_replicas_ready()

        # Bring up the cross-replica EP process group exactly once, before the
        # polling thread starts. Doing it here (rather than lazily inside
        # enter_balloon) means:
        #  - Failures fail-fast and propagate to the caller (the training job)
        #    instead of being retried silently every tick.
        #  - The scheduler request queue on each replica is hit only once for
        #    init_weights_update_group, eliminating the case where a stuck
        #    NCCL handshake also blocks /kunserve/status probes.
        # If the initial status fetch or layout planning fails, we surface the
        # error here. The replicas must be healthy by the time the
        # AgentLoopManager calls start().
        plan = self._ensure_layout_plan(initial_statuses)
        await self._ensure_process_group(plan)

        # Eager-capture the GLOBAL CUDA graph + register cross-replica EP
        # bundle on every replica BEFORE starting the polling loop. This
        # converts the slow path (~30s) inside prepare_balloon into a no-op
        # later, so when retract_decode finally fires the BALLOON entry
        # latency drops from tens of seconds to <1s.
        # Failures here are downgraded to a warning: the controller still
        # starts, and BALLOON entry will fall back to the lazy capture path
        # the first time it is triggered.
        if self.eager_warmup:
            try:
                await self._warmup_balloon(plan)
                self._warmup_done = True
            except Exception as exc:
                self._emit(
                    "eager warmup failed; will fall back to lazy capture in "
                    "prepare_balloon. error=%r" % exc
                )

        self._running = True
        self._thread_started.clear()
        self._thread = threading.Thread(
            target=self._thread_main,
            daemon=True,
            name="kunserve-controller",
        )
        self._thread.start()
        started = await asyncio.to_thread(
            self._thread_started.wait,
            max(5.0, self.poll_interval + 5.0),
        )
        if not started or self._loop is None:
            self._running = False
            raise RuntimeError("KunServeController thread failed to start.")

    async def _wait_for_replicas_ready(self) -> list[dict[str, Any]]:
        deadline = time.time() + max(
            float(getattr(replica, "max_start_wait_time", 300.0))
            for replica in self._replicas
        )
        attempt = 0
        last_error: Optional[BaseException] = None
        while True:
            attempt += 1
            try:
                statuses = await self._fetch_statuses()
                self._emit(
                    "replicas ready after %d status probe(s): %s"
                    % (attempt, self._summarize_statuses(statuses))
                )
                return statuses
            except Exception as exc:
                last_error = exc
                now = time.time()
                if now >= deadline:
                    break
                if attempt == 1 or attempt % 10 == 0:
                    self._emit(
                        "waiting for replica /kunserve/status endpoints "
                        "to become ready: attempt=%d error=%r" % (attempt, exc)
                    )
                await asyncio.sleep(
                    min(
                        5.0,
                        max(
                            float(getattr(replica, "retry_delay", 2.0))
                            for replica in self._replicas
                        ),
                    )
                )

        raise RuntimeError(
            "replicas did not become ready before KunServe manager startup "
            f"deadline; last_error={last_error!r}"
        )

    async def stop(self) -> None:
        self._emit("stopping")
        self._running = False
        if self._loop is not None and self._loop.is_running():
            if self._stop_event is not None:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            else:
                self._loop.call_soon_threadsafe(lambda: None)
        if self._thread is not None:
            await asyncio.to_thread(
                self._thread.join,
                max(10.0, self.poll_interval + 10.0),
            )
            if self._thread.is_alive():
                raise RuntimeError("KunServeController thread did not stop in time.")
            self._thread = None
        # Mirror the gating in _poll_loop: only tear down the cross-replica PG
        # when restore is enabled. Otherwise the process is exiting and the
        # kernel handles cleanup.
        if self.enable_restore and self._process_group_initialized:
            await self._destroy_process_group(force=True)
        self._emit("stopped")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._stop_event = asyncio.Event()
        self._thread_started.set()
        try:
            loop.run_until_complete(self._poll_loop())
        except Exception:
            logger.exception("[KunServeController] thread crashed.")
        finally:
            self._stop_event = None
            self._loop = None
            self._running = False
            loop.close()

    async def tick(self) -> list[dict[str, Any]]:
        statuses = await self._fetch_statuses()
        self._tick_count += 1
        self._check_stale_decode_statuses(statuses)
        self._write_bw_status_sample(statuses)
        status_summary = self._summarize_statuses(statuses)
        if self._tick_count <= 5 or self._tick_count % 30 == 0:
            self._emit(f"tick#{self._tick_count} statuses: {status_summary}")
        if any(status.get("state") == "prepared" for status in statuses):
            logger.warning(
                "[KunServeController] detected leftover prepared state, forcing rollback: %s",
                self._summarize_statuses(statuses),
            )
            await self._rollback_prepared_replicas(statuses)
            statuses = await self._fetch_statuses()

        self._balloon_active = any(
            status.get("state") == "balloon" for status in statuses
        )
        self._log_asymmetric_balloon_risk(statuses)
        enter_ok, enter_reason = self._should_enter_balloon(statuses)
        if not self._balloon_active:
            decision = f"enter={enter_ok} reason={enter_reason}"
        if not self._balloon_active and enter_ok:
            self._last_decision = decision
            self._emit(f"enter conditions met: {enter_reason}")
            await self.enter_balloon(statuses=statuses)
            return await self._fetch_statuses()
        if not self._balloon_active and enter_reason:
            if decision != self._last_decision:
                self._emit(f"keep local runtime: {enter_reason}")
                self._last_decision = decision
        # RESTORE is gated behind enable_restore. The first end-to-end pass
        # only needs LOCAL -> BALLOON; the runtime stays in BALLOON until the
        # rollout job exits. The restore decision logic below is preserved
        # unchanged so it can be turned back on as a future optimization.
        if self.enable_restore:
            restore_ok, restore_reason = self._should_restore_balloon(statuses)
            if self._balloon_active:
                decision = f"restore={restore_ok} reason={restore_reason}"
            if self._balloon_active and restore_ok:
                self._last_decision = decision
                self._emit(f"restore conditions met: {restore_reason}")
                await self.restore_balloon(require_idle=True)
                return await self._fetch_statuses()
            if self._balloon_active and restore_reason:
                if decision != self._last_decision:
                    self._emit(f"keep balloon runtime: {restore_reason}")
                    self._last_decision = decision
        return statuses

    def _write_bw_status_sample(self, statuses: Sequence[dict[str, Any]]) -> None:
        """Write a lightweight bw_throughput-compatible status sample.

        The old verl wrapper generated bw_throughput.jsonl from an external
        scraper.  After moving the control plane into the standalone
        kunserve_manager, keep producing the fields needed by the existing
        analysis scripts from the manager's regular /kunserve/status poll.
        GPU SM/HBM fields are intentionally best-effort and may be absent; the
        plotting code already tolerates samples that contain only
        running/token_usage/throughput.
        """

        if not self.write_bw_log:
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            row: dict[str, Any] = {"ts": time.time()}
            for idx, status in enumerate(statuses, start=1):
                suffix = f"sglang{idx}"
                row[f"running_{suffix}"] = float(
                    status.get("num_running_requests", 0) or 0
                )
                row[f"waiting_{suffix}"] = float(
                    status.get("num_waiting_requests", 0) or 0
                )
                row[f"token_usage_{suffix}"] = float(
                    status.get("token_usage", 0.0) or 0.0
                )
                row[f"throughput_{suffix}"] = float(
                    status.get("gen_throughput", 0.0) or 0.0
                )
                row[f"state_{suffix}"] = status.get("state")
                row[f"variant_{suffix}"] = status.get("runtime_variant")
                row[f"forward_ct_{suffix}"] = status.get("scheduler_forward_ct")
                row[f"cur_batch_{suffix}"] = status.get("scheduler_cur_batch_mode")
                row[f"last_batch_{suffix}"] = status.get(
                    "scheduler_last_batch_mode"
                )
                row[f"health_code_{suffix}"] = status.get("kunserve_health_code")
            with self._bw_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("failed to write bw status sample", exc_info=True)

    def _check_stale_decode_statuses(self, statuses: Sequence[dict[str, Any]]) -> None:
        if self._stale_decode_warn_seconds <= 0:
            return

        now = time.time()
        active_indices = set()
        for idx, status in enumerate(statuses):
            try:
                running = int(status.get("num_running_requests", 0) or 0)
                forward_ct = int(status.get("scheduler_forward_ct"))
            except (TypeError, ValueError):
                self._decode_progress_state.pop(idx, None)
                continue

            is_active_decode = status.get("state") == "balloon" and running > 0
            if not is_active_decode:
                self._decode_progress_state.pop(idx, None)
                continue

            active_indices.add(idx)
            prev = self._decode_progress_state.get(idx)
            if prev is None or prev[0] != forward_ct:
                self._decode_progress_state[idx] = (forward_ct, now, 0.0)
                status["kunserve_health_code"] = "OK"
                continue

            _, unchanged_since, last_emit = prev
            unchanged_for = now - unchanged_since
            if unchanged_for < self._stale_decode_warn_seconds:
                status["kunserve_health_code"] = "OK"
                continue

            status["kunserve_health_code"] = "KUNSERVE_STALE_DECODE"
            status["kunserve_stale_decode_seconds"] = round(unchanged_for, 3)
            if last_emit <= 0 or now - last_emit >= self._stale_decode_warn_seconds:
                self._decode_progress_state[idx] = (forward_ct, unchanged_since, now)
                self._emit(
                    "KUNSERVE_STALE_DECODE replica=%d unchanged_for=%.1fs "
                    "forward_ct=%d running=%s waiting=%s state=%s variant=%s "
                    "cur=%s/%s last=%s/%s"
                    % (
                        idx,
                        unchanged_for,
                        forward_ct,
                        status.get("num_running_requests"),
                        status.get("num_waiting_requests"),
                        status.get("state"),
                        status.get("runtime_variant"),
                        status.get("scheduler_cur_batch_mode"),
                        status.get("scheduler_cur_batch_size"),
                        status.get("scheduler_last_batch_mode"),
                        status.get("scheduler_last_batch_size"),
                    )
                )

        for idx in list(self._decode_progress_state.keys()):
            if idx not in active_indices:
                self._decode_progress_state.pop(idx, None)

    def _log_asymmetric_balloon_risk(self, statuses: Sequence[dict[str, Any]]) -> None:
        if not self._balloon_active:
            self._last_asymmetric_balloon_signature = None
            return

        running_waiting = tuple(
            (
                int(status.get("num_running_requests", 0)),
                int(status.get("num_waiting_requests", 0)),
            )
            for status in statuses
        )
        any_idle = any(
            running == 0 and waiting == 0 for running, waiting in running_waiting
        )
        any_busy = any(
            running > 0 or waiting > 0 for running, waiting in running_waiting
        )
        if any_idle and any_busy:
            if running_waiting != self._last_asymmetric_balloon_signature:
                logger.warning(
                    "[KunServeController] balloon runtime asymmetric drain detected: %s. "
                    "This path requires scheduler Phase E keepalive so idle replicas "
                    "continue participating in cross-replica MoE collectives.",
                    self._summarize_statuses(statuses),
                )
                self._last_asymmetric_balloon_signature = running_waiting
            return

        self._last_asymmetric_balloon_signature = None

    def _build_layout_payloads(
        self, plan: KunServeLayoutPlan, *, capture_cuda_graph: bool = True
    ) -> list[dict[str, Any]]:
        """Build the per-replica payload sent to /kunserve/{warmup,prepare}_balloon.

        Both warmup and prepare share the exact same per-replica configuration
        (target_variant, runtime_ep_size, active_local_expert_mapping, etc.),
        so we generate it once and the call site picks the endpoint.
        """
        effective_capture_cuda_graph = bool(
            capture_cuda_graph and self.runtime_backend.should_capture_global_graph()
        )
        backend_config = {
            "exchange_mode": self.runtime_backend.exchange_mode,
            "local_ep_size": plan.local_ep_size,
            "num_replicas": len(self._replicas),
            "global_world_size": plan.global_world_size,
        }
        pg_names: dict[str, str] = {"global": self.group_name}
        # Phase F: when lane subgroups are initialized, surface their
        # names so the model_runner can pass the right per-lane handle
        # to CrossReplicaStandardDispatcher.  Empty dict means the
        # dispatcher falls back to the global group for everything.
        for lane_idx, lane_name in self.lane_group_names.items():
            pg_names[f"lane_{int(lane_idx)}"] = lane_name
        payloads: list[dict[str, Any]] = []
        for replica_idx in range(len(self._replicas)):
            payloads.append(
                {
                    "target_variant": "global",
                    "runtime_ep_size": plan.global_world_size,
                    "runtime_rank_offset": replica_idx * plan.local_ep_size,
                    "dispatch_rank_offset": replica_idx * plan.local_ep_size,
                    "active_local_expert_mapping": plan.replica_active_mappings[
                        replica_idx
                    ],
                    "physical_to_logical_map": plan.global_physical_to_logical_map,
                    "process_group_name": self.group_name,
                    "capture_cuda_graph": effective_capture_cuda_graph,
                    "kunserve_comm_backend": self.runtime_backend.comm_backend,
                    "capture_policy": self.runtime_backend.capture_policy,
                    "kunserve_pg_names": pg_names,
                    "kunserve_backend_config": backend_config,
                }
            )
        return payloads

    async def _warmup_balloon(self, plan: KunServeLayoutPlan) -> None:
        """Dispatch /kunserve/warmup_balloon to every replica, in parallel.

        The replicas register the GLOBAL FusedMoE bundle and capture the
        GLOBAL CUDA graph but do NOT change `_balloon_state`; engines stay
        in LOCAL mode and continue serving requests. Returning successfully
        means a subsequent prepare_balloon -> commit_balloon transition will
        skip the heavy capture step.
        """
        payloads = self._build_layout_payloads(plan, capture_cuda_graph=True)
        capture_cuda_graph = bool(payloads and payloads[0].get("capture_cuda_graph"))
        self._emit(
            "warmup balloon: world=%d retained=%d offload=%d comm_backend=%s "
            "capture_policy=%s capture_graph=%s"
            % (
                plan.global_world_size,
                plan.retained_local_experts,
                plan.offload_local_experts,
                self.runtime_backend.comm_backend,
                self.runtime_backend.capture_policy,
                capture_cuda_graph,
            )
        )
        logger.warning(
            "[KUNSERVE-MS] WARMUP dispatch warmup_balloon: world=%d retained=%d "
            "offload=%d comm_backend=%s capture_policy=%s capture_graph=%s",
            plan.global_world_size,
            plan.retained_local_experts,
            plan.offload_local_experts,
            self.runtime_backend.comm_backend,
            self.runtime_backend.capture_policy,
            capture_cuda_graph,
        )
        results = await asyncio.gather(
            *[
                replica.warmup_balloon(payload)
                for replica, payload in zip(self._replicas, payloads, strict=True)
            ],
            return_exceptions=True,
        )
        errors: list[str] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"{self._replicas[idx].name}: {result!r}")
                continue
            if not _response_succeeded(result):
                errors.append(f"{self._replicas[idx].name}: {_response_error(result)}")
        if errors:
            raise RuntimeError(f"warmup_balloon failed: {'; '.join(errors)}")
        logger.warning(
            "[KUNSERVE-MS] WARMUP done: replicas=%d capture_graph=%s (state stays LOCAL)",
            len(self._replicas),
            capture_cuda_graph,
        )

    async def enter_balloon(
        self, *, statuses: Optional[Sequence[dict[str, Any]]] = None
    ) -> list[dict[str, Any]]:
        current_statuses = (
            list(statuses) if statuses is not None else await self._fetch_statuses()
        )
        plan = self._ensure_layout_plan(current_statuses)
        # Process group is brought up once in start(); _ensure_process_group is
        # a no-op if already initialized but kept here as a safety net in case
        # this controller is reused without a fresh start() call.
        await self._ensure_process_group(plan)

        # When warmup already ran, prepare_balloon's ensure_cuda_graph_variant_captured
        # short-circuits via has_captured_variant("global"), so this becomes a fast
        # state-flip + the much smaller commit_balloon work below.
        prepare_payloads = self._build_layout_payloads(plan, capture_cuda_graph=True)
        prepare_capture_graph = bool(
            prepare_payloads and prepare_payloads[0].get("capture_cuda_graph")
        )

        self._emit(
            "preparing balloon: offload=%d retained=%d world=%d "
            "comm_backend=%s capture_policy=%s capture_graph=%s payloads=%s"
            % (
                plan.offload_local_experts,
                plan.retained_local_experts,
                plan.global_world_size,
                self.runtime_backend.comm_backend,
                self.runtime_backend.capture_policy,
                prepare_capture_graph,
                [
                    {
                        "replica": replica_idx,
                        "runtime_rank_offset": payload["runtime_rank_offset"],
                        "active_local_expert_mapping": payload[
                            "active_local_expert_mapping"
                        ],
                    }
                    for replica_idx, payload in enumerate(prepare_payloads)
                ],
            )
        )

        logger.warning(
            "[KUNSERVE-MS] BALLOON dispatch prepare_balloon: offload=%d retained=%d "
            "world=%d comm_backend=%s capture_policy=%s capture_graph=%s",
            plan.offload_local_experts,
            plan.retained_local_experts,
            plan.global_world_size,
            self.runtime_backend.comm_backend,
            self.runtime_backend.capture_policy,
            prepare_capture_graph,
        )
        prepare_results = await asyncio.gather(
            *[
                replica.prepare_balloon(payload)
                for replica, payload in zip(
                    self._replicas, prepare_payloads, strict=True
                )
            ]
        )
        prepared_indices = [
            idx
            for idx, result in enumerate(prepare_results)
            if _response_succeeded(result)
        ]
        if len(prepared_indices) != len(self._replicas):
            await self._restore_replicas(prepared_indices, require_idle=False)
            errors = [
                f"{self._replicas[idx].name}: {_response_error(result)}"
                for idx, result in enumerate(prepare_results)
                if not _response_succeeded(result)
            ]
            raise RuntimeError(f"prepare_balloon failed: {'; '.join(errors)}")
        logger.warning(
            "[KUNSERVE-MS] BALLOON prepare_balloon ok: replicas=%d (graph capture done)",
            len(prepared_indices),
        )

        logger.warning(
            "[KUNSERVE-MS] BALLOON dispatch commit_balloon: offload=%d",
            plan.offload_local_experts,
        )
        commit_results = await asyncio.gather(
            *[
                replica.commit_balloon(
                    {
                        "target_variant": "global",
                        "offload_local_experts": plan.offload_local_experts,
                        "require_prepared": True,
                    }
                )
                for replica in self._replicas
            ]
        )
        commit_successes = [
            _commit_response_succeeded(
                result,
                target_variant="global",
                offload_local_experts=plan.offload_local_experts,
            )
            for result in commit_results
        ]
        for idx, (result, success) in enumerate(zip(commit_results, commit_successes)):
            if success and not _response_succeeded(result):
                logger.warning(
                    "[KunServeController] treating commit_balloon response as success because replica already reached target balloon state: replica=%s error=%s",
                    self._replicas[idx].name,
                    _response_error(result),
                )
        committed_indices = [
            idx for idx, success in enumerate(commit_successes) if success
        ]
        if len(committed_indices) != len(self._replicas):
            await self._restore_replicas(committed_indices, require_idle=False)
            errors = [
                f"{self._replicas[idx].name}: {_response_error(result)}"
                for idx, result in enumerate(commit_results)
                if not commit_successes[idx]
            ]
            raise RuntimeError(f"commit_balloon failed: {'; '.join(errors)}")

        self._balloon_active = True
        # Read back per-replica added_kv_slots from commit responses for the
        # milestone log so the user immediately sees how much capacity was
        # gained. Falls back to "?" if the field is not present.
        per_replica_added = []
        for idx, result in enumerate(commit_results):
            outputs = _unwrap_output_list(result)
            status = (
                outputs[0].get("status")
                if outputs and isinstance(outputs[0], dict)
                else None
            )
            added = (status or {}).get("added_kv_slots", "?")
            cap = (status or {}).get("max_total_num_tokens", "?")
            per_replica_added.append(f"r{idx}(added={added} max_total={cap})")
        logger.warning(
            "[KUNSERVE-MS] BALLOON entered: offload=%d retained=%d per_replica=[%s]",
            plan.offload_local_experts,
            plan.retained_local_experts,
            " ".join(per_replica_added),
        )
        self._emit(
            "entered balloon mode: offload_local_experts=%d retained_local_experts=%d"
            % (
                plan.offload_local_experts,
                plan.retained_local_experts,
            )
        )
        return await self._fetch_statuses()

    async def restore_balloon(self, *, require_idle: bool) -> list[dict[str, Any]]:
        self._emit(f"restoring balloon require_idle={require_idle}")
        await self._restore_replicas(
            list(range(len(self._replicas))), require_idle=require_idle
        )
        self._balloon_active = False
        self._emit("restored local runtime")
        return await self._fetch_statuses()

    async def _poll_loop(self) -> None:
        try:
            while self._running:
                try:
                    await self.tick()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("[KunServeController] poll iteration failed.")
                if self._running:
                    try:
                        if self._stop_event is None:
                            await asyncio.sleep(self.poll_interval)
                        else:
                            await asyncio.wait_for(
                                self._stop_event.wait(),
                                timeout=self.poll_interval,
                            )
                    except asyncio.TimeoutError:
                        pass
        finally:
            # When enable_restore is on, drain BALLOON and tear down the
            # cross-replica PG cleanly on shutdown. When it is off, the
            # rollout job is exiting anyway and the kernel will reclaim the
            # PG/sockets; touching them here only adds failure surface.
            if self.enable_restore:
                if self._balloon_active:
                    try:
                        logger.info(
                            "[KunServeController] poll loop exiting while balloon is active, restoring replicas first"
                        )
                        await self._restore_replicas(
                            list(range(len(self._replicas))),
                            require_idle=False,
                        )
                        self._balloon_active = False
                    except Exception:
                        logger.exception(
                            "[KunServeController] failed to restore replicas during shutdown."
                        )
                await self._destroy_process_group()

    async def _fetch_statuses(self) -> list[dict[str, Any]]:
        raw_statuses = await asyncio.gather(
            *[replica.get_balloon_status() for replica in self._replicas]
        )
        return [_unwrap_status_response(raw) for raw in raw_statuses]

    def _ensure_layout_plan(
        self, statuses: Sequence[dict[str, Any]]
    ) -> KunServeLayoutPlan:
        if self._layout_plan is None:
            self._layout_plan = build_layout_plan_from_statuses(
                statuses,
                offload_local_experts=self.offload_local_experts,
                num_replicas=len(self._replicas),
            )
        return self._layout_plan

    async def _ensure_process_group(self, plan: KunServeLayoutPlan) -> None:
        if self._process_group_initialized:
            return
        master_address = self._replicas[0].host

        # Each retry uses a fresh master_port AND a fresh group_name suffix.
        # Background:
        #  - get_free_port() closes the probe socket before the HTTP RPC
        #    reaches the replica, so the port can be stolen in between
        #    (TOCTOU). When that happens we get EADDRINUSE on the rank-0 side.
        #  - torch.distributed registers groups by name globally; a partially
        #    failed init may leave stale state under the same name, so reusing
        #    the old name on retry tends to fail the same way.
        # Reusing a fresh (port, group_name) pair sidesteps both.
        last_errors: list[str] = []
        attempts = max(1, self._pg_init_max_attempts)
        for attempt in range(attempts):
            try:
                # Prefer kernel-reported free port; fall back to a wide random
                # range when the address cannot be bound from this process.
                master_port, _ = get_free_port(master_address)
            except OSError:
                master_port = random.randint(40000, 60000)
            attempt_group_name = (
                f"{self._base_group_name}_v{int(time.time())}_{attempt}"
            )
            logger.info(
                "[KunServeController] init process group attempt=%d/%d: "
                "master=%s:%d world=%d group=%s backend=%s",
                attempt + 1,
                attempts,
                master_address,
                master_port,
                plan.global_world_size,
                attempt_group_name,
                self.backend,
            )
            init_results = await asyncio.gather(
                *[
                    replica.init_weights_update_group(
                        {
                            "master_address": master_address,
                            "master_port": master_port,
                            "rank_offset": replica_idx * plan.local_ep_size,
                            "world_size": plan.global_world_size,
                            "group_name": attempt_group_name,
                            "backend": self.backend,
                        }
                    )
                    for replica_idx, replica in enumerate(self._replicas)
                ],
                return_exceptions=True,
            )
            attempt_errors = []
            ok = True
            for idx, result in enumerate(init_results):
                if isinstance(result, Exception):
                    ok = False
                    attempt_errors.append(f"{self._replicas[idx].name}: {result!r}")
                    continue
                if not bool(result.get("success")):
                    ok = False
                    attempt_errors.append(
                        f"{self._replicas[idx].name}: "
                        f"{result.get('message', 'unknown error')}"
                    )
            if ok:
                self.group_name = attempt_group_name
                self._process_group_initialized = True
                logger.warning(
                    "[KUNSERVE-MS] PG ready group=%s master=%s:%d world=%d backend=%s attempts=%d",
                    attempt_group_name,
                    master_address,
                    master_port,
                    plan.global_world_size,
                    self.backend,
                    attempt + 1,
                )
                # Phase F: also build lane subgroups for the sglang
                # backend.  These are NCCL groups of size num_replicas
                # whose members are [replica0_tp_rank=L, replica1_tp_rank=L]
                # for each L in 0..local_ep_size-1.  By default failure is
                # fatal because silently falling back to the global group makes
                # perf traces look valid while they are not using the Phase F
                # lane path.  Set KUNSERVE_ALLOW_GLOBAL_GROUP_FALLBACK=1 only
                # for explicit fallback debugging.
                if self.runtime_backend.comm_backend == "sglang" and not (
                    os.environ.get("KUNSERVE_DISABLE_LANE_SUBGROUPS", "")
                    in ("1", "true", "True", "yes")
                ):
                    try:
                        await self._ensure_lane_subgroups(
                            plan=plan,
                            master_address=master_address,
                            base_suffix=attempt_group_name,
                        )
                    except Exception as exc:
                        if os.environ.get(
                            "KUNSERVE_ALLOW_GLOBAL_GROUP_FALLBACK", ""
                        ) in ("1", "true", "True", "yes"):
                            logger.warning(
                                "[KUNSERVE-MS] Phase F lane subgroup init failed: "
                                "%r -- falling back to global-group dispatch/combine",
                                exc,
                            )
                            self.lane_group_names = {}
                        else:
                            logger.error(
                                "[KUNSERVE-MS] Phase F lane subgroup init failed: "
                                "%r -- aborting global PG bring-up. Set "
                                "KUNSERVE_ALLOW_GLOBAL_GROUP_FALLBACK=1 to use "
                                "the older global-group fallback path.",
                                exc,
                            )
                            await self._destroy_process_group(force=True)
                            raise
                return

            last_errors = attempt_errors
            logger.warning(
                "[KunServeController] init process group attempt %d/%d failed: %s",
                attempt + 1,
                attempts,
                "; ".join(attempt_errors),
            )
            # Best-effort destroy on whichever side may have half-init'd, so
            # the next retry doesn't trip the idempotent shortcut on success.
            try:
                await asyncio.gather(
                    *[
                        replica.destroy_weights_update_group(attempt_group_name)
                        for replica in self._replicas
                    ],
                    return_exceptions=True,
                )
            except Exception:
                logger.exception(
                    "[KunServeController] best-effort destroy after failed init raised"
                )
            if attempt < attempts - 1:
                await asyncio.sleep(self._pg_init_retry_delay)

        raise RuntimeError(
            "init_weights_update_group failed after "
            f"{attempts} attempts: {'; '.join(last_errors)}"
        )

    async def _ensure_lane_subgroups(
        self,
        *,
        plan: KunServeLayoutPlan,
        master_address: str,
        base_suffix: str,
    ) -> None:
        """Phase F: initialize lane subgroups on top of the global group.

        Each lane L is a 2-replica NCCL group of size ``num_replicas``
        whose members are the tp_rank=L workers from every replica:

            lane 0 = [replica0.tp_rank=0, replica1.tp_rank=0] = [rank0, rank2]
            lane 1 = [replica0.tp_rank=1, replica1.tp_rank=1] = [rank1, rank3]

        The HTTP RPC is sent to every TP worker on every replica, but
        the ``lane_only_tp_rank=L`` argument makes only the matching
        workers actually rendezvous.  Non-participating workers record a
        ``None`` sentinel so the dispatcher can detect them.

        On success self.lane_group_names is populated and downstream
        warmup_balloon payloads carry the names so the model_runner
        resolves them and passes the right lane handle to the
        CrossReplicaStandardDispatcher.
        """
        local_ep_size = int(plan.local_ep_size)
        num_replicas = len(self._replicas)
        if num_replicas != plan.global_world_size // local_ep_size:
            raise RuntimeError(
                f"Phase F lane init: world_size mismatch "
                f"global={plan.global_world_size} local_ep={local_ep_size} "
                f"num_replicas={num_replicas}"
            )
        new_lane_names: dict[int, str] = {}
        attempts = max(1, self._pg_init_max_attempts)
        try:
            for lane_idx in range(local_ep_size):
                last_failures: list[str] = []
                for attempt in range(attempts):
                    # Allocate a fresh port and group name per retry.  Lane
                    # subgroup init has the same TOCTOU risk as the global
                    # PG, and stale partial groups must not be reused.
                    try:
                        lane_port, _ = get_free_port(master_address)
                    except OSError:
                        lane_port = random.randint(40000, 60000)
                    lane_group_name = (
                        f"kunserve_lane{lane_idx}_{base_suffix}_a{attempt}"
                    )
                    logger.info(
                        "[KunServeController] init lane subgroup lane=%d "
                        "attempt=%d/%d: master=%s:%d world=%d group=%s "
                        "backend=%s",
                        lane_idx,
                        attempt + 1,
                        attempts,
                        master_address,
                        lane_port,
                        num_replicas,
                        lane_group_name,
                        self.backend,
                    )
                    init_results = await asyncio.gather(
                        *[
                            replica.init_weights_update_group(
                                {
                                    "master_address": master_address,
                                    "master_port": lane_port,
                                    # rank_offset is unused for lane init
                                    # because explicit_group_rank is set.
                                    "rank_offset": 0,
                                    "world_size": num_replicas,
                                    "group_name": lane_group_name,
                                    "backend": self.backend,
                                    # Only the matching tp_rank actually
                                    # rendezvouses on this lane.
                                    "lane_only_tp_rank": int(lane_idx),
                                    # The participating worker's rank inside
                                    # the 2-rank lane subgroup equals its
                                    # replica index.
                                    "explicit_group_rank": int(replica_idx),
                                }
                            )
                            for replica_idx, replica in enumerate(self._replicas)
                        ],
                        return_exceptions=True,
                    )
                    failures: list[str] = []
                    for idx, result in enumerate(init_results):
                        if isinstance(result, Exception):
                            failures.append(
                                f"{self._replicas[idx].name}: {result!r}"
                            )
                            continue
                        message = str(result.get("message", "unknown error"))
                        if not bool(result.get("success")):
                            failures.append(
                                f"{self._replicas[idx].name}: {message}"
                            )
                        elif "Skipped non-participating" in message:
                            failures.append(
                                f"{self._replicas[idx].name}: lane {lane_idx} "
                                f"returned a non-participating TP result: {message}"
                            )
                    if not failures:
                        new_lane_names[lane_idx] = lane_group_name
                        logger.warning(
                            "[KUNSERVE-MS] Phase F lane subgroup ready lane=%d "
                            "group=%s world=%d master=%s:%d attempts=%d",
                            lane_idx,
                            lane_group_name,
                            num_replicas,
                            master_address,
                            lane_port,
                            attempt + 1,
                        )
                        break

                    last_failures = failures
                    logger.warning(
                        "[KUNSERVE-MS] Phase F lane subgroup lane=%d "
                        "attempt=%d/%d failed: %s",
                        lane_idx,
                        attempt + 1,
                        attempts,
                        "; ".join(failures),
                    )
                    await asyncio.gather(
                        *[
                            replica.destroy_weights_update_group(lane_group_name)
                            for replica in self._replicas
                        ],
                        return_exceptions=True,
                    )
                    if attempt < attempts - 1:
                        await asyncio.sleep(self._pg_init_retry_delay)
                else:
                    raise RuntimeError(
                        f"lane {lane_idx} subgroup init failed after "
                        f"{attempts} attempts: {'; '.join(last_failures)}"
                    )
        except Exception:
            if new_lane_names:
                await asyncio.gather(
                    *[
                        replica.destroy_weights_update_group(lane_group_name)
                        for replica in self._replicas
                        for lane_group_name in new_lane_names.values()
                    ],
                    return_exceptions=True,
                )
            raise
        self.lane_group_names = new_lane_names

    async def _destroy_process_group(self, *, force: bool = False) -> None:
        if not self._process_group_initialized and not force:
            return
        logger.info(
            "[KunServeController] destroying process group force=%s group=%s",
            force,
            self.group_name,
        )
        group_names = [self.group_name]
        group_names.extend(
            lane_name
            for lane_name in self.lane_group_names.values()
            if lane_name not in group_names
        )
        await asyncio.gather(
            *[
                replica.destroy_weights_update_group(group_name)
                for replica in self._replicas
                for group_name in group_names
            ],
            return_exceptions=True,
        )
        self._process_group_initialized = False
        self.lane_group_names = {}
        logger.info("[KunServeController] process group destroyed.")

    def _should_enter_balloon(
        self, statuses: Sequence[dict[str, Any]]
    ) -> tuple[bool, str]:
        if not any(bool(status.get("expand_requested")) for status in statuses):
            return False, "no replica requested expansion"
        if any(status.get("state") not in ("local", "prepared") for status in statuses):
            return False, "at least one replica is not in local/prepared state"
        if any(status.get("moe_weight_vmm_enabled") is False for status in statuses):
            return (
                False,
                "MoE weight VMM is unavailable on at least one replica; balloon expert offload is unsupported for this runtime",
            )
        running_ok = all(
            int(status.get("num_running_requests", 0))
            >= self.min_running_requests_per_replica
            for status in statuses
        )
        if not running_ok:
            return (
                False,
                f"running requests below threshold {self.min_running_requests_per_replica}",
            )
        return True, "expand_requested and running thresholds satisfied"

    def _should_restore_balloon(
        self, statuses: Sequence[dict[str, Any]]
    ) -> tuple[bool, str]:
        should_restore = all(
            int(status.get("num_running_requests", 0)) == 0
            and int(status.get("num_waiting_requests", 0)) == 0
            for status in statuses
        )
        if should_restore:
            return True, "all replicas are drained"
        return False, "at least one replica still has running or waiting requests"

    async def _rollback_prepared_replicas(
        self, statuses: Sequence[dict[str, Any]]
    ) -> None:
        prepared_indices = [
            idx
            for idx, status in enumerate(statuses)
            if status.get("state") == "prepared"
        ]
        if prepared_indices:
            logger.warning(
                "[KunServeController] rolling back prepared replicas=%s",
                list(prepared_indices),
            )
            await self._restore_replicas(prepared_indices, require_idle=False)

    async def _restore_replicas(
        self, indices: Sequence[int], *, require_idle: bool
    ) -> None:
        if not indices:
            return
        logger.info(
            "[KunServeController] restoring replicas=%s require_idle=%s",
            list(indices),
            require_idle,
        )
        results = await asyncio.gather(
            *[
                self._replicas[idx].restore_from_balloon({"require_idle": require_idle})
                for idx in indices
            ],
            return_exceptions=True,
        )
        failures = []
        for idx, result in zip(indices, results, strict=True):
            if isinstance(result, Exception):
                failures.append(f"{self._replicas[idx].name}: {result!r}")
            elif not _response_succeeded(result):
                failures.append(
                    f"{self._replicas[idx].name}: {_response_error(result)}"
                )
        if failures:
            raise RuntimeError(f"restore_from_balloon failed: {'; '.join(failures)}")
