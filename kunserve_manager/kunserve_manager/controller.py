from __future__ import annotations

import asyncio
import json
import logging
import random
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence
from urllib.parse import urlsplit

import aiohttp

import socket


def get_free_port(host: str = "") -> tuple[int, socket.socket]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host or "", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port, sock


logger = logging.getLogger(__name__)


async def _read_async_response(resp: aiohttp.ClientResponse) -> dict[str, Any]:
    if resp.status == 204 or resp.content_length == 0:
        return {}

    try:
        return await resp.json(content_type=None)
    except Exception:
        try:
            text = await resp.text()
        except Exception:
            return {}
        return {
            "content_type": resp.headers.get("Content-Type", ""),
            "text": text,
        }


class KunServeReplicaClient(Protocol):
    name: str
    host: str

    async def get_balloon_status(self) -> dict[str, Any]: ...

    async def prepare_balloon(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def warmup_balloon(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def commit_balloon(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def restore_from_balloon(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def init_weights_update_group(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]: ...

    async def destroy_weights_update_group(self, group_name: str) -> dict[str, Any]: ...


def _parse_server_address(server_address: str) -> tuple[str, int]:
    parts = urlsplit(f"http://{server_address}")
    if parts.hostname is None or parts.port is None:
        raise ValueError(f"Invalid server address: {server_address}")
    return parts.hostname, parts.port


def _unwrap_status_response(raw: Any) -> dict[str, Any]:
    if isinstance(raw, list):
        if not raw:
            raise ValueError("Empty balloon status response.")
        return dict(raw[0])
    if isinstance(raw, dict):
        return dict(raw)
    raise TypeError(f"Unsupported balloon status response type: {type(raw)!r}")


def _unwrap_output_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [dict(item) for item in raw]
    if isinstance(raw, dict):
        return [dict(raw)]
    raise TypeError(f"Unsupported RPC response type: {type(raw)!r}")


def _response_succeeded(raw: Any) -> bool:
    outputs = _unwrap_output_list(raw)
    return bool(outputs) and all(bool(item.get("success")) for item in outputs)


def _response_error(raw: Any) -> str:
    outputs = _unwrap_output_list(raw)
    for item in outputs:
        if not item.get("success", False):
            return str(item.get("message", "unknown error"))
    return "unknown error"


def _status_matches_balloon_target(
    raw: Any, *, target_variant: str, offload_local_experts: int
) -> bool:
    try:
        status = _unwrap_status_response(raw)
    except (TypeError, ValueError):
        return False

    return (
        str(status.get("state", "")).lower() == "balloon"
        and str(status.get("runtime_variant", "")).lower()
        == str(target_variant).lower()
        and int(status.get("offloaded_local_experts", -1)) == int(offload_local_experts)
    )


def _commit_response_succeeded(
    raw: Any, *, target_variant: str, offload_local_experts: int
) -> bool:
    if _response_succeeded(raw):
        return True

    outputs = _unwrap_output_list(raw)
    return bool(outputs) and all(
        _status_matches_balloon_target(
            item.get("status"),
            target_variant=target_variant,
            offload_local_experts=offload_local_experts,
        )
        for item in outputs
    )


def build_complementary_physical_to_logical_map(
    base_physical_to_logical_map: Sequence[Sequence[int]],
    *,
    local_ep_size: int,
    retained_local_experts: int,
) -> list[list[int]]:
    if local_ep_size <= 0:
        raise ValueError(f"local_ep_size must be positive, got {local_ep_size}")
    if retained_local_experts <= 0:
        raise ValueError(
            f"retained_local_experts must be positive, got {retained_local_experts}"
        )

    normalized = [list(map(int, row)) for row in base_physical_to_logical_map]
    if not normalized:
        raise ValueError("base_physical_to_logical_map must be non-empty")

    num_physical_experts = len(normalized[0])
    if any(len(row) != num_physical_experts for row in normalized):
        raise ValueError("All physical_to_logical rows must have the same length.")
    if num_physical_experts % local_ep_size != 0:
        raise ValueError(
            f"num_physical_experts={num_physical_experts} is not divisible by local_ep_size={local_ep_size}"
        )

    local_chunk = num_physical_experts // local_ep_size
    if retained_local_experts * 2 != local_chunk:
        raise ValueError(
            "Complementary two-replica balloon requires a symmetric half split per local rank."
        )

    merged: list[list[int]] = []
    for layer_row in normalized:
        global_row: list[int] = []
        for local_rank in range(local_ep_size):
            base = local_rank * local_chunk
            global_row.extend(layer_row[base : base + retained_local_experts])
        for local_rank in range(local_ep_size):
            base = local_rank * local_chunk + (local_chunk - retained_local_experts)
            global_row.extend(layer_row[base : base + retained_local_experts])
        if len(global_row) != num_physical_experts:
            raise ValueError(
                f"Expected merged row length {num_physical_experts}, got {len(global_row)}"
            )
        merged.append(global_row)
    return merged


@dataclass
class KunServeLayoutPlan:
    local_ep_size: int
    local_routed_experts: int
    retained_local_experts: int
    offload_local_experts: int
    global_world_size: int
    global_physical_to_logical_map: list[list[int]]
    replica_active_mappings: list[list[int]]


@dataclass
class KunServeHttpReplicaClient:
    name: str
    host: str
    port: int
    model_path: str
    timeout: float = 60.0
    # warmup_balloon does GLOBAL cuda graph capture which empirically takes
    # 5-10 minutes on H20 (deepgemm precompile + LL Buffer creation +
    # 35 batch sizes). The default 60s × 3 attempts (180s) is way too short —
    # in ab_20260506_172822 it triggered a false-positive "warmup failed"
    # while the capture was actually mid-flight. Override that one RPC.
    warmup_timeout: float = 600.0
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_start_wait_time: float = 300.0
    max_connections: int = 64

    def __post_init__(self) -> None:
        # Control-plane clients only talk to already-running HTTP servers.
        # They must not instantiate SGLang ServerArgs here because the controller
        # runs in a non-GPU Ray actor process where accelerator probing fails.
        self._base_url = f"http://{self.host}:{self.port}"
        logger.info(
            "[KunServeHttpReplicaClient] configured control-plane HTTP client for %s at %s",
            self.name,
            self._base_url,
        )
        print(
            f"[KunServeHttpReplicaClient:{self.name}] configured at {self._base_url}",
            flush=True,
        )

    @asynccontextmanager
    async def _get_session(self, timeout: Optional[float] = None):
        connector = aiohttp.TCPConnector(
            limit=max(1, self.max_connections),
            limit_per_host=max(1, self.max_connections // 4),
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        effective_timeout = self.timeout if timeout is None else float(timeout)
        client_timeout = aiohttp.ClientTimeout(total=effective_timeout)
        session = aiohttp.ClientSession(connector=connector, timeout=client_timeout)
        try:
            yield session
        finally:
            if not session.closed:
                await session.close()

    async def _request(
        self,
        endpoint: str,
        payload: Optional[dict[str, Any]] = None,
        *,
        method: str = "POST",
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}/{endpoint}"
        should_trace = endpoint != "kunserve/status"
        if should_trace:
            print(
                f"[KunServeHttpReplicaClient:{self.name}] {method.upper()} {endpoint} payload={payload or {}}",
                flush=True,
            )

        for attempt in range(self.max_attempts):
            try:
                async with self._get_session(timeout=timeout) as session:
                    if method.upper() == "GET":
                        async with session.get(url) as response:
                            response.raise_for_status()
                            result = await _read_async_response(response)
                            if should_trace:
                                print(
                                    f"[KunServeHttpReplicaClient:{self.name}] {endpoint} response={result}",
                                    flush=True,
                                )
                            return result
                    async with session.post(url, json=payload or {}) as response:
                        response.raise_for_status()
                        result = await _read_async_response(response)
                        if should_trace:
                            print(
                                f"[KunServeHttpReplicaClient:{self.name}] {endpoint} response={result}",
                                flush=True,
                            )
                        return result
            except asyncio.TimeoutError:
                logger.warning(
                    "[KunServeHttpReplicaClient] %s %s timed out (%d/%d)",
                    self.name,
                    endpoint,
                    attempt + 1,
                    self.max_attempts,
                )
            except aiohttp.ClientConnectorError:
                logger.warning(
                    "[KunServeHttpReplicaClient] %s %s connection error (%d/%d)",
                    self.name,
                    endpoint,
                    attempt + 1,
                    self.max_attempts,
                )
            except aiohttp.ClientResponseError as exc:
                logger.error(
                    "[KunServeHttpReplicaClient] %s %s HTTP error: %s",
                    self.name,
                    endpoint,
                    exc,
                )
                raise
            except Exception as exc:
                logger.error(
                    "[KunServeHttpReplicaClient] %s %s unexpected error: %s",
                    self.name,
                    endpoint,
                    exc,
                )
                if attempt == self.max_attempts - 1:
                    raise

            if attempt < self.max_attempts - 1:
                await asyncio.sleep(self.retry_delay * (2**attempt))

        raise RuntimeError(
            f"[KunServeHttpReplicaClient] {self.name} failed to call {endpoint} "
            f"after {self.max_attempts} attempts"
        )

    async def get_balloon_status(self) -> dict[str, Any]:
        return await self._request("kunserve/status", method="GET")

    async def prepare_balloon(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("kunserve/prepare_balloon", payload)

    async def warmup_balloon(self, payload: dict[str, Any]) -> dict[str, Any]:
        # warmup_balloon includes GLOBAL cuda graph capture which can take
        # 5-10 minutes on H20. Use the dedicated warmup_timeout (default 600s)
        # instead of the per-RPC default (60s) to avoid spuriously aborting
        # an in-flight capture.
        return await self._request(
            "kunserve/warmup_balloon", payload, timeout=self.warmup_timeout
        )

    async def commit_balloon(self, payload: dict[str, Any]) -> dict[str, Any]:
        # commit_balloon does borrow_tail (per-layer cuMemUnmap) + KV expand
        # (cuMemMap into the KV region for hundreds of donor segments). On
        # 4-rank kunserve at 32 offloaded experts × 48 layers this measured
        # ~67 s end-to-end (see ab_20260429_155927). Reuse warmup_timeout so
        # we don't trip the 60 s per-RPC default mid-mapping.
        return await self._request(
            "kunserve/commit_balloon", payload, timeout=self.warmup_timeout
        )

    async def restore_from_balloon(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("kunserve/restore_from_balloon", payload)

    async def init_weights_update_group(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._request("init_weights_update_group", payload)

    async def destroy_weights_update_group(self, group_name: str) -> dict[str, Any]:
        return await self._request(
            "destroy_weights_update_group", {"group_name": group_name}
        )


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
        enable_restore: bool = False,
        pg_init_max_attempts: int = 8,
        pg_init_retry_delay: float = 1.0,
        eager_warmup: bool = True,
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
        self.backend = backend
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

    def _emit(self, message: str) -> None:
        print(f"[KunServeController] {message}", flush=True)
        logger.warning("[KunServeController] %s", message)

    def _summarize_statuses(self, statuses: Sequence[dict[str, Any]]) -> str:
        parts = []
        for idx, status in enumerate(statuses):
            parts.append(
                "r%d(state=%s variant=%s expand=%s running=%s waiting=%s offloaded=%s slots=%s)"
                % (
                    idx,
                    status.get("state"),
                    status.get("runtime_variant"),
                    status.get("expand_requested"),
                    status.get("num_running_requests"),
                    status.get("num_waiting_requests"),
                    status.get("offloaded_local_experts"),
                    status.get("added_kv_slots"),
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
            "starting: replicas=%d poll_interval=%.2fs min_running=%d group=%s backend=%s enable_restore=%s"
            % (
                len(self._replicas),
                self.poll_interval,
                self.min_running_requests_per_replica,
                self.group_name,
                self.backend,
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
            with self._bw_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("failed to write bw status sample", exc_info=True)

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
                    "This is the current high-risk condition for cross-replica MoE collectives "
                    "because there is not yet a dummy-participation tick.",
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
                    "capture_cuda_graph": bool(capture_cuda_graph),
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
        self._emit(
            "warmup balloon: world=%d retained=%d offload=%d (capturing GLOBAL graph in parallel)"
            % (
                plan.global_world_size,
                plan.retained_local_experts,
                plan.offload_local_experts,
            )
        )
        logger.warning(
            "[KUNSERVE-MS] WARMUP dispatch warmup_balloon: world=%d retained=%d offload=%d",
            plan.global_world_size,
            plan.retained_local_experts,
            plan.offload_local_experts,
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
            "[KUNSERVE-MS] WARMUP done: replicas=%d (GLOBAL graph cached, state stays LOCAL)",
            len(self._replicas),
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

        self._emit(
            "preparing balloon: offload=%d retained=%d world=%d payloads=%s"
            % (
                plan.offload_local_experts,
                plan.retained_local_experts,
                plan.global_world_size,
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
            "[KUNSERVE-MS] BALLOON dispatch prepare_balloon: offload=%d retained=%d world=%d",
            plan.offload_local_experts,
            plan.retained_local_experts,
            plan.global_world_size,
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
        if self._layout_plan is not None:
            return self._layout_plan

        local_maps = [
            status.get("local_physical_to_logical_map") for status in statuses
        ]
        if any(local_map is None for local_map in local_maps):
            raise ValueError("Balloon status is missing local_physical_to_logical_map.")
        if local_maps[0] != local_maps[1]:
            raise ValueError(
                "Replicas do not agree on the baseline physical_to_logical expert layout."
            )

        local_ep_size = int(statuses[0]["local_ep_size"])
        routed_by_layer = {
            int(layer_id): int(count)
            for layer_id, count in statuses[0]["local_routed_experts_per_layer"].items()
        }
        routed_values = set(routed_by_layer.values())
        if len(routed_values) != 1:
            raise ValueError(
                "Current KunServe controller requires all MoE layers to expose the same local routed expert count."
            )
        local_routed_experts = routed_values.pop()

        offload_local_experts = (
            self.offload_local_experts
            if self.offload_local_experts is not None
            else local_routed_experts // 2
        )
        if offload_local_experts <= 0 or offload_local_experts >= local_routed_experts:
            raise ValueError(
                f"Invalid offload_local_experts={offload_local_experts} for local_routed_experts={local_routed_experts}"
            )
        retained_local_experts = local_routed_experts - offload_local_experts
        if retained_local_experts != offload_local_experts:
            raise ValueError(
                "Current KunServe controller requires a symmetric half split to preserve the original expert count."
            )

        global_map = build_complementary_physical_to_logical_map(
            local_maps[0],
            local_ep_size=local_ep_size,
            retained_local_experts=retained_local_experts,
        )
        replica_active_mappings = [
            list(range(retained_local_experts)),
            list(
                range(
                    local_routed_experts - retained_local_experts, local_routed_experts
                )
            ),
        ]
        self._layout_plan = KunServeLayoutPlan(
            local_ep_size=local_ep_size,
            local_routed_experts=local_routed_experts,
            retained_local_experts=retained_local_experts,
            offload_local_experts=offload_local_experts,
            global_world_size=local_ep_size * len(self._replicas),
            global_physical_to_logical_map=global_map,
            replica_active_mappings=replica_active_mappings,
        )
        logger.info(
            "[KunServeController] layout plan ready: local_ep_size=%d local_routed=%d retained=%d offload=%d",
            local_ep_size,
            local_routed_experts,
            retained_local_experts,
            offload_local_experts,
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

    async def _destroy_process_group(self, *, force: bool = False) -> None:
        if not self._process_group_initialized and not force:
            return
        logger.info(
            "[KunServeController] destroying process group force=%s group=%s",
            force,
            self.group_name,
        )
        await asyncio.gather(
            *[
                replica.destroy_weights_update_group(self.group_name)
                for replica in self._replicas
            ],
            return_exceptions=True,
        )
        self._process_group_initialized = False
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
