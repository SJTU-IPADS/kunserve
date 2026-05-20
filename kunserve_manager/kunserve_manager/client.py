"""HTTP client and response helpers for KunServe replica control-plane RPCs."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional, Protocol
from urllib.parse import urlsplit

import aiohttp

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
