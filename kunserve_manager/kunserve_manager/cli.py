"""CLI entry for the kunserve sidecar manager.

This is a thin wrapper around :class:`KunServeController` that reads
arguments from argv (with env-var fallbacks for the deployer's
convenience) and runs the controller until the process is interrupted.

Design rules (do not regress these):

* No leader-election among replicas. The manager is a single, separately
  launched process. The deployer (verl, a CLI invocation, etc.) is
  responsible for ensuring exactly one manager exists per group.
* No shared discovery file. Replica addresses come in via ``--replica``.
* No sglang-internal imports. The manager only speaks HTTP to replicas.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from typing import Optional

from kunserve_manager.controller import KunServeController

logger = logging.getLogger("kunserve_manager")


def _env_or(name: str, default: Optional[str]) -> Optional[str]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw or default


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kunserve-manager",
        description=(
            "Standalone control plane for a group of KunServe-enabled sglang "
            "replicas. Polls /kunserve/status on each replica, drives "
            "LOCAL→BALLOON transitions, and tears them down at shutdown."
        ),
    )
    p.add_argument(
        "--replica",
        action="append",
        default=[],
        metavar="HOST:PORT",
        help=(
            "Replica HTTP address, repeatable. Exactly two are required for "
            "the current 2-replica balloon implementation. Falls back to "
            "$KUNSERVE_MANAGER_REPLICAS (comma-separated) if not given."
        ),
    )
    p.add_argument(
        "--model-path",
        default=_env_or("KUNSERVE_MANAGER_MODEL_PATH", None),
        help="Model path used to derive expert layout (also $KUNSERVE_MANAGER_MODEL_PATH).",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=float(_env_or("KUNSERVE_MANAGER_POLL_INTERVAL", "2.0") or "2.0"),
        help="Status poll interval (seconds, default 2.0).",
    )
    p.add_argument(
        "--min-running-requests-per-replica",
        type=int,
        default=int(
            _env_or("KUNSERVE_MANAGER_MIN_RUNNING_REQUESTS_PER_REPLICA", "1") or "1"
        ),
        help="Per-replica running threshold before BALLOON entry is allowed.",
    )
    p.add_argument(
        "--offload-local-experts",
        type=int,
        default=(
            int(_env_or("KUNSERVE_MANAGER_OFFLOAD_LOCAL_EXPERTS", "0") or "0") or None
        ),
        help="Number of local experts to offload at BALLOON commit (default: layout-derived).",
    )
    p.add_argument(
        "--group-name",
        default=_env_or("KUNSERVE_MANAGER_GROUP_NAME", "kunserve_global_ep"),
        help="NCCL/process group name shared across replicas.",
    )
    p.add_argument(
        "--backend",
        default=_env_or("KUNSERVE_MANAGER_BACKEND", "nccl"),
        help=(
            "init_weights_update_group backend (default: nccl; comm_backend=sglang "
            "maps the default to kunserve_pynccl)."
        ),
    )
    p.add_argument(
        "--comm-backend",
        choices=("deepep", "sglang"),
        default=_env_or("KUNSERVE_MANAGER_COMM_BACKEND", "deepep"),
        help=(
            "KunServe GLOBAL MoE communication backend. 'deepep' keeps the "
            "existing DeepEP dispatcher path; 'sglang' uses the "
            "correctness-first CrossReplicaStandardDispatcher (default: deepep)."
        ),
    )
    p.add_argument(
        "--capture-policy",
        choices=("auto", "fixed_padded", "disabled"),
        default=_env_or("KUNSERVE_MANAGER_CAPTURE_POLICY", "auto"),
        help=(
            "GLOBAL CUDA graph capture policy. For comm-backend=sglang this "
            "currently resolves to disabled because the initial dispatcher uses "
            "dynamic all-gather/all-reduce collectives."
        ),
    )
    p.add_argument(
        "--enable-restore",
        action="store_true",
        default=_env_or("KUNSERVE_MANAGER_ENABLE_RESTORE", "0") in ("1", "true"),
        help="Allow BALLOON→LOCAL when both replicas are idle (default off).",
    )
    p.add_argument(
        "--no-eager-warmup",
        dest="eager_warmup",
        action="store_false",
        default=_env_or("KUNSERVE_MANAGER_EAGER_WARMUP", "1") in ("1", "true"),
        help="Skip /kunserve/warmup_balloon at startup; rely on lazy capture.",
    )
    p.add_argument(
        "--log-level",
        default=_env_or("KUNSERVE_MANAGER_LOG_LEVEL", "INFO"),
        help="Python logging level (DEBUG/INFO/WARNING/...).",
    )
    p.add_argument(
        "--output-dir",
        default=_env_or(
            "KUNSERVE_MANAGER_OUTPUT_DIR",
            _env_or("SGLANG_KUNSERVE_OUTPUT_DIR", "/workspace/sglang/output"),
        ),
        help=(
            "Directory for manager-generated artifacts such as "
            "bw_throughput.jsonl (also $KUNSERVE_MANAGER_OUTPUT_DIR)."
        ),
    )
    p.add_argument(
        "--no-bw-log",
        dest="write_bw_log",
        action="store_false",
        default=_env_or("KUNSERVE_MANAGER_WRITE_BW_LOG", "1")
        in ("1", "true", "yes", "on"),
        help=(
            "Disable manager-generated bw_throughput.jsonl. Use this when an "
            "external scraper is already writing the same file."
        ),
    )
    return p


def _resolve_replicas(args: argparse.Namespace) -> list[str]:
    if args.replica:
        return list(args.replica)
    raw = _env_or("KUNSERVE_MANAGER_REPLICAS", "")
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return []


async def _run(controller: KunServeController) -> None:
    stop_event = asyncio.Event()

    def _signal_stop():
        if not stop_event.is_set():
            logger.info("kunserve_manager received shutdown signal")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _signal_stop)
        except (NotImplementedError, RuntimeError):
            # Windows / non-main-thread cases — fall back to the default
            # KeyboardInterrupt path.
            pass

    await controller.start()
    logger.info("kunserve_manager controller started; waiting for shutdown")
    try:
        await stop_event.wait()
    finally:
        try:
            await controller.stop()
        except Exception as exc:
            logger.warning("controller.stop raised: %r", exc)


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    replicas = _resolve_replicas(args)
    if len(replicas) != 2:
        parser.error(
            f"need exactly 2 --replica entries (or KUNSERVE_MANAGER_REPLICAS=a,b); got {len(replicas)}: {replicas}"
        )
    if not args.model_path:
        parser.error("--model-path is required (or KUNSERVE_MANAGER_MODEL_PATH)")

    controller = KunServeController.from_server_addresses(
        replicas,
        model_path=args.model_path,
        poll_interval=args.poll_interval,
        min_running_requests_per_replica=args.min_running_requests_per_replica,
        offload_local_experts=args.offload_local_experts,
        group_name=args.group_name,
        backend=args.backend,
        comm_backend=args.comm_backend,
        capture_policy=args.capture_policy,
        enable_restore=args.enable_restore,
        eager_warmup=args.eager_warmup,
        output_dir=args.output_dir,
        write_bw_log=args.write_bw_log,
    )

    logger.info(
        "kunserve_manager starting: replicas=%s model_path=%s poll=%.2fs "
        "group=%s backend=%s effective_backend=%s comm_backend=%s capture_policy=%s eager_warmup=%s "
        "enable_restore=%s output_dir=%s bw_log=%s",
        replicas,
        args.model_path,
        args.poll_interval,
        args.group_name,
        args.backend,
        controller.backend,
        args.comm_backend,
        args.capture_policy,
        args.eager_warmup,
        args.enable_restore,
        args.output_dir,
        args.write_bw_log,
    )
    try:
        asyncio.run(_run(controller))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
