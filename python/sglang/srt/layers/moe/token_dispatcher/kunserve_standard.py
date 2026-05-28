from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import (
    kunserve_lane_reduce_scatter_then_tp_all_reduce,
)
from sglang.srt.kunserve_forward_timing import (
    kunserve_detailed_timing_enabled,
    kunserve_timing_scope,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopKOutput,
    TopKOutputChecker,
)

logger = logging.getLogger(__name__)


def _detail_scope(enabled: bool, event: str, **fields: Any):
    if enabled:
        return kunserve_timing_scope(event, **fields)
    return nullcontext()


def _group_world_size(group: Any) -> int:
    if hasattr(group, "world_size"):
        return int(group.world_size)
    return int(dist.get_world_size(group=group))


def _group_rank(group: Any) -> int:
    if hasattr(group, "rank"):
        return int(group.rank)
    return int(dist.get_rank(group=group))


def _all_gather_into_tensor(group: Any, output: torch.Tensor, input_: torch.Tensor):
    if hasattr(group, "all_gather_into_tensor"):
        group.all_gather_into_tensor(output, input_)
    else:
        dist.all_gather_into_tensor(output, input_, group=group)


def _all_gather_padded_tensor(group: Any, tensor: torch.Tensor) -> list[torch.Tensor]:
    world = _group_world_size(group)
    out = torch.empty(
        (world * int(tensor.shape[0]), *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    _all_gather_into_tensor(group, out, tensor.contiguous())
    return list(out.chunk(world, dim=0))


def _reduce_scatter_tensor(group: Any, output: torch.Tensor, input_: torch.Tensor):
    if hasattr(group, "reduce_scatter_tensor"):
        group.reduce_scatter_tensor(output, input_)
    else:
        dist.reduce_scatter_tensor(
            output,
            input_,
            op=dist.ReduceOp.SUM,
            group=group,
        )


def _all_reduce(group: Any, tensor: torch.Tensor):
    if hasattr(group, "all_reduce"):
        group.all_reduce(tensor)
    else:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)


def _capture_state_text() -> str:
    try:
        return "capturing" if torch.cuda.is_current_stream_capturing() else "eager"
    except Exception as exc:
        return f"capture_state_err={exc!r}"


def _tensor_meta(tensor: Optional[torch.Tensor]) -> str:
    if not isinstance(tensor, torch.Tensor):
        return "None"
    try:
        ptr = hex(int(tensor.data_ptr()))
    except Exception as exc:
        ptr = f"<ptr_err:{exc!r}>"
    try:
        contiguous = bool(tensor.is_contiguous())
    except Exception:
        contiguous = False
    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} ptr={ptr} contiguous={contiguous}"
    )


def _group_name(group: Any) -> str:
    if group is None:
        return "None"
    for attr in ("unique_name", "name"):
        value = getattr(group, attr, None)
        if value is not None:
            return str(value)
    return type(group).__name__


def _group_unique_name(group: Any) -> Optional[str]:
    value = getattr(group, "unique_name", None)
    return str(value) if value is not None else None


class CrossReplicaStandardDispatcher(BaseDispatcher):
    """KunServe GLOBAL dispatcher implemented with dense collectives.

    Two operating modes:

    * **Dynamic eager** (default, when ``capture_max_m`` is None or buffers
      have not been provisioned): all-gather sizes, pad to per-step max,
      dispatch/combine with dynamic shapes. Correctness-first, NOT
      CUDA-graph-safe. Used for prefill, mismatched batch sizes, and any path
      that does not go through cuda graph replay.  When the manager provides
      KunServe PyNccl groups, eager calls route through PyNccl directly while
      CUDA graph capture uses registered collectives; raw torch distributed
      remains only as a fallback.

    * **Fixed-padded static** (when ``capture_max_m`` is provided): the
      constructor pre-allocates static buffers sized for ``capture_max_m``
      rows per rank. During actual CUDA graph capture the dispatcher uses
      fixed-shape collectives and static slice copies over those buffers, no
      host syncs, no ``torch.empty_like`` inside the hot path. The fixed shapes
      are necessary for CUDA graph capture.  CUDA graph replay is only enabled
      when the active cross-replica groups expose SGLang registered/PyNccl
      collectives.

    The FusedMoE contract is preserved in both modes: ``dispatch`` takes
    ``hidden_states[local_m, H]`` for this replica and ``combine`` returns
    ``[local_m, H]``.

    Cross-replica shape invariant for the static path: at graph capture time
    every global rank must be capturing the same ``(variant, bs)`` graph,
    which guarantees that all replicas observe the same ``local_m``. At
    replay/runtime, the idle keepalive batch must be padded to the same
    captured ``bs`` so collectives stay in lockstep. The static path
    explicitly zeros the padding rows of the source buffer so any
    ``local_m <= capture_max_m`` is well-defined.
    """

    def __init__(
        self,
        *,
        group: Any,
        moe_runner_config: MoeRunnerConfig,
        local_expert_mapping: torch.Tensor,
        local_ep_size: int,
        replica_rank: Optional[int] = None,
        global_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        capture_max_m: Optional[int] = None,
        lane_group: Optional[Any] = None,
        local_tp_group: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if group is None:
            raise ValueError("CrossReplicaStandardDispatcher requires a process group.")
        if local_ep_size <= 0:
            raise ValueError(f"local_ep_size must be positive, got {local_ep_size}.")

        self.group = group
        self.local_ep_size = int(local_ep_size)
        self.global_rank = (
            int(global_rank)
            if global_rank is not None
            else _group_rank(group)
        )
        self.world_size = (
            int(world_size)
            if world_size is not None
            else _group_world_size(group)
        )
        if self.world_size % self.local_ep_size != 0:
            raise ValueError(
                "CrossReplicaStandardDispatcher requires global world size to be "
                f"divisible by local_ep_size, got world={self.world_size}, "
                f"local_ep_size={self.local_ep_size}."
            )

        self.num_replicas = self.world_size // self.local_ep_size
        self.replica_rank = (
            int(replica_rank)
            if replica_rank is not None
            else self.global_rank // self.local_ep_size
        )
        self.lane_rank = self.global_rank % self.local_ep_size

        # Phase F: optional lane subgroup (cross-replica 2-rank group
        # containing only the workers in this lane) and the local TP
        # group (intra-replica 2-rank group).  When both are available
        # dispatch uses lane_group.all_gather_into_tensor to avoid the
        # ``[A, A, B, B]`` redundancy of the global group, and combine
        # uses lane_group.reduce_scatter_tensor + the model layer's normal
        # local TP all_reduce
        # instead of a global all_reduce so each rank only receives the
        # union slice it actually needs.  When either is None the
        # dispatcher transparently falls back to the Phase D path on
        # the global runtime_group.
        self.lane_group = lane_group
        self.local_tp_group = local_tp_group
        self.phase_f_enabled = (
            lane_group is not None and local_tp_group is not None
        )

        self.num_experts = int(moe_runner_config.num_experts)
        self.top_k = int(moe_runner_config.top_k)
        self.num_local_experts = int(moe_runner_config.num_local_experts)
        self.hidden_size = (
            int(moe_runner_config.hidden_size)
            if moe_runner_config.hidden_size is not None
            else None
        )
        self.params_dtype = moe_runner_config.params_dtype
        self.local_expert_mapping = self._normalize_local_expert_mapping(
            local_expert_mapping
        )
        self.active_local_expert_mapping = self.local_expert_mapping

        # State saved by dispatch and consumed by the matching combine.  A
        # FusedMoE layer calls dispatch -> run_moe_core -> combine
        # synchronously, so a single in-flight state per dispatcher is enough.
        self._last_local_m: Optional[int] = None
        self._last_max_m: Optional[int] = None
        self._last_slice_start: Optional[int] = None
        self._logged_shape: bool = False
        # KUNSERVE-DBG probe: per-dispatcher call counter so we can sample
        # numerical stats at logarithmic intervals (call 1, 5, 20, ...)
        # without spamming the log.  Enabled by env var KUNSERVE_DISPATCH_PROBE=1.
        # IMPORTANT: scheduler subprocess's logger.warning is invisible from
        # the Ray driver capture; we must append to KUNSERVE_DETAIL_LOG file
        # directly (same trick as _kunserve_ms in model_runner.py).
        import os as _os
        self._probe_enabled: bool = _os.environ.get(
            "KUNSERVE_DISPATCH_PROBE", ""
        ) in ("1", "true", "True", "yes")
        self._probe_detail_log_path: Optional[str] = _os.environ.get(
            "KUNSERVE_DETAIL_LOG"
        )
        self._dispatch_call_count: int = 0
        self._combine_call_count: int = 0
        self._static_dispatch_logged: bool = False
        self._static_after_gather_logged: bool = False
        self._static_combine_logged: bool = False
        self._static_combine_mode_logged: bool = False
        self._static_mapping_mismatch_logged: bool = False
        self._probe_milestones = {1, 5, 20, 100, 500, 2000}
        static_combine_mode = str(
            _os.environ.get("KUNSERVE_STATIC_COMBINE_MODE", "reduce_scatter")
        ).strip().lower()
        enable_composite_fusion = _os.environ.get(
            "KUNSERVE_STATIC_COMBINE_TP_ALLREDUCE_FUSION", ""
        ) in ("1", "true", "True", "yes", "on")
        if enable_composite_fusion and static_combine_mode == "reduce_scatter":
            static_combine_mode = "reduce_scatter_tp_all_reduce"
        if static_combine_mode not in (
            "reduce_scatter",
            "reduce_scatter_tp_all_reduce",
            "all_reduce",
        ):
            logger.warning(
                "Invalid KUNSERVE_STATIC_COMBINE_MODE=%r; using reduce_scatter.",
                static_combine_mode,
            )
            static_combine_mode = "reduce_scatter"
        self._static_combine_mode = static_combine_mode
        self._allow_static_tp_allreduce_fusion: bool = False
        # Phase G P1: dispatch ncclGroup batching.  When enabled, the 3
        # all-gather calls (hidden / topk_ids / topk_weights) are fused
        # into a single ncclGroupStart/End bracket, saving 2 NCCL kernel
        # launches per layer.  Default on; set to 0 to fall back to the
        # 3-separate-launches path for A/B comparison.
        self._dispatch_ncclgroup_enabled: bool = _os.environ.get(
            "KUNSERVE_DISPATCH_NCCL_GROUP", "1"
        ) in ("1", "true", "True", "yes", "on")
        self._dispatch_ncclgroup_logged: bool = False
        # One-shot init diagnostic so we can confirm env-var propagation
        # to the scheduler subprocess from the file content.
        self._probe_log(
            f"probe_init enabled={self._probe_enabled} "
            f"detail_log={self._probe_detail_log_path!r} "
            f"rank={self.global_rank} replica={self.replica_rank} "
            f"lane={self.lane_rank} phase_f={self.phase_f_enabled} "
            f"dispatch_ncclgroup={self._dispatch_ncclgroup_enabled}"
        )

        # Static buffers for the fixed-padded capture path.  We pre-allocate
        # in the constructor (default cuda pool, not the cuda graph private
        # pool) so the addresses are stable across multiple
        # (variant, batch_size) graph captures.
        self._capture_max_m: Optional[int] = (
            int(capture_max_m) if capture_max_m else None
        )
        self._static_buffers_ready: bool = False
        self._buf_padded_hidden: Optional[torch.Tensor] = None
        self._buf_padded_topk_ids: Optional[torch.Tensor] = None
        self._buf_padded_topk_weights: Optional[torch.Tensor] = None
        self._buf_gathered_hidden: Optional[torch.Tensor] = None
        self._buf_gathered_topk_ids: Optional[torch.Tensor] = None
        self._buf_gathered_topk_weights: Optional[torch.Tensor] = None
        self._buf_union_hidden: Optional[torch.Tensor] = None
        self._buf_union_topk_ids: Optional[torch.Tensor] = None
        self._buf_union_topk_weights: Optional[torch.Tensor] = None
        # Separate output buffer for the remapped topk_ids so we never have
        # to reassign self._buf_union_topk_ids (which would break the next
        # graph capture by replacing the recorded buffer pointer).
        self._buf_union_topk_ids_remapped: Optional[torch.Tensor] = None
        self._neg_one_int32: Optional[torch.Tensor] = None

        if self._capture_max_m is not None and self.hidden_size is not None:
            self._allocate_static_buffers()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _normalize_local_expert_mapping(self, mapping: torch.Tensor) -> torch.Tensor:
        if not isinstance(mapping, torch.Tensor):
            mapping = torch.tensor(mapping)
        if mapping.dim() != 1:
            raise ValueError("local_expert_mapping must be a 1D tensor.")
        if int(mapping.numel()) != self.num_experts:
            raise ValueError(
                "local_expert_mapping must contain one entry per dispatch-domain "
                f"expert, got {int(mapping.numel())} vs {self.num_experts}."
            )
        return mapping.to(dtype=torch.int32)

    def _mapping_on(self, device: torch.device) -> torch.Tensor:
        if self.local_expert_mapping.device != device:
            old_device = self.local_expert_mapping.device
            self.local_expert_mapping = self.local_expert_mapping.to(
                device=device, non_blocking=True
            )
            self.active_local_expert_mapping = self.local_expert_mapping
            self._probe_log(
                f"mapping_moved rank={self.global_rank} replica={self.replica_rank} "
                f"lane={self.lane_rank} old_device={old_device} "
                f"new_device={self.local_expert_mapping.device} "
                f"mapping={_tensor_meta(self.local_expert_mapping)}"
            )
        return self.local_expert_mapping

    def _allocate_static_buffers(self) -> None:
        """Allocate persistent buffers for the fixed-padded capture path.

        Allocated eagerly in the constructor so the buffers live in the
        default caching allocator pool, not inside any cuda graph private
        pool.  Reused across every (variant, batch_size) capture and across
        replays.

        Phase F shrinks the all-gather receive buffer from ``world * M``
        rows to ``num_replicas * M`` rows because the lane subgroup is
        used instead of the global group.  In that case the lane-select
        step becomes a no-op and union buffers alias the gather buffers.
        """
        if self._static_buffers_ready:
            return
        if self._capture_max_m is None or self.hidden_size is None:
            return
        M = self._capture_max_m
        H = self.hidden_size
        K = self.top_k
        W = self.world_size
        NR = self.num_replicas
        device = torch.device("cuda", torch.cuda.current_device())

        # hidden_states usually inherits from params_dtype (bf16 in this
        # config).  topk weights are float32 by sglang convention; topk ids
        # are int32 after the normalization done in dispatch.  These match
        # the live tensors produced by StandardTopKOutput; the _ensure
        # check inside _dispatch_static would catch a mismatch.
        hidden_dtype = self.params_dtype or torch.bfloat16
        weight_dtype = torch.float32
        id_dtype = torch.int32

        # Per-rank source buffers (input to all_gather_into_tensor).
        self._buf_padded_hidden = torch.zeros(
            (M, H), dtype=hidden_dtype, device=device
        )
        self._buf_padded_topk_ids = torch.full(
            (M, K), -1, dtype=id_dtype, device=device
        )
        self._buf_padded_topk_weights = torch.zeros(
            (M, K), dtype=weight_dtype, device=device
        )

        # Gather destination shape depends on which group we all_gather on.
        gather_world = NR if self.phase_f_enabled else W
        self._buf_gathered_hidden = torch.zeros(
            (gather_world * M, H), dtype=hidden_dtype, device=device
        )
        self._buf_gathered_topk_ids = torch.full(
            (gather_world * M, K), -1, dtype=id_dtype, device=device
        )
        self._buf_gathered_topk_weights = torch.zeros(
            (gather_world * M, K), dtype=weight_dtype, device=device
        )

        if self.phase_f_enabled:
            # gather output IS already the union; alias the union buffers
            # to the gather buffers so the lane-select copies become
            # no-ops and the graph records fewer ops.
            self._buf_union_hidden = self._buf_gathered_hidden
            self._buf_union_topk_ids = self._buf_gathered_topk_ids
            self._buf_union_topk_weights = self._buf_gathered_topk_weights
        else:
            self._buf_union_hidden = torch.zeros(
                (NR * M, H), dtype=hidden_dtype, device=device
            )
            self._buf_union_topk_ids = torch.full(
                (NR * M, K), -1, dtype=id_dtype, device=device
            )
            self._buf_union_topk_weights = torch.zeros(
                (NR * M, K), dtype=weight_dtype, device=device
            )
        self._buf_union_topk_ids_remapped = torch.full(
            (NR * M, K), -1, dtype=id_dtype, device=device
        )
        # Phase F combine scratch: receives the per-replica slice after
        # lane reduce_scatter.  Sized [M, H], one slice.
        self._buf_combine_local_slice = torch.zeros(
            (M, H), dtype=hidden_dtype, device=device
        )

        self._neg_one_int32 = torch.full((), -1, dtype=id_dtype, device=device)

        # Move the expert mapping to device now so the static path never
        # reassigns self.local_expert_mapping during capture.
        self._mapping_on(device)

        self._static_buffers_ready = True
        self._probe_log(
            f"static_buffers_ready rank={self.global_rank} replica={self.replica_rank} "
            f"lane={self.lane_rank} capture_max_m={M} hidden={H} top_k={K} "
            f"world={W} num_replicas={NR} phase_f={self.phase_f_enabled} "
            f"padded_hidden={_tensor_meta(self._buf_padded_hidden)} "
            f"gathered_hidden={_tensor_meta(self._buf_gathered_hidden)} "
            f"union_hidden={_tensor_meta(self._buf_union_hidden)} "
            f"mapping={_tensor_meta(self.local_expert_mapping)} "
            f"group={_group_name(self.group)} lane_group={_group_name(self.lane_group)}"
        )
        logger.info(
            "[KUNSERVE-MS] CrossReplicaStandardDispatcher static buffers ready: "
            "capture_max_m=%d hidden=%d top_k=%d world=%d num_replicas=%d "
            "rank=%d lane=%d replica=%d hidden_dtype=%s phase_f=%s",
            M,
            H,
            K,
            W,
            NR,
            self.global_rank,
            self.lane_rank,
            self.replica_rank,
            hidden_dtype,
            self.phase_f_enabled,
        )

    # ------------------------------------------------------------------
    # path selection
    # ------------------------------------------------------------------

    def _use_static_path(self) -> bool:
        """Return True iff we should take the fixed-shape static path.

        SGLang runs two warmup forwards under ``model_capture_mode()`` before
        entering ``torch.cuda.CUDAGraph`` capture.  KunServe fixed-padded
        GLOBAL path includes non-TP PyNccl collectives, so keep those warmups
        on the eager dynamic path and switch to static buffers only while the
        CUDA stream is actually being captured.  Replay does not call back into
        this Python dispatcher.
        """
        if not self._static_buffers_ready:
            return False
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            return False

    def _require_static_graph_collective_group(self, group: Any, op_name: str) -> None:
        """Fail early if the static CUDA graph path would hit torch.distributed.

        The fixed-padded path is only meant to record KunServe PyNccl/SGLang
        registered collectives.  A silent fallback to raw torch distributed
        inside capture can hang or record non-replayable work.
        """
        try:
            capturing = bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            capturing = False
        if not capturing:
            return
        if group is None or not hasattr(group, op_name):
            raise RuntimeError(
                "KunServe static GLOBAL CUDA graph requires a group exposing "
                f"{op_name}; got {_group_name(group)}."
            )
        if not bool(getattr(group, "kunserve_graph_safe", False)):
            raise RuntimeError(
                "KunServe static GLOBAL CUDA graph requires registered/PyNccl "
                f"collectives; group={_group_name(group)} op={op_name} would "
                "fall back to raw torch.distributed."
            )

    # ------------------------------------------------------------------
    # dynamic (eager) path - preserved from the correctness-first version
    # ------------------------------------------------------------------

    def _all_gather_sizes(self, local_m: int, device: torch.device) -> torch.Tensor:
        """All-gather sizes across the appropriate group for the eager path.

        Phase F uses the lane subgroup (size num_replicas) since the
        eager dispatch's all-gather also runs on the lane.  Phase D
        fallback uses the full global group and validates the TP
        within-replica invariant.
        """
        local_size = torch.tensor([int(local_m)], dtype=torch.int64, device=device)
        if self.phase_f_enabled:
            ws = int(self.num_replicas)
            gather_group = self.lane_group
        else:
            ws = int(self.world_size)
            gather_group = self.group
        sizes = torch.empty(ws, dtype=local_size.dtype, device=device)
        _all_gather_into_tensor(gather_group, sizes, local_size)

        if not self.phase_f_enabled:
            # In SGLang's TP/EP=local_ep_size baseline, ranks in the same
            # replica carry the same token batch.  The cross-replica
            # Standard-like algorithm relies on that invariant because
            # lane 0 and lane 1 produce partial sums for the same union-
            # token order before all-reduce.  In Phase F mode we don't
            # see the within-replica peer in this group, so the
            # invariant is checked implicitly by the lane subgroup itself
            # plus the Phase E scheduler-level bs negotiation.
            sizes_cpu = sizes.detach().cpu().tolist()
            for replica_idx in range(self.num_replicas):
                base = replica_idx * self.local_ep_size
                replica_sizes = sizes_cpu[base : base + self.local_ep_size]
                if len(set(replica_sizes)) != 1:
                    raise RuntimeError(
                        "CrossReplicaStandardDispatcher requires all local EP "
                        "ranks inside a replica to see the same token count. "
                        f"replica={replica_idx} sizes={replica_sizes} "
                        f"all_sizes={sizes_cpu}"
                    )
        return sizes

    def _pad_dim0(
        self,
        tensor: torch.Tensor,
        *,
        max_m: int,
        pad_value: float | int = 0,
    ) -> torch.Tensor:
        local_m = int(tensor.shape[0])
        if local_m == max_m:
            return tensor.contiguous()
        out = torch.full(
            (max_m, *tensor.shape[1:]),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if local_m > 0:
            out[:local_m].copy_(tensor)
        return out

    def _all_gather_padded(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        if self.phase_f_enabled:
            ws = int(self.num_replicas)
            gather_group = self.lane_group
        else:
            ws = int(self.world_size)
            gather_group = self.group
        gathered = _all_gather_padded_tensor(gather_group, tensor)
        return gathered

    def _select_lane_segments(self, gathered: list[torch.Tensor]) -> torch.Tensor:
        if self.phase_f_enabled:
            # Phase F: the lane subgroup already returned only the
            # ``num_replicas`` segments we need, in replica-index order.
            # Just concat.
            return torch.cat(list(gathered), dim=0).contiguous()
        segments = [
            gathered[replica_idx * self.local_ep_size + self.lane_rank]
            for replica_idx in range(self.num_replicas)
        ]
        return torch.cat(segments, dim=0).contiguous()

    def _remap_topk_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        mapping = self._mapping_on(topk_ids.device)
        remapped = torch.full_like(topk_ids, -1, dtype=torch.int32)
        valid = (topk_ids >= 0) & (topk_ids < self.num_experts)
        if bool(valid.any()):
            remapped[valid] = mapping[topk_ids[valid].to(dtype=torch.long)]
        return remapped

    def _probe_log(self, message: str) -> None:
        """Append a probe line directly to KUNSERVE_DETAIL_LOG.

        We can't rely on logger.warning here because dispatcher code runs
        inside the SGLang scheduler subprocess, whose stdout/stderr is not
        captured by Ray.  Mirror the _kunserve_ms file-append trick from
        model_runner.py so probe events actually land in
        ``kunserve_sglang_detail.log``.  Never raises.
        """
        if not self._probe_detail_log_path:
            return
        try:
            import datetime as _dt
            import os as _os
            ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            line = f"[{ts} pid={_os.getpid()}] [KUNSERVE-DBG] {message}\n"
            with open(self._probe_detail_log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:
            pass

    @staticmethod
    def _stats(t: torch.Tensor) -> str:
        """Cheap numerical fingerprint for probe logs.  Includes mean/max/min,
        finite-only mean (to detect nan/inf masking real values), nan/inf
        counts.  Synchronous to ensure values are read after the previous
        collective completes -- only call in probe paths."""
        try:
            tf = t.detach().float()
            n_nan = int(torch.isnan(tf).sum().item())
            n_inf = int(torch.isinf(tf).sum().item())
            finite = tf[torch.isfinite(tf)]
            if finite.numel() > 0:
                return (
                    f"shape={tuple(t.shape)} dtype={t.dtype} "
                    f"mean={finite.mean().item():.4e} "
                    f"absmax={finite.abs().max().item():.4e} "
                    f"nan={n_nan} inf={n_inf}"
                )
            return f"shape={tuple(t.shape)} dtype={t.dtype} all_non_finite nan={n_nan} inf={n_inf}"
        except Exception as exc:
            return f"shape={tuple(t.shape)} dtype={t.dtype} stats_err={exc!r}"

    def _should_log_dynamic_path(self, local_m: int, call: int) -> bool:
        if not self._probe_detail_log_path:
            return False
        capture_m = int(self._capture_max_m or 0)
        # Always log the large eager path that cannot be represented by the
        # fixed-padded decode graph, plus a few early/milestone calls so the
        # next failure log shows whether we reached dispatch/combine/reduce.
        return (
            int(local_m) > max(capture_m, 1024)
            or call <= 3
            or call in self._probe_milestones
        )

    def _dispatch_dynamic(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput, local_m: int
    ) -> StandardDispatchOutput:
        call = self._dispatch_call_count + 1
        detail_timing = kunserve_detailed_timing_enabled()
        timing_fields = {
            "dispatcher": "kunserve_standard",
            "path": "dynamic",
            "rank": int(self.global_rank),
            "replica_rank": int(self.replica_rank),
            "lane_rank": int(self.lane_rank),
            "local_m": int(local_m),
            "phase_f": bool(self.phase_f_enabled),
        }
        with _detail_scope(
            detail_timing, "kunserve_dispatch_dynamic_gather_sizes", **timing_fields
        ):
            sizes = self._all_gather_sizes(local_m, hidden_states.device)
        max_m = int(sizes.max().item()) if sizes.numel() > 0 else local_m
        timing_fields["max_m"] = int(max_m)
        log_dynamic = self._should_log_dynamic_path(local_m, call)
        if log_dynamic:
            try:
                sizes_text = sizes.detach().cpu().tolist()
            except Exception as exc:
                sizes_text = f"<sizes_err:{exc!r}>"
            self._probe_log(
                f"dynamic_dispatch_enter call={call} rank={self.global_rank} "
                f"replica={self.replica_rank} lane={self.lane_rank} "
                f"state={_capture_state_text()} phase_f={self.phase_f_enabled} "
                f"local_m={local_m} max_m={max_m} sizes={sizes_text} "
                f"capture_max_m={self._capture_max_m} "
                f"hidden={_tensor_meta(hidden_states)} "
                f"topk_ids={_tensor_meta(topk_output.topk_ids)} "
                f"topk_weights={_tensor_meta(topk_output.topk_weights)} "
                f"router_logits={_tensor_meta(topk_output.router_logits)} "
                f"group={_group_name(self.group)} lane_group={_group_name(self.lane_group)}"
            )

        with _detail_scope(detail_timing, "kunserve_dispatch_dynamic_pad", **timing_fields):
            padded_hidden = self._pad_dim0(hidden_states, max_m=max_m, pad_value=0)
            padded_topk_ids = self._pad_dim0(
                topk_output.topk_ids, max_m=max_m, pad_value=-1
            )
            padded_topk_weights = self._pad_dim0(
                topk_output.topk_weights, max_m=max_m, pad_value=0
            )

        with _detail_scope(
            detail_timing, "kunserve_dispatch_dynamic_all_gather", **timing_fields
        ):
            gathered_hidden = self._all_gather_padded(padded_hidden)
            gathered_topk_ids = self._all_gather_padded(padded_topk_ids)
            gathered_topk_weights = self._all_gather_padded(padded_topk_weights)

        with _detail_scope(
            detail_timing, "kunserve_dispatch_dynamic_lane_select", **timing_fields
        ):
            union_hidden = self._select_lane_segments(gathered_hidden)
            union_topk_ids = self._select_lane_segments(gathered_topk_ids)
            union_topk_weights = self._select_lane_segments(gathered_topk_weights)

        router_logits = topk_output.router_logits
        if (
            isinstance(router_logits, torch.Tensor)
            and router_logits.dim() >= 1
            and int(router_logits.shape[0]) == local_m
        ):
            with _detail_scope(
                detail_timing,
                "kunserve_dispatch_dynamic_router_logits_gather",
                **timing_fields,
            ):
                padded_router_logits = self._pad_dim0(
                    router_logits, max_m=max_m, pad_value=0
                )
                router_logits = self._select_lane_segments(
                    self._all_gather_padded(padded_router_logits)
                )

        self._last_local_m = local_m
        self._last_max_m = max_m
        self._last_slice_start = self.replica_rank * max_m

        if not self._logged_shape:
            logger.warning(
                "[KUNSERVE-MS] CrossReplicaStandardDispatcher active (dynamic): "
                "rank=%d world=%d local_ep=%d replica=%d lane=%d "
                "local_m=%d max_m=%d union_m=%d capture_max_m=%s phase_f=%s",
                self.global_rank,
                self.world_size,
                self.local_ep_size,
                self.replica_rank,
                self.lane_rank,
                local_m,
                max_m,
                int(union_hidden.shape[0]),
                self._capture_max_m,
                self.phase_f_enabled,
            )
            self._logged_shape = True

        with _detail_scope(detail_timing, "kunserve_dispatch_dynamic_remap", **timing_fields):
            remapped_topk = self._remap_topk_ids(union_topk_ids)

        self._dispatch_call_count = call
        if log_dynamic:
            self._probe_log(
                f"dynamic_dispatch_after_gather call={call} rank={self.global_rank} "
                f"replica={self.replica_rank} lane={self.lane_rank} "
                f"state={_capture_state_text()} local_m={local_m} max_m={max_m} "
                f"padded_hidden={_tensor_meta(padded_hidden)} "
                f"union_hidden={_tensor_meta(union_hidden)} "
                f"union_topk_ids={_tensor_meta(union_topk_ids)} "
                f"union_topk_weights={_tensor_meta(union_topk_weights)} "
                f"remapped_topk={_tensor_meta(remapped_topk)} "
                f"router_logits={_tensor_meta(router_logits)}"
            )
        if self._probe_enabled and self._dispatch_call_count in self._probe_milestones:
            # Histogram of remapped_topk: how many tokens have a valid local
            # row index (0..num_local-1) vs invalid (-1).  If too few valid
            # tokens, the topk routing is mismatched with what this rank
            # actually holds.
            try:
                rt = remapped_topk.detach().to(torch.int64)
                valid_count = int(((rt >= 0) & (rt < self.num_local_experts)).sum().item())
                neg_count = int((rt == -1).sum().item())
                total = int(rt.numel())
                # Also log the unique global expert ids that DID get routed to
                # this rank (so we can verify they match the rank's intended
                # expert range).  Limit to first 16 unique to keep log small.
                global_routed = union_topk_ids[(rt >= 0) & (rt < self.num_local_experts)]
                if global_routed.numel() > 0:
                    uniq, counts = torch.unique(
                        global_routed.detach().to(torch.int64), return_counts=True
                    )
                    uniq = uniq.tolist()[:16]
                    counts = counts.tolist()[:16]
                    routed_summary = list(zip(uniq, counts))
                else:
                    routed_summary = []
            except Exception as exc:
                valid_count = neg_count = total = -1
                routed_summary = f"<err: {exc!r}>"
            self._probe_log(
                f"dispatch_probe call={self._dispatch_call_count} "
                f"rank={self.global_rank} replica={self.replica_rank} "
                f"lane={self.lane_rank} phase_f={self.phase_f_enabled} "
                f"local_m={local_m} max_m={max_m} | "
                f"hidden_in={self._stats(hidden_states)} | "
                f"union_hidden={self._stats(union_hidden)} | "
                f"union_topk_ids={self._stats(union_topk_ids.float())} | "
                f"remapped_topk={self._stats(remapped_topk.float())} | "
                f"routing valid={valid_count}/{total} neg={neg_count} "
                f"routed_global_experts={routed_summary}"
            )

        return StandardDispatchOutput(
            hidden_states=union_hidden,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=union_topk_weights,
                topk_ids=remapped_topk,
                router_logits=router_logits,
            ),
        )

    def _combine_dynamic(self, combine_input: StandardCombineInput) -> torch.Tensor:
        (hidden_states,) = combine_input
        hidden_states = hidden_states.contiguous()
        call = self._combine_call_count + 1
        detail_timing = kunserve_detailed_timing_enabled()
        timing_fields = {
            "dispatcher": "kunserve_standard",
            "path": "dynamic",
            "rank": int(self.global_rank),
            "replica_rank": int(self.replica_rank),
            "lane_rank": int(self.lane_rank),
            "local_m": int(self._last_local_m or 0),
            "max_m": int(self._last_max_m or 0),
            "phase_f": bool(self.phase_f_enabled),
        }
        log_dynamic = self._should_log_dynamic_path(int(self._last_local_m), call)
        if log_dynamic:
            self._probe_log(
                f"dynamic_combine_enter call={call} rank={self.global_rank} "
                f"replica={self.replica_rank} lane={self.lane_rank} "
                f"state={_capture_state_text()} phase_f={self.phase_f_enabled} "
                f"last_local_m={self._last_local_m} last_max_m={self._last_max_m} "
                f"input={_tensor_meta(hidden_states)} "
                f"group={_group_name(self.group)} lane_group={_group_name(self.lane_group)}"
            )

        if self.phase_f_enabled:
            # Phase F: lane reduce_scatter only.
            # Input ``[NR*max_m, H]`` is reduced over the lane subgroup
            # and scattered by replica chunk so each lane member keeps
            # its replica's [max_m, H] slice covering half the expert
            # logical space (lane0: experts 0..63, lane1: experts 64..127
            # in the 2-replica case).  The TP all_reduce that combines
            # lane0's partial with lane1's partial is NOT done here —
            # forward_normal in the model layer already calls
            # tensor_model_parallel_all_reduce after experts().  Doing it
            # here too would double-reduce and overflow after many layers.
            #
            max_m = int(self._last_max_m)
            H = hidden_states.shape[1]
            hidden_for_reduce = hidden_states
            reduce_dtype = hidden_states.dtype
            local_slice = torch.empty(
                (max_m, H),
                dtype=reduce_dtype,
                device=hidden_states.device,
            )
            with _detail_scope(
                detail_timing, "kunserve_combine_dynamic_reduce_scatter", **timing_fields
            ):
                _reduce_scatter_tensor(self.lane_group, local_slice, hidden_for_reduce)
            with _detail_scope(
                detail_timing, "kunserve_combine_dynamic_slice", **timing_fields
            ):
                result = local_slice[: int(self._last_local_m)].contiguous()
            self._combine_call_count = call
            if log_dynamic:
                self._probe_log(
                    f"dynamic_combine_after_reduce call={call} rank={self.global_rank} "
                    f"replica={self.replica_rank} lane={self.lane_rank} "
                    f"state={_capture_state_text()} reduce_dtype={reduce_dtype} "
                    f"reduced={_tensor_meta(local_slice)} result={_tensor_meta(result)}"
                )
            if (
                self._probe_enabled
                and self._combine_call_count in self._probe_milestones
            ):
                self._probe_log(
                    f"combine_probe call={self._combine_call_count} "
                    f"rank={self.global_rank} replica={self.replica_rank} "
                    f"lane={self.lane_rank} phase_f=True "
                    f"last_local_m={int(self._last_local_m)} last_max_m={max_m} | "
                    f"post_expert_union={self._stats(hidden_states)} | "
                    f"reduce_scatter_out={self._stats(local_slice)} | "
                    f"sliced={self._stats(result)}"
                )
            return result

        # Phase D fallback: global all_reduce on the union, then slice.
        # WARNING: this path has the same double-TP-all_reduce hazard as
        # Phase F had before the fix above.  forward_normal will do a
        # tensor_model_parallel_all_reduce after experts(), which doubles
        # the already-complete sum returned here.  Phase D is not currently
        # triggered when Phase F lane subgroups initialise successfully.
        # If Phase D needs to be re-enabled, fix by either (a) migrating
        # to lane groups, or (b) suppressing forward_normal's all_reduce
        # when GLOBAL bundle is active.
        with _detail_scope(
            detail_timing, "kunserve_combine_dynamic_all_reduce", **timing_fields
        ):
            _all_reduce(self.group, hidden_states)
        start = int(self._last_slice_start)
        end = start + int(self._last_local_m)
        with _detail_scope(detail_timing, "kunserve_combine_dynamic_slice", **timing_fields):
            return hidden_states[start:end].contiguous()

    # ------------------------------------------------------------------
    # static (graph-safe) path - the actual Phase D contribution
    # ------------------------------------------------------------------

    def _dispatch_static(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput, local_m: int
    ) -> StandardDispatchOutput:
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights

        M = self._capture_max_m
        detail_timing = kunserve_detailed_timing_enabled()
        timing_fields = {
            "dispatcher": "kunserve_standard",
            "path": "static",
            "rank": int(self.global_rank),
            "replica_rank": int(self.replica_rank),
            "lane_rank": int(self.lane_rank),
            "local_m": int(local_m),
            "max_m": int(M),
            "phase_f": bool(self.phase_f_enabled),
        }
        if not self._static_dispatch_logged:
            self._probe_log(
                f"static_dispatch_enter rank={self.global_rank} replica={self.replica_rank} "
                f"lane={self.lane_rank} state={_capture_state_text()} "
                f"local_m={local_m} capture_max_m={M} phase_f={self.phase_f_enabled} "
                f"hidden_in={_tensor_meta(hidden_states)} "
                f"topk_ids_in={_tensor_meta(topk_ids)} "
                f"topk_weights_in={_tensor_meta(topk_weights)} "
                f"mapping={_tensor_meta(self.local_expert_mapping)} "
                f"padded_hidden={_tensor_meta(self._buf_padded_hidden)} "
                f"gathered_hidden={_tensor_meta(self._buf_gathered_hidden)} "
                f"union_hidden={_tensor_meta(self._buf_union_hidden)} "
                f"remap_buf={_tensor_meta(self._buf_union_topk_ids_remapped)} "
                f"group={_group_name(self.group)} lane_group={_group_name(self.lane_group)}"
            )
            self._static_dispatch_logged = True
        if local_m > M:
            raise RuntimeError(
                "CrossReplicaStandardDispatcher static path requires "
                f"local_m={local_m} <= capture_max_m={M}; ensure the captured "
                "batch size set spans all decode shapes."
            )

        # 1) Pad source buffers.  Zero/fill everything first so any
        #    local_m <= M is well-defined: padding rows produce zero
        #    contribution and topk_ids=-1 makes the runner skip them.
        with _detail_scope(detail_timing, "kunserve_dispatch_static_pad", **timing_fields):
            self._buf_padded_hidden.zero_()
            self._buf_padded_hidden[:local_m].copy_(hidden_states)
            self._buf_padded_topk_ids.fill_(-1)
            self._buf_padded_topk_ids[:local_m].copy_(
                topk_ids.to(self._buf_padded_topk_ids.dtype)
            )
            self._buf_padded_topk_weights.zero_()
            self._buf_padded_topk_weights[:local_m].copy_(
                topk_weights.to(self._buf_padded_topk_weights.dtype)
            )

        # 2) Cross-replica all-gather.
        #
        # Phase F: gather across the LANE subgroup (size num_replicas).
        #          The receive buffer ends up as ``[A_pad, B_pad]`` --
        #          already the union; no lane select needed.
        #
        # Phase D fallback: gather across the global group (size world).
        #          The receive buffer is ``[A,A,B,B]`` and we copy out
        #          the lane segments below.
        if self.phase_f_enabled:
            gather_group = self.lane_group
        else:
            gather_group = self.group
        self._require_static_graph_collective_group(
            gather_group, "all_gather_into_tensor"
        )
        use_grouped = (
            self._dispatch_ncclgroup_enabled
            and hasattr(gather_group, "grouped_all_gather_into_tensor")
        )
        if use_grouped and not self._dispatch_ncclgroup_logged:
            self._probe_log(
                f"dispatch_ncclgroup_active rank={self.global_rank} "
                f"replica={self.replica_rank} lane={self.lane_rank} "
                f"phase_f={self.phase_f_enabled} "
                f"gather_group={_group_name(gather_group)}"
            )
            self._dispatch_ncclgroup_logged = True
        with _detail_scope(
            detail_timing, "kunserve_dispatch_static_all_gather", **timing_fields
        ):
            if use_grouped:
                # Phase G P1: 3 all-gathers fused into one ncclGroup.
                gather_group.grouped_all_gather_into_tensor(
                    [
                        (self._buf_gathered_hidden, self._buf_padded_hidden),
                        (self._buf_gathered_topk_ids, self._buf_padded_topk_ids),
                        (
                            self._buf_gathered_topk_weights,
                            self._buf_padded_topk_weights,
                        ),
                    ]
                )
            else:
                _all_gather_into_tensor(
                    gather_group,
                    self._buf_gathered_hidden,
                    self._buf_padded_hidden,
                )
                _all_gather_into_tensor(
                    gather_group,
                    self._buf_gathered_topk_ids,
                    self._buf_padded_topk_ids,
                )
                _all_gather_into_tensor(
                    gather_group,
                    self._buf_gathered_topk_weights,
                    self._buf_padded_topk_weights,
                )

        # 3) Lane select.  Skipped in Phase F because the lane-subgroup
        #    gather already produced the union directly into
        #    _buf_union_* (aliased to _buf_gathered_*).
        if not self.phase_f_enabled:
            with _detail_scope(
                detail_timing, "kunserve_dispatch_static_lane_select", **timing_fields
            ):
                for replica_idx in range(self.num_replicas):
                    src_base = (
                        replica_idx * self.local_ep_size + self.lane_rank
                    ) * M
                    dst_base = replica_idx * M
                    self._buf_union_hidden[dst_base : dst_base + M].copy_(
                        self._buf_gathered_hidden[src_base : src_base + M]
                    )
                    self._buf_union_topk_ids[dst_base : dst_base + M].copy_(
                        self._buf_gathered_topk_ids[src_base : src_base + M]
                    )
                    self._buf_union_topk_weights[dst_base : dst_base + M].copy_(
                        self._buf_gathered_topk_weights[src_base : src_base + M]
                    )

        if not self._static_after_gather_logged:
            self._probe_log(
                f"static_dispatch_after_gather rank={self.global_rank} replica={self.replica_rank} "
                f"lane={self.lane_rank} state={_capture_state_text()} "
                f"phase_f={self.phase_f_enabled} gather_group="
                f"{_group_name(gather_group)} gathered_hidden={_tensor_meta(self._buf_gathered_hidden)} "
                f"gathered_topk_ids={_tensor_meta(self._buf_gathered_topk_ids)} "
                f"union_hidden={_tensor_meta(self._buf_union_hidden)} "
                f"union_topk_ids={_tensor_meta(self._buf_union_topk_ids)}"
            )
            self._static_after_gather_logged = True

        # 4) Branch-free expert id remap.  Out-of-range or negative ids
        #    become -1; valid ids index into the local expert mapping.
        #    The result is written via copy_ into the pre-allocated remap
        #    buffer so the buffer's pointer stays stable across every
        #    (variant, bs) capture — we must NEVER do
        #    `self._buf_union_topk_ids = ...` because that rebinds the
        #    attribute to a graph-private tensor and corrupts later
        #    captures.
        mapping = self.local_expert_mapping  # already on device
        union_ids = self._buf_union_topk_ids
        if mapping.device != union_ids.device or mapping.dtype != torch.int32:
            if not self._static_mapping_mismatch_logged:
                self._probe_log(
                    f"STATIC_DISPATCH_MAPPING_MISMATCH rank={self.global_rank} "
                    f"replica={self.replica_rank} lane={self.lane_rank} "
                    f"state={_capture_state_text()} mapping={_tensor_meta(mapping)} "
                    f"union_ids={_tensor_meta(union_ids)} expected_dtype=torch.int32 "
                    f"group={_group_name(self.group)} lane_group={_group_name(self.lane_group)}"
                )
                self._static_mapping_mismatch_logged = True
            raise RuntimeError(
                "CrossReplicaStandardDispatcher static mapping must be a "
                "torch.int32 tensor on the same CUDA device as union topk ids; "
                f"mapping device={mapping.device} dtype={mapping.dtype}, "
                f"union_ids device={union_ids.device} dtype={union_ids.dtype}."
            )
        if int(mapping.numel()) != int(self.num_experts):
            if not self._static_mapping_mismatch_logged:
                self._probe_log(
                    f"STATIC_DISPATCH_MAPPING_SIZE_MISMATCH rank={self.global_rank} "
                    f"mapping={_tensor_meta(mapping)} num_experts={self.num_experts}"
                )
                self._static_mapping_mismatch_logged = True
            raise RuntimeError(
                "CrossReplicaStandardDispatcher static mapping has wrong length: "
                f"{int(mapping.numel())} vs num_experts={self.num_experts}."
            )
        with _detail_scope(detail_timing, "kunserve_dispatch_static_remap", **timing_fields):
            valid = (union_ids >= 0) & (union_ids < self.num_experts)
            safe_ids = torch.clamp(
                union_ids, min=0, max=self.num_experts - 1
            ).to(dtype=torch.long)
            looked = mapping[safe_ids].to(union_ids.dtype)
            self._buf_union_topk_ids_remapped.copy_(
                torch.where(valid, looked, self._neg_one_int32)
            )

        self._last_local_m = local_m
        self._last_max_m = M
        self._last_slice_start = self.replica_rank * M

        return StandardDispatchOutput(
            hidden_states=self._buf_union_hidden,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=self._buf_union_topk_weights,
                topk_ids=self._buf_union_topk_ids_remapped,
                # router_logits is informational at this layer; expert
                # compute only consumes topk_weights/topk_ids.  Pass through
                # the original local-shape tensor; downstream code that
                # cares already handles None.
                router_logits=topk_output.router_logits,
            ),
        )

    def _combine_static(self, combine_input: StandardCombineInput) -> torch.Tensor:
        (hidden_states,) = combine_input
        hidden_states = hidden_states.contiguous()
        detail_timing = kunserve_detailed_timing_enabled()
        timing_fields = {
            "dispatcher": "kunserve_standard",
            "path": "static",
            "rank": int(self.global_rank),
            "replica_rank": int(self.replica_rank),
            "lane_rank": int(self.lane_rank),
            "local_m": int(self._last_local_m or 0),
            "max_m": int(self._last_max_m or 0),
            "phase_f": bool(self.phase_f_enabled),
        }

        if not self._static_combine_logged:
            self._probe_log(
                f"static_combine_enter rank={self.global_rank} replica={self.replica_rank} "
                f"lane={self.lane_rank} state={_capture_state_text()} "
                f"phase_f={self.phase_f_enabled} last_local_m={self._last_local_m} "
                f"last_max_m={self._last_max_m} last_slice_start={self._last_slice_start} "
                f"post_expert_hidden={_tensor_meta(hidden_states)} "
                f"combine_slice_buf={_tensor_meta(getattr(self, '_buf_combine_local_slice', None))} "
                f"group={_group_name(self.group)} lane_group={_group_name(self.lane_group)}"
            )
            self._static_combine_logged = True

        if self.phase_f_enabled:
            # Phase F combine: lane collective + local slice.
            #
            # Input shape: [NR*M, H] -- partial expert sum from this rank's
            # local_experts over the whole union batch.
            #
            # reduce_scatter is the bandwidth-optimal operation here: each rank
            # only needs its own replica chunk after summing the lane peers.
            # all_reduce remains available via KUNSERVE_STATIC_COMBINE_MODE for
            # quick rollback if a driver/NCCL stack regresses registered
            # reduce_scatter replay.
            #   lane0 rank → experts 0..63 partial for its replica's tokens
            #   lane1 rank → experts 64..127 partial for its replica's tokens
            #
            # The TP all_reduce that combines lane0's partial with lane1's
            # is NOT done here.  forward_normal in the model layer already
            # calls tensor_model_parallel_all_reduce after experts().
            # Doing it here too would double-reduce every layer and overflow
            # after many MoE layers.
            M = int(self._capture_max_m)
            timing_fields["combine_mode"] = self._static_combine_mode
            if not self._static_combine_mode_logged:
                self._probe_log(
                    f"static_combine_mode rank={self.global_rank} "
                    f"replica={self.replica_rank} lane={self.lane_rank} "
                    f"mode={self._static_combine_mode} phase_f={self.phase_f_enabled} "
                    f"lane_group={_group_name(self.lane_group)} "
                    f"slice_buf={_tensor_meta(getattr(self, '_buf_combine_local_slice', None))}"
                )
                self._static_combine_mode_logged = True
            if self._static_combine_mode in (
                "reduce_scatter",
                "reduce_scatter_tp_all_reduce",
            ):
                local_slice = self._buf_combine_local_slice
                if (
                    local_slice is None
                    or tuple(local_slice.shape) != (M, int(hidden_states.shape[1]))
                    or local_slice.dtype != hidden_states.dtype
                    or local_slice.device != hidden_states.device
                ):
                    raise RuntimeError(
                        "KunServe static reduce_scatter combine buffer mismatch: "
                        f"slice={_tensor_meta(local_slice)} "
                        f"input={_tensor_meta(hidden_states)} M={M}."
                    )
                self._require_static_graph_collective_group(
                    self.lane_group, "reduce_scatter_tensor"
                )
                if (
                    self._static_combine_mode == "reduce_scatter_tp_all_reduce"
                    and self._allow_static_tp_allreduce_fusion
                ):
                    lane_group_name = _group_unique_name(self.lane_group)
                    tp_group_name = _group_unique_name(self.local_tp_group)
                    if lane_group_name is None or tp_group_name is None:
                        raise RuntimeError(
                            "KunServe static TP all-reduce fusion requires registered "
                            f"group names; lane_group={_group_name(self.lane_group)} "
                            f"local_tp_group={_group_name(self.local_tp_group)}."
                        )
                    if not hasattr(self.local_tp_group, "_all_reduce_in_place"):
                        raise RuntimeError(
                            "KunServe static TP all-reduce fusion requires the "
                            "SGLang TP GroupCoordinator, not a raw ProcessGroup."
                        )
                    with _detail_scope(
                        detail_timing,
                        "kunserve_combine_static_lane_reduce_scatter_tp_all_reduce",
                        **timing_fields,
                    ):
                        kunserve_lane_reduce_scatter_then_tp_all_reduce(
                            local_slice,
                            hidden_states,
                            lane_group_name,
                            tp_group_name,
                        )
                    with _detail_scope(
                        detail_timing, "kunserve_combine_static_slice", **timing_fields
                    ):
                        result = local_slice[: int(self._last_local_m)].contiguous()
                    try:
                        result._kunserve_tp_allreduce_done = True
                    except Exception:
                        pass
                    return result

                with _detail_scope(
                    detail_timing,
                    "kunserve_combine_static_lane_reduce_scatter",
                    **timing_fields,
                ):
                    _reduce_scatter_tensor(self.lane_group, local_slice, hidden_states)
                with _detail_scope(
                    detail_timing, "kunserve_combine_static_slice", **timing_fields
                ):
                    return local_slice[: int(self._last_local_m)].contiguous()

            self._require_static_graph_collective_group(self.lane_group, "all_reduce")
            with _detail_scope(
                detail_timing,
                "kunserve_combine_static_lane_all_reduce",
                **timing_fields,
            ):
                _all_reduce(self.lane_group, hidden_states)
            start = int(self.replica_rank) * M
            end = start + int(self._last_local_m)
            with _detail_scope(
                detail_timing, "kunserve_combine_static_slice", **timing_fields
            ):
                return hidden_states[start:end].contiguous()

        # Phase D fallback: global all_reduce + slice.  Each (variant,
        # bs) graph allocates its own partial-output tensor in the
        # graph private pool, so all_reduce-in-place targets a stable
        # address per graph.
        # WARNING: same double-TP-all_reduce hazard as _combine_dynamic
        # Phase D; see note there.  Not triggered when Phase F is active.
        with _detail_scope(
            detail_timing, "kunserve_combine_static_global_all_reduce", **timing_fields
        ):
            self._require_static_graph_collective_group(self.group, "all_reduce")
            _all_reduce(self.group, hidden_states)

        start = self.replica_rank * self._capture_max_m
        end = start + int(self._last_local_m)
        with _detail_scope(detail_timing, "kunserve_combine_static_slice", **timing_fields):
            return hidden_states[start:end].contiguous()

    # ------------------------------------------------------------------
    # public BaseDispatcher API
    # ------------------------------------------------------------------

    def set_static_tp_allreduce_fusion_enabled(self, enabled: bool) -> None:
        self._allow_static_tp_allreduce_fusion = bool(enabled)

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> StandardDispatchOutput:
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise NotImplementedError(
                "CrossReplicaStandardDispatcher currently supports only "
                f"StandardTopKOutput, got {type(topk_output)!r}."
            )

        local_m = int(hidden_states.shape[0])
        if self._use_static_path():
            return self._dispatch_static(hidden_states, topk_output, local_m)
        return self._dispatch_dynamic(hidden_states, topk_output, local_m)

    def combine(self, combine_input: StandardCombineInput) -> torch.Tensor:
        if (
            self._last_local_m is None
            or self._last_max_m is None
            or self._last_slice_start is None
        ):
            raise RuntimeError(
                "CrossReplicaStandardDispatcher.combine called before dispatch."
            )

        if self._use_static_path():
            return self._combine_static(combine_input)
        return self._combine_dynamic(combine_input)
