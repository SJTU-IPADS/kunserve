from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist

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


class CrossReplicaStandardDispatcher(BaseDispatcher):
    """KunServe GLOBAL dispatcher implemented with plain torch collectives.

    Two operating modes:

    * **Dynamic eager** (default, when ``capture_max_m`` is None or buffers
      have not been provisioned): all-gather sizes, pad to per-step max,
      dispatch/combine with dynamic shapes. Correctness-first, NOT
      CUDA-graph-safe. Used for prefill, mismatched batch sizes, and any path
      that does not go through cuda graph replay.

    * **Fixed-padded capture** (when ``capture_max_m`` is provided): the
      constructor pre-allocates static buffers sized for ``capture_max_m``
      rows per rank. While ``torch.cuda.is_current_stream_capturing()`` is
      True the dispatcher uses ``all_gather_into_tensor`` + static slice
      copies + ``all_reduce`` over those buffers, no host syncs, no
      ``torch.empty_like`` inside the hot path. This is what makes
      ``capture_policy=fixed_padded`` Phase D possible for the sglang
      backend.

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
        group: dist.ProcessGroup,
        moe_runner_config: MoeRunnerConfig,
        local_expert_mapping: torch.Tensor,
        local_ep_size: int,
        replica_rank: Optional[int] = None,
        global_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        capture_max_m: Optional[int] = None,
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
            else int(dist.get_rank(group=group))
        )
        self.world_size = (
            int(world_size)
            if world_size is not None
            else int(dist.get_world_size(group=group))
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
            self.local_expert_mapping = self.local_expert_mapping.to(
                device=device, non_blocking=True
            )
            self.active_local_expert_mapping = self.local_expert_mapping
        return self.local_expert_mapping

    def _allocate_static_buffers(self) -> None:
        """Allocate persistent buffers for the fixed-padded capture path.

        Allocated eagerly in the constructor so the buffers live in the
        default caching allocator pool, not inside any cuda graph private
        pool.  Reused across every (variant, batch_size) capture and across
        replays.
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

        self._buf_padded_hidden = torch.zeros(
            (M, H), dtype=hidden_dtype, device=device
        )
        self._buf_padded_topk_ids = torch.full(
            (M, K), -1, dtype=id_dtype, device=device
        )
        self._buf_padded_topk_weights = torch.zeros(
            (M, K), dtype=weight_dtype, device=device
        )

        self._buf_gathered_hidden = torch.zeros(
            (W * M, H), dtype=hidden_dtype, device=device
        )
        self._buf_gathered_topk_ids = torch.full(
            (W * M, K), -1, dtype=id_dtype, device=device
        )
        self._buf_gathered_topk_weights = torch.zeros(
            (W * M, K), dtype=weight_dtype, device=device
        )

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

        self._neg_one_int32 = torch.full((), -1, dtype=id_dtype, device=device)

        # Move the expert mapping to device now so the static path never
        # reassigns self.local_expert_mapping during capture.
        self._mapping_on(device)

        self._static_buffers_ready = True
        logger.info(
            "[KUNSERVE-MS] CrossReplicaStandardDispatcher static buffers ready: "
            "capture_max_m=%d hidden=%d top_k=%d world=%d num_replicas=%d "
            "rank=%d lane=%d replica=%d hidden_dtype=%s",
            M,
            H,
            K,
            W,
            NR,
            self.global_rank,
            self.lane_rank,
            self.replica_rank,
            hidden_dtype,
        )

    # ------------------------------------------------------------------
    # path selection
    # ------------------------------------------------------------------

    def _use_static_path(self) -> bool:
        """Return True iff we should take the graph-safe static path.

        Only chosen while a cuda graph is being captured.  Eager forward
        (BALLOON without captured graph, or batch sizes outside capture_bs)
        always falls through to the dynamic path.
        """
        if not self._static_buffers_ready:
            return False
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            return False

    # ------------------------------------------------------------------
    # dynamic (eager) path - preserved from the correctness-first version
    # ------------------------------------------------------------------

    def _all_gather_sizes(self, local_m: int, device: torch.device) -> torch.Tensor:
        local_size = torch.tensor([int(local_m)], dtype=torch.int64, device=device)
        gathered = [torch.empty_like(local_size) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_size, group=self.group)
        sizes = torch.cat(gathered, dim=0)

        # In SGLang's TP/EP=local_ep_size baseline, ranks in the same replica
        # carry the same token batch.  The cross-replica Standard-like
        # algorithm relies on that invariant because lane 0 and lane 1 produce
        # partial sums for the same union-token order before all-reduce.
        sizes_cpu = sizes.detach().cpu().tolist()
        for replica_idx in range(self.num_replicas):
            base = replica_idx * self.local_ep_size
            replica_sizes = sizes_cpu[base : base + self.local_ep_size]
            if len(set(replica_sizes)) != 1:
                raise RuntimeError(
                    "CrossReplicaStandardDispatcher requires all local EP ranks "
                    "inside a replica to see the same token count. "
                    f"replica={replica_idx} sizes={replica_sizes} all_sizes={sizes_cpu}"
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
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=self.group)
        return gathered

    def _select_lane_segments(self, gathered: list[torch.Tensor]) -> torch.Tensor:
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

    def _dispatch_dynamic(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput, local_m: int
    ) -> StandardDispatchOutput:
        sizes = self._all_gather_sizes(local_m, hidden_states.device)
        max_m = int(sizes.max().item()) if sizes.numel() > 0 else local_m

        padded_hidden = self._pad_dim0(hidden_states, max_m=max_m, pad_value=0)
        padded_topk_ids = self._pad_dim0(
            topk_output.topk_ids, max_m=max_m, pad_value=-1
        )
        padded_topk_weights = self._pad_dim0(
            topk_output.topk_weights, max_m=max_m, pad_value=0
        )

        gathered_hidden = self._all_gather_padded(padded_hidden)
        gathered_topk_ids = self._all_gather_padded(padded_topk_ids)
        gathered_topk_weights = self._all_gather_padded(padded_topk_weights)

        union_hidden = self._select_lane_segments(gathered_hidden)
        union_topk_ids = self._select_lane_segments(gathered_topk_ids)
        union_topk_weights = self._select_lane_segments(gathered_topk_weights)

        router_logits = topk_output.router_logits
        if (
            isinstance(router_logits, torch.Tensor)
            and router_logits.dim() >= 1
            and int(router_logits.shape[0]) == local_m
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
                "local_m=%d max_m=%d union_m=%d capture_max_m=%s",
                self.global_rank,
                self.world_size,
                self.local_ep_size,
                self.replica_rank,
                self.lane_rank,
                local_m,
                max_m,
                int(union_hidden.shape[0]),
                self._capture_max_m,
            )
            self._logged_shape = True

        return StandardDispatchOutput(
            hidden_states=union_hidden,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=union_topk_weights,
                topk_ids=self._remap_topk_ids(union_topk_ids),
                router_logits=router_logits,
            ),
        )

    def _combine_dynamic(self, combine_input: StandardCombineInput) -> torch.Tensor:
        (hidden_states,) = combine_input
        hidden_states = hidden_states.contiguous()
        dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM, group=self.group)

        start = int(self._last_slice_start)
        end = start + int(self._last_local_m)
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
        if local_m > M:
            raise RuntimeError(
                "CrossReplicaStandardDispatcher static path requires "
                f"local_m={local_m} <= capture_max_m={M}; ensure the captured "
                "batch size set spans all decode shapes."
            )

        # 1) Pad source buffers.  Zero/fill everything first so any
        #    local_m <= M is well-defined: padding rows produce zero
        #    contribution and topk_ids=-1 makes the runner skip them.
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

        # 2) Cross-replica all-gather into pre-allocated contiguous buffers.
        dist.all_gather_into_tensor(
            self._buf_gathered_hidden,
            self._buf_padded_hidden,
            group=self.group,
        )
        dist.all_gather_into_tensor(
            self._buf_gathered_topk_ids,
            self._buf_padded_topk_ids,
            group=self.group,
        )
        dist.all_gather_into_tensor(
            self._buf_gathered_topk_weights,
            self._buf_padded_topk_weights,
            group=self.group,
        )

        # 3) Lane select with static slice copies.  num_replicas is a
        #    Python int known at capture time, so this loop unrolls cleanly
        #    into a small number of recorded copies.
        for replica_idx in range(self.num_replicas):
            src_base = (replica_idx * self.local_ep_size + self.lane_rank) * M
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
        valid = (union_ids >= 0) & (union_ids < self.num_experts)
        safe_ids = torch.clamp(union_ids, min=0).to(dtype=torch.long)
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
        # Each (variant, bs) graph allocates its own partial-output tensor
        # in the graph private pool, so all_reduce-in-place targets a
        # stable address per graph.
        hidden_states = hidden_states.contiguous()
        dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM, group=self.group)

        start = self.replica_rank * self._capture_max_m
        end = start + int(self._last_local_m)
        return hidden_states[start:end].contiguous()

    # ------------------------------------------------------------------
    # public BaseDispatcher API
    # ------------------------------------------------------------------

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
