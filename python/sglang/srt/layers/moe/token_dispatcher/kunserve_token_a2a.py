"""KunServe Phase G: token-level all-to-all dispatcher.

Replaces the dense lane.all_gather + lane.reduce_scatter pattern from
:class:`CrossReplicaStandardDispatcher` with sparse token-level a2a.
Each token's K expert assignments are sent ONLY to the ranks whose
experts those assignments target.  The original top-k width is preserved
for the MoE runner; columns not owned by the destination peer are masked
as expert=-1, weight=0.  Expert outputs are then sent back to the origin
rank and reduced per-token in fp32, matching the numerical behavior of a
single-replica EP=4 fused MoE.

Lane-level a2a (2-rank groups) is used rather than full 4-way a2a to
keep the existing TP-all_reduce-handles-lane-combine architecture.
For each lane, the lane peers exchange tokens; expert compute happens
on each rank for its local experts; the lane's contribution to each
origin token is summed and returned.  ``forward_normal``'s downstream
``tp_all_reduce`` then combines lane 0 + lane 1 partials into the
final MoE output.

Toggled by env var ``KUNSERVE_PHASE_G=1``.  Eager mode only — CUDA
graph capture support is future work (Phase G2).
"""
from __future__ import annotations

import datetime as _dt
import logging
import os as _os
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


class CrossReplicaTokenA2ADispatcher(BaseDispatcher):
    """KunServe Phase G GLOBAL dispatcher.

    Numerical equivalence: each origin token's K expert contributions
    are summed in fp32 on the origin rank in the same order as a
    single-replica EP=N fused MoE.  Compared to Phase F
    (lane.all_gather + lane.reduce_scatter + tp_all_reduce), only the
    final tp_all_reduce bf16 step remains as a noise source.

    Constructor mirrors :class:`CrossReplicaStandardDispatcher`.
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
        lane_group: Optional[dist.ProcessGroup] = None,
        local_tp_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        if group is None:
            raise ValueError("CrossReplicaTokenA2ADispatcher requires a process group.")
        if lane_group is None:
            raise ValueError(
                "CrossReplicaTokenA2ADispatcher requires lane_group; "
                "Phase G operates on lane subgroups only."
            )

        self.group = group  # 4-rank runtime_group, used only for diagnostic
        self.lane_group = lane_group  # 2-rank lane group for a2a
        self.local_tp_group = local_tp_group  # 2-rank local TP, unused here

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
        self.num_replicas = self.world_size // self.local_ep_size
        self.replica_rank = (
            int(replica_rank)
            if replica_rank is not None
            else self.global_rank // self.local_ep_size
        )
        self.lane_rank = self.global_rank % self.local_ep_size
        self.lane_world = int(dist.get_world_size(group=lane_group))  # = num_replicas
        self.lane_local_rank = int(dist.get_rank(group=lane_group))  # 0 or 1

        self.num_experts = int(moe_runner_config.num_experts)
        self.top_k = int(moe_runner_config.top_k)
        self.num_local_experts = int(moe_runner_config.num_local_experts)
        self.hidden_size = (
            int(moe_runner_config.hidden_size)
            if moe_runner_config.hidden_size is not None
            else None
        )
        self.params_dtype = moe_runner_config.params_dtype

        # Local expert mapping: tensor of shape [num_experts] mapping
        # dispatch-domain expert id -> compact local row index, or -1 when
        # the expert is not resident on this rank.
        self.local_expert_mapping = self._normalize_local_expert_mapping(
            local_expert_mapping
        )
        self.active_local_expert_mapping = self.local_expert_mapping

        # State saved by dispatch for combine to consume.
        self._last_local_m: Optional[int] = None
        self._last_send_counts: Optional[torch.Tensor] = None
        self._last_recv_counts: Optional[torch.Tensor] = None
        self._last_flat_t: Optional[torch.Tensor] = None  # origin token idx per send entry
        self._last_flat_k: Optional[torch.Tensor] = None  # origin k idx per send entry
        self._last_flat_weight: Optional[torch.Tensor] = None
        self._last_topk_weights: Optional[torch.Tensor] = None  # [M, K]
        self._last_num_send: Optional[int] = None
        self._last_num_recv: Optional[int] = None
        self._last_recv_expert_ids: Optional[torch.Tensor] = None

        # Probe / debug log path.
        self._detail_log_path: Optional[str] = _os.environ.get("KUNSERVE_DETAIL_LOG")
        self._dispatch_call_count = 0
        self._combine_call_count = 0

        self._log(
            f"phase_g_init rank={self.global_rank} replica={self.replica_rank} "
            f"lane={self.lane_rank} lane_world={self.lane_world} "
            f"num_local_experts={self.num_local_experts} top_k={self.top_k}"
        )

    # ----------------------------------------------------------
    # logging
    # ----------------------------------------------------------
    def _log(self, message: str) -> None:
        if not self._detail_log_path:
            return
        try:
            ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            line = (
                f"[{ts} pid={_os.getpid()}] [KUNSERVE-DBG] phase_g {message}\n"
            )
            with open(self._detail_log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:
            pass

    # ----------------------------------------------------------
    # construction / mapping helpers
    # ----------------------------------------------------------
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

    def _map_received_expert_ids(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Map dispatch-domain expert ids to this rank's compact local rows."""
        mapping = self._mapping_on(expert_ids.device)
        safe_ids = expert_ids.to(torch.long).clamp_(min=0, max=self.num_experts - 1)
        mapped = mapping[safe_ids].to(torch.int32)
        valid = (
            (expert_ids >= 0)
            & (expert_ids < self.num_experts)
            & (mapped >= 0)
        )
        return torch.where(
            valid,
            mapped,
            torch.full_like(mapped, -1, dtype=torch.int32),
        )

    # ----------------------------------------------------------
    # routing helper: which lane peer owns each (t, k) assignment?
    # ----------------------------------------------------------
    def _compute_owner_in_lane(
        self, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        """For each (t, k) in topk_ids, determine which lane-local rank
        owns the assigned expert.  Returns int32 tensor shape [M, K]
        with values in {0, 1, ..., lane_world-1} or -1 for invalid.

        Mapping logic: each lane covers ``num_local_experts × lane_world``
        physical experts (with a "swap" pattern in our setup).  We use
        ``self.local_expert_mapping`` to determine if an assignment is
        local; for cross-lane-peer assignments we use the global EP-rank
        from the SWAP map to find the lane-local rank index.

        In our 2-replica setup:
            lane 0 = [global rank 0 = lane_local 0, global rank 2 = lane_local 1]
            lane 1 = [global rank 1 = lane_local 0, global rank 3 = lane_local 1]
        and the swap places:
            rank 0 = physical experts [0..31]
            rank 1 = physical experts [32..63]
            rank 2 = physical experts [64..95]
            rank 3 = physical experts [96..127]

        For a token's topk_ids (PHYSICAL expert ids after the SWAP
        applied by topk), owner_global_rank = expert_id // num_local.
        Then owner_in_lane = (owner_global_rank % lane_world) where
        lane_world = num_replicas.  Equivalently:
            owner_in_lane = owner_global_rank // local_ep_size
        (because global_rank = replica_idx * local_ep_size + lane_rank,
        so owner_global_rank % local_ep_size == lane_rank means
        this token's lane peer doesn't own it; we need to map to the
        peer-on-this-lane via replica_idx).

        For correctness with the existing dispatcher_local_expert_mapping
        we cross-check: if owner_in_lane==self.lane_local_rank, the entry
        is local; otherwise it goes to the lane peer.
        """
        # owner_global_rank: which of the 4 ranks holds this expert
        owner_global = torch.where(
            (topk_ids >= 0) & (topk_ids < self.num_experts),
            topk_ids // self.num_local_experts,  # 0..world-1
            torch.full_like(topk_ids, -1),
        )
        # owner_in_lane: which lane-local rank (0 or 1) within OUR lane
        # holds it.  Only meaningful when owner_global's lane_rank matches
        # our lane_rank (= same lane membership).  Cross-lane assignments
        # are handled by the OTHER lane's dispatcher (we just ignore here
        # and let tp_all_reduce combine).
        owner_lane_rank = owner_global % self.local_ep_size  # lane_rank
        owner_replica_idx = owner_global // self.local_ep_size  # replica_idx
        owner_in_lane = torch.where(
            owner_lane_rank == self.lane_rank,
            owner_replica_idx,  # which replica's lane-rank
            torch.full_like(owner_global, -1),  # not on our lane
        )
        return owner_in_lane

    # ----------------------------------------------------------
    # dispatch: lane a2a forward
    # ----------------------------------------------------------
    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> StandardDispatchOutput:
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise NotImplementedError(
                "CrossReplicaTokenA2ADispatcher currently supports only "
                f"StandardTopKOutput, got {type(topk_output)!r}."
            )

        device = hidden_states.device
        topk_ids = topk_output.topk_ids  # [M, K] physical expert ids (or -1)
        topk_weights = topk_output.topk_weights  # [M, K] float32

        M, _ = topk_ids.shape
        K = int(topk_ids.shape[1])
        H = int(hidden_states.shape[1])

        # === Step 1: routing ===
        owner_in_lane = self._compute_owner_in_lane(topk_ids)  # [M, K]
        valid_mask = owner_in_lane >= 0  # only assignments on OUR lane

        # Preserve the original top-k width for the MoE runner.  For each
        # destination lane peer, send each origin token at most once with its
        # full [K] metadata row; columns not owned by that peer are masked out
        # as expert=-1, weight=0.  This keeps Triton on the same top-k shape as
        # the base model while still avoiding dense cross-replica all-gather.
        flat_t_parts = []
        send_expert_parts = []
        send_weight_parts = []
        send_count_values = []
        for peer in range(self.lane_world):
            peer_mask = owner_in_lane == peer  # [M, K]
            peer_rows = peer_mask.any(dim=1)   # [M]
            peer_t = torch.nonzero(peer_rows, as_tuple=False).flatten()
            peer_topk_ids = topk_ids[peer_t].to(torch.int32)
            peer_topk_weights = topk_weights[peer_t].to(torch.float32)
            peer_keep = peer_mask[peer_t]
            peer_send_ids = torch.where(
                peer_keep,
                peer_topk_ids,
                torch.full_like(peer_topk_ids, -1),
            )
            peer_send_weights = torch.where(
                peer_keep,
                peer_topk_weights,
                torch.zeros_like(peer_topk_weights),
            )
            flat_t_parts.append(peer_t.to(torch.int64))
            send_expert_parts.append(peer_send_ids)
            send_weight_parts.append(peer_send_weights)
            send_count_values.append(int(peer_t.numel()))

        flat_t = torch.cat(flat_t_parts, dim=0) if flat_t_parts else torch.empty(
            (0,), dtype=torch.int64, device=device
        )
        send_expert_global = (
            torch.cat(send_expert_parts, dim=0).contiguous()
            if send_expert_parts
            else torch.empty((0, K), dtype=torch.int32, device=device)
        )
        send_weight = (
            torch.cat(send_weight_parts, dim=0).contiguous()
            if send_weight_parts
            else torch.empty((0, K), dtype=torch.float32, device=device)
        )
        send_counts = torch.tensor(
            send_count_values, dtype=torch.int64, device=device
        )

        # === Step 2: exchange counts ===
        recv_counts = torch.empty(
            self.lane_world, dtype=torch.int64, device=device
        )
        dist.all_to_all_single(recv_counts, send_counts, group=self.lane_group)

        # Need host-side counts to size receive buffers.  Eager-mode sync.
        send_counts_cpu = send_counts.tolist()
        recv_counts_cpu = recv_counts.tolist()
        num_send = int(sum(send_counts_cpu))
        num_recv = int(sum(recv_counts_cpu))

        # === Step 3: pack and a2a send hidden + metadata ===
        send_hidden = hidden_states[flat_t].contiguous()                  # [num_send, H]

        recv_hidden = torch.empty(
            (num_recv, H), dtype=hidden_states.dtype, device=device
        )
        recv_expert_global = torch.empty(
            (num_recv, K), dtype=torch.int32, device=device
        )
        recv_weight = torch.empty(
            (num_recv, K), dtype=torch.float32, device=device
        )

        # These collectives must run on every lane rank even when this rank has
        # no local send/recv rows.  A busy peer may still be doing a self-only
        # transfer, and skipping locally would desynchronize the collective
        # sequence and deadlock the lane.
        dist.all_to_all_single(
            recv_hidden, send_hidden,
            output_split_sizes=recv_counts_cpu,
            input_split_sizes=send_counts_cpu,
            group=self.lane_group,
        )
        dist.all_to_all_single(
            recv_expert_global, send_expert_global,
            output_split_sizes=recv_counts_cpu,
            input_split_sizes=send_counts_cpu,
            group=self.lane_group,
        )
        dist.all_to_all_single(
            recv_weight, send_weight,
            output_split_sizes=recv_counts_cpu,
            input_split_sizes=send_counts_cpu,
            group=self.lane_group,
        )
        recv_expert_local = self._map_received_expert_ids(recv_expert_global)

        # === Step 4: save state for combine ===
        self._last_local_m = M
        self._last_send_counts = send_counts
        self._last_recv_counts = recv_counts
        self._last_flat_t = flat_t
        self._last_flat_k = None
        self._last_flat_weight = None
        self._last_topk_weights = topk_weights  # for backup / debugging
        self._last_num_send = num_send
        self._last_num_recv = num_recv
        self._last_recv_expert_ids = recv_expert_local

        self._dispatch_call_count += 1
        if self._dispatch_call_count in (1, 5, 20, 100, 500):
            try:
                invalid_recv = (
                    int(((recv_expert_global >= 0) & (recv_expert_local < 0)).sum().item())
                    if num_recv > 0
                    else 0
                )
            except Exception:
                invalid_recv = -1
            self._log(
                f"dispatch call={self._dispatch_call_count} rank={self.global_rank} "
                f"M={M} K={K} num_valid_lane={int(valid_mask.sum().item())} "
                f"num_send={num_send} num_recv={num_recv} "
                f"send_counts={send_counts_cpu} recv_counts={recv_counts_cpu} "
                f"invalid_recv_experts={invalid_recv}"
            )

        # === Step 5: format runner input ===
        # Keep the original top-k width.  Masked columns remain expert=-1 and
        # weight=0, matching StandardDispatcher's non-local expert convention.
        runner_topk_ids = recv_expert_local.to(torch.int32)                # [num_recv, K]
        runner_topk_weights = recv_weight                                  # [num_recv, K]

        return StandardDispatchOutput(
            hidden_states=recv_hidden,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=runner_topk_weights,
                topk_ids=runner_topk_ids,
                router_logits=None,
            ),
        )

    # ----------------------------------------------------------
    # combine: lane a2a backward + scatter-add at origin
    # ----------------------------------------------------------
    def combine(self, combine_input: StandardCombineInput) -> torch.Tensor:
        if self._last_local_m is None:
            raise RuntimeError(
                "CrossReplicaTokenA2ADispatcher.combine called before dispatch."
            )

        (post_expert,) = combine_input  # [num_recv, H] expert outputs (already weighted by topk_weights inside kernel)
        post_expert = post_expert.contiguous()

        device = post_expert.device
        H = int(post_expert.shape[1])
        M = self._last_local_m
        num_send = self._last_num_send
        num_recv = self._last_num_recv

        send_counts_cpu = self._last_send_counts.tolist()
        recv_counts_cpu = self._last_recv_counts.tolist()

        # === Step 1: a2a back (inverse of dispatch) ===
        # The output we need is shaped [num_send, H] in the same order
        # as the dispatch-time send buffer (so we can scatter_add back
        # to origin token indices using flat_t).
        recv_back = torch.empty(
            (num_send, H), dtype=post_expert.dtype, device=device
        )
        # Note swapped split sizes: send to peers from recv side, recv into
        # send-shaped buffer.  Keep this collective unconditional for the same
        # reason as the dispatch-side data a2a above.
        dist.all_to_all_single(
            recv_back, post_expert,
            output_split_sizes=send_counts_cpu,
            input_split_sizes=recv_counts_cpu,
            group=self.lane_group,
        )

        # === Step 2: scatter_add into origin token buffer ===
        # Each recv_back row is one destination peer's partial MoE output for
        # flat_t[i].  The runner has already reduced that peer's local top-k
        # subset with topk_weights applied; here we sum peer partials per origin
        # token in fp32.
        out_fp32 = torch.zeros((M, H), dtype=torch.float32, device=device)
        if num_send > 0:
            # Upcast to fp32 for the reduction (matches single-replica EP
            # numerical behavior).
            recv_back_fp32 = recv_back.to(torch.float32)
            out_fp32.index_add_(0, self._last_flat_t, recv_back_fp32)

        result = out_fp32.to(post_expert.dtype)

        self._combine_call_count += 1
        if self._combine_call_count in (1, 5, 20, 100, 500):
            try:
                absmax = float(result.float().abs().max().item())
                nan_count = int(torch.isnan(result).sum().item())
            except Exception:
                absmax = float("nan")
                nan_count = -1
            self._log(
                f"combine call={self._combine_call_count} rank={self.global_rank} "
                f"M={M} num_send={num_send} num_recv={num_recv} "
                f"result_shape={tuple(result.shape)} absmax={absmax:.4e} nan={nan_count}"
            )

        return result
