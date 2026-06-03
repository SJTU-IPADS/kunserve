"""KunServe DeepEP link — M0: pure reference for cross-replica balloon routing.

This module is the **precision/transport-independent specification** of how a
token's top-k logical experts must be routed across the two-replica GLOBAL EP
world after BALLOON. It contains NO torch.distributed / GPU / DeepEP code — it
is a CPU reference used by:

  * unit tests (``test_kunserve_routing_ref_m0.py``) to pin down correctness of
    the existing layout math in ``balloon_utils`` / ``kunserve_manager.layout``;
  * later milestones (M1+) as the ground-truth oracle the real DeepEP path's
    output must match.

The single correctness property we care about (the one that, if wrong, makes
DeepEP route to the wrong owner and corrupt outputs):

    For every logical expert ``e``, after the static remap ``e -> GLOBAL
    physical id p``, the owner rank ``p // num_local_physical`` is exactly the
    rank that retains ``e``'s weights, at local row ``p % num_local_physical``.

Equivalently: GLOBAL-split routing computes the *same* MoE output as a plain
single-device full-expert MoE — the half-split + dispatch must not change the
math, only where each expert runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from sglang.srt.model_executor.balloon_utils import (
    build_dispatcher_physical_expert_mapping,
)


@dataclass
class GlobalBalloonLayout:
    """Resolved two-replica GLOBAL EP layout (single MoE layer).

    All ids are in the GLOBAL physical dispatch domain unless named ``logical``.
    """

    num_logical_experts: int
    num_physical_experts: int
    local_ep_size: int
    num_replicas: int
    retained_local_experts: int
    global_world_size: int
    num_local_physical: int  # physical experts retained per GLOBAL rank
    global_physical_to_logical: list[int]  # GLOBAL physical id -> logical id
    logical_to_physical: list[int]  # static remap: logical id -> GLOBAL physical id
    # per GLOBAL rank: GLOBAL-physical-id -> compact local row, or -1 (from balloon_utils)
    dispatcher_physical_mapping: list[torch.Tensor]

    def owner_rank(self, logical_id: int) -> int:
        return self.logical_to_physical[logical_id] // self.num_local_physical

    def local_row(self, logical_id: int) -> int:
        return self.logical_to_physical[logical_id] % self.num_local_physical


def build_complementary_global_p2l(
    base_physical_to_logical_row: Sequence[int],
    *,
    local_ep_size: int,
    retained_local_experts: int,
) -> list[int]:
    """Independent re-derivation of the symmetric half-split GLOBAL physical->logical
    map for a single layer (mirrors ``kunserve_manager.layout.
    build_complementary_physical_to_logical_map``; kept self-contained so this
    reference does not depend on the sidecar package, and so the test can
    cross-check two independent implementations).

    Layout per local rank: [rank0 prefix..][rank1 prefix..] then
    [rank0 suffix..][rank1 suffix..], i.e. replica 0 retains the prefix half of
    every local rank's chunk, replica 1 retains the suffix half.
    """
    layer_row = list(map(int, base_physical_to_logical_row))
    num_physical = len(layer_row)
    if num_physical % local_ep_size != 0:
        raise ValueError("num_physical not divisible by local_ep_size")
    local_chunk = num_physical // local_ep_size
    if retained_local_experts * 2 != local_chunk:
        raise ValueError("symmetric half split required: retained*2 == local_chunk")

    global_row: list[int] = []
    for local_rank in range(local_ep_size):  # replica 0: prefix halves
        base = local_rank * local_chunk
        global_row.extend(layer_row[base : base + retained_local_experts])
    for local_rank in range(local_ep_size):  # replica 1: suffix halves
        base = local_rank * local_chunk + (local_chunk - retained_local_experts)
        global_row.extend(layer_row[base : base + retained_local_experts])
    assert len(global_row) == num_physical
    return global_row


def build_global_balloon_layout(
    base_physical_to_logical_row: Sequence[int],
    *,
    local_ep_size: int,
    retained_local_experts: int,
    num_replicas: int = 2,
) -> GlobalBalloonLayout:
    if num_replicas != 2:
        raise ValueError("M0 reference only models the symmetric two-replica split.")

    layer_row = list(map(int, base_physical_to_logical_row))
    num_physical = len(layer_row)
    num_logical = max(layer_row) + 1
    global_p2l = build_complementary_global_p2l(
        layer_row,
        local_ep_size=local_ep_size,
        retained_local_experts=retained_local_experts,
    )
    global_world_size = local_ep_size * num_replicas
    if num_physical % global_world_size != 0:
        raise ValueError("num_physical not divisible by global_world_size")
    num_local_physical = num_physical // global_world_size

    # static remap: logical id -> GLOBAL physical id (inverse of global_p2l).
    logical_to_physical = [-1] * num_logical
    for phys, logical in enumerate(global_p2l):
        if logical_to_physical[logical] != -1:
            raise ValueError(f"logical expert {logical} appears twice in GLOBAL map")
        logical_to_physical[logical] = phys
    if any(p < 0 for p in logical_to_physical):
        raise ValueError("GLOBAL map does not cover every logical expert exactly once")

    # per-rank GLOBAL-physical -> compact local row (reuse the real runtime helper).
    dispatcher_physical_mapping = [
        build_dispatcher_physical_expert_mapping(
            num_physical_experts=num_physical,
            runtime_ep_rank=r,
            active_local_expert_mapping=list(range(num_local_physical)),
        )
        for r in range(global_world_size)
    ]

    return GlobalBalloonLayout(
        num_logical_experts=num_logical,
        num_physical_experts=num_physical,
        local_ep_size=local_ep_size,
        num_replicas=num_replicas,
        retained_local_experts=retained_local_experts,
        global_world_size=global_world_size,
        num_local_physical=num_local_physical,
        global_physical_to_logical=global_p2l,
        logical_to_physical=logical_to_physical,
        dispatcher_physical_mapping=dispatcher_physical_mapping,
    )


def reference_moe_plain(
    hidden: torch.Tensor,  # [M, H]
    topk_logical: torch.Tensor,  # [M, K] logical expert ids
    topk_weights: torch.Tensor,  # [M, K]
    expert_weight_by_logical: torch.Tensor,  # [N, H, H]
) -> torch.Tensor:
    """Plain top-k MoE (each token through its top-k logical experts). Oracle."""
    M, K = topk_logical.shape
    out = torch.zeros_like(hidden)
    for t in range(M):
        for k in range(K):
            e = int(topk_logical[t, k].item())
            if e < 0:
                continue
            out[t] += topk_weights[t, k] * (hidden[t] @ expert_weight_by_logical[e])
    return out


def reference_moe_global_routed(
    hidden: torch.Tensor,  # [M, H]
    topk_logical: torch.Tensor,  # [M, K]
    topk_weights: torch.Tensor,  # [M, K]
    expert_weight_by_logical: torch.Tensor,  # [N, H, H]
    layout: GlobalBalloonLayout,
) -> torch.Tensor:
    """MoE computed via the GLOBAL balloon split: each (token, k) is handled by
    the unique rank that retains the logical expert, using that rank's LOCAL
    weight buffer indexed by compact local row. Summing across ranks must equal
    ``reference_moe_plain`` — this is the M0 correctness check end-to-end
    (static remap + owner + local-row indexing together).
    """
    M, K = topk_logical.shape
    H = hidden.shape[1]

    # Each rank's local weight buffer, ordered by compact local row:
    #   local_weights[r][row] = W[ global_p2l[r*num_local_physical + row] ]
    local_weights = []
    for r in range(layout.global_world_size):
        rows = []
        for row in range(layout.num_local_physical):
            phys = r * layout.num_local_physical + row
            logical = layout.global_physical_to_logical[phys]
            rows.append(expert_weight_by_logical[logical])
        local_weights.append(torch.stack(rows, dim=0))  # [L, H, H]

    out = torch.zeros_like(hidden)
    for t in range(M):
        for k in range(K):
            e = int(topk_logical[t, k].item())
            if e < 0:
                continue
            phys = layout.logical_to_physical[e]
            r = phys // layout.num_local_physical
            row = int(layout.dispatcher_physical_mapping[r][phys].item())
            assert row >= 0, f"rank {r} does not map physical {phys} (logical {e})"
            out[t] += topk_weights[t, k] * (hidden[t] @ local_weights[r][row])
    return out
