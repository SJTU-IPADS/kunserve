"""Expert layout planning helpers for the KunServe sidecar manager."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class KunServeLayoutPlan:
    """Per-run two-replica expert sharing plan.

    ``local_*`` fields describe each pre-BALLOON SGLang replica.  ``global_*``
    fields describe the temporary cross-replica GLOBAL EP world used while
    BALLOON is active.  ``replica_active_mappings`` is the compact local expert
    slice retained by each replica after half of the local experts are loaned to
    the peer.
    """

    local_ep_size: int
    local_routed_experts: int
    retained_local_experts: int
    offload_local_experts: int
    global_world_size: int
    global_physical_to_logical_map: list[list[int]]
    replica_active_mappings: list[list[int]]


def build_complementary_physical_to_logical_map(
    base_physical_to_logical_map: Sequence[Sequence[int]],
    *,
    local_ep_size: int,
    retained_local_experts: int,
) -> list[list[int]]:
    """Build the GLOBAL physical->logical expert map for two replicas.

    The current KunServe implementation assumes a symmetric half split.  For
    each local EP rank we first place the prefix half retained by replica 0, then
    the suffix half retained by replica 1.  That keeps the total physical expert
    count unchanged while making the GLOBAL dispatch domain cover both replicas'
    retained rows.
    """

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


def build_layout_plan_from_statuses(
    statuses: Sequence[dict[str, Any]],
    *,
    offload_local_experts: Optional[int],
    num_replicas: int = 2,
) -> KunServeLayoutPlan:
    """Derive a KunServe layout plan from replica ``/kunserve/status`` payloads."""

    if len(statuses) != num_replicas:
        raise ValueError(
            f"Expected {num_replicas} replica status payloads, got {len(statuses)}."
        )

    local_maps = [status.get("local_physical_to_logical_map") for status in statuses]
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

    resolved_offload = (
        int(offload_local_experts)
        if offload_local_experts is not None
        else local_routed_experts // 2
    )
    if resolved_offload <= 0 or resolved_offload >= local_routed_experts:
        raise ValueError(
            f"Invalid offload_local_experts={resolved_offload} for local_routed_experts={local_routed_experts}"
        )
    retained_local_experts = local_routed_experts - resolved_offload
    if retained_local_experts != resolved_offload:
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
        list(range(local_routed_experts - retained_local_experts, local_routed_experts)),
    ]
    plan = KunServeLayoutPlan(
        local_ep_size=local_ep_size,
        local_routed_experts=local_routed_experts,
        retained_local_experts=retained_local_experts,
        offload_local_experts=resolved_offload,
        global_world_size=local_ep_size * num_replicas,
        global_physical_to_logical_map=global_map,
        replica_active_mappings=replica_active_mappings,
    )
    logger.info(
        "[KunServeController] layout plan ready: local_ep_size=%d local_routed=%d retained=%d offload=%d",
        plan.local_ep_size,
        plan.local_routed_experts,
        plan.retained_local_experts,
        plan.offload_local_experts,
    )
    return plan
