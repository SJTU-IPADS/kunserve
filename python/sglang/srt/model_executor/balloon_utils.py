from __future__ import annotations

from typing import Optional, Sequence, Union

import torch


def slice_rank_local_logical_expert_ids(
    *,
    physical_to_logical_map: Union[torch.Tensor, Sequence[Sequence[int]]],
    layer_id: int,
    moe_ep_rank: int,
    num_local_physical_experts: int,
) -> torch.Tensor:
    if not isinstance(physical_to_logical_map, torch.Tensor):
        physical_to_logical_map = torch.tensor(physical_to_logical_map)

    if physical_to_logical_map.dim() != 2 or physical_to_logical_map.numel() == 0:
        raise ValueError(
            "physical_to_logical_map must be a non-empty 2D tensor or sequence."
        )
    if layer_id < 0 or layer_id >= int(physical_to_logical_map.shape[0]):
        raise ValueError(
            f"layer_id {layer_id} is outside [0, {int(physical_to_logical_map.shape[0])})."
        )
    if num_local_physical_experts <= 0:
        raise ValueError(
            "num_local_physical_experts must be positive, got "
            f"{num_local_physical_experts}"
        )

    start = int(moe_ep_rank) * int(num_local_physical_experts)
    end = start + int(num_local_physical_experts)
    if start < 0 or end > int(physical_to_logical_map.shape[1]):
        raise ValueError(
            "Requested rank-local physical expert slice is outside "
            "physical_to_logical_map."
        )

    return physical_to_logical_map[layer_id, start:end].to(dtype=torch.int32).cpu()


def build_dispatcher_local_expert_mapping(
    *,
    num_logical_experts: int,
    local_logical_expert_ids: Union[torch.Tensor, Sequence[int]],
    active_local_expert_mapping: Union[torch.Tensor, Sequence[int]],
) -> torch.Tensor:
    if num_logical_experts <= 0:
        raise ValueError(
            f"num_logical_experts must be positive, got {num_logical_experts}"
        )

    if not isinstance(local_logical_expert_ids, torch.Tensor):
        local_logical_expert_ids = torch.tensor(local_logical_expert_ids)
    if not isinstance(active_local_expert_mapping, torch.Tensor):
        active_local_expert_mapping = torch.tensor(active_local_expert_mapping)

    if local_logical_expert_ids.dim() != 1 or local_logical_expert_ids.numel() == 0:
        raise ValueError(
            "local_logical_expert_ids must be a non-empty 1D tensor or sequence."
        )
    if (
        active_local_expert_mapping.dim() != 1
        or active_local_expert_mapping.numel() == 0
    ):
        raise ValueError(
            "active_local_expert_mapping must be a non-empty 1D tensor or sequence."
        )

    local_logical_expert_ids = local_logical_expert_ids.to(dtype=torch.int32).cpu()
    active_local_expert_mapping = active_local_expert_mapping.to(dtype=torch.int32).cpu()

    start = int(active_local_expert_mapping[0].item())
    expected = torch.arange(
        start,
        start + active_local_expert_mapping.numel(),
        dtype=active_local_expert_mapping.dtype,
    )
    if not torch.equal(active_local_expert_mapping, expected):
        raise ValueError(
            "active_local_expert_mapping must describe a contiguous local expert slice."
        )

    max_local_physical_idx = int(active_local_expert_mapping[-1].item())
    if max_local_physical_idx >= int(local_logical_expert_ids.numel()):
        raise ValueError(
            "active_local_expert_mapping refers to a local physical expert row "
            "outside local_logical_expert_ids."
        )

    dispatcher_local_expert_mapping = torch.full(
        (num_logical_experts,),
        -1,
        dtype=torch.int32,
    )

    for compact_local_idx, local_physical_idx in enumerate(
        active_local_expert_mapping.tolist()
    ):
        logical_expert_id = int(local_logical_expert_ids[local_physical_idx].item())
        if logical_expert_id < 0 or logical_expert_id >= num_logical_experts:
            raise ValueError(
                f"logical expert id {logical_expert_id} is outside "
                f"[0, {num_logical_experts})."
            )
        if int(dispatcher_local_expert_mapping[logical_expert_id].item()) != -1:
            raise ValueError(
                f"logical expert id {logical_expert_id} is assigned more than once."
            )
        dispatcher_local_expert_mapping[logical_expert_id] = compact_local_idx

    return dispatcher_local_expert_mapping


def resolve_balloon_kv_slots_to_expand(
    *,
    max_slots_from_donor: int,
    kv_vmm_headroom_slots: int,
    num_slots_to_expand: Optional[int],
) -> int:
    max_slots_from_donor = int(max_slots_from_donor)
    kv_vmm_headroom_slots = int(kv_vmm_headroom_slots)
    if max_slots_from_donor < 0:
        raise ValueError(
            f"max_slots_from_donor must be non-negative, got {max_slots_from_donor}"
        )
    if kv_vmm_headroom_slots < 0:
        raise ValueError(
            "kv_vmm_headroom_slots must be non-negative, got "
            f"{kv_vmm_headroom_slots}"
        )

    if num_slots_to_expand is None:
        return min(max_slots_from_donor, kv_vmm_headroom_slots)

    requested_slots = int(num_slots_to_expand)
    if requested_slots < 0:
        raise ValueError(
            f"num_slots_to_expand must be non-negative, got {num_slots_to_expand}"
        )
    if requested_slots > max_slots_from_donor:
        raise ValueError(
            "Requested "
            f"{requested_slots} balloon KV slots but donor segments only support "
            f"{max_slots_from_donor}."
        )
    if requested_slots > kv_vmm_headroom_slots:
        raise ValueError(
            "Requested "
            f"{requested_slots} balloon KV slots but KV VMM reserve only supports "
            f"{kv_vmm_headroom_slots}."
        )
    return requested_slots


def resolve_balloon_kv_slots_to_whole_donor_segments(
    *,
    requested_slots: int,
    donor_segment_sizes: Sequence[int],
    allocation_row_bytes: Sequence[int],
    page_size: int,
) -> int:
    requested_slots = int(requested_slots)
    page_size = int(page_size)
    if requested_slots < 0:
        raise ValueError(f"requested_slots must be non-negative, got {requested_slots}")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if requested_slots == 0:
        return 0

    donor_segment_sizes = [int(size) for size in donor_segment_sizes]
    allocation_row_bytes = [int(size) for size in allocation_row_bytes]
    if any(size <= 0 for size in donor_segment_sizes):
        raise ValueError("donor_segment_sizes must contain only positive values.")
    if any(size <= 0 for size in allocation_row_bytes):
        raise ValueError("allocation_row_bytes must contain only positive values.")
    if not donor_segment_sizes or not allocation_row_bytes:
        return 0

    candidate_slots = requested_slots // page_size * page_size
    while candidate_slots > 0:
        donor_idx = 0
        donor_bytes_used = 0
        required_boundary_bytes = 0
        fits = True

        for row_bytes in allocation_row_bytes:
            required_boundary_bytes += candidate_slots * row_bytes
            while (
                donor_bytes_used < required_boundary_bytes
                and donor_idx < len(donor_segment_sizes)
            ):
                donor_bytes_used += donor_segment_sizes[donor_idx]
                donor_idx += 1
            if donor_bytes_used != required_boundary_bytes:
                fits = False
                break

        if fits:
            return candidate_slots
        candidate_slots -= page_size

    return 0
