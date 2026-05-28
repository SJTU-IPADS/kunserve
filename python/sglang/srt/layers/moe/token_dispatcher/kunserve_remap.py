"""Phase G P2: fused dispatch topk_ids remap kernel for CrossReplicaStandardDispatcher.

Replaces a 5–7 op chain inside ``_dispatch_static``
(``(ids >= 0) & (ids < N)`` → ``clamp`` → ``to(long)`` → ``mapping[safe_ids]``
→ ``where(...)`` → ``copy_(...)``) with a single Triton kernel call.

This is a *KunServe-specific* optimization: the remap converts the global
expert ids gathered into the cross-replica union (``_buf_union_topk_ids``)
into per-rank local expert ids (``_buf_union_topk_ids_remapped``).  The
baseline (2x independent TP=2) never does this remap because there is no
cross-replica union; it's pure overhead unique to KunServe GLOBAL EP.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _kunserve_remap_topk_ids_kernel(
    ids_ptr,          # *int32, shape [n_elements]
    mapping_ptr,      # *int32, shape [num_experts]
    out_ptr,          # *int32, shape [n_elements]
    n_elements,
    num_experts,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    ids = tl.load(ids_ptr + offsets, mask=mask, other=0)
    # An id is "valid" if it is a real global expert id (0..num_experts-1).
    # ``-1`` (padding rows produced by dispatch pad / lane all-gather) and
    # any other out-of-range id falls through to -1 in the output, matching
    # the original torch ``where(valid, looked, -1)`` semantics.
    valid = (ids >= 0) & (ids < num_experts)
    safe_ids = tl.where(valid, ids, 0)
    looked = tl.load(mapping_ptr + safe_ids, mask=mask, other=0)
    result = tl.where(valid, looked, -1)
    tl.store(out_ptr + offsets, result, mask=mask)


def fused_remap_topk_ids(
    union_ids: torch.Tensor,
    mapping: torch.Tensor,
    num_experts: int,
    out: torch.Tensor,
) -> None:
    """P2: single-kernel fused remap.

    Semantically equivalent to:

        valid = (union_ids >= 0) & (union_ids < num_experts)
        safe = torch.clamp(union_ids, 0, num_experts - 1).long()
        looked = mapping[safe].to(union_ids.dtype)
        out.copy_(torch.where(valid, looked, -1))

    All tensors must be int32 and on the same CUDA device.  The output is
    written in place into ``out`` (matches the call site that uses a
    pre-allocated, pointer-stable ``_buf_union_topk_ids_remapped``).
    """
    if union_ids.numel() == 0:
        return
    assert union_ids.is_cuda and mapping.is_cuda and out.is_cuda
    assert union_ids.dtype == torch.int32
    assert mapping.dtype == torch.int32
    assert out.dtype == torch.int32
    assert union_ids.numel() == out.numel()
    assert mapping.numel() == int(num_experts)

    n_elements = int(union_ids.numel())
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _kunserve_remap_topk_ids_kernel[grid](
        union_ids,
        mapping,
        out,
        n_elements,
        int(num_experts),
        BLOCK_SIZE=BLOCK_SIZE,
    )
