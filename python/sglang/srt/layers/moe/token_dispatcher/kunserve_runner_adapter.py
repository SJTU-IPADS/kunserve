"""KunServe DeepEP link — M4 skeleton: DeepEP grouped <-> triton sorted adapter.

STATUS: SKELETON / NOT IMPLEMENTED (milestone M4). This module fixes the
*contract* between the DeepEP low-latency dispatch output and the triton
``fused_moe`` runner so the bf16 path (``dispatch_dtype=bf16``,
``expert_runner=triton``) can be built without re-deriving the layout.

Why an adapter is needed
------------------------
DeepEP LL ``low_latency_dispatch`` returns tokens **grouped per local expert**
with a masked valid count per expert (``masked_m``). The deep_gemm grouped FP8
GEMM consumes that layout natively, so the fp8 path needs NO adapter. The triton
``fused_moe`` kernel instead wants tokens **flattened and sorted by expert id**
plus the ``moe_align_block_size`` metadata (``sorted_token_ids``, ``expert_ids``,
``num_tokens_post_pad``). This module converts grouped->sorted before the triton
expert GEMM and sorted->grouped after, so DeepEP combine can scatter outputs
back to origin tokens.

CUDA-graph note: both conversions must be static-shape (use the capture-time
``num_max_dispatch_tokens_per_rank`` bound + masking), never data-dependent host
syncs — same discipline as the rest of the GLOBAL path.

M4 TODO: the exact shapes/dtypes below must be confirmed against
``token_dispatcher/deepep.py`` ``_DeepEPDispatcherImplLowLatency.dispatch_b`` /
``combine_a`` outputs before implementing — do not assume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DeepEPGroupedTokens:
    """DeepEP LL dispatch output for THIS rank's local experts (bf16 path).

    Fields are the M4 contract; exact shapes to be confirmed against deepep.py.
    """

    # Received hidden states grouped by local expert, padded per expert to the
    # capture-time bound. Shape: [num_local_experts, max_tokens_per_expert, hidden].
    x: torch.Tensor
    # Valid token count per local expert (DeepEP ``masked_m``). [num_local_experts].
    counts: torch.Tensor
    # Per-token routing weight to apply at combine. [num_local_experts, max_tokens_per_expert].
    topk_weights: torch.Tensor
    # Opaque DeepEP combine handle (returned by dispatch, consumed by combine).
    handle: Optional[object] = None


@dataclass
class TritonSortedTokens:
    """Input layout the triton ``fused_moe`` grouped GEMM expects."""

    sorted_x: torch.Tensor  # [num_padded_tokens, hidden], tokens sorted by expert
    sorted_token_ids: torch.Tensor  # [num_padded_tokens] -> index back into grouped x
    expert_ids: torch.Tensor  # [num_blocks] expert id per block
    num_tokens_post_pad: torch.Tensor  # scalar
    block_m: int


def deepep_grouped_to_triton_sorted(
    grouped: DeepEPGroupedTokens,
    *,
    block_m: int,
) -> TritonSortedTokens:
    """Gather the per-expert grouped DeepEP tokens into the flat expert-sorted,
    block-padded layout the triton fused_moe kernel consumes (``moe_align``
    semantics), carrying an index map so outputs can be scattered back.

    NOT IMPLEMENTED (M4).
    """
    raise NotImplementedError(
        "KunServe DeepEP bf16/triton adapter (deepep_grouped_to_triton_sorted) "
        "is milestone M4; see kunserve_manager/deepep_link.md."
    )


def triton_sorted_to_deepep_grouped(
    expert_out_sorted: torch.Tensor,
    *,
    ref: TritonSortedTokens,
    grouped: DeepEPGroupedTokens,
) -> torch.Tensor:
    """Scatter the triton expert outputs (in sorted layout) back to the DeepEP
    per-expert grouped layout expected by ``low_latency_combine``.

    NOT IMPLEMENTED (M4).
    """
    raise NotImplementedError(
        "KunServe DeepEP bf16/triton adapter (triton_sorted_to_deepep_grouped) "
        "is milestone M4; see kunserve_manager/deepep_link.md."
    )
