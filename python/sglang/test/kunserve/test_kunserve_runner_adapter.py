"""M4 skeleton tests: DeepEP grouped <-> triton sorted adapter contract.

The adapter itself is NOT implemented yet (milestone M4); these tests pin the
module's surface (importable, dataclasses + functions exist, unimplemented funcs
fail loudly) and carry an xfail placeholder describing the correctness check to
fill in when M4 lands.
"""
from __future__ import annotations

import pytest
import torch

from sglang.srt.layers.moe.token_dispatcher import kunserve_runner_adapter as A


def test_module_surface_exists():
    assert hasattr(A, "DeepEPGroupedTokens")
    assert hasattr(A, "TritonSortedTokens")
    assert callable(A.deepep_grouped_to_triton_sorted)
    assert callable(A.triton_sorted_to_deepep_grouped)


def test_unimplemented_funcs_fail_loudly():
    grouped = A.DeepEPGroupedTokens(
        x=torch.zeros(2, 4, 8),
        counts=torch.zeros(2, dtype=torch.int32),
        topk_weights=torch.zeros(2, 4),
    )
    with pytest.raises(NotImplementedError, match="M4"):
        A.deepep_grouped_to_triton_sorted(grouped, block_m=16)
    ref = A.TritonSortedTokens(
        sorted_x=torch.zeros(0, 8),
        sorted_token_ids=torch.zeros(0, dtype=torch.int32),
        expert_ids=torch.zeros(0, dtype=torch.int32),
        num_tokens_post_pad=torch.zeros((), dtype=torch.int32),
        block_m=16,
    )
    with pytest.raises(NotImplementedError, match="M4"):
        A.triton_sorted_to_deepep_grouped(torch.zeros(0, 8), ref=ref, grouped=grouped)


@pytest.mark.xfail(reason="M4 not implemented: grouped<->sorted round-trip", strict=True)
def test_m4_round_trip_contract():
    # When M4 lands, this should hold: converting DeepEP grouped tokens to the
    # triton sorted layout and applying an identity expert, then scattering back,
    # reproduces the input grouped tokens (weighted) exactly.
    grouped = A.DeepEPGroupedTokens(
        x=torch.randn(2, 4, 8),
        counts=torch.tensor([3, 2], dtype=torch.int32),
        topk_weights=torch.ones(2, 4),
    )
    sorted_in = A.deepep_grouped_to_triton_sorted(grouped, block_m=16)
    back = A.triton_sorted_to_deepep_grouped(
        sorted_in.sorted_x, ref=sorted_in, grouped=grouped
    )
    # only the valid (masked) rows must match
    assert torch.allclose(back, grouped.x)


if __name__ == "__main__":
    test_module_surface_exists()
    test_unimplemented_funcs_fail_loudly()
    print("KunServe runner-adapter skeleton tests: surface OK (M4 round-trip xfail)")
