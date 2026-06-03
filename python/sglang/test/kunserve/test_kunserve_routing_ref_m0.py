"""M0 unit tests: cross-replica balloon routing / static-remap correctness.

Pure CPU. Validates (does NOT blindly trust) the existing balloon layout math
by composing the real runtime helpers (``balloon_utils``) through the M0
reference and asserting the end-to-end correctness invariant + numeric
equivalence to a plain full-expert MoE.

Run:
    python -m pytest python/sglang/test/kunserve/test_kunserve_routing_ref_m0.py -q
or standalone:
    python python/sglang/test/kunserve/test_kunserve_routing_ref_m0.py
"""
from __future__ import annotations

import torch

from sglang.srt.layers.moe.token_dispatcher.kunserve_routing_ref import (
    build_complementary_global_p2l,
    build_global_balloon_layout,
    reference_moe_global_routed,
    reference_moe_plain,
)

# (num_logical_experts == num_physical pre-balloon, local_ep_size) configs.
# base map per layer is identity [0..N-1] (each replica holds full experts,
# split across local_ep_size ranks); symmetric half split.
CONFIGS = [
    (8, 2),     # tiny
    (16, 2),    # small
    (128, 2),   # the doc's 2-replica TP=EP=2, 128-expert example
    (64, 2),
]


def _layout(num_experts: int, local_ep_size: int):
    base_row = list(range(num_experts))  # pre-balloon identity physical->logical
    local_chunk = num_experts // local_ep_size
    retained = local_chunk // 2  # symmetric half split
    return build_global_balloon_layout(
        base_row,
        local_ep_size=local_ep_size,
        retained_local_experts=retained,
        num_replicas=2,
    )


def test_complementary_coverage():
    for num_experts, ep in CONFIGS:
        lay = _layout(num_experts, ep)
        # every logical expert covered exactly once across the GLOBAL map
        assert sorted(lay.global_physical_to_logical) == list(range(num_experts))
        # global world = local_ep_size * 2; physical evenly split
        assert lay.global_world_size == ep * 2
        assert lay.num_local_physical * lay.global_world_size == num_experts
        # the two replicas (rank groups [0..ep-1] and [ep..2ep-1]) are complementary
        L = lay.num_local_physical
        r0 = set(lay.global_physical_to_logical[: ep * L])
        r1 = set(lay.global_physical_to_logical[ep * L :])
        assert r0.isdisjoint(r1)
        assert r0 | r1 == set(range(num_experts))


def test_remap_owner_and_local_row_consistency():
    for num_experts, ep in CONFIGS:
        lay = _layout(num_experts, ep)
        L = lay.num_local_physical
        for e in range(num_experts):
            p = lay.logical_to_physical[e]
            r = lay.owner_rank(e)
            # owner derived from physical//L matches the slice that holds e
            assert lay.global_physical_to_logical[p] == e
            assert r * L <= p < (r + 1) * L
            # that rank's dispatcher mapping resolves p to a valid local row;
            # all OTHER ranks return -1 for p (exactly one owner).
            owners = [
                rr
                for rr in range(lay.global_world_size)
                if int(lay.dispatcher_physical_mapping[rr][p].item()) >= 0
            ]
            assert owners == [r], f"e={e} p={p} owners={owners} expected {[r]}"
            assert int(lay.dispatcher_physical_mapping[r][p].item()) == lay.local_row(e)


def test_dispatcher_mapping_partition():
    # Union of each rank's mapped physical ids == all physical ids, disjoint.
    for num_experts, ep in CONFIGS:
        lay = _layout(num_experts, ep)
        seen = torch.zeros(num_experts, dtype=torch.int64)
        for rr in range(lay.global_world_size):
            mapped = (lay.dispatcher_physical_mapping[rr] >= 0).nonzero().flatten()
            seen[mapped] += 1
        assert torch.all(seen == 1), "every physical id owned by exactly one rank"


def test_cross_check_manager_layout():
    # Independent re-derivation must equal the sidecar manager's implementation.
    try:
        from kunserve_manager.layout import (
            build_complementary_physical_to_logical_map,
        )
    except Exception:
        import pytest

        pytest.skip("kunserve_manager not importable in this env")
        return
    for num_experts, ep in CONFIGS:
        base_row = list(range(num_experts))
        retained = (num_experts // ep) // 2
        ours = build_complementary_global_p2l(
            base_row, local_ep_size=ep, retained_local_experts=retained
        )
        theirs = build_complementary_physical_to_logical_map(
            [base_row], local_ep_size=ep, retained_local_experts=retained
        )[0]
        assert ours == theirs


def test_numeric_equivalence_global_vs_plain():
    # The half-split + dispatch must not change the math: GLOBAL-routed MoE
    # output == plain full-expert MoE output.
    torch.manual_seed(0)
    H, K = 16, 4
    for num_experts, ep in CONFIGS:
        lay = _layout(num_experts, ep)
        M = 37
        hidden = torch.randn(M, H, dtype=torch.float64)
        weights = torch.rand(M, K, dtype=torch.float64)
        # distinct top-k logical experts per token
        topk = torch.stack(
            [torch.randperm(num_experts)[:K] for _ in range(M)]
        ).to(torch.int64)
        W = torch.randn(num_experts, H, H, dtype=torch.float64)

        plain = reference_moe_plain(hidden, topk, weights, W)
        routed = reference_moe_global_routed(hidden, topk, weights, W, lay)
        assert torch.allclose(plain, routed, atol=1e-9, rtol=1e-9), (
            f"mismatch for num_experts={num_experts} ep={ep}: "
            f"max|Δ|={(plain - routed).abs().max().item()}"
        )


if __name__ == "__main__":
    test_complementary_coverage()
    test_remap_owner_and_local_row_consistency()
    test_dispatcher_mapping_partition()
    test_cross_check_manager_layout()
    test_numeric_equivalence_global_vs_plain()
    print("M0 routing-reference tests: ALL PASSED")
