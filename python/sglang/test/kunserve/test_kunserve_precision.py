"""Unit tests for the KunServe DeepEP precision policy.

Pure CPU (no torch/GPU/process-group). Pins the refactor of the old hard
"deepep => deep_gemm" assert into the (fp8->deep_gemm / bf16->triton) policy.

Run:
    python -m pytest python/sglang/test/kunserve/test_kunserve_precision.py -q
or standalone:
    python python/sglang/test/kunserve/test_kunserve_precision.py
"""
from __future__ import annotations

import pytest

from sglang.srt.model_executor.kunserve_precision import (
    KunServePrecisionPolicy,
    resolve_kunserve_precision_policy,
)


def test_deep_gemm_defaults_to_fp8():
    p = resolve_kunserve_precision_policy(moe_runner_backend="deep_gemm")
    assert p == KunServePrecisionPolicy(dispatch_dtype="fp8", expert_runner="deep_gemm")
    assert p.use_fp8_dispatch is True
    assert isinstance(p.describe(), str) and p.describe()


def test_triton_defaults_to_bf16():
    p = resolve_kunserve_precision_policy(moe_runner_backend="triton")
    assert p == KunServePrecisionPolicy(dispatch_dtype="bf16", expert_runner="triton")
    assert p.use_fp8_dispatch is False


def test_explicit_matching_dtype_ok():
    assert (
        resolve_kunserve_precision_policy(
            moe_runner_backend="deep_gemm", dispatch_dtype="fp8"
        ).dispatch_dtype
        == "fp8"
    )
    assert (
        resolve_kunserve_precision_policy(
            moe_runner_backend="triton", dispatch_dtype="bf16"
        ).dispatch_dtype
        == "bf16"
    )


def test_incoherent_dtype_runner_rejected():
    # deep_gemm is FP8-only; bf16 with it is incoherent
    with pytest.raises(ValueError, match="Incoherent"):
        resolve_kunserve_precision_policy(
            moe_runner_backend="deep_gemm", dispatch_dtype="bf16"
        )
    # triton is the bf16 path; fp8 with it is incoherent
    with pytest.raises(ValueError, match="Incoherent"):
        resolve_kunserve_precision_policy(
            moe_runner_backend="triton", dispatch_dtype="fp8"
        )


def test_auto_and_none_runner_rejected():
    for bad in ("auto", None, "", "cutlass"):
        with pytest.raises(ValueError, match="deep_gemm.*triton|triton.*deep_gemm"):
            resolve_kunserve_precision_policy(moe_runner_backend=bad)


def test_mooncake_disallows_bf16_triton():
    # allow_bf16_triton=False models mooncake (only fp8/deep_gemm validated).
    with pytest.raises(ValueError, match="bf16/triton"):
        resolve_kunserve_precision_policy(
            moe_runner_backend="triton",
            a2a_backend_value="mooncake",
            allow_bf16_triton=False,
        )
    # but fp8/deep_gemm is fine for mooncake
    p = resolve_kunserve_precision_policy(
        moe_runner_backend="deep_gemm",
        a2a_backend_value="mooncake",
        allow_bf16_triton=False,
    )
    assert p.expert_runner == "deep_gemm" and p.dispatch_dtype == "fp8"


def test_case_and_whitespace_insensitive():
    p = resolve_kunserve_precision_policy(
        moe_runner_backend="  Deep_GEMM ", dispatch_dtype=" FP8 "
    )
    assert p == KunServePrecisionPolicy(dispatch_dtype="fp8", expert_runner="deep_gemm")


def test_backward_compatible_default_is_fp8_deep_gemm():
    # The pre-refactor behavior: deepep + deep_gemm was the only accepted combo.
    # It must still resolve to exactly fp8/deep_gemm with no overrides.
    p = resolve_kunserve_precision_policy(
        moe_runner_backend="deep_gemm", a2a_backend_value="deepep"
    )
    assert (p.dispatch_dtype, p.expert_runner) == ("fp8", "deep_gemm")


if __name__ == "__main__":
    test_deep_gemm_defaults_to_fp8()
    test_triton_defaults_to_bf16()
    test_explicit_matching_dtype_ok()
    test_incoherent_dtype_runner_rejected()
    test_auto_and_none_runner_rejected()
    test_mooncake_disallows_bf16_triton()
    test_case_and_whitespace_insensitive()
    test_backward_compatible_default_is_fp8_deep_gemm()
    print("KunServe precision-policy tests: ALL PASSED")
