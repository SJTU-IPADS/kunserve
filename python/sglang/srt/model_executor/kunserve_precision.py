"""KunServe DeepEP link — precision policy.

Refactors the old hard "kunserve_comm_backend=deepep => moe_runner_backend must
be deep_gemm" assert in ``model_runner.register_balloon_global_runtime_bundle``
into an explicit, testable policy that also recognizes the bf16 path.

Two coherent combinations (and only two):

    dispatch_dtype="fp8"   <->  expert_runner="deep_gemm"   (current, native)
    dispatch_dtype="bf16"  <->  expert_runner="triton"      (M4, future)

DeepEP fp8 dispatch feeds the deep_gemm grouped FP8 GEMM directly; bf16 dispatch
(``use_fp8=False``) must feed the triton fused_moe runner via the M4 grouped->
sorted adapter. ``deep_gemm`` is FP8-only and ``triton`` here is the bf16 path,
so the dispatch dtype and the expert runner are not independent — picking one
fixes the other. The policy validates the pair and rejects everything else
(notably ``auto``, which previously crashed during warmup/cuda-graph capture).

This module is pure (no torch / sglang internals) so it can be unit-tested
without a GPU or a cross-replica process group.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# (dispatch_dtype, expert_runner) -> human description of the path.
_VALID_COMBINATIONS = {
    ("fp8", "deep_gemm"): "fp8 dispatch + deep_gemm grouped FP8 GEMM (native)",
    ("bf16", "triton"): "bf16 dispatch + triton fused_moe (M4 adapter)",
}

# expert_runner -> the dispatch dtype it implies, when dispatch_dtype is unset.
_RUNNER_TO_DISPATCH = {
    "deep_gemm": "fp8",
    "triton": "bf16",
}


@dataclass(frozen=True)
class KunServePrecisionPolicy:
    """Resolved precision for the GLOBAL (cross-replica) DeepEP MoE path."""

    dispatch_dtype: str  # "fp8" | "bf16"
    expert_runner: str  # "deep_gemm" | "triton"

    @property
    def use_fp8_dispatch(self) -> bool:
        """Value for DeepEP ``low_latency_dispatch(..., use_fp8=...)``."""
        return self.dispatch_dtype == "fp8"

    def describe(self) -> str:
        return _VALID_COMBINATIONS[(self.dispatch_dtype, self.expert_runner)]


def _norm(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    return s or None


def resolve_kunserve_precision_policy(
    *,
    moe_runner_backend: Optional[object],
    dispatch_dtype: Optional[object] = None,
    a2a_backend_value: Optional[str] = None,
    allow_bf16_triton: bool = True,
) -> KunServePrecisionPolicy:
    """Resolve and validate the (dispatch_dtype, expert_runner) policy.

    Args:
        moe_runner_backend: server_args.moe_runner_backend ("deep_gemm" | "triton"
            | "auto" | None). Determines the expert runner.
        dispatch_dtype: optional explicit override ("fp8" | "bf16"); if None it is
            derived from the runner (deep_gemm->fp8, triton->bf16).
        a2a_backend_value: for error messages only (e.g. "deepep" / "mooncake").
        allow_bf16_triton: whether the bf16/triton combination is permitted for
            this a2a backend. Mooncake is currently only validated for
            fp8/deep_gemm, so callers pass False for it.

    Raises:
        ValueError: on an unsupported or incoherent combination (the replacement
            for the old hard assert), with the same actionable guidance.
    """
    runner = _norm(moe_runner_backend)
    requested = _norm(dispatch_dtype)

    backend_label = (
        f"moe_a2a_backend={a2a_backend_value!r} " if a2a_backend_value else ""
    )

    if runner not in _RUNNER_TO_DISPATCH:
        raise ValueError(
            "Balloon GLOBAL bundle with kunserve_comm_backend='deepep' "
            f"{backend_label}requires server_args.moe_runner_backend to be "
            "'deep_gemm' (fp8 path) or 'triton' (bf16 path); got "
            f"{moe_runner_backend!r}. Leaving the runner as 'auto' can crash the "
            "scheduler during warmup/cuda-graph capture. Pass "
            "engine_kwargs.sglang.moe_runner_backend=deep_gemm."
        )

    derived = _RUNNER_TO_DISPATCH[runner]
    resolved_dispatch = requested or derived

    combo = (resolved_dispatch, runner)
    if combo not in _VALID_COMBINATIONS:
        valid = ", ".join(
            f"(dispatch_dtype={d}, expert_runner={r})"
            for (d, r) in _VALID_COMBINATIONS
        )
        raise ValueError(
            "Incoherent KunServe DeepEP precision: "
            f"dispatch_dtype={resolved_dispatch!r} with moe_runner_backend={runner!r}. "
            f"deep_gemm is FP8-only and triton is the bf16 path; valid pairs are: "
            f"{valid}."
        )

    if combo == ("bf16", "triton") and not allow_bf16_triton:
        raise ValueError(
            f"{backend_label}does not support the bf16/triton path; "
            "only fp8/deep_gemm is validated for it. Use "
            "moe_runner_backend=deep_gemm, or use moe_a2a_backend=deepep for the "
            "bf16 path."
        )

    return KunServePrecisionPolicy(
        dispatch_dtype=resolved_dispatch,
        expert_runner=runner,
    )
