from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import (
    ceil_div,
    dispose_tensor,
    get_bool_env_var,
    is_cuda,
    is_hip,
    is_npu,
)
from sglang.srt.utils.offloader import get_offloader

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_cuda = is_cuda()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if not (_is_npu or _is_hip) and _is_cuda:
    from sgl_kernel import silu_and_mul


_MASKED_GEMM_FAST_ACT = get_bool_env_var("SGLANG_MASKED_GEMM_FAST_ACT")
_DEEPGEMM_ON_H20 = get_bool_env_var("SGLANG_DEEPGEMM_ON_H20")


# TODO(kaixih@nvidia): ideally we should merge this logic into
# `fill_gateup_input_triton_kernel` to directly generate e8m0 scale.
@torch.compile(disable=_is_hip or _is_npu)
def _cast_to_e8m0_with_rounding_up(x: torch.Tensor) -> torch.Tensor:
    temp = x.to(torch.float32).view(torch.int32)
    exp = torch.bitwise_right_shift(temp, 23)
    mant = torch.bitwise_and(temp, 0x7FFFFF)
    is_ru = torch.logical_and(
        torch.logical_and((mant > 0), (exp != 0xFE)),
        ~torch.logical_and((exp == 0), (mant <= 0x400000)),
    )
    exp = torch.where(is_ru, exp + 1, exp)
    new_x = exp.to(torch.uint8).view(torch.int)
    return new_x.transpose(1, 2).contiguous().transpose(1, 2)


def copy_list_to_gpu_no_ce(arr: List[int]):
    from sgl_kernel.elementwise import copy_to_gpu_no_ce

    tensor_cpu = torch.tensor(arr, dtype=torch.int32, device="cpu")
    tensor_gpu = torch.empty_like(tensor_cpu, device="cuda")
    copy_to_gpu_no_ce(tensor_cpu, tensor_gpu)
    return tensor_gpu


@dataclass
class DeepGemmRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    use_masked_gemm: bool
    masked_m: Optional[torch.Tensor] = None
    expected_m: Optional[int] = None
    m_indices: Optional[torch.Tensor] = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM


@dataclass
class DeepGemmRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM


@dataclass
class DeepGemmMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    use_fp8: bool
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None


class DeepGemmRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        assert self.config.activation == "silu"
        assert self.config.is_gated

    def run(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> DeepGemmRunnerOutput:
        if not runner_input.use_masked_gemm:
            hidden_states = self._run_contiguous_gemm(
                runner_input, quant_info, running_state
            )
        else:
            hidden_states = self._run_masked_gemm(
                runner_input, quant_info, running_state
            )
        return DeepGemmRunnerOutput(hidden_states=hidden_states)

    def _run_contiguous_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.ep_moe.kernels import tma_align_input_scale
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        all_tokens = running_state["all_tokens"]
        hidden_states_device = running_state["hidden_states_device"]
        hidden_states_dtype = running_state["hidden_states_dtype"]
        hidden_states_shape = running_state["hidden_states_shape"]
        m_indices = runner_input.m_indices

        N = quant_info.w13_weight.size(1)
        K = hidden_states_shape[1]
        scale_block_size = 128

        w13_weight_fp8 = (
            quant_info.w13_weight,
            quant_info.w13_scale,
        )
        w2_weight_fp8 = (quant_info.w2_weight, quant_info.w2_scale)

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            hidden_states_scale = tma_align_input_scale(hidden_states_scale)
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (hidden_states, hidden_states_scale),
            w13_weight_fp8,
            gateup_output,
            m_indices,
        )

        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        down_input = torch.empty(
            (
                all_tokens,
                N // 2,
            ),
            device=gateup_output.device,
            dtype=torch.bfloat16,
        )
        silu_and_mul(gateup_output.view(-1, N), down_input)
        del gateup_output

        down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
            down_input,
            scale_block_size,
            column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        del down_input

        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            down_input_scale = tma_align_input_scale(down_input_scale)

        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (down_input_fp8, down_input_scale),
            w2_weight_fp8,
            down_output,
            m_indices,
        )

        return down_output

    def _run_masked_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.moe.ep_moe.kernels import (
            silu_and_mul_masked_post_quant_fwd,
        )
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_8bit,
        )

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale

        hidden_states_device = running_state["hidden_states_device"]

        # GroupGemm-0
        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            if hidden_states_scale.dtype != torch.int:
                b, s_mn, s_k = hidden_states_scale.shape
                assert (
                    s_mn % 4 == 0 and s_k % 4 == 0
                ), f"scales must be aligned to 4, but got ({b}, {s_mn}, {s_k})"
                hidden_states_scale = _cast_to_e8m0_with_rounding_up(
                    hidden_states_scale
                )
        else:
            hidden_states_scale = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(
                hidden_states_scale
            )

        num_groups, m, k = hidden_states.shape
        n = w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (hidden_states, hidden_states_scale),
            (w13_weight, w13_scale),
            gateup_output,
            masked_m,
            expected_m,
        )
        # [MASKED-REF] in-situ GEMM-0 parity: dequant THIS rank's fp8 activation +
        # w13 (per-128x128 block scales) and matmul a clean bf16 reference for one
        # valid (group,token), diff vs the kernel's gateup_output. Localizes whether
        # the masked-GEMM fp8 scale consumption is wrong (-> diverging values, the
        # confirmed garbling mechanism). Uses runner_input.hidden_states_scale (the
        # ORIGINAL logical scale, before the line-254 TMA reorder). KUNSERVE_MASKED_REF
        # gated, ct<=2.
        try:
            import os as _os
            if _os.environ.get("KUNSERVE_DETAIL_LOG") and getattr(type(self), "_kun_mg_ct", 0) < 2:
                import datetime as _dt
                _hsc0 = runner_input.hidden_states_scale  # logical [ng, m, k//128]
                _g = int((masked_m > 0).nonzero()[0].item())
                _ng, _m, _k = hidden_states.shape
                _bk = _k // 128
                _nw = w13_weight.shape[1]                            # 1536
                _bn = _nw // 128
                _w = (w13_weight[_g].float().view(_bn, 128, _bk, 128)
                      * w13_scale[_g].float().view(_bn, 1, _bk, 1)).view(_nw, _k)
                # test first AND last valid token of group _g (token-index/scale
                # misalignment in the packed [.., m, ..] buffer would hit t=last not t=0)
                _last = int(masked_m[_g].item()) - 1
                for _t in sorted(set([0, max(_last, 0)])):
                    _hx = (hidden_states[_g, _t].float().view(_bk, 128)
                           * _hsc0[_g, _t].float().view(_bk, 1)).view(_k)
                    _ref = (_hx @ _w.t()).to(torch.bfloat16)
                    _act = gateup_output[_g, _t]
                    _d = (_ref.float() - _act.float()).abs()
                    with open(_os.environ["KUNSERVE_DETAIL_LOG"], "a", encoding="utf-8") as _f:
                        _f.write(
                            f"[{_dt.datetime.now()} pid={_os.getpid()}] [KUNSERVE-DBG] "
                            f"[MASKED-REF] g={_g} t={_t} masked_m_g={_last + 1} "
                            f"hsc_shape={tuple(_hsc0.shape)} "
                            f"gemm0_max_abs_diff={_d.max().item():.4f} gemm0_mean_abs_diff={_d.mean().item():.5f} "
                            f"ref_mean_abs={_ref.float().abs().mean().item():.4f} "
                            f"act_mean_abs={_act.float().abs().mean().item():.4f} "
                            f"ref[:4]={_ref[:4].float().tolist()} act[:4]={_act[:4].float().tolist()}\n"
                        )
        except Exception as _e:
            try:
                import os as _os2
                with open(_os2.environ.get("KUNSERVE_DETAIL_LOG","/dev/null"), "a", encoding="utf-8") as _f:
                    _f.write(f"[KUNSERVE-DBG] [MASKED-REF] ERROR: {_e!r}\n")
            except Exception:
                pass
        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        # Act
        scale_block_size = 128
        if _MASKED_GEMM_FAST_ACT:
            down_input, down_input_scale = sglang_per_token_group_quant_8bit(
                x=gateup_output,
                dst_dtype=torch.float8_e4m3fn,
                group_size=scale_block_size,
                masked_m=masked_m,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                fuse_silu_and_mul=True,
                enable_v2=True,
            )
        else:
            down_input = torch.empty(
                (
                    gateup_output.shape[0],
                    gateup_output.shape[1],
                    gateup_output.shape[2] // 2,
                ),
                device=hidden_states_device,
                dtype=torch.float8_e4m3fn,
            )
            down_input_scale = torch.empty(
                (
                    gateup_output.shape[0],
                    gateup_output.shape[1],
                    gateup_output.shape[2] // 2 // scale_block_size,
                ),
                device=hidden_states_device,
                dtype=torch.float32,
            )
            silu_and_mul_masked_post_quant_fwd(
                gateup_output,
                down_input,
                down_input_scale,
                scale_block_size,
                masked_m,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
        # [MASKED-GEMM] valid-row probe (part 1): NaN in GEMM-0 output, VALID rows
        # only (rows [0, masked_m[g]) per group). Padding rows are uninitialized so
        # whole-tensor NaN is expected+harmless; only valid-row NaN is the bug.
        _kun_gateup_valid_nan = None
        try:
            import os as _os
            if _os.environ.get("KUNSERVE_DETAIL_LOG") and getattr(type(self), "_kun_mg_ct", 0) < 40:
                _ng, _mtot = gateup_output.shape[0], gateup_output.shape[1]
                _vm = (torch.arange(_mtot, device=gateup_output.device).unsqueeze(0)
                       < masked_m.unsqueeze(1))
                _gv = gateup_output[_vm]
                _kun_gateup_valid_nan = bool(torch.isnan(_gv).any().item()) if _gv.numel() else False
        except Exception:
            _kun_gateup_valid_nan = None
        del gateup_output

        # GroupGemm-1
        n = w2_weight.shape[1]

        if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            down_input_scale = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(
                down_input_scale
            )

        down_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )

        down_gemm_overlap_args = running_state.get("down_gemm_overlap_args", None)
        if down_gemm_overlap_args is None:
            gemm_overlap_args_dict = {}
        else:
            down_gemm_overlap_args.start_event.record()
            max_block_n = (
                160 if (_DEEPGEMM_ON_H20 and runner_input.expected_m <= 64) else 256
            )
            gemm_overlap_args_dict = {
                "overlap_args": down_gemm_overlap_args,
                "max_block_n": max_block_n,
            }

        deep_gemm_return_value = deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (down_input, down_input_scale),
            (w2_weight, w2_scale),
            down_output,
            masked_m,
            expected_m,
            **gemm_overlap_args_dict,
        )
        meta_overlap_args = running_state.get("meta_overlap_args", None)
        if meta_overlap_args is not None:
            block_m, threshold = deep_gemm_return_value
            meta_overlap_args["block_m"] = block_m
            meta_overlap_args["threshold"] = threshold

        # KunServe NaN-padding hygiene (NOT the garbling fix — see below). down_output
        # is torch.empty so its PADDING rows ([masked_m[g], m) per local expert) are
        # NaN. Zeroing them is harmless hygiene, but it does NOT fix the GLOBAL-LL
        # garbling: 2026-06-15 A/B (deepep_ll_graph_fp8_20260615_023450, ZERO_PAD=1)
        # still produced 46/47 finish=length. So low_latency_combine does NOT actually
        # propagate the padding NaN (output is garbled TEXT, not NaN). The real bug is
        # elsewhere (leading hypothesis: slow fp8 accumulation drift over ~10K decode
        # steps — short requests finish coherently, long ones garble). Default OFF to
        # avoid hot-path overhead; KUNSERVE_ZERO_PAD=1 re-enables for experiments.
        if os.environ.get("KUNSERVE_ZERO_PAD", "0") != "0":
            _pad_mask = (
                torch.arange(down_output.shape[1], device=down_output.device).unsqueeze(0)
                >= masked_m.unsqueeze(1)
            )  # [num_groups, m] True on padding rows
            down_output.masked_fill_(_pad_mask.unsqueeze(-1), 0.0)

        # KunServe [MASKED-GEMM] probe: in M2 this masked runner fires ONLY for the
        # GLOBAL LL bundle (LOCAL is StandardDispatcher post-balloon). Comm + remap +
        # fp8-dispatch + TP are all exonerated by parity_deepep_ll_remap.sh, so the
        # last suspect for the GLOBAL-LL garbling is this masked GEMM numeric path.
        # Dump per-stage norms / NaN / masked_m / scale shapes so we can tell a
        # kernel blow-up (NaN/huge) from a subtle scale-layout error (norms sane but
        # output still wrong). KUNSERVE_DETAIL_LOG-gated, sampled.
        try:
            import os as _os
            _dbg = _os.environ.get("KUNSERVE_DETAIL_LOG")
            if _dbg:
                _ct = getattr(type(self), "_kun_mg_ct", 0) + 1
                type(self)._kun_mg_ct = _ct
                if _ct <= 40:
                    import datetime as _dt
                    _mm = masked_m
                    # VALID rows only ([0, masked_m[g]) per group) — the rows combine
                    # actually reads. Padding-row NaN is harmless; valid-row NaN = bug.
                    _vm = (torch.arange(down_output.shape[1], device=down_output.device).unsqueeze(0)
                           < _mm.unsqueeze(1))
                    _dv = down_output[_vm]
                    _dnan_valid = bool(torch.isnan(_dv).any().item()) if _dv.numel() else False
                    _dmean_valid = float(_dv.abs()[~torch.isnan(_dv.abs())].mean().item()) if _dv.numel() else 0.0
                    _dnan_all = bool(torch.isnan(down_output).any().item())
                    _mm_sum = int(_mm.sum().item())
                    _mm_head = _mm[:8].tolist()
                    with open(_dbg, "a", encoding="utf-8") as _f:
                        _f.write(
                            f"[{_dt.datetime.now()} pid={_os.getpid()}] [KUNSERVE-DBG] "
                            f"[MASKED-GEMM] ct={_ct} num_groups={num_groups} m={m} "
                            f"masked_m_sum={_mm_sum} masked_m_head={_mm_head} "
                            f"w13={tuple(w13_weight.shape)} w13_scale={tuple(w13_scale.shape)} "
                            f"gateup_valid_nan={_kun_gateup_valid_nan} "
                            f"down_valid_nan={_dnan_valid} down_valid_mean_abs={_dmean_valid:.4f} "
                            f"down_all_nan={_dnan_all} expected_m={expected_m}\n"
                        )
        except Exception:
            pass

        return down_output

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM


@register_pre_permute("standard", "deep_gemm")
def pre_permute_standard_to_deep_gemm(
    dispatch_output: StandardDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    from sglang.srt.layers.moe.ep_moe.kernels import moe_ep_deepgemm_preprocess

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )
    topk_weights, topk_ids, _ = topk_output

    hidden_states_shape = hidden_states.shape
    hidden_states_dtype = hidden_states.dtype
    hidden_states_device = hidden_states.device
    hidden_states_ref = hidden_states

    topk_weights, topk_ids = topk_weights, topk_ids

    # PreReorder
    masked_m, expected_m, src2dst, hidden_states, hidden_states_scale = (
        moe_ep_deepgemm_preprocess(
            topk_ids,
            runner_config.num_local_experts,
            hidden_states,
            runner_config.top_k,
            quant_info.block_shape,
        )
    )

    dispose_tensor(hidden_states_ref)

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["hidden_states_device"] = hidden_states_device
    running_state["src2dst"] = src2dst

    return DeepGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m,
    )


@register_post_permute("deep_gemm", "standard")
def post_permute_deep_gemm_to_standard(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.ep_moe.kernels import post_reorder_triton_kernel
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states_shape = running_state["hidden_states_shape"]
    hidden_states_dtype = running_state["hidden_states_dtype"]
    hidden_states_device = running_state["hidden_states_device"]
    src2dst = running_state["src2dst"]
    topk_ids = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]

    output = torch.zeros(
        hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
    )
    post_reorder_triton_kernel[(hidden_states_shape[0],)](
        runner_output.hidden_states,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        runner_config.top_k,
        hidden_states_shape[1],
        BLOCK_SIZE=512,
    )

    dispose_tensor(runner_output.hidden_states)

    if runner_config.routed_scaling_factor is not None:
        output *= runner_config.routed_scaling_factor

    return StandardCombineInput(
        hidden_states=output,
    )


@register_pre_permute("deepep_ll", "deep_gemm")
def pre_permute_deepep_ll_to_deep_gemm(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    hidden_states, hidden_states_scale, topk_ids, topk_weights, masked_m, expected_m = (
        dispatch_output
    )

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states.shape
    running_state["hidden_states_dtype"] = hidden_states.dtype
    running_state["hidden_states_device"] = hidden_states.device

    return DeepGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m,
    )


@register_post_permute("deep_gemm", "deepep_ll")
def post_permute_deep_gemm_to_deepep_ll(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_pre_permute("deepep_normal", "deep_gemm")
def pre_permute_deepep_normal_to_deep_gemm(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    from sglang.srt.layers.moe.ep_moe.kernels import ep_scatter

    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output
    assert runner_config.activation == "silu"

    all_tokens = sum(num_recv_tokens_per_expert)
    running_state["all_tokens"] = all_tokens

    K = hidden_states.shape[1]

    hidden_states_shape = hidden_states.shape
    hidden_states_device = hidden_states.device
    hidden_states_dtype = hidden_states.dtype

    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_device"] = hidden_states_device
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    input_tensor = torch.empty(
        (all_tokens, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        # TODO check whether need `zeros`
        input_tensor_scale = torch.zeros(
            (ceil_div(K // 128, 4), all_tokens),
            device=hidden_states.device,
            dtype=torch.int,
        ).transpose(0, 1)
    else:
        input_tensor_scale = torch.empty(
            (all_tokens, K // 128),
            device=hidden_states.device,
            dtype=torch.float32,
        )
    m_indices = torch.empty(all_tokens, device=hidden_states.device, dtype=torch.int32)
    output_index = torch.empty_like(topk_ids)

    if get_offloader().forbid_copy_engine_usage:
        num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(
            num_recv_tokens_per_expert
        )
    else:
        num_recv_tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)

    ep_scatter(
        hidden_states,
        hidden_states_scale,
        topk_ids,
        num_recv_tokens_per_expert_gpu,
        expert_start_loc,
        input_tensor,
        input_tensor_scale,
        m_indices,
        output_index,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )
    dispose_tensor(hidden_states)
    dispose_tensor(hidden_states_scale)

    running_state["output_index"] = output_index

    return DeepGemmRunnerInput(
        hidden_states=input_tensor,
        hidden_states_scale=input_tensor_scale,
        use_masked_gemm=False,
        m_indices=m_indices,
    )


@register_post_permute("deep_gemm", "deepep_normal")
def post_permute_deep_gemm_to_deepep_normal(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPNormalCombineInput:
    from sglang.srt.layers.moe.ep_moe.kernels import ep_gather
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    hidden_states = runner_output.hidden_states
    topk_ids = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]
    output_index = running_state["output_index"]

    gather_out = torch.empty(
        running_state["hidden_states_shape"],
        device=running_state["hidden_states_device"],
        dtype=torch.bfloat16,
    )
    ep_gather(hidden_states, topk_ids, topk_weights, output_index, gather_out)

    return DeepEPNormalCombineInput(
        hidden_states=gather_out,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )
