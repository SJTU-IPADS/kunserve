from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_tp_group,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    get_local_dp_buffer,
    is_allocation_symmetric,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutput, TopKOutputChecker
from sglang.srt.layers.moe.utils import (
    get_moe_runner_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.utils.common import get_bool_env_var, is_hip, is_sm120_supported

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


try:
    if is_sm120_supported():
        from flashinfer import fp4_quantize
    else:
        from sgl_kernel import scaled_fp4_quant as fp4_quantize

    from flashinfer import fp4_quantize as fp4_quantize_flashinfer
except ImportError:
    fp4_quantize = None


class StandardDispatchOutput(NamedTuple):
    """Standard dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_output: TopKOutput

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.STANDARD


assert isinstance(StandardDispatchOutput, DispatchOutput)


class StandardCombineInput(NamedTuple):
    """Standard combine input."""

    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.STANDARD


assert isinstance(StandardCombineInput, CombineInput)


class StandardDispatcher(BaseDispatcher):

    def __init__(
        self,
        moe_runner_config: MoeRunnerConfig,
        moe_ep_size: Optional[int] = None,
        moe_ep_rank: Optional[int] = None,
        local_expert_mapping: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.moe_ep_size = (
            moe_ep_size
            if moe_ep_size is not None
            else get_moe_expert_parallel_world_size()
        )
        self.enable_flashinfer_cutlass_moe = (
            get_moe_runner_backend().is_flashinfer_cutlass()
        )
        self.num_experts = moe_runner_config.num_experts
        self.num_local_shared_experts = moe_runner_config.num_fused_shared_experts
        self.num_local_routed_experts = (
            moe_runner_config.num_local_experts - self.num_local_shared_experts
        )
        self.moe_ep_rank = (
            moe_ep_rank if moe_ep_rank is not None else get_moe_expert_parallel_rank()
        )
        self.local_expert_mapping = self._init_local_expert_mapping(
            local_expert_mapping
        )
        self.active_local_expert_mapping = self.local_expert_mapping

    def _init_local_expert_mapping(
        self, local_expert_mapping: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if local_expert_mapping is None:
            return None
        if not isinstance(local_expert_mapping, torch.Tensor):
            local_expert_mapping = torch.tensor(local_expert_mapping)
        if local_expert_mapping.dim() != 1:
            raise ValueError("local_expert_mapping must be a 1D tensor.")
        if local_expert_mapping.shape[0] != self.num_experts:
            raise ValueError(
                "local_expert_mapping must have one entry for every global expert."
            )
        return local_expert_mapping.to(dtype=torch.int32)

    def _get_or_create_local_expert_mapping(
        self, device: torch.device
    ) -> Optional[torch.Tensor]:
        if self.local_expert_mapping is None:
            self.local_expert_mapping = torch.full(
                (self.num_experts,), -1, dtype=torch.int32, device=device
            )
            self.local_expert_mapping[
                self.moe_ep_rank
                * self.num_local_routed_experts : (self.moe_ep_rank + 1)
                * self.num_local_routed_experts
            ] = torch.arange(
                0, self.num_local_routed_experts, dtype=torch.int32, device=device
            )

            if self.num_local_shared_experts > 0:
                self.local_expert_mapping[-self.num_local_shared_experts :] = (
                    torch.arange(
                        self.num_local_routed_experts,
                        self.num_local_routed_experts + self.num_local_shared_experts,
                        dtype=torch.int32,
                        device=device,
                    )
                )
            self.active_local_expert_mapping = self.local_expert_mapping
        elif self.local_expert_mapping.device != device:
            self.local_expert_mapping = self.local_expert_mapping.to(
                device=device, non_blocking=True
            )
            self.active_local_expert_mapping = self.local_expert_mapping
        return self.local_expert_mapping

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> StandardDispatchOutput:

        if should_use_flashinfer_cutlass_moe_fp4_allgather():
            # all-gather fp4 hidden states
            from flashinfer import nvfp4_block_scale_interleave

            global_scale = self.quant_config.get("input_global_scale", None)
            assert global_scale is not None, "input_global_scale is not set"
            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

            # Quantize before comm, swizzle after.
            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                if hidden_states.shape[0] > 0:
                    x, x_sf = fp4_quantize_flashinfer(
                        hidden_states, global_scale, is_sf_swizzled_layout=False
                    )
                else:
                    x_col = hidden_states.shape[1]
                    x = torch.zeros(
                        0, x_col // 2, dtype=torch.uint8, device=hidden_states.device
                    )
                    x_sf = torch.zeros(
                        0, x_col // 16, dtype=torch.uint8, device=hidden_states.device
                    )
            topk_weights, topk_ids, x, x_sf = get_tp_group().all_gatherv(
                [topk_weights, topk_ids, x, x_sf], sizes=get_dp_global_num_tokens()
            )
            # TODO: fuse into cutlass moe
            x_sf = nvfp4_block_scale_interleave(x_sf)

            hidden_states = x
            hidden_states_scale = x_sf
            topk_output = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=topk_output.router_logits,  # never tested
            )
        else:
            hidden_states = hidden_states
            hidden_states_scale = None

        if (
            self.moe_ep_size > 1
            and not self.enable_flashinfer_cutlass_moe
            and TopKOutputChecker.format_is_standard(topk_output)
        ):
            self._get_or_create_local_expert_mapping(topk_output.topk_ids.device)

        # KUNSERVE [STD-MAP] probe: [EP-REDUCE] proved the all-reduce works
        # (tp_ws=ep_ws=2, both ranks summed) yet the summed MoE output is still
        # ~62% of DeepEP's -> experts are DROPPED before the reduce. Prime suspect
        # is this remap: topk_ids are PHYSICAL (post ExpertLocationDispatchInfo),
        # and the naive rank*N local_expert_mapping may send experts that ARE local
        # to -1. Capture the ORIGINAL ids here (pre-remap) so the post-remap block
        # can report how many survive. D2H .item() is illegal during capture.
        import os as _os

        _stdmap_orig = None
        if (
            _os.environ.get("KUNSERVE_DETAIL_LOG")
            and not torch.cuda.is_current_stream_capturing()
            and getattr(type(self), "_kun_stdmap_ct", 0) < 4
            and self.local_expert_mapping is not None
            and TopKOutputChecker.format_is_standard(topk_output)
        ):
            _stdmap_orig = topk_output.topk_ids

        if self.local_expert_mapping is not None and not _use_aiter:
            self._get_or_create_local_expert_mapping(topk_output.topk_ids.device)
            if TopKOutputChecker.format_is_standard(topk_output):
                topk_output = topk_output._replace(
                    topk_ids=self.local_expert_mapping[topk_output.topk_ids]
                )
            elif TopKOutputChecker.format_is_triton_kernels(topk_output):
                raise NotImplementedError()

        if _stdmap_orig is not None:
            try:
                import datetime as _dt

                type(self)._kun_stdmap_ct = getattr(type(self), "_kun_stdmap_ct", 0) + 1
                _m = self.local_expert_mapping
                _ntok = int(_stdmap_orig.shape[0]) if _stdmap_orig.dim() >= 1 else -1
                _total = int(_stdmap_orig.numel())
                # rightful-local = original physical id in this rank's owned window
                _lo = self.moe_ep_rank * self.num_local_routed_experts
                _hi = _lo + self.num_local_routed_experts
                _rightful = int(
                    ((_stdmap_orig >= _lo) & (_stdmap_orig < _hi)).sum().item()
                )
                _survive = int((topk_output.topk_ids >= 0).sum().item())
                _nz = (_m >= 0).nonzero().flatten()
                _glo_lo = int(_nz.min().item()) if _nz.numel() else -1
                _glo_hi = int(_nz.max().item()) if _nz.numel() else -1
                _omin = int(_stdmap_orig.min().item())
                _omax = int(_stdmap_orig.max().item())
                # Dump the SAME tokens' raw physical topk_ids on every rank: if
                # rank0 and rank1 disagree on token-0's 8 experts, the topk /
                # ExpertLocationDispatchInfo routing is rank-inconsistent (each
                # rank computes a different routing -> all-reduce sums garbage).
                _t0 = _stdmap_orig[0].tolist() if _ntok > 0 else []
                _t1 = _stdmap_orig[1].tolist() if _ntok > 1 else []
                with open(
                    _os.environ["KUNSERVE_DETAIL_LOG"], "a", encoding="utf-8"
                ) as _fp:
                    _fp.write(
                        f"[{_dt.datetime.now()} pid={_os.getpid()}] [KUNSERVE-DBG] "
                        f"[STD-MAP] ep_rank={self.moe_ep_rank} num_experts={self.num_experts} "
                        f"num_local_routed={self.num_local_routed_experts} "
                        f"map_valid={int((_m >= 0).sum().item())} "
                        f"map_global_range=[{_glo_lo},{_glo_hi}] "
                        f"orig_id_range=[{_omin},{_omax}] "
                        f"rightful_local={_rightful} survive={_survive} total={_total} "
                        f"ntok={_ntok} survive_per_tok={_survive/max(_ntok,1):.2f} "
                        f"rightful_per_tok={_rightful/max(_ntok,1):.2f} "
                        f"orig_tok0={_t0} orig_tok1={_t1}\n"
                    )
            except Exception:
                pass

        return StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            topk_output=topk_output,
        )

    def combine(self, combine_input: StandardCombineInput) -> torch.Tensor:
        (hidden_states,) = combine_input
        if should_use_flashinfer_cutlass_moe_fp4_allgather():
            hidden_states, global_hidden_states = get_local_dp_buffer(), hidden_states
            get_tp_group().reduce_scatterv(
                global_hidden_states,
                output=hidden_states,
                sizes=get_dp_global_num_tokens(),
            )
        return hidden_states
