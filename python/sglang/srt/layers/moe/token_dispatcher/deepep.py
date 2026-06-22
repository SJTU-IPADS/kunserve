from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple, Union

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    BaseDispatcherConfig,
    CombineInput,
    CombineInputFormat,
    DispatcherBaseHooks,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    get_deepep_config,
    get_moe_runner_backend,
    is_tbo_enabled,
)
from sglang.srt.utils import (
    get_bool_env_var,
    is_blackwell,
    is_hip,
    is_npu,
    load_json_config,
)

_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import CombineOverlapArgs

try:
    from deep_ep import Buffer, Config

    if not _is_npu:
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

    use_deepep = True
except ImportError:
    use_deepep = False

from enum import Enum, IntEnum, auto

import torch
import torch.distributed as dist

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

logger = logging.getLogger(__name__)


class DeepEPPDispatchHooks(DispatcherBaseHooks):

    def __call__(self, dispatcher: BaseDispatcher):
        for hook_fun in self.hook_dict.values():
            hook_fun(dispatcher)


class DeepEPNormalDispatchOutput(NamedTuple):
    """DeepEP normal dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_NORMAL


class DeepEPLLDispatchOutput(NamedTuple):
    """DeepEP low latency dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    masked_m: torch.Tensor
    expected_m: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_LL


assert isinstance(DeepEPNormalDispatchOutput, DispatchOutput)
assert isinstance(DeepEPLLDispatchOutput, DispatchOutput)


class DeepEPNormalCombineInput(NamedTuple):
    """DeepEP normal combine input."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_NORMAL


class DeepEPLLCombineInput(NamedTuple):
    """DeepEP low latency combine input."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_LL


assert isinstance(DeepEPNormalCombineInput, CombineInput)
assert isinstance(DeepEPLLCombineInput, CombineInput)


class DeepEPDispatchMode(IntEnum):
    NORMAL = auto()
    LOW_LATENCY = auto()


@dataclass
class _DeepEPBufferEntry:
    buffer: Buffer
    hidden_size: int
    param_bytes: int
    num_max_dispatch_tokens_per_rank: int
    num_experts: int


class DeepEPBuffer:
    _buffer_cache: dict[int, dict[DeepEPDispatchMode, _DeepEPBufferEntry]] = {}
    _dispatch_mode_by_group: dict[int, DeepEPDispatchMode] = {}
    _default_dispatch_mode: Optional[DeepEPDispatchMode] = None

    @staticmethod
    def _group_key(group: dist.ProcessGroup) -> int:
        return id(group)

    @classmethod
    def _resolve_dispatch_mode(
        cls,
        group: dist.ProcessGroup,
        deepep_mode: DeepEPMode,
        dispatch_mode: Optional[DeepEPDispatchMode],
    ) -> DeepEPDispatchMode:
        if dispatch_mode is not None:
            return dispatch_mode
        if deepep_mode == DeepEPMode.NORMAL:
            return DeepEPDispatchMode.NORMAL
        if deepep_mode == DeepEPMode.LOW_LATENCY:
            return DeepEPDispatchMode.LOW_LATENCY
        if deepep_mode == DeepEPMode.AUTO:
            group_key = cls._group_key(group)
            return cls._dispatch_mode_by_group.get(
                group_key,
                cls._default_dispatch_mode or DeepEPDispatchMode.NORMAL,
            )
        raise NotImplementedError(f"Unsupported DeepEP mode: {deepep_mode}")

    @classmethod
    def get_deepep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        param_bytes: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int = -1,
        num_experts: int = -1,
        dispatch_mode: Optional[DeepEPDispatchMode] = None,
    ):
        resolved_dispatch_mode = cls._resolve_dispatch_mode(
            group=group,
            deepep_mode=deepep_mode,
            dispatch_mode=dispatch_mode,
        )
        group_key = cls._group_key(group)
        cls._dispatch_mode_by_group[group_key] = resolved_dispatch_mode
        group_cache = cls._buffer_cache.setdefault(group_key, {})
        entry = group_cache.get(resolved_dispatch_mode)
        if entry is not None:
            if entry.hidden_size != hidden_size or entry.param_bytes != param_bytes:
                raise RuntimeError(
                    "DeepEPBuffer cache was initialized with a different hidden_size or param_bytes "
                    f"for group {group_key} and mode {resolved_dispatch_mode}."
                )
            if (
                resolved_dispatch_mode == DeepEPDispatchMode.LOW_LATENCY
                and (
                    entry.num_max_dispatch_tokens_per_rank
                    != num_max_dispatch_tokens_per_rank
                    or entry.num_experts != num_experts
                )
            ):
                raise RuntimeError(
                    "DeepEPBuffer low-latency cache was initialized with different token/expert settings "
                    f"for group {group_key}."
                )
            return entry.buffer

        resolved_deepep_mode = (
            DeepEPMode.NORMAL
            if resolved_dispatch_mode == DeepEPDispatchMode.NORMAL
            else DeepEPMode.LOW_LATENCY
        )

        num_nvl_bytes, num_rdma_bytes = 0, 0
        if resolved_deepep_mode.enable_normal():
            hidden_bytes = hidden_size * param_bytes
            for config in (
                DeepEPConfig.get_instance().normal_dispatch_config
                or Buffer.get_dispatch_config(group.size()),
                DeepEPConfig.get_instance().normal_combine_config
                or Buffer.get_combine_config(group.size()),
            ):
                num_nvl_bytes = max(
                    config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                    num_nvl_bytes,
                )
                num_rdma_bytes = max(
                    config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                    num_rdma_bytes,
                )
        if resolved_deepep_mode.enable_low_latency():
            assert num_max_dispatch_tokens_per_rank != -1
            assert num_experts != -1 and num_experts % group.size() == 0
            num_rdma_bytes = max(
                Buffer.get_low_latency_rdma_size_hint(
                    num_max_dispatch_tokens_per_rank,
                    hidden_size,
                    group.size(),
                    num_experts,
                ),
                num_rdma_bytes,
            )

        # We should calculate num_qps_per_rank consistently with DeepEP's test script logic:
        if resolved_deepep_mode == DeepEPMode.NORMAL:
            # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_internode.py#L235
            num_qps_per_rank = DeepEPConfig.get_instance().num_sms
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
            # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_low_latency.py#L176
            num_qps_per_rank = num_experts // group.size()
        else:
            raise NotImplementedError

        if not _is_npu:
            total_num_sms = torch.cuda.get_device_properties(
                device="cuda"
            ).multi_processor_count
            if (
                (resolved_deepep_mode != DeepEPMode.LOW_LATENCY)
                and not is_tbo_enabled()
                and (DeepEPConfig.get_instance().num_sms < total_num_sms // 2)
            ):
                logger.warning(
                    f"Only use {DeepEPConfig.get_instance().num_sms} SMs for DeepEP communication. "
                    f"This may result in highly suboptimal performance. "
                    f"Consider using --deepep-config to change the behavior."
                )

        # [M1-BUF] ground-truth probe right before the crashing Buffer() sync.
        # The isolated repro matches every Buffer param + GPU masking and PASSES,
        # so the differentiator is the real process's device/group state. Dump it.
        try:
            import os as _os, datetime as _dt
            from torch import distributed as _d
            _p = _os.environ.get("KUNSERVE_DETAIL_LOG")
            if _p:
                try:
                    _ranks = _d.get_process_group_ranks(group)
                except Exception:
                    _ranks = "?"
                try:
                    _bk = _d.get_backend(group)
                except Exception:
                    _bk = "?"
                try:
                    from sglang.srt.layers.dp_attention import (
                        get_is_extend_in_batch as _gie,
                    )
                    _isext = _gie()
                except Exception:
                    _isext = "?"
                with open(_p, "a", encoding="utf-8") as _f:
                    _f.write(
                        f"[{_dt.datetime.now()} pid={_os.getpid()}] [KUNSERVE-DBG] "
                        f"[M1-BUF] pre-Buffer is_extend={_isext} "
                        f"resolved_dispatch_mode={resolved_dispatch_mode} "
                        f"ll={resolved_deepep_mode.enable_low_latency()} "
                        f"nvl={num_nvl_bytes} rdma={num_rdma_bytes} qps={num_qps_per_rank} "
                        f"grp_size={group.size()} grp_rank={group.rank()} ranks={_ranks} "
                        f"backend={_bk} cur_dev={torch.cuda.current_device()} "
                        f"dev_count={torch.cuda.device_count()} "
                        f"CVD={_os.environ.get('CUDA_VISIBLE_DEVICES')}\n"
                    )
        except Exception:
            pass

        # How many DeepEP buffers already exist on THIS group? If a NORMAL buffer
        # was created earlier (fwd=1) and we are now creating the LL buffer
        # (fwd=2), this is the 2nd NVSHMEM-using buffer on the same group ->
        # suspected double-init deadlock. Log the existing modes.
        _existing = list(group_cache.keys())
        buffer = Buffer(
            group,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode=resolved_deepep_mode.enable_low_latency(),
            num_qps_per_rank=num_qps_per_rank,
            # TODO can be false when unneeded
            allow_mnnvl=True,
        )
        try:
            import os as _os2, datetime as _dt2
            _p2 = _os2.environ.get("KUNSERVE_DETAIL_LOG")
            if _p2:
                with open(_p2, "a", encoding="utf-8") as _f2:
                    _f2.write(
                        f"[{_dt2.datetime.now()} pid={_os2.getpid()}] [KUNSERVE-DBG] "
                        f"[M1-BUF] POST Buffer() returned mode={resolved_dispatch_mode} "
                        f"existing_modes_before={_existing}\n"
                    )
        except Exception:
            pass
        group_cache[resolved_dispatch_mode] = _DeepEPBufferEntry(
            buffer=buffer,
            hidden_size=hidden_size,
            param_bytes=param_bytes,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            num_experts=num_experts,
        )
        return buffer

    @classmethod
    def clean_buffer(
        cls,
        group: Optional[dist.ProcessGroup] = None,
        dispatch_mode: Optional[DeepEPDispatchMode] = None,
    ):
        if group is None:
            return
        group_key = cls._group_key(group)
        resolved_dispatch_mode = dispatch_mode or cls._dispatch_mode_by_group.get(
            group_key
        )
        if resolved_dispatch_mode != DeepEPDispatchMode.LOW_LATENCY:
            return
        entry = cls._buffer_cache.get(group_key, {}).get(resolved_dispatch_mode)
        if entry is None or not entry.buffer.low_latency_mode:
            return
        entry.buffer.clean_low_latency_buffer(
            entry.num_max_dispatch_tokens_per_rank,
            entry.hidden_size,
            entry.num_experts,
        )

    @classmethod
    def set_dispatch_mode_as_normal(cls, group: Optional[dist.ProcessGroup] = None):
        if group is None:
            cls._default_dispatch_mode = DeepEPDispatchMode.NORMAL
            return
        cls._dispatch_mode_by_group[cls._group_key(group)] = DeepEPDispatchMode.NORMAL

    @classmethod
    def set_dispatch_mode_as_low_latency(
        cls, group: Optional[dist.ProcessGroup] = None
    ):
        if group is None:
            cls._default_dispatch_mode = DeepEPDispatchMode.LOW_LATENCY
            return
        cls._dispatch_mode_by_group[cls._group_key(group)] = (
            DeepEPDispatchMode.LOW_LATENCY
        )

    @classmethod
    def set_dispatch_mode(
        cls,
        mode: Union[DeepEPMode, DeepEPDispatchMode],
        group: Optional[dist.ProcessGroup] = None,
    ):
        if mode == DeepEPDispatchMode.LOW_LATENCY or (
            isinstance(mode, DeepEPMode) and mode.is_low_latency()
        ):
            cls.set_dispatch_mode_as_low_latency(group=group)
        elif mode == DeepEPDispatchMode.NORMAL or (
            isinstance(mode, DeepEPMode) and mode.is_normal()
        ):
            cls.set_dispatch_mode_as_normal(group=group)
        else:
            raise Exception("unsupported mode")


class DeepEPConfig(BaseDispatcherConfig):
    _instance = None

    def __init__(self):
        config_str = get_deepep_config()
        if config_str:
            config_parsed = load_json_config(config_str)
            if torch.distributed.get_rank() == 0:
                logger.info(f"Use DeepEP Config: {config_parsed}")
            config_dispatch = config_parsed["normal_dispatch"]
            config_combine = config_parsed["normal_combine"]

            self.normal_dispatch_config = Config(**config_dispatch)
            self.normal_combine_config = Config(**config_combine)

            assert config_dispatch["num_sms"] == config_combine["num_sms"]
            self.num_sms = config_dispatch["num_sms"]
        else:
            self.normal_dispatch_config = None
            self.normal_combine_config = None
            self.num_sms = Buffer.num_sms

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DeepEPConfig()
        return cls._instance


class _DeepEPDispatcherImplBase:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        deepep_mode: DeepEPMode,
    ):
        if not use_deepep:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )

        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode

        self.params_bytes = 2
        # A large value will lead to large memory occupation, thus users should change it accordingly
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        # DeepEP internode_ll dispatch uses FINISHED_SUM_TAG=1024
        # and the logic requires num-tokens-sent-from-one-rank-to-another-rank less than it
        assert self.num_max_dispatch_tokens_per_rank <= 1024

        self.handle = None

        self.quant_config: Optional[dict] = None

        self.overlap_args: Optional[CombineOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        raise NotImplementedError

    def dispatch_b(self, *args, **kwargs):
        raise NotImplementedError

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError

    def combine_b(self, *args, **kwargs):
        raise NotImplementedError

    def _get_buffer(self):
        raise NotImplementedError

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ) -> None:
        self.overlap_args = combine_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        self.overlap_args = None
        self.meta_overlap_args = None


class _DeepEPDispatcherImplNormal(_DeepEPDispatcherImplBase):
    def __init__(self, async_finish: bool, **kwargs):
        super().__init__(**kwargs)

        self.async_finish = async_finish
        self.src2dst = None
        self.quant_config = {}

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(torch.int64)
        if (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and not get_moe_runner_backend().is_cutlass()
            and not envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        ):
            # TODO hard code 128 block quant,use fp8 communication
            hidden_states = sglang_per_token_group_quant_fp8(
                hidden_states,
                128,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
        previous_event = Buffer.capture() if self.async_finish else None
        return hidden_states, topk_ids, topk_weights, previous_event

    def dispatch_b(self, hidden_states, topk_ids, topk_weights, previous_event):
        (
            hidden_states,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
            event,
        ) = self._dispatch_core(hidden_states, topk_ids, topk_weights, previous_event)
        event.current_stream_wait() if self.async_finish else ()

        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        else:
            hidden_states_scale = None

        return DeepEPNormalDispatchOutput(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
        )

    def _dispatch_core(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        previous_event,
    ):
        buffer = self._get_buffer()
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            topk_ids,
            self.num_experts,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=previous_event is not None,
        )
        # FIXME: `handle` should be transmitted with tokens from dispatch to combine.
        # However, doing this would incur an unknown synchronization error, but keeping
        # `handle` as a member variable works.

        (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            self.handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=(previous_event is not None) and self.async_finish,
            expert_alignment=128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1,
            config=DeepEPConfig.get_instance().normal_dispatch_config,
        )
        get_global_expert_distribution_recorder().on_deepep_dispatch_normal(
            num_recv_tokens_per_expert,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )

        return (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            event,
        )

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):

        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM or _use_aiter or _is_npu:
            output = hidden_states
        else:
            raise NotImplementedError()  # triton runner was supported but it's temporarily disabled

        previous_event = Buffer.capture() if self.async_finish else None
        return output, previous_event

    def combine_b(self, output, previous_event):
        hidden_states, event = self._combine_core(output, previous_event)
        event.current_stream_wait() if self.async_finish else ()
        self.handle = None
        self.src2dst = None
        return hidden_states

    def _combine_core(self, x: torch.Tensor, previous_event):
        buffer = self._get_buffer()
        combined_x, _, event = buffer.combine(
            x,
            self.handle,
            async_finish=self.async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
            config=DeepEPConfig.get_instance().normal_combine_config,
        )
        return combined_x, event

    def _get_buffer(self):
        DeepEPBuffer.set_dispatch_mode_as_normal(group=self.group)

        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            dispatch_mode=DeepEPDispatchMode.NORMAL,
        )


class _DeepEPDispatcherImplLowLatency(_DeepEPDispatcherImplBase):
    def __init__(self, return_recv_hook: bool, **kwargs):
        super().__init__(**kwargs)

        """
        num_max_dispatch_tokens_per_rank: the actual batch size in the decoding engine should be less than 256
        https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
        """
        self.return_recv_hook = return_recv_hook
        self.device_module = torch.get_device_module()
        self.quant_config = {}

    def _ll_async_finish(self) -> bool:
        """Return the async_finish value for DeepEP LL dispatch/combine."""
        if self.return_recv_hook:
            return False

        override = os.environ.get("KUNSERVE_DEEPEP_LL_ASYNC_FINISH")
        if override is not None and override.strip() != "":
            return override.strip().lower() not in ("0", "false", "no", "off")

        # DeepEP low_latency_dispatch creates/returns a DeepEP event when
        # async_finish=True. That path invalidates PyTorch CUDA graph capture in
        # GLOBAL LL warmup on H20. Keep eager behavior unchanged, but use the
        # synchronous DeepEP path while the stream is being captured so we can
        # test whether the event/async layer is the graph blocker.
        try:
            if (
                os.environ.get("KUNSERVE_DEEPEP_LL_GRAPH_SYNC", "1") != "0"
                and torch.cuda.is_current_stream_capturing()
            ):
                return False
        except Exception:
            pass

        return True

    def _wait_ll_recv(self, event, hook) -> None:
        if self.return_recv_hook:
            hook()
            return
        if event is not None and getattr(event, "event", None) is not None:
            event.current_stream_wait()

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        buffer = self._get_buffer()
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(torch.int64)
        expected_m = (
            hidden_states.shape[0] * buffer.group_size * topk_ids.shape[1]
            + self.num_experts
        ) // self.num_experts
        hidden_states, masked_m, event, hook = self._dispatch_core(
            hidden_states,
            topk_ids,
        )
        return (
            hidden_states,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
            event,
            hook,
        )

    def dispatch_b(
        self,
        hidden_states,
        topk_ids,
        topk_weights,
        masked_m,
        expected_m,
        event,
        hook,
    ):
        # Path-B probe: localize the LL-dispatch hang. The cross-replica
        # low_latency_dispatch comm happens inside hook() (return_recv_hook=True)
        # or in low_latency_dispatch itself (False). Mark enter/exit of the wait
        # so a hang shows as ENTER with no EXIT. KUNSERVE_DETAIL_LOG-gated, capped.
        import os as _os

        _dbg = _os.environ.get("KUNSERVE_DETAIL_LOG")
        _ct = getattr(type(self), "_kun_ll_dispb_ct", 0) + 1
        type(self)._kun_ll_dispb_ct = _ct
        if _dbg and _ct <= 40:
            try:
                import datetime as _dt

                with open(_dbg, "a", encoding="utf-8") as _f:
                    _f.write(
                        f"[{_dt.datetime.now()} pid={_os.getpid()}] [KUNSERVE-DBG] "
                        f"[LL-DISP] recv_wait ENTER ct={_ct} "
                        f"return_recv_hook={self.return_recv_hook}\n"
                    )
            except Exception:
                pass

        self._wait_ll_recv(event, hook)

        if _dbg and _ct <= 40:
            try:
                import datetime as _dt

                with open(_dbg, "a", encoding="utf-8") as _f:
                    _f.write(
                        f"[{_dt.datetime.now()} pid={_os.getpid()}] [KUNSERVE-DBG] "
                        f"[LL-DISP] recv_wait EXIT ct={_ct}\n"
                    )
            except Exception:
                pass

        get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency(
            masked_m
        )

        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        else:
            hidden_states_scale = None

        deepep_output = DeepEPLLDispatchOutput(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
        )
        return deepep_output

    def _dispatch_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        use_nvfp4 = use_fp8 = False
        input_global_scale = self.quant_config.get("input_global_scale", None)
        if input_global_scale is not None:
            use_nvfp4 = True
        elif not envs.SGLANG_DEEPEP_BF16_DISPATCH.get():
            use_fp8 = True

        buffer = self._get_buffer()
        ll_async_finish = self._ll_async_finish()

        # Path-B instrumentation: dump the exact LL dispatch inputs + bracket the
        # call so a hang shows as pre with no post (and we see the topk routing).
        import os as _os

        _dbg = _os.environ.get("KUNSERVE_DETAIL_LOG")
        _llct = getattr(type(self), "_kun_ll_core_ct", 0) + 1
        type(self)._kun_ll_core_ct = _llct
        _log_ll = bool(_dbg) and _llct <= 30
        if _log_ll:
            try:
                import datetime as _dt

                _ti = topk_ids
                _tmin = int(_ti.min().item())
                _tmax = int(_ti.max().item())
                _nneg = int((_ti < 0).sum().item())
                _noor = int((_ti >= self.num_experts).sum().item())
                with open(_dbg, "a", encoding="utf-8") as _f:
                    _f.write(
                        f"[{_dt.datetime.now()} pid={_os.getpid()}] [KUNSERVE-DBG] "
                        f"[LL-CORE] ct={_llct} PRE low_latency_dispatch "
                        f"x={tuple(hidden_states.shape)} topk={tuple(_ti.shape)} "
                        f"topk_min={_tmin} topk_max={_tmax} n_neg={_nneg} "
                        f"n_oor={_noor} num_max={self.num_max_dispatch_tokens_per_rank} "
                        f"num_experts={self.num_experts} use_fp8={use_fp8} "
                        f"async_finish={ll_async_finish} "
                        f"return_recv_hook={self.return_recv_hook}\n"
                    )
            except Exception:
                pass

        # Phase E probe: catch the GLOBAL-LL decode garbling moment. The capped
        # [LL-CORE] above only covers the first 30 calls (warmup/prefill). Here we
        # fire specifically when topk has any -1 row (n_neg>0) regardless of call
        # index, dumping per-rank token count + WHICH token rows are fully masked.
        # If two TP-partner ranks (same replica) report different token counts or
        # different fully-masked rows on the same decode step => asymmetric-load /
        # idle-keepalive (Phase E) is corrupting the LL fixed-capacity packing.
        if _dbg:
            try:
                _ti2 = topk_ids
                _nneg2 = int((_ti2 < 0).sum().item())
                if _nneg2 > 0:
                    _act = getattr(type(self), "_kun_ll_anom_ct", 0) + 1
                    type(self)._kun_ll_anom_ct = _act
                    if _act <= 80:
                        import datetime as _dt

                        _full_masked = (
                            (_ti2 < 0).all(dim=1).nonzero(as_tuple=False).flatten().tolist()
                        )
                        with open(_dbg, "a", encoding="utf-8") as _f:
                            _f.write(
                                f"[{_dt.datetime.now()} pid={_os.getpid()}] [KUNSERVE-DBG] "
                                f"[LL-ANOM] ct={_act} n_tokens={hidden_states.shape[0]} "
                                f"n_neg={_nneg2} fully_masked_rows={_full_masked[:16]} "
                                f"(n_full={len(_full_masked)}) num_max="
                                f"{self.num_max_dispatch_tokens_per_rank}\n"
                            )
            except Exception:
                pass

        packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
            buffer.low_latency_dispatch(
                hidden_states,
                topk_ids,
                self.num_max_dispatch_tokens_per_rank,
                self.num_experts,
                use_fp8=use_fp8,
                **(dict(use_nvfp4=True) if use_nvfp4 else dict()),
                **(
                    dict(x_global_scale=input_global_scale)
                    if input_global_scale is not None
                    else dict()
                ),
                async_finish=ll_async_finish,
                return_recv_hook=self.return_recv_hook,
                round_scale=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
                use_ue8m0=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
            )
        )
        if _log_ll:
            try:
                import datetime as _dt

                with open(_dbg, "a", encoding="utf-8") as _f:
                    _f.write(
                        f"[{_dt.datetime.now()} pid={_os.getpid()}] [KUNSERVE-DBG] "
                        f"[LL-CORE] ct={_llct} POST low_latency_dispatch returned "
                        f"(host call done; recv via "
                        f"{'hook' if self.return_recv_hook else 'event.wait'})\n"
                    )
            except Exception:
                pass
        return packed_recv_hidden, self.packed_recv_count, event, hook

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        # Probe: compare combine's topk vs dispatch's ([LL-CORE]) + dump per-local
        # -expert recv counts (masked_m) so we can see if tokens landed on the
        # wrong experts (garbled GLOBAL LL decode). Capped, KUNSERVE_DETAIL_LOG.
        import os as _os

        _dbg = _os.environ.get("KUNSERVE_DETAIL_LOG")
        _cmbct = getattr(type(self), "_kun_ll_cmb_ct", 0) + 1
        type(self)._kun_ll_cmb_ct = _cmbct
        if _dbg and _cmbct <= 20:
            try:
                import datetime as _dt

                _rc = getattr(self, "packed_recv_count", None)
                _rc_list = _rc.tolist() if _rc is not None else None
                # D8 inputs: topk_weights row-sum should be ~1 per token; -1 in topk_ids
                # marks padding. Misaligned/zeroed weights or x/topk mismatch -> garble.
                _tw = topk_weights.float()
                _rowsum = _tw.sum(dim=-1)
                _xin = hidden_states
                _xnan = bool(torch.isnan(_xin).any().item()) if _xin is not None else None
                with open(_dbg, "a", encoding="utf-8") as _f:
                    _f.write(
                        f"[{_dt.datetime.now()} pid={_os.getpid()}] [KUNSERVE-DBG] "
                        f"[LL-CMB] ct={_cmbct} combine topk_min={int(topk_ids.min().item())} "
                        f"topk_max={int(topk_ids.max().item())} "
                        f"topk_shape={tuple(topk_ids.shape)} "
                        f"tw_shape={tuple(topk_weights.shape)} "
                        f"tw_rowsum_mean={_rowsum.mean().item():.4f} "
                        f"tw_rowsum_min={_rowsum.min().item():.4f} tw_rowsum_max={_rowsum.max().item():.4f} "
                        f"tw_min={_tw.min().item():.4f} tw_max={_tw.max().item():.4f} "
                        f"x_in_shape={tuple(_xin.shape) if _xin is not None else None} x_in_nan={_xnan} "
                        f"num_experts={self.num_experts} "
                        f"packed_recv_count(per_local_expert)={_rc_list}\n"
                    )
            except Exception:
                pass

        # [LL-MARKER] 方案 A: 诊断 run（KUNSERVE_MARKER=1，会破坏本次生成）。把 combine
        # 输入每个 local group g 全写成它的 global dispatch_id = ep_rank*num_local + g，
        # 那么 combine 正确时 out[t][c] == Σ_{topk_ids[t,k]>=0} topk_weights[t,k]*topk_ids[t,k]。
        # 在 combine_b 验证。若不符 = combine 把 token 映射到了错的 (expert,slot) → 实锤 H1。
        # 这是唯一绕开 input-faithful + 保范数盲点的检查（marker 受控、可逐 token 验算）。
        self._kun_marker = None
        try:
            import os as _os2
            if _os2.environ.get("KUNSERVE_MARKER") == "1" and getattr(type(self), "_kun_mk_ct", 0) < 8:
                type(self)._kun_mk_ct = getattr(type(self), "_kun_mk_ct", 0) + 1
                _nl = hidden_states.shape[0]
                _myrank = self.group.rank()
                _gids = (torch.arange(_nl, device=hidden_states.device, dtype=torch.float32)
                         + float(_myrank * _nl)).view(_nl, 1, 1)
                hidden_states[:] = _gids.to(hidden_states.dtype)   # 覆盖 combine 输入为 marker
                self._kun_marker = (topk_ids.detach().clone(), topk_weights.detach().clone())
        except Exception:
            self._kun_marker = None

        hidden_states, event, hook = self._combine_core(
            hidden_states,
            topk_ids,
            topk_weights,
        )
        return hidden_states, event, hook

    def combine_b(self, hidden_states, event, hook):
        overlap_args = self.overlap_args
        if overlap_args is not None:
            overlap_args.stream.wait_stream(self.device_module.current_stream())

        self._wait_ll_recv(event, hook)

        if overlap_args is not None:
            self.device_module.current_stream().wait_stream(overlap_args.stream)

        # [LL-CMB-OUT] D8 output: combine result (per-token MoE output) after wait.
        # If runner down_output was correct ([MASKED-REF2] small) but THIS is garbage
        # -> the low_latency_combine weighting/alignment/reduce (D8) is the bug.
        import os as _os
        _dbg = _os.environ.get("KUNSERVE_DETAIL_LOG")
        _oct = getattr(type(self), "_kun_ll_cmbout_ct", 0) + 1
        type(self)._kun_ll_cmbout_ct = _oct
        if _dbg and _oct <= 20:
            try:
                import datetime as _dt
                _h = hidden_states
                _f0 = _h.float()
                with open(_dbg, "a", encoding="utf-8") as _ff:
                    _ff.write(
                        f"[{_dt.datetime.now()} pid={_os.getpid()}] [KUNSERVE-DBG] "
                        f"[LL-CMB-OUT] ct={_oct} out_shape={tuple(_h.shape)} "
                        f"out_mean_abs={_f0.abs().mean().item():.4f} out_max_abs={_f0.abs().max().item():.3f} "
                        f"out_nan={bool(torch.isnan(_h).any().item())} "
                        f"row0[:4]={_f0.reshape(-1, _h.shape[-1])[0,:4].tolist()}\n"
                    )
            except Exception:
                pass

        # [LL-MARKER] 验证 combine 的 token 映射（配对 combine_a 注入的 marker）。
        _mk = getattr(self, "_kun_marker", None)
        if _mk is not None:
            self._kun_marker = None
            try:
                import datetime as _dt
                _tids, _tws = _mk                       # dispatch ids + weights, [n_tok, topk]
                _valid = (_tids >= 0).float()
                # 正确时 out[t][c] = Σ_k w[t,k] * dispatch_id_k（marker = dispatch_id）
                _predict = (_valid * _tws.float() * _tids.float().clamp(min=0)).sum(-1)  # [n_tok]
                _actual = hidden_states.float().reshape(_predict.shape[0], -1)[:, 0]      # out[t][0]
                _d = (_predict - _actual).abs()
                _bad = int((_d > 0.5).sum().item())
                _wt = int(_d.argmax().item())
                with open(_dbg or "/dev/null", "a", encoding="utf-8") as _ff:
                    _ff.write(
                        f"[{_dt.datetime.now()} pid={_os.getpid()}] [KUNSERVE-DBG] "
                        f"[LL-MARKER] n_tok={_predict.shape[0]} map_max_abs_diff={_d.max().item():.3f} "
                        f"map_mean_abs_diff={_d.mean().item():.4f} n_bad(>0.5)={_bad} "
                        f"worst_t={_wt} predict={_predict[_wt].item():.3f} actual={_actual[_wt].item():.3f} "
                        f"worst_topk_ids={_tids[_wt].tolist()} worst_w={[round(x,3) for x in _tws[_wt].tolist()]}\n"
                    )
            except Exception as _e:
                try:
                    with open(_os.environ.get("KUNSERVE_DETAIL_LOG", "/dev/null"), "a", encoding="utf-8") as _ff:
                        _ff.write(f"[KUNSERVE-DBG] [LL-MARKER] ERROR: {_e!r}\n")
                except Exception:
                    pass

        return hidden_states

    def _combine_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        buffer = self._get_buffer()
        overlap_args = self.overlap_args
        meta_overlap_args = self.meta_overlap_args

        ctx = nullcontext()
        if overlap_args is not None:
            overlap_args.stream.wait_event(overlap_args.wait_event)
            ctx = torch.cuda.stream(overlap_args.stream)

            if is_blackwell():
                overlap_args_dict = dict(
                    overlap=overlap_args.overlap,
                    src_signals=overlap_args.signal,
                    src_signal_expect_value=overlap_args.threshold,
                )
            else:
                overlap_args_dict = dict(
                    overlap=overlap_args.overlap,
                    packed_recv_count=self.packed_recv_count,
                    comp_signal=overlap_args.signal,
                    block_m=meta_overlap_args["block_m"],
                    threshold=meta_overlap_args["threshold"],
                    num_sms=overlap_args.num_sms,
                )
        else:
            overlap_args_dict = {}
        ll_async_finish = self._ll_async_finish()

        with ctx:
            combined_hidden_states, event, hook = buffer.low_latency_combine(
                x=hidden_states,
                topk_idx=topk_ids,
                topk_weights=topk_weights,
                handle=self.handle,
                async_finish=ll_async_finish,
                return_recv_hook=self.return_recv_hook,
                **overlap_args_dict,
            )

        self.packed_recv_count = self.handle = None
        return combined_hidden_states, event, hook

    def _get_buffer(self):
        DeepEPBuffer.set_dispatch_mode_as_low_latency(group=self.group)
        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            dispatch_mode=DeepEPDispatchMode.LOW_LATENCY,
        )


@dataclass
class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class DeepEPDispatcher(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        super().__init__()

        self.deepep_mode = deepep_mode

        common_kwargs = dict(
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
        )

        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _DeepEPDispatcherImplLowLatency(
                return_recv_hook=return_recv_hook,
                **common_kwargs,
            )
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher = _DeepEPDispatcherImplNormal(
                async_finish=async_finish,
                **common_kwargs,
            )

        self._stage = _Stage.INITIAL
        self._deepep_dispatch_hooks = DeepEPPDispatchHooks()

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)
        if self._deepep_dispatch_hooks is not None:
            self._deepep_dispatch_hooks(self)
        ret = self.dispatch_b()
        return ret

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        inner_state = self._get_impl().dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state

    def dispatch_b(self):
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl().dispatch_b(*inner_state)

    def combine(
        self,
        combine_input: CombineInput,
    ) -> torch.Tensor:
        self.combine_a(combine_input)
        ret = self.combine_b()
        return ret

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        hidden_states, topk_ids, topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl().combine_a(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = inner_state

    def combine_b(self):
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl().combine_b(*inner_state)

    def _get_impl(self) -> _DeepEPDispatcherImplBase:
        is_extend_in_batch = get_is_extend_in_batch()
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
        if resolved_deepep_mode == DeepEPMode.NORMAL:
            return self._normal_dispatcher
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage
        self._stage = new_stage

    def set_quant_config(self, quant_config: dict):
        super().set_quant_config(quant_config)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_quant_config(quant_config)
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_quant_config(quant_config)

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ):
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )

    def clear_overlap_args(self):
        super().clear_overlap_args()
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.clear_overlap_args()
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.clear_overlap_args()

    def register_deepep_dispatch_hook(self, hook):
        return self._deepep_dispatch_hooks.register_hook(hook)
