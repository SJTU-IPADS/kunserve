# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run the model with cuda graph and torch.compile."""

from __future__ import annotations

import bisect
import datetime
import gc
import inspect
import logging
import os
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import tqdm
from torch.profiler import ProfilerActivity, profile

from sglang.srt.batch_overlap.two_batch_overlap import TboCudaGraphRunnerPlugin
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    graph_capture,
    set_pdmux_status,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_cp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
from sglang.srt.layers.moe.utils import get_deepep_mode, get_moe_a2a_backend
from sglang.srt.kunserve_forward_timing import (
    kunserve_cuda_graph_timing_capture,
    kunserve_graph_internal_timing_enabled,
    kunserve_graph_internal_timing_interval,
    kunserve_timing_log,
    kunserve_timing_scope,
)
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.input_buffers import GraphInputBuffers
from sglang.srt.multiplex.pdmux_context import get_current_stream_idx, get_stream_groups
from sglang.srt.utils import (
    empty_context,
    get_available_gpu_memory,
    get_bool_env_var,
    is_hip,
    log_info_on_rank0,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_compile
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False

_is_hip = is_hip()

logger = logging.getLogger(__name__)


def _kunserve_graph_log(message: str, *args) -> None:
    path = os.environ.get("KUNSERVE_DETAIL_LOG")
    try:
        logger.info(message, *args)
    except Exception:
        pass
    if not path:
        return
    try:
        rendered = message % args if args else message
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(
                f"[{ts} pid={os.getpid()}] [KUNSERVE-DBG] cuda_graph {rendered}\n"
            )
    except Exception:
        pass


if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

# Detect whether the current forward pass is in capture mode
is_capture_mode = False


def get_is_capture_mode():
    return is_capture_mode


@contextmanager
def model_capture_mode():
    global is_capture_mode
    is_capture_mode = True

    yield

    is_capture_mode = False


@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    """
    Optimize garbage collection during CUDA graph capture.
    Clean up, then freeze all remaining objects from being included
    in future collections if GC is disabled during capture.
    """
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
            gc.collect()


def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, MultiPlatformOp):
            if reverse:
                sub.leave_torch_compile()
            else:
                sub.enter_torch_compile(num_tokens=num_tokens)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)


@contextmanager
def patch_model(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    """Patch the model to make it compatible with with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            _to_torch(model, reverse=False, num_tokens=num_tokens)
            backup_ca_comm = tp_group.ca_comm
            # Use custom-allreduce here.
            # We found the custom allreduce is much faster than the built-in allreduce in torch,
            # even with ENABLE_INTRA_NODE_COMM=1.
            # tp_group.ca_comm = None
            yield torch.compile(
                torch.no_grad()(model.forward),
                mode=os.environ.get(
                    "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
                ),
                dynamic=_is_hip and get_bool_env_var("SGLANG_TORCH_DYNAMIC_SHAPE"),
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True, num_tokens=num_tokens)
            tp_group.ca_comm = backup_ca_comm


def set_torch_compile_config():
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

    # FIXME: tmp workaround
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024

    monkey_patch_torch_compile()


def get_batch_sizes_to_capture(model_runner: ModelRunner, num_tokens_per_bs=1):
    server_args = model_runner.server_args
    capture_bs = server_args.cuda_graph_bs

    if max(capture_bs) > model_runner.req_to_token_pool.size:
        # In some cases (e.g., with a small GPU or --max-running-requests), the #max-running-requests
        # is very small. We add more values here to make sure we capture the maximum bs.
        capture_bs += [model_runner.req_to_token_pool.size]

    mul_base = 1

    if server_args.enable_two_batch_overlap:
        mul_base *= 2
        num_tokens_per_bs = 1  # tbo not test, set num_tokens_per_bs to 1

    if require_gathered_buffer(server_args):
        mul_base *= get_attention_tp_size()

    if mul_base % get_attention_cp_size() != 0:
        mul_base *= get_attention_cp_size()

    # Model input token count = bs * num_tokens_per_bs; must be a multiple of attn_tp_size.
    capture_bs = [bs for bs in capture_bs if bs * num_tokens_per_bs % mul_base == 0]

    capture_bs = [bs for bs in capture_bs if bs <= model_runner.req_to_token_pool.size]
    capture_bs = list(sorted(set(capture_bs)))
    assert len(capture_bs) > 0 and capture_bs[0] > 0, f"{capture_bs=}"
    compile_bs = (
        [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs]
        if server_args.enable_torch_compile
        else []
    )
    return capture_bs, compile_bs


# Reuse this memory pool across all cuda graph runners.
global_graph_memory_pool = None


def get_global_graph_memory_pool():
    return global_graph_memory_pool


def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val


class CudaGraphRunner:
    """A CudaGraphRunner runs the forward pass of a model with cuda graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.enable_two_batch_overlap = (
            model_runner.server_args.enable_two_batch_overlap
        )
        self.speculative_algorithm = model_runner.server_args.speculative_algorithm
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size
        self.enable_pdmux = model_runner.server_args.enable_pdmux
        self._kunserve_graph_padding_log_ct = 0

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm = self.dllm_config is not None

        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.num_tokens_per_bs = 1
        if (
            model_runner.spec_algorithm.is_eagle()
            or model_runner.spec_algorithm.is_standalone()
            or model_runner.spec_algorithm.is_ngram()
        ):
            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen")
            else:
                self.capture_forward_mode = ForwardMode.TARGET_VERIFY
                self.num_tokens_per_bs = (
                    self.model_runner.server_args.speculative_num_draft_tokens
                )
        elif self.is_dllm:
            self.capture_forward_mode = ForwardMode.DLLM_EXTEND
            self.num_tokens_per_bs = self.dllm_config.block_size

        # Batch sizes to capture
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(
            model_runner, self.num_tokens_per_bs
        )
        log_info_on_rank0(logger, f"Capture cuda graph bs {self.capture_bs}")
        if KTRANSFORMERS_AVAILABLE:
            KTMoEWrapper.set_capture_batch_sizes(self.capture_bs)

        # If returning hidden states is enabled, set initial capture hidden mode to full to avoid double-capture on startup
        if model_runner.server_args.enable_return_hidden_states:
            self.capture_hidden_mode = CaptureHiddenMode.FULL

        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.model_runner.attn_backend.init_cuda_graph_state(
            self.max_bs, self.max_num_token
        )

        # Init PDMux if needed
        self.maybe_init_pdmux()
        self.seq_len_fill_value = (
            self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
            if self.dllm_config is None
            else self.dllm_config.block_size
        )

        self.encoder_len_fill_value = 0

        if self.enable_torch_compile:
            set_torch_compile_config()

        if self.model_runner.server_args.enable_lora:
            self.model_runner.lora_manager.init_cuda_graph_batch_info(
                max_bs_in_cuda_graph=self.max_bs,
                num_tokens_per_bs=self.num_tokens_per_bs,
            )

        enable_mamba_track = (
            self.model_runner.server_args.enable_mamba_extra_buffer()
            and self.model_runner.spec_algorithm.is_none()
        )

        if self.require_gathered_buffer:
            assert self.require_mlp_tp_gather or self.require_attn_tp_gather
        self.buffers: GraphInputBuffers = GraphInputBuffers.create(
            device=self.device,
            max_bs=self.max_bs,
            max_num_token=self.max_num_token,
            hidden_size=self.model_runner.model_config.hidden_size,
            vocab_size=self.model_runner.model_config.vocab_size,
            dtype=self.model_runner.model_config.dtype,
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            is_encoder_decoder=self.is_encoder_decoder,
            require_mlp_tp_gather=self.require_mlp_tp_gather,
            seq_len_fill_value=self.seq_len_fill_value,
            encoder_len_fill_value=self.encoder_len_fill_value,
            num_tokens_per_bs=self.num_tokens_per_bs,
            cache_loc_dtype=self._cache_loc_dtype(),
            enable_mamba_track=enable_mamba_track,
        )

        self.tbo_plugin = TboCudaGraphRunnerPlugin()

        # Speculative_inference
        if (
            model_runner.spec_algorithm.is_eagle3()
            and model_runner.eagle_use_aux_hidden_state
        ):
            self.model_runner.model.set_eagle3_layers_to_capture()

        # Capture
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def maybe_init_pdmux(self):
        if self.enable_pdmux:
            self.stream_groups = get_stream_groups()
            for attn_backend in self.model_runner.decode_attn_backend_group:
                attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)

    def _normalize_runtime_variant(self, variant: Optional[Union[str, object]] = None):
        runtime_variant = (
            variant
            if variant is not None
            else self.model_runner.get_cuda_graph_runtime_variant()
        )
        runtime_variant = getattr(runtime_variant, "value", runtime_variant)
        return str(runtime_variant).lower()

    def _graph_key(
        self,
        variant: Optional[Union[str, object]],
        bs: int,
        stream_idx: Optional[int] = None,
    ):
        normalized_variant = self._normalize_runtime_variant(variant)
        if stream_idx is None:
            return (normalized_variant, int(bs))
        return (int(stream_idx), normalized_variant, int(bs))

    def has_captured_variant(self, variant: Optional[Union[str, object]]) -> bool:
        normalized_variant = self._normalize_runtime_variant(variant)
        return any(
            (
                key[0] == normalized_variant
                if len(key) == 2
                else key[1] == normalized_variant
            )
            for key in self.graphs
        )

    def get_captured_variants(self) -> list[str]:
        variants = set()
        for key in self.graphs:
            if len(key) == 2:
                variants.add(key[0])
            else:
                variants.add(key[1])
        return sorted(variants)

    def drop_captured_variant(self, variant: Optional[Union[str, object]]) -> int:
        """Drop all graphs for one runtime variant.

        KunServe GLOBAL changes expert/KV VMM mappings during balloon commit.
        A graph captured before that transition can still contain old device
        pointers.  Dropping the variant lets commit recapture against the final
        post-balloon memory layout.
        """
        normalized_variant = self._normalize_runtime_variant(variant)
        keys_to_drop = [
            key
            for key in self.graphs
            if (
                key[0] == normalized_variant
                if len(key) == 2
                else key[1] == normalized_variant
            )
        ]
        for key in keys_to_drop:
            self.graphs.pop(key, None)
            self.output_buffers.pop(key, None)
        if keys_to_drop and normalized_variant == "global":
            _kunserve_graph_log(
                "drop_captured_variant variant=%s dropped=%d remaining_variants=%s",
                normalized_variant,
                len(keys_to_drop),
                self.get_captured_variants(),
            )
        return len(keys_to_drop)

    def ensure_variant_captured(self, variant: Optional[Union[str, object]]) -> None:
        normalized_variant = self._normalize_runtime_variant(variant)
        if self.has_captured_variant(normalized_variant):
            return
        with model_capture_mode():
            self.capture(variants=[normalized_variant], clear_existing=False)

    def _cache_loc_dtype(self):
        return torch.int64

    def _kunserve_graph_bs_override(self, forward_batch: ForwardBatch) -> Optional[int]:
        try:
            runtime_variant = self.model_runner.get_cuda_graph_runtime_variant()
        except Exception:
            return None
        if runtime_variant != "global":
            return None
        if str(getattr(self.model_runner, "_balloon_state", "local")) != "balloon":
            return None
        try:
            if not (
                forward_batch.forward_mode.is_decode()
                or forward_batch.forward_mode.is_idle()
            ):
                return None
        except Exception:
            return None
        getter = getattr(self.model_runner, "get_balloon_step_graph_bs_override", None)
        if getter is None:
            return None
        try:
            target_bs = getter()
        except Exception:
            return None
        if target_bs is None:
            return None
        target_bs = int(target_bs)
        if target_bs <= 0:
            return None
        raw_bs = int(getattr(forward_batch, "batch_size", 0) or 0)
        if target_bs < raw_bs:
            return None
        return target_bs

    def can_run(self, forward_batch: ForwardBatch):
        if not self.model_runner.is_cuda_graph_replay_enabled():
            return False
        graph_bs_override = self._kunserve_graph_bs_override(forward_batch)
        if self.require_mlp_tp_gather:
            cuda_graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                else max(forward_batch.global_num_tokens_cpu)
            )
        else:
            cuda_graph_bs = forward_batch.batch_size
        if graph_bs_override is not None:
            cuda_graph_bs = max(int(cuda_graph_bs), int(graph_bs_override))

        runtime_variant = self.model_runner.get_cuda_graph_runtime_variant()
        stream_idx = get_current_stream_idx() if self.enable_pdmux else None
        if self.disable_padding:
            graph_key = self._graph_key(runtime_variant, cuda_graph_bs, stream_idx)
            is_bs_supported = graph_key in self.graphs
        else:
            index = bisect.bisect_left(self.capture_bs, cuda_graph_bs)
            if index >= len(self.capture_bs):
                is_bs_supported = False
            else:
                padded_bs = self.capture_bs[index]
                graph_key = self._graph_key(runtime_variant, padded_bs, stream_idx)
                is_bs_supported = graph_key in self.graphs

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        # NOTE: cuda graph cannot handle mixed batch (encoder_len = 0)
        # If mixed batch cannot be supported, then encoder_lens can be removed in cuda graph
        # because the full_text_row_masked_out_mask tensor will always be ones
        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )

        requested_capture_hidden_mode = max(
            forward_batch.capture_hidden_mode,
            (
                forward_batch.spec_info.capture_hidden_mode
                if getattr(forward_batch.spec_info, "capture_hidden_mode", None)
                is not None
                else CaptureHiddenMode.NULL
            ),
        )
        capture_hidden_mode_matches = (
            requested_capture_hidden_mode == CaptureHiddenMode.NULL
            or requested_capture_hidden_mode == self.capture_hidden_mode
        )
        is_tbo_supported = (
            forward_batch.can_run_tbo if self.enable_two_batch_overlap else True
        )

        is_ngram_supported = (
            (
                forward_batch.batch_size * self.num_tokens_per_bs
                == forward_batch.input_ids.numel()
            )
            if self.model_runner.spec_algorithm.is_ngram()
            else True
        )

        return (
            is_bs_supported
            and is_encoder_lens_supported
            and is_tbo_supported
            and capture_hidden_mode_matches
            and is_ngram_supported
        )

    def _init_profile_context_and_memory_record(self):
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        )
        torch.cuda.memory._record_memory_history()
        return profile_context

    def _post_process_after_profile(self, prof_context):
        torch.cuda.memory._dump_snapshot(f"cuda_graph_runner_memory_usage.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        log_message = (
            "Sorted by CUDA Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total", row_limit=10
            )
            + "\n\nSorted by CPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="cpu_time_total", row_limit=10
            )
            + "\n\nMemory Usage is saved to cuda_graph_runner_memory_usage.pickle\n"
        )
        logger.info(log_message)

    def capture(
        self,
        *,
        variants: Optional[list[str]] = None,
        clear_existing: bool = True,
    ) -> None:
        if clear_existing:
            self.graphs = {}
            self.output_buffers = {}
        profile_context = empty_context()
        if self.enable_profile_cuda_graph:
            profile_context = self._init_profile_context_and_memory_record()
        capture_variants = variants or self.model_runner.get_cuda_graph_capture_variants()
        capture_variants = [self._normalize_runtime_variant(variant) for variant in capture_variants]

        def _capture_one_stream(
            variant: str, stream_idx: Optional[int] = None
        ):
            avail_mem = get_available_gpu_memory(
                self.model_runner.device,
                self.model_runner.gpu_id,
                empty_cache=False,
            )
            # Reverse the order to enable better memory sharing across cuda graphs.
            capture_range = (
                tqdm.tqdm(list(reversed(self.capture_bs)))
                if get_tensor_model_parallel_rank() == 0
                else reversed(self.capture_bs)
            )
            for i, bs in enumerate(capture_range):
                if get_tensor_model_parallel_rank() == 0:
                    avail_mem = get_available_gpu_memory(
                        self.model_runner.device,
                        self.model_runner.gpu_id,
                        empty_cache=False,
                    )
                    capture_range.set_description(
                        f"Capturing batches ({variant=} {bs=} {avail_mem=:.2f} GB)"
                    )

                with patch_model(
                    self.model_runner.model,
                    bs in self.compile_bs,
                    num_tokens=bs * self.num_tokens_per_bs,
                    tp_group=self.model_runner.tp_group,
                ) as forward:
                    (
                        graph,
                        output_buffers,
                    ) = self.capture_one_batch_size(bs, forward, stream_idx)
                    # For pd_multiplexing, we need to save the graph and output buffers
                    key = self._graph_key(variant, bs, stream_idx)
                    self.graphs[key] = graph
                    self.output_buffers[key] = output_buffers

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc):
            with profile_context as prof:
                if not self.enable_pdmux:
                    with graph_capture() as graph_capture_context:
                        self.stream = graph_capture_context.stream
                        for variant in capture_variants:
                            with self.model_runner.cuda_graph_capture_variant_scope(
                                variant
                            ):
                                _capture_one_stream(variant)
                else:
                    set_pdmux_status(False)
                    for i, sg in enumerate(self.stream_groups):
                        with graph_capture(stream=sg[1]) as graph_capture_context:
                            self.stream = graph_capture_context.stream
                            for variant in capture_variants:
                                with self.model_runner.cuda_graph_capture_variant_scope(
                                    variant
                                ):
                                    _capture_one_stream(variant, i)

        if self.enable_profile_cuda_graph:
            self._post_process_after_profile(prof)

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.model_runner.server_args.enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        graph_fn = (
            partial(memory_saver_adapter.cuda_graph, tag=GPU_MEMORY_TYPE_CUDA_GRAPH)
            if memory_saver_adapter.enabled
            else self.device_module.graph
        )
        with graph_fn(cuda_graph=graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _create_device_graph(self):
        return torch.cuda.CUDAGraph()

    def capture_one_batch_size(
        self, bs: int, forward: Callable, stream_idx: Optional[int] = None
    ):
        buffers: GraphInputBuffers = self.buffers
        graph = self._create_device_graph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs
        try:
            runtime_variant_for_log = self.model_runner.get_cuda_graph_runtime_variant()
        except Exception:
            runtime_variant_for_log = None
        if runtime_variant_for_log == "global":
            _kunserve_graph_log(
                "capture_one_batch_size enter variant=%s bs=%d num_tokens=%d "
                "stream_idx=%s tp_rank=%s graph=%s stream=%s",
                runtime_variant_for_log,
                bs,
                num_tokens,
                stream_idx,
                getattr(self.model_runner, "tp_rank", None),
                hex(id(graph)),
                stream,
            )

        # Graph inputs
        input_ids = buffers.input_ids[:num_tokens]
        req_pool_indices = buffers.req_pool_indices[:bs]
        seq_lens = buffers.seq_lens[:bs]
        seq_lens_cpu = buffers.seq_lens_cpu[:bs]
        out_cache_loc = buffers.out_cache_loc[:num_tokens]
        positions = buffers.positions[:num_tokens]
        if self.is_encoder_decoder:
            encoder_lens = buffers.encoder_lens[:bs]
        else:
            encoder_lens = None
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        next_token_logits_buffer = buffers.next_token_logits_buffer[:num_tokens]
        buffers.num_token_non_padded[...] = num_tokens

        # pipeline parallelism
        if self.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        if self.require_mlp_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.dp_size
        elif self.require_attn_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens
        else:
            global_dp_buffer_len = None

        spec_info = self.get_spec_info(num_tokens)
        if self.capture_hidden_mode != CaptureHiddenMode.FULL:
            self.capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        if self.model_runner.server_args.enable_lora:
            # It is safe to capture CUDA graph using empty LoRA id, as the LoRA kernels will always be launched whenever
            # `--enable-lora` is set to True (and return immediately if the LoRA id is empty for perf optimization).
            lora_ids = [None] * bs
        else:
            lora_ids = None

        # mamba state tracking
        mamba_track_indices = (
            buffers.mamba_track_indices[:bs]
            if buffers.mamba_track_indices is not None
            else None
        )
        mamba_track_mask = (
            buffers.mamba_track_mask[:bs]
            if buffers.mamba_track_mask is not None
            else None
        )

        if stream_idx is None:
            attn_backend = self.model_runner.attn_backend
        else:
            assert self.enable_pdmux
            attn_backend = self.model_runner.decode_attn_backend_group[stream_idx]

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            orig_seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            mamba_track_seqlens=None,  # Prefill only
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            num_token_non_padded=buffers.num_token_non_padded,
            global_forward_mode=self.capture_forward_mode,
            lora_ids=lora_ids,
        )
        self.tbo_plugin.capture_one_batch_size(forward_batch, num_tokens=num_tokens)

        if lora_ids is not None:
            self.model_runner.lora_manager.prepare_lora_batch(forward_batch)

        # Attention backend
        attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )

        # Run and capture
        def run_once():
            # Clean intermediate result cache for DP attention
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            kwargs = {}
            if (
                self.pp_size > 1
                and "pp_proxy_tensors" in inspect.signature(forward).parameters
            ):
                kwargs["pp_proxy_tensors"] = PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )

            logits_output_or_pp_proxy_tensors = forward(
                input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
            return logits_output_or_pp_proxy_tensors

        self.deepep_adapter.capture(is_extend_in_batch=False)

        skip_pre_capture_warmup = False
        try:
            runtime_variant = self.model_runner.get_cuda_graph_runtime_variant()
            skip_pre_capture_warmup = runtime_variant == "global" and bool(
                getattr(self.model_runner, "_kunserve_skip_global_pre_capture_warmup", True)
            )
        except Exception:
            runtime_variant = None
            skip_pre_capture_warmup = False

        if skip_pre_capture_warmup:
            # KunServe GLOBAL uses a different MoE dispatcher/runtime bundle and
            # non-TP registered collectives.  The two generic SGLang warmup
            # forwards run under graph_capture() but before torch.cuda.CUDAGraph
            # capture; their outputs are discarded, and they have repeatedly
            # poisoned CUDA before the actual graph capture begins.  We already
            # preheat the KunServe communicators explicitly, so go straight to
            # the real capture where the dispatcher takes its fixed static path.
            self.device_module.synchronize()
            self.model_runner.tp_group.barrier()
            logger.info(
                "Skip pre-capture warmup forwards for KunServe GLOBAL cuda graph."
            )
            _kunserve_graph_log(
                "skip_pre_capture_warmup variant=%s bs=%d num_tokens=%d "
                "stream_idx=%s tp_rank=%s",
                runtime_variant,
                bs,
                num_tokens,
                stream_idx,
                getattr(self.model_runner, "tp_rank", None),
            )
        else:
            for _ in range(2):
                self.device_module.synchronize()
                self.model_runner.tp_group.barrier()
                run_once()

        if get_global_graph_memory_pool() is None:
            set_global_graph_memory_pool(self.device_module.graph_pool_handle())
        # Set graph pool id globally to be able to use symmetric memory
        set_graph_pool_id(get_global_graph_memory_pool())
        if runtime_variant_for_log == "global":
            _kunserve_graph_log(
                "capture_graph_begin variant=%s bs=%d num_tokens=%d "
                "stream_idx=%s tp_rank=%s pool=%s",
                runtime_variant_for_log,
                bs,
                num_tokens,
                stream_idx,
                getattr(self.model_runner, "tp_rank", None),
                get_global_graph_memory_pool(),
            )
        try:
            graph_key = self._graph_key(runtime_variant_for_log, bs, stream_idx)
            with kunserve_cuda_graph_timing_capture(str(graph_key)):
                out = self._capture_graph(
                    graph, get_global_graph_memory_pool(), stream, run_once
                )
        except Exception as exc:
            if runtime_variant_for_log == "global":
                _kunserve_graph_log(
                    "capture_graph_failed variant=%s bs=%d num_tokens=%d "
                    "stream_idx=%s tp_rank=%s exc=%r",
                    runtime_variant_for_log,
                    bs,
                    num_tokens,
                    stream_idx,
                    getattr(self.model_runner, "tp_rank", None),
                    exc,
                )
            raise
        if runtime_variant_for_log == "global":
            _kunserve_graph_log(
                "capture_graph_complete variant=%s bs=%d num_tokens=%d "
                "stream_idx=%s tp_rank=%s",
                runtime_variant_for_log,
                bs,
                num_tokens,
                stream_idx,
                getattr(self.model_runner, "tp_rank", None),
            )

        return graph, out

    def recapture_if_needed(self, forward_batch: ForwardBatch):

        # If the required capture_hidden_mode changes, we need to recapture the graph

        # These are the different factors that can influence the capture_hidden_mode
        capture_hidden_mode_required_by_forward_batch = (
            forward_batch.capture_hidden_mode
        )
        capture_hidden_mode_required_by_spec_info = (
            getattr(forward_batch.spec_info, "capture_hidden_mode", None)
            or CaptureHiddenMode.NULL
        )
        capture_hidden_mode_required_for_returning_hidden_states = (
            CaptureHiddenMode.FULL
            if self.model_runner.server_args.enable_return_hidden_states
            else CaptureHiddenMode.NULL
        )

        # Determine the highest capture_hidden_mode required
        # (If we have FULL, we can emulate LAST or NULL)
        # (If we have LAST, we can emulate NULL)
        required_capture_hidden_mode = max(
            capture_hidden_mode_required_by_forward_batch,
            capture_hidden_mode_required_by_spec_info,
            capture_hidden_mode_required_for_returning_hidden_states,
        )

        # If the current hidden mode is no longer aligned with the required hidden mode, we need to set it to what is required and re-capture
        if self.capture_hidden_mode != required_capture_hidden_mode:
            self.capture_hidden_mode = required_capture_hidden_mode
            self.capture()

    def replay_prepare(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        buffers = self.buffers
        self.recapture_if_needed(forward_batch)

        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.num_tokens_per_bs

        # Pad
        graph_bs_override = self._kunserve_graph_bs_override(forward_batch)
        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = (
                max_num_tokens / self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                else max_num_tokens
            )
            if graph_bs_override is not None:
                max_batch_size = max(int(max_batch_size), int(graph_bs_override))
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:
            replay_bs = (
                max(int(raw_bs), int(graph_bs_override))
                if graph_bs_override is not None
                else raw_bs
            )
            index = bisect.bisect_left(self.capture_bs, replay_bs)
        bs = self.capture_bs[index]

        seq_lens_cpu = buffers.populate_from_forward_batch(
            forward_batch=forward_batch,
            raw_bs=raw_bs,
            raw_num_token=raw_num_token,
            bs=bs,
            seq_len_fill_value=self.seq_len_fill_value,
            require_gathered_buffer=self.require_gathered_buffer,
            num_tokens_per_bs=self.num_tokens_per_bs,
            nsa_enable_prefill_cp=self.nsa_enable_prefill_cp,
            enable_num_token_non_padded_flag=enable_num_token_non_padded(
                self.model_runner.server_args
            ),
            pp_proxy_tensors=pp_proxy_tensors,
        )
        replay_seq_lens_sum = forward_batch.seq_lens_sum + (
            bs - raw_bs
        ) * self.seq_len_fill_value
        if self._kunserve_patch_global_graph_padding(
            buffers=buffers,
            forward_batch=forward_batch,
            raw_bs=raw_bs,
            raw_num_token=raw_num_token,
            bs=bs,
        ):
            replay_seq_lens_sum = forward_batch.seq_lens_sum + (bs - raw_bs)
            if seq_lens_cpu is not None:
                seq_lens_cpu = buffers.seq_lens_cpu[:bs]
        if self.enable_two_batch_overlap:
            self.tbo_plugin.replay_prepare(
                forward_mode=self.capture_forward_mode,
                bs=bs,
                num_token_non_padded=len(forward_batch.input_ids),
                spec_info=forward_batch.spec_info,
            )
        if forward_batch.forward_mode.is_idle() and forward_batch.spec_info is not None:
            forward_batch.spec_info.custom_mask = buffers.custom_mask
        # Attention backend
        if self.enable_pdmux:
            stream_idx = get_current_stream_idx()
            attn_backend = self.model_runner.decode_attn_backend_group[stream_idx]
        else:
            attn_backend = self.model_runner.attn_backend
        attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            buffers.req_pool_indices[:bs],
            buffers.seq_lens[:bs],
            replay_seq_lens_sum,
            buffers.encoder_lens[:bs] if self.is_encoder_decoder else None,
            self.capture_forward_mode,
            forward_batch.spec_info,
            seq_lens_cpu=seq_lens_cpu,
        )

        # Store fields
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs

    def _kunserve_patch_global_graph_padding(
        self,
        *,
        buffers: GraphInputBuffers,
        forward_batch: ForwardBatch,
        raw_bs: int,
        raw_num_token: int,
        bs: int,
    ) -> bool:
        """Redirect KunServe GLOBAL graph padding rows to reserved dummy state.

        SGLang CUDA graph replay may use a larger captured bucket than the live
        decode batch, e.g. replaying bs=8 for raw_bs=6.  The stock padding path
        resets seq_lens/out_cache_loc but leaves req_pool_indices[raw_bs:bs]
        untouched.  After a request finishes, those stale entries can point at
        a freed req_to_token row whose KV indices are invalid, poisoning CUDA in
        the next graph replay.  KunServe already reserves a phantom req_pool row
        and dummy KV slot for keepalive; use the same pair for padding rows.
        """
        if int(bs) <= int(raw_bs):
            return False
        try:
            runtime_variant = self.model_runner.get_cuda_graph_runtime_variant()
        except Exception:
            runtime_variant = getattr(
                self.model_runner, "_balloon_runtime_variant", "local"
            )
        if str(runtime_variant) != "global":
            return False
        if str(getattr(self.model_runner, "_balloon_state", "local")) != "balloon":
            return False

        pad_rows = int(bs) - int(raw_bs)
        pad_tokens = pad_rows * int(self.num_tokens_per_bs)
        pad_req_idx = getattr(
            self.model_runner, "_kunserve_keepalive_phantom_req_idx", None
        )
        pad_kv_slot = getattr(
            self.model_runner, "_kunserve_keepalive_dummy_kv_slot", None
        )
        fallback_req_idx = None
        if pad_req_idx is None and int(raw_bs) > 0:
            try:
                fallback_req_idx = int(forward_batch.req_pool_indices[0].item())
            except Exception:
                fallback_req_idx = None

        req_idx_value = (
            int(pad_req_idx)
            if pad_req_idx is not None
            else (int(fallback_req_idx) if fallback_req_idx is not None else 0)
        )
        kv_value = int(pad_kv_slot) if pad_kv_slot is not None else 0

        buffers.req_pool_indices[raw_bs:bs].fill_(req_idx_value)
        buffers.seq_lens[raw_bs:bs].fill_(1)
        buffers.seq_lens_cpu[raw_bs:bs].fill_(1)
        token_end = int(raw_num_token) + int(pad_tokens)
        buffers.input_ids[raw_num_token:token_end].zero_()
        buffers.positions[raw_num_token:token_end].zero_()
        buffers.mrope_positions[:, raw_num_token:token_end].zero_()
        buffers.out_cache_loc[raw_num_token:token_end].fill_(kv_value)

        self._kunserve_graph_padding_log_ct += 1
        log_ct = self._kunserve_graph_padding_log_ct
        if log_ct in (1, 10, 100) or log_ct % 1000 == 0:
            _kunserve_graph_log(
                "global_padding_patch count=%d raw_bs=%d bs=%d pad_rows=%d "
                "raw_num_token=%d pad_tokens=%d req_idx=%d dummy_kv=%d "
                "phantom_available=%s dummy_available=%s tp_rank=%s",
                log_ct,
                int(raw_bs),
                int(bs),
                int(pad_rows),
                int(raw_num_token),
                int(pad_tokens),
                int(req_idx_value),
                int(kv_value),
                pad_req_idx is not None,
                pad_kv_slot is not None,
                getattr(self.model_runner, "tp_rank", None),
            )
        return True

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        self.deepep_adapter.replay()

        replay_fields = {
            "runtime_variant": self.model_runner.get_cuda_graph_runtime_variant(),
            "raw_batch_size": getattr(forward_batch, "batch_size", None),
            "graph_bs_override": self._kunserve_graph_bs_override(forward_batch),
            "skip_attn_backend_init": bool(skip_attn_backend_init),
        }
        if not skip_attn_backend_init:
            with kunserve_timing_scope("cuda_graph_replay_prepare", **replay_fields):
                self.replay_prepare(forward_batch, pp_proxy_tensors)
        else:
            # In speculative decoding, these two fields are still needed.
            self.buffers.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.buffers.positions[: self.raw_num_token].copy_(forward_batch.positions)

        # Replay
        if self.enable_pdmux:
            graph_key = self._graph_key(
                self.model_runner.get_cuda_graph_runtime_variant(),
                self.bs,
                get_current_stream_idx(),
            )
        else:
            graph_key = self._graph_key(
                self.model_runner.get_cuda_graph_runtime_variant(), self.bs
            )
        try:
            runtime_variant_for_log = self.model_runner.get_cuda_graph_runtime_variant()
        except Exception:
            runtime_variant_for_log = None
        if runtime_variant_for_log == "global":
            replay_log_ct = int(getattr(self, "_kunserve_global_replay_log_ct", 0)) + 1
            self._kunserve_global_replay_log_ct = replay_log_ct
            if replay_log_ct in (1, 2, 3, 10, 100) or replay_log_ct % 1000 == 0:
                _kunserve_graph_log(
                    "replay variant=global count=%d graph_key=%s raw_bs=%d bs=%d "
                    "raw_num_token=%d replay_enabled=%s state=%s tp_rank=%s",
                    replay_log_ct,
                    graph_key,
                    int(self.raw_bs),
                    int(self.bs),
                    int(self.raw_num_token),
                    bool(self.model_runner.is_cuda_graph_replay_enabled()),
                    getattr(self.model_runner, "_balloon_state", None),
                    getattr(self.model_runner, "tp_rank", None),
                )
        kunserve_timing_log(
            "cuda_graph_replay_selected",
            graph_key=str(graph_key),
            raw_bs=int(self.raw_bs),
            bs=int(self.bs),
            raw_num_token=int(self.raw_num_token),
            **replay_fields,
        )
        replay_ct = int(getattr(self, "_kunserve_graph_internal_replay_ct", 0)) + 1
        self._kunserve_graph_internal_replay_ct = replay_ct
        internal_interval = kunserve_graph_internal_timing_interval()
        sample_internal = (
            kunserve_graph_internal_timing_enabled()
            and (replay_ct <= 3 or replay_ct % internal_interval == 0)
        )
        graph_start_event = graph_end_event = None
        if sample_internal:
            try:
                graph_start_event = torch.cuda.Event(enable_timing=True)
                graph_end_event = torch.cuda.Event(enable_timing=True)
                graph_start_event.record()
            except Exception:
                graph_start_event = graph_end_event = None
        with kunserve_timing_scope(
            "cuda_graph_replay_launch",
            graph_key=str(graph_key),
            raw_bs=int(self.raw_bs),
            bs=int(self.bs),
            raw_num_token=int(self.raw_num_token),
            **replay_fields,
        ):
            self.graphs[graph_key].replay()
        if graph_end_event is not None:
            try:
                graph_end_event.record()
            except Exception:
                graph_start_event = graph_end_event = None
        self._kunserve_last_replay_timing_meta = (
            {
                "graph_key": str(graph_key),
                "graph_replay_count": int(replay_ct),
                "graph_internal_sample": bool(sample_internal),
                "graph_internal_interval": int(internal_interval),
                "graph_start_event": graph_start_event,
                "graph_end_event": graph_end_event,
                "raw_bs": int(self.raw_bs),
                "bs": int(self.bs),
                "raw_num_token": int(self.raw_num_token),
                **replay_fields,
            }
            if sample_internal
            else None
        )
        output = self.output_buffers[graph_key]

        if isinstance(output, LogitsProcessorOutput):
            if self.is_dllm:
                next_token_logits = None
                full_logits = output.full_logits[: self.raw_num_token]
            else:
                full_logits = None
                next_token_logits = output.next_token_logits[: self.raw_num_token]

            return LogitsProcessorOutput(
                next_token_logits=next_token_logits,
                full_logits=full_logits,
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
                customized_info=output.customized_info,
            )
        else:
            assert isinstance(output, PPProxyTensors)
            return PPProxyTensors({k: v[: self.bs] for k, v in output.tensors.items()})

    def get_spec_info(self, num_tokens: int):
        spec_info = None
        if (
            self.model_runner.spec_algorithm.is_eagle()
            or self.model_runner.spec_algorithm.is_standalone()
        ):
            from sglang.srt.speculative.eagle_info import EagleVerifyInput

            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen.")
            else:
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=self.buffers.custom_mask,
                    positions=None,
                    retrive_index=None,
                    retrive_next_token=None,
                    retrive_next_sibling=None,
                    retrive_cum_len=None,
                    spec_steps=self.model_runner.server_args.speculative_num_steps,
                    topk=self.model_runner.server_args.speculative_eagle_topk,
                    draft_token_num=self.model_runner.server_args.speculative_num_draft_tokens,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    seq_lens_sum=None,
                    seq_lens_cpu=None,
                )

        elif self.model_runner.spec_algorithm.is_ngram():
            from sglang.srt.speculative.ngram_info import NgramVerifyInput

            spec_info = NgramVerifyInput(
                draft_token=None,
                tree_mask=self.buffers.custom_mask,
                positions=None,
                retrive_index=None,
                retrive_next_token=None,
                retrive_next_sibling=None,
                draft_token_num=self.num_tokens_per_bs,
            )
            spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

        return spec_info


CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "Possible solutions:\n"
    "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
    "2. set --cuda-graph-max-bs to a smaller value (e.g., 16)\n"
    "3. disable torch compile by not using --enable-torch-compile\n"
    "4. disable CUDA graph by --disable-cuda-graph. (Not recommended. Huge performance loss)\n"
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)


class DeepEPCudaGraphRunnerAdapter:
    def __init__(self):
        # Record DeepEP mode used during capture to ensure replay consistency
        self._captured_deepep_mode = None

    def capture(self, is_extend_in_batch: bool):
        if not get_moe_a2a_backend().is_deepep():
            return
        self._captured_deepep_mode = get_deepep_mode().resolve(
            is_extend_in_batch=is_extend_in_batch
        )
        DeepEPBuffer.set_dispatch_mode(self._captured_deepep_mode)

    def replay(self):
        if not get_moe_a2a_backend().is_deepep():
            return
        assert self._captured_deepep_mode is not None
        DeepEPBuffer.set_dispatch_mode(self._captured_deepep_mode)
