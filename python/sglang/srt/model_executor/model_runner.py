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
"""ModelRunner runs the forward passes of the models."""

import copy
import datetime
import gc
import inspect
import json
import logging
import os
import socket
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.configs import (
    BailingHybridConfig,
    FalconH1Config,
    GraniteMoeHybridConfig,
    JetNemotronConfig,
    JetVLMConfig,
    KimiLinearConfig,
    Lfm2Config,
    Lfm2MoeConfig,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3NextConfig,
)
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import AttentionArch, ModelConfig, ModelImpl
from sglang.srt.configs.update_config import adjust_config_with_unaligned_cpu_tp
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
from sglang.srt.distributed import (
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
    set_torch_symm_mem_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.environ import envs
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionMetrics,
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    build_expert_location_metadata_from_mapping,
    compute_initial_expert_location_metadata,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    attn_backend_wrapper,
)
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_group,
    initialize_dp_attention,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    get_global_experts_capturer,
    set_global_experts_capturer,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import sanity_check_mm_pad_shift_value
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.balloon_utils import (
    build_dispatcher_physical_expert_mapping,
    resolve_balloon_kv_slots_to_expand,
)
from sglang.srt.model_executor.cuda_graph_runner import (
    CudaGraphRunner,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.hook_manager import register_forward_hooks
from sglang.srt.model_executor.input_buffers import GraphInputBuffers
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
    PiecewiseCudaGraphRunner,
)
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    register_memory_region,
    trigger_init_weights_send_group_for_remote_instance_request,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    MultiprocessingSerializer,
    cpu_has_amx_support,
    dynamic_import,
    empty_context,
    enable_show_time_cost,
    get_available_gpu_memory,
    get_cpu_ids_by_node,
    get_local_ip_auto,
    init_custom_process_group,
    is_hip,
    is_host_cpu_arm64,
    is_npu,
    log_info_on_rank0,
    monkey_patch_p2p_access_check,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_tp_gather,
    reserve_rope_cache_for_long_sequences,
    set_cuda_arch,
    slow_rank_detector,
)
from sglang.srt.utils.nvtx_pytorch_hooks import PytHooks
from sglang.srt.utils.offloader import (
    create_offloader_from_server_args,
    get_offloader,
    set_offloader,
)
from sglang.srt.utils.patch_torch import (
    monkey_patch_torch_reductions,
    register_sgl_tp_rank,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.weight_checker import WeightChecker
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu_arm64 = is_host_cpu_arm64()

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend

    init_npu_backend()

MLA_ATTENTION_BACKENDS = [
    "aiter",
    "flashinfer",
    "fa3",
    "fa4",
    "triton",
    "flashmla",
    "cutlass_mla",
    "trtllm_mla",
    "ascend",
    "nsa",
]

CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS = [
    "flashinfer",
    "fa3",
    "fa4",
    "flashmla",
    "cutlass_mla",
    "trtllm_mla",
]

TORCH_DTYPE_TO_KV_CACHE_STR = {
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e4m3fnuz: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
    torch.bfloat16: "bf16",
}


def add_mla_attention_backend(backend_name):
    if backend_name not in MLA_ATTENTION_BACKENDS:
        MLA_ATTENTION_BACKENDS.append(backend_name)
        logger.info(f"Added {backend_name} to MLA_ATTENTION_BACKENDS.")


def add_chunked_prefix_cache_attention_backend(backend_name):
    if backend_name not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS:
        CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS.append(backend_name)
        logger.info(
            f"Added {backend_name} to CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS."
        )


# Detect stragger ranks in model loading
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480  # leave more time for post data processing


logger = logging.getLogger(__name__)


def _kunserve_ms(message: str, *args) -> None:
    """Emit a [KUNSERVE-MS] milestone line.

    The scheduler runs as a forked subprocess of the SGLangHttpServer Ray
    actor; its stdout/stderr does not propagate to Ray's (SGLangHttpServer
    pid=...) capture stream, so logger.warning() alone is invisible from the
    training driver's log. To make milestones reliably visible regardless of
    fork/IPC plumbing we ALSO append them directly to the file pointed at by
    the KUNSERVE_DETAIL_LOG env var. The file write is best-effort and never
    raises. Signature mirrors logger.warning(msg, *args) so callsites can be
    a drop-in replacement.
    """
    logger.warning(message, *args)
    path = os.environ.get("KUNSERVE_DETAIL_LOG")
    if not path:
        return
    try:
        rendered = message % args if args else message
    except Exception:
        rendered = message
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        line = f"[{ts} pid={os.getpid()}] {rendered}\n"
        # Append mode; concurrent writers from different scheduler subprocesses
        # may interleave whole lines but POSIX guarantees atomicity for small
        # writes so individual lines never tear.
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception:
        pass


def resolve_language_model(model: nn.Module) -> nn.Module:
    model_cls_name = model.__class__.__name__
    if model_cls_name == "Qwen3OmniMoeForConditionalGeneration":
        return model.thinker.model
    return model.model


class RankZeroFilter(logging.Filter):
    """Filter that only allows INFO level logs from rank 0, but allows all other levels from any rank."""

    def __init__(self, is_rank_zero):
        super().__init__()
        self.is_rank_zero = is_rank_zero

    def filter(self, record):
        if record.levelno == logging.INFO:
            return self.is_rank_zero
        return True


@dataclass
class ModelRunnerOutput:
    logits_output: Union[LogitsProcessorOutput, PPProxyTensors]
    can_run_graph: bool
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None


class ModelRunner(ModelRunnerKVCacheMixin):
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,
        attn_cp_rank: Optional[int] = None,
        moe_dp_rank: Optional[int] = None,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        draft_model_idx: Optional[int] = None,
    ):
        # Parse args
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.moe_ep_rank = moe_ep_rank
        self.moe_ep_size = moe_ep_size
        self.dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.attn_cp_rank = attn_cp_rank
        self.attn_cp_size = server_args.attn_cp_size
        self.moe_dp_rank = moe_dp_rank
        self.moe_dp_size = server_args.moe_dp_size
        self.model_config = model_config
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.is_generation = model_config.is_generation
        self.is_multimodal = model_config.is_multimodal
        self.is_multimodal_chunked_prefill_supported = (
            model_config.is_multimodal_chunked_prefill_supported
        )
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid_swa = model_config.is_hybrid_swa
        self.is_hybrid_swa_compress = model_config.is_hybrid_swa_compress
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.attention_chunk_size = model_config.attention_chunk_size
        self.forward_pass_id = 0
        self.init_new_workspace = False
        self.draft_model_idx = draft_model_idx

        self.remote_instance_transfer_engine = None
        self.remote_instance_transfer_engine_session_id = ""
        self.remote_instance_transfer_engine_weight_info = None
        # auxiliary hidden capture mode. TODO: expose this to server args?
        self.eagle_use_aux_hidden_state = False
        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
            # load draft config
            draft_model_config = ModelConfig.from_server_args(
                server_args,
                model_path=(server_args.speculative_draft_model_path),
                model_revision=server_args.speculative_draft_model_revision,
                is_draft_model=True,
            )
            self.eagle_use_aux_hidden_state = True

            try:
                # get the aux layer from draft model config
                eagle_config = getattr(
                    draft_model_config.hf_config, "eagle_config", None
                )
                self.eagle_use_aux_hidden_state = eagle_config.get(
                    "use_aux_hidden_state", True
                )
                self.eagle_aux_hidden_state_layer_ids = eagle_config[
                    "eagle_aux_hidden_state_layer_ids"
                ]
            except:
                # if there is no aux layer, set to None
                self.eagle_aux_hidden_state_layer_ids = None

        # Apply the rank zero filter to logger
        if server_args.show_time_cost:
            enable_show_time_cost()

        # Model-specific adjustment
        self.model_specific_adjustment()

        # Set the global server_args in the scheduler process
        set_global_server_args_for_scheduler(server_args)
        global_server_args = get_global_server_args()

        # FIXME: hacky set `use_mla_backend`
        global_server_args.use_mla_backend = self.use_mla_backend

        # Init OpenMP threads binding for CPU
        if self.device == "cpu":
            self.init_threads_binding()

        # Initialize MooncakeTransferEngine
        self.init_shared_mooncake_transfer_engine()

        # Get memory before model loading
        min_per_gpu_memory = self.init_torch_distributed()

        # Init forward stream for overlap schedule
        self.forward_stream = torch.get_device_module(self.device).Stream()

        # CPU offload
        set_offloader(create_offloader_from_server_args(server_args, dp_rank=dp_rank))

        self._weight_checker = WeightChecker(model_runner=self)

        if envs.SGLANG_DETECT_SLOW_RANK.get():
            slow_rank_detector.execute()

        # Init mindspore running environment when model impl is "mindspore"
        self.init_mindspore_runner()

        # Update deep gemm configure
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            deep_gemm_wrapper.update_deep_gemm_config(gpu_id, server_args)

        # Initialize the model runner
        self.initialize(min_per_gpu_memory)
        self.check_quantized_moe_compatibility()

        if self.is_multimodal:
            sanity_check_mm_pad_shift_value(self.model_config.vocab_size)

        # Temporary cached values
        self.support_pp = (
            "pp_proxy_tensors" in inspect.signature(self.model.forward).parameters
        )

        if self.pp_size > 1:
            assert (
                self.support_pp
            ), "Pipeline Parallel is not compatible with this model."

        # For weight updates
        self._model_update_group = {}
        self._weights_send_group = {}

    def init_mindspore_runner(self):
        # Init the mindspore runner
        # for now, there is only some communication initialization work
        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE and _is_npu:
            from sglang.srt.model_executor.mindspore_runner import init_ms_distributed

            init_ms_distributed(
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                server_args=self.server_args,
                port=self.dist_port,
            )

    def initialize(self, min_per_gpu_memory: float):
        server_args = self.server_args

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        if self.server_args.remote_instance_weight_loader_use_transfer_engine():
            self.remote_instance_init_transfer_engine()

        if not self.is_draft_worker:
            set_global_expert_location_metadata(
                compute_initial_expert_location_metadata(
                    server_args=server_args,
                    model_config=self.model_config,
                    moe_ep_rank=self.moe_ep_rank,
                )
            )
            if self.tp_rank == 0 and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get():
                logger.info(
                    f"Initial expert_location_metadata: {get_global_expert_location_metadata()}"
                )

            set_global_expert_distribution_recorder(
                ExpertDistributionRecorder.init_new(
                    server_args,
                    get_global_expert_location_metadata(),
                    rank=self.tp_rank,
                )
            )

        # Expert parallelism
        self.eplb_manager = (
            EPLBManager(self)
            if self.server_args.enable_eplb and (not self.is_draft_worker)
            else None
        )
        self.expert_location_updater = ExpertLocationUpdater()

        (
            ElasticEPStateManager.init(self.server_args)
            if self.server_args.elastic_ep_backend
            else None
        )
        # Load the model
        self.sampler = create_sampler()
        self.load_model()

        if (
            self.server_args.remote_instance_weight_loader_use_transfer_engine()
            and self.remote_instance_transfer_engine is not None
            and self.remote_instance_transfer_engine_weight_info is None
        ):
            self.remote_instance_transfer_engine_weight_info = register_memory_region(
                self.model, self.remote_instance_transfer_engine
            )

        # For MTP models like DeepSeek-V3 or GLM-4.5, the MTP layer(s) are used separately as draft
        # models for speculative decoding. In those cases, `num_nextn_predict_layers` is used to
        # determine the number of layers.
        model_has_mtp_layers = self.model_config.num_nextn_predict_layers is not None
        model_num_layers = (
            self.model_config.num_nextn_predict_layers
            if self.is_draft_worker and model_has_mtp_layers
            else max(
                self.model_config.num_hidden_layers,
                self.model_config.num_attention_layers,
            )
        )
        if self.model_config.hf_config.architectures[0] == "MiMoV2MTP":
            model_num_layers = 1
        elif self.model_config.hf_config.architectures[0] == "Step3p5MTP":
            model_num_layers = 1
        self.start_layer = getattr(self.model, "start_layer", 0)
        self.end_layer = getattr(self.model, "end_layer", model_num_layers)
        self.num_effective_layers = self.end_layer - self.start_layer

        # For LoopCoder models, each loop has its own layer_id, so we need to multiply by loop_num
        loop_num = getattr(self.model_config.hf_config, "loop_num", 1)
        if loop_num > 1:
            self.num_effective_layers = self.num_effective_layers * loop_num

        assert (
            (not model_has_mtp_layers)
            or (self.spec_algorithm.is_none())
            or (
                (not self.spec_algorithm.is_none())
                and (self.num_effective_layers == model_num_layers)
            )
        ), "PP is not compatible with MTP models."

        # Consider PP, so use start_layer and end_layer.
        full_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "full_attention_layer_ids")
            and layer_idx in self.model_config.full_attention_layer_ids
        ]
        swa_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "swa_attention_layer_ids")
            and layer_idx in self.model_config.swa_attention_layer_ids
        ]
        # Update back to model_config.
        self.model_config.swa_attention_layer_ids = swa_attention_layer_ids
        self.model_config.full_attention_layer_ids = full_attention_layer_ids

        # Apply torchao quantization
        torchao_applied = getattr(self.model, "torchao_applied", False)
        # In layered loading, torchao may have been applied
        if not torchao_applied:
            apply_torchao_config_to_model(
                self.model, get_global_server_args().torchao_config
            )

        # Apply torch TP if the model supports it
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.tp_size > 1 and supports_torch_tp:
            self.apply_torch_tp()

        # Init lora
        if server_args.enable_lora:
            self.init_lora_manager()

        # Init Double Sparsity
        if server_args.enable_double_sparsity:
            if server_args.ds_heavy_channel_type is None:
                raise ValueError(
                    "Please specify the heavy channel type for double sparsity optimization."
                )
            self.init_double_sparsity_channel_config(server_args.ds_heavy_channel_type)

        # Enable batch invariant mode
        if server_args.enable_deterministic_inference:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            enable_batch_invariant_mode()

        # Deduce KV cache dtype
        self.configure_kv_cache_dtype()

        # Init memory pool and attention backends
        self.init_memory_pool(min_per_gpu_memory)

        # Init max running requests
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests // (self.dp_size)
            ),
            self.req_to_token_pool.size,
        )

        # Init routed experts capturer
        self.init_routed_experts_capturer()
        self.init_balloon_runtime_state()

        if self.device == "cuda" or self.device == "musa":
            self.init_cublas()
            self.init_attention_backend()
            self.kernel_warmup()
            self.init_device_graphs()
        elif self.device in ["npu", "cpu"]:
            self.init_attention_backend()
            self.init_device_graphs()
        else:
            self.graph_runner = None
            self.graph_mem_usage = 0
            self.init_attention_backend()

        if server_args.forward_hooks:
            register_forward_hooks(self.model, server_args.forward_hooks)

        if self.eagle_use_aux_hidden_state:
            self.model.set_eagle3_layers_to_capture(
                self.eagle_aux_hidden_state_layer_ids
            )

        # Initialize piecewise CUDA graph
        self.init_piecewise_cuda_graphs()

        self.prealloc_symmetric_memory_pool()

    def init_routed_experts_capturer(self):
        if not self.server_args.disable_shared_experts_fusion and hasattr(
            self.model, "num_fused_shared_experts"
        ):
            num_fused_shared_experts = self.model.num_fused_shared_experts
        else:
            num_fused_shared_experts = 0

        set_global_experts_capturer(
            RoutedExpertsCapturer.create(
                enable=get_global_server_args().enable_return_routed_experts,
                model_config=self.model_config,
                num_fused_shared_experts=num_fused_shared_experts,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def init_balloon_runtime_state(self):
        local_metadata = None
        if not self.is_draft_worker:
            expert_location_metadata = get_global_expert_location_metadata()
            if expert_location_metadata is not None:
                local_metadata = copy.deepcopy(expert_location_metadata)

        self._balloon_state = "local"
        self._balloon_runtime_variant = "local"
        self._balloon_prepared_variant = None
        self._balloon_graph_replay_enabled = True
        # Phase E per-step override: scheduler flips this on when the
        # cross-replica bs negotiation detects an asymmetric busy/busy
        # step that would otherwise replay graphs with mismatched NCCL
        # collective shapes.  Read by is_cuda_graph_replay_enabled().
        self._balloon_step_force_eager: bool = False
        # Phase E keepalive KV scratch slot.  Allocated once at
        # commit_balloon time so the keepalive batch's out_cache_loc
        # points at a dedicated dummy slot instead of slot 0.  Without
        # this, captured DECODE graph replay (forward_mode=IDLE still
        # matches is_cuda_graph() and triggers graph replay) would
        # call set_kv_buffer at out_cache_loc=0 and clobber whichever
        # real request currently owns slot 0.  Storing the alloc
        # tensor (not just the int) lets us free it cleanly on a
        # future restore.
        self._kunserve_keepalive_dummy_kv_slot_tensor: Optional[torch.Tensor] = None
        self._kunserve_keepalive_dummy_kv_slot: Optional[int] = None
        self._balloon_local_expert_location_metadata = local_metadata
        self._balloon_global_expert_location_metadata = None
        self._balloon_prepared_active_mappings: Dict[int, torch.Tensor] = {}
        self._balloon_unused_donors = None
        self._balloon_added_slots = 0
        self._balloon_offloaded_local_experts = 0
        self._balloon_last_error = None
        self._balloon_process_group_name = None
        self._balloon_kunserve_comm_backend = "deepep"
        self._balloon_capture_policy = "auto"
        self._balloon_fused_moe_layers: Optional[List[torch.nn.Module]] = None

        # KunServe LOCAL/GLOBAL split:
        #
        # The original concern: with moe_a2a_backend=deepep + deepep_mode=auto,
        # FusedMoE.__init__ creates a LOCAL DeepEP dispatcher whose first
        # decode forward initializes NVSHMEM for the local TP group; if
        # register_balloon_global_runtime_bundle later builds a GLOBAL DeepEP
        # dispatcher in LL mode, NVSHMEM's per-process singleton trips on the
        # `nvshmem_rank == internode::init(...)` assert (one process can only
        # init one NVSHMEM context).
        #
        # Why we don't need the override anymore: with
        # SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL=1 (the default in this repo),
        # register_balloon_global_runtime_bundle below builds the GLOBAL
        # dispatcher with DeepEPMode.NORMAL only — NORMAL skips NVSHMEM init
        # entirely (DeepEP buffer.py:96 gate). So LOCAL is free to use the
        # default DeepEP+DeepGEMM path that baseline (FP8 + deepep + deep_gemm
        # without kunserve) is known to run correctly. Forcing LOCAL to
        # StandardDispatcher+Triton was producing ~25% degenerate outputs in
        # ab_20260507_163750 (sample-7 r1 first request, 4 tokens of garbage)
        # while a same-config baseline (ab_20260509_015613/baseline) was clean.
        #
        # If someone ever flips SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL=0 to use
        # LL for GLOBAL (e.g., once IBGDA + peermem are properly set up), the
        # double-init assert returns and the override needs to be re-enabled.
        # Gate the override on that condition only.
        if (
            envs.SGLANG_EXPERIMENTAL_VMM_MOE_WEIGHTS.get()
            and not envs.SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL.get()
        ):
            self._force_local_bundle_to_standard_dispatcher()

    def _force_local_bundle_to_standard_dispatcher(self) -> None:
        """Replace each FusedMoE layer's LOCAL bundle with a StandardDispatcher.

        Called when KunServe is in play (SGLANG_EXPERIMENTAL_VMM_MOE_WEIGHTS=1).
        Standard dispatch keeps fixed shapes (cuda-graph friendly) and runs
        without NVSHMEM. LOCAL intentionally uses the Triton runner instead of
        reusing the original DeepGEMM runner: the standard->deep_gemm permute
        path is unsafe for EP-local execution with -1 non-local expert ids and
        can corrupt hidden states before any KunServe GLOBAL commit happens.
        The companion ``reduce_results=True`` flag tells FusedMoE.forward_impl
        to finish the TP all-reduce internally; qwen3_moe.forward_deepep does
        not add one on top, and the per-bundle dispatcher abstraction means
        model-level code does not need to know which mode the layer is using.
        """
        if self.is_draft_worker:
            return
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            FusedMoERuntimeVariant,
        )
        from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatcher,
        )
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        layers = self._iter_fused_moe_layers()
        if not layers:
            return

        for layer in layers:
            local_runner = MoeRunner(
                MoeRunnerBackend.TRITON,
                layer.local_bundle.moe_runner_config,
            )
            standard = StandardDispatcher(
                layer.local_bundle.moe_runner_config,
                moe_ep_size=layer.local_bundle.moe_ep_size,
                moe_ep_rank=layer.local_bundle.moe_ep_rank,
                # local_expert_mapping is built lazily by StandardDispatcher
                # on first dispatch using moe_ep_rank — leave None so it
                # picks up the correct mapping for the live layer.
                local_expert_mapping=None,
            )
            # Re-register so self._runtime_bundles[LOCAL] points at the new
            # bundle. Do not reuse the original runner here: in KunServe FP8
            # runs it is DeepGEMM, while LOCAL standard dispatch has already
            # converted non-local experts to -1 ids. The Triton standard path
            # is the normal local EP fallback and keeps LOCAL independent from
            # DeepEP/NVSHMEM.
            layer.register_runtime_bundle(
                variant=FusedMoERuntimeVariant.LOCAL,
                moe_runner_config=layer.local_bundle.moe_runner_config,
                dispatcher=standard,
                runner=local_runner,
                moe_ep_size=layer.local_bundle.moe_ep_size,
                moe_ep_rank=layer.local_bundle.moe_ep_rank,
                moe_tp_size=layer.local_bundle.moe_tp_size,
                moe_tp_rank=layer.local_bundle.moe_tp_rank,
                num_local_experts=layer.local_bundle.num_local_experts,
                # Standard dispatch leaves partial sums on each rank, so
                # FusedMoE.forward_impl must run tp_group all-reduce.
                reduce_results=True,
            )
            # Refresh self.dispatcher / self.reduce_results / etc. on the
            # FusedMoE layer so the very next forward (graph capture or live)
            # actually picks up the new bundle.
            layer.switch_runtime_bundle(FusedMoERuntimeVariant.LOCAL)
        # Use _kunserve_ms (writes to KUNSERVE_DETAIL_LOG and goes through
        # logger.warning) instead of logger.info — sglang's Ray-actor stdout
        # capture suppresses INFO-level lines, which made it impossible to
        # confirm this override actually fired in the previous run.
        _kunserve_ms(
            "[KUNSERVE-MS] forced LOCAL bundle to StandardDispatcher "
            "+ Triton runner (reduce_results=True) on %d FusedMoE layers; "
            "NVSHMEM will only init when the GLOBAL bundle is created.",
            len(layers),
        )

    def _iter_fused_moe_layers(self) -> List[torch.nn.Module]:
        if self._balloon_fused_moe_layers is None:
            from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

            self._balloon_fused_moe_layers = [
                module
                for module in self.model.modules()
                if isinstance(module, FusedMoE)
            ]
        return self._balloon_fused_moe_layers

    def _normalize_balloon_variant(self, variant: Union[str, object]) -> str:
        variant_value = getattr(variant, "value", variant)
        variant_value = str(variant_value).lower()
        if variant_value not in ("local", "global"):
            raise ValueError(f"Unsupported balloon runtime variant: {variant}")
        return variant_value

    def _resolve_balloon_process_group(self, process_group_name: Optional[str]):
        """Look up an initialized PG by name.

        Returns ``None`` for both the "not set" case and the Phase F
        "this TP rank did not participate" case (the latter stores
        ``None`` under the name as a sentinel — see
        ``init_weights_update_group`` with ``lane_only_tp_rank``).
        """
        if process_group_name is None:
            return None
        if process_group_name in self._model_update_group:
            return self._model_update_group[process_group_name]
        if process_group_name in self._weights_send_group:
            return self._weights_send_group[process_group_name]
        raise ValueError(
            f"Process group '{process_group_name}' is not initialized on this rank."
        )

    def negotiate_balloon_step_bs(self, local_bs: int) -> Tuple[int, int]:
        """Phase E: cross-replica per-step batch-size negotiation.

        Every rank in the cross-replica runtime_group must call this in
        lockstep at the same scheduler step.  Returns
        ``(max_bs, min_bs)`` of the gathered ``local_bs`` values --
        callers use the pair to derive:

        * ``negotiated_bs = max_bs`` -- the size every idle replica
          should pad its keepalive batch to so the captured GLOBAL
          graph can replay in lockstep on all 4 ranks.
        * ``lockstep_safe = (min_bs == 0) or (min_bs == max_bs)`` -- if
          False the world is split into busy-but-different-bs and graph
          replay would mismatch across ranks; the scheduler must force
          eager forward on this step.  Returning a pair (instead of
          just ``max``) is what makes the decision symmetric: every
          rank sees the same ``(max, min)`` tuple and reaches the same
          force-eager conclusion.

        Idle replicas pass ``local_bs=0``; if ``max_bs > 0`` they must
        build a keepalive batch of that size.

        Returns ``(local_bs, local_bs)`` (i.e. skips the collective)
        when:

        - balloon state is not ``balloon`` (LOCAL forward needs no sync),
        - the runtime_group has not been resolved on this rank yet,
        - or the kunserve backend is something that does its own per-step
          sync (DeepEP path - keepalive there is handled by the existing
          DeepEP normal mode buffer setup).
        """
        if str(self._balloon_state) != "balloon":
            return int(local_bs), int(local_bs)
        if str(self._balloon_kunserve_comm_backend or "").lower() != "sglang":
            return int(local_bs), int(local_bs)
        try:
            runtime_group = self._resolve_balloon_process_group(
                self._balloon_process_group_name
            )
        except Exception:
            return int(local_bs), int(local_bs)
        if runtime_group is None:
            return int(local_bs), int(local_bs)
        try:
            device = torch.device("cuda", torch.cuda.current_device())
            local_t = torch.tensor(
                [int(local_bs)], dtype=torch.int32, device=device
            )
            world = int(dist.get_world_size(group=runtime_group))
            all_t = torch.empty(world, dtype=torch.int32, device=device)
            dist.all_gather_into_tensor(all_t, local_t, group=runtime_group)
            # Pull both max and min in one device->host sync so every
            # rank reaches the same force-eager decision below.
            return int(all_t.max().item()), int(all_t.min().item())
        except Exception as exc:
            # If the collective fails (e.g. group destroyed mid-shutdown),
            # fall back to the local bs.  The dispatcher's dynamic eager
            # path can still handle whatever shape arrives.
            _kunserve_ms(
                "[KUNSERVE-MS] negotiate_balloon_step_bs failed: %r (local_bs=%d) "
                "-- falling back to local-only sizing",
                exc,
                int(local_bs),
            )
            return int(local_bs), int(local_bs)

    def _default_balloon_physical_to_logical_map(self):
        metadata = (
            self._balloon_local_expert_location_metadata
            or get_global_expert_location_metadata()
        )
        if metadata is None:
            return None
        return metadata.physical_to_logical_map_cpu.tolist()

    def _resolve_balloon_rank(
        self,
        *,
        explicit_rank: Optional[int],
        rank_offset: Optional[int],
        default_rank: int,
    ) -> int:
        if explicit_rank is not None:
            return int(explicit_rank)
        if rank_offset is not None:
            return int(rank_offset) + int(self.tp_rank)
        return int(default_rank)

    def _normalize_balloon_active_mappings(
        self,
        *,
        retained_local_experts: Optional[int] = None,
        active_local_expert_mapping: Optional[List[int]] = None,
        active_local_expert_mapping_by_layer: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[int, torch.Tensor]:
        fused_layers = self._iter_fused_moe_layers()
        if not fused_layers:
            return {}

        base_mapping = None
        if active_local_expert_mapping_by_layer is not None:
            mappings = {
                int(layer_id): torch.tensor(mapping, dtype=torch.int32)
                for layer_id, mapping in active_local_expert_mapping_by_layer.items()
            }
        else:
            if active_local_expert_mapping is not None:
                base_mapping = torch.tensor(
                    active_local_expert_mapping, dtype=torch.int32
                )
            elif retained_local_experts is not None:
                if retained_local_experts <= 0:
                    raise ValueError(
                        f"retained_local_experts must be positive, got {retained_local_experts}"
                    )
                base_mapping = torch.arange(retained_local_experts, dtype=torch.int32)
            else:
                raise ValueError(
                    "Either retained_local_experts or an active_local_expert_mapping must be provided."
                )
            mappings = {
                int(layer.layer_id): base_mapping.clone() for layer in fused_layers
            }

        for layer in fused_layers:
            if layer.layer_id not in mappings:
                raise ValueError(
                    f"Missing active_local_expert_mapping for fused MoE layer {layer.layer_id}."
                )
            mapping = mappings[layer.layer_id]
            if mapping.dim() != 1 or mapping.numel() == 0:
                raise ValueError(
                    f"Layer {layer.layer_id} active_local_expert_mapping must be a non-empty 1D tensor."
                )
            expected = torch.arange(
                int(mapping[0].item()),
                int(mapping[0].item()) + mapping.numel(),
                dtype=mapping.dtype,
            )
            if not torch.equal(mapping.cpu(), expected.cpu()):
                raise ValueError(
                    "Current balloon runtime only supports contiguous active_local_expert_mapping per layer."
                )
        return mappings

    def _switch_all_fused_moe_runtime_bundles(self, variant: str) -> None:
        normalized_variant = self._normalize_balloon_variant(variant)
        if not self._iter_fused_moe_layers():
            self._balloon_runtime_variant = normalized_variant
            return
        for layer in self._iter_fused_moe_layers():
            layer.switch_runtime_bundle(normalized_variant)
        self._balloon_runtime_variant = normalized_variant

    @contextmanager
    def cuda_graph_capture_variant_scope(self, variant: str):
        previous_variant = self.get_cuda_graph_runtime_variant()
        normalized_variant = self._normalize_balloon_variant(variant)
        if previous_variant != normalized_variant:
            self._switch_all_fused_moe_runtime_bundles(normalized_variant)
        try:
            yield
        finally:
            if previous_variant != normalized_variant:
                self._switch_all_fused_moe_runtime_bundles(previous_variant)

    def get_cuda_graph_capture_variants(self) -> List[str]:
        variants = ["local"]
        fused_layers = self._iter_fused_moe_layers()
        if fused_layers and all(
            getattr(layer, "global_bundle", None) is not None for layer in fused_layers
        ):
            variants.append("global")
        return variants

    def get_cuda_graph_runtime_variant(self) -> str:
        return self._normalize_balloon_variant(self._balloon_runtime_variant)

    def is_cuda_graph_replay_enabled(self) -> bool:
        if getattr(self, "_balloon_step_force_eager", False):
            return False
        return bool(self._balloon_graph_replay_enabled)

    def set_balloon_step_force_eager(self, force: bool) -> None:
        """Phase E hook: when the cross-replica bs negotiation detects an
        asymmetric busy/busy step (e.g. replica A's batch padded to bs=64
        but replica B's padded to bs=128), both replicas must skip graph
        replay for this single step.  Otherwise each rank would replay
        its own (variant, bs) graph whose captured NCCL collectives have
        different shapes -> NCCL all_gather_into_tensor size mismatch ->
        silent hang.  The scheduler flips this on before run_batch and
        back off after the forward returns.
        """
        self._balloon_step_force_eager = bool(force)

    def _update_live_expert_location_metadata(self, metadata) -> None:
        if self.is_draft_worker or metadata is None:
            _kunserve_ms(
                "[KUNSERVE-DBG] _update_live_expert_location_metadata SKIPPED "
                "is_draft_worker=%s metadata_is_none=%s tp_rank=%s",
                self.is_draft_worker,
                metadata is None,
                getattr(self, "tp_rank", None),
            )
            return
        live_metadata = get_global_expert_location_metadata()
        if live_metadata is None:
            _kunserve_ms(
                "[KUNSERVE-DBG] _update_live_expert_location_metadata SKIPPED: "
                "live_metadata is None tp_rank=%s",
                getattr(self, "tp_rank", None),
            )
            return

        # Snapshot a few entries before/after to verify the in-place update
        # actually flips logical->physical from LOCAL (identity) to GLOBAL
        # (complementary). The captured cuda graph reads from this storage at
        # replay time, so a missing flip here is the same as the topk kernel
        # baking in identity values.
        live_map = getattr(live_metadata, "logical_to_rank_dispatch_physical_map", None)
        other_map = getattr(metadata, "logical_to_rank_dispatch_physical_map", None)
        before = None
        after_other = None
        if live_map is not None and live_map.numel() >= 65:
            before = (
                int(live_map[0, 0].item()),
                int(live_map[0, 32].item()),
                int(live_map[0, 64].item()),
            )
        if other_map is not None and other_map.numel() >= 65:
            after_other = (
                int(other_map[0, 0].item()),
                int(other_map[0, 32].item()),
                int(other_map[0, 64].item()),
            )
        _kunserve_ms(
            "[KUNSERVE-DBG] _update_live_expert_location_metadata BEFORE: "
            "tp_rank=%s live_map_is_none=%s other_map_is_none=%s "
            "live[layer0,(0,32,64)]=%s other[layer0,(0,32,64)]=%s",
            getattr(self, "tp_rank", None),
            live_map is None,
            other_map is None,
            before,
            after_other,
        )

        live_metadata.update(metadata, list(range(live_metadata.num_layers)))

        live_map_after = getattr(
            live_metadata, "logical_to_rank_dispatch_physical_map", None
        )
        after = None
        if live_map_after is not None and live_map_after.numel() >= 65:
            after = (
                int(live_map_after[0, 0].item()),
                int(live_map_after[0, 32].item()),
                int(live_map_after[0, 64].item()),
            )
        _kunserve_ms(
            "[KUNSERVE-DBG] _update_live_expert_location_metadata AFTER: "
            "tp_rank=%s live[layer0,(0,32,64)]=%s same_storage=%s",
            getattr(self, "tp_rank", None),
            after,
            (
                (live_map is live_map_after)
                if (live_map is not None and live_map_after is not None)
                else None
            ),
        )

    def _build_balloon_global_metadata(
        self,
        *,
        physical_to_logical_map=None,
        runtime_ep_size: Optional[int] = None,
        moe_ep_rank: Optional[int] = None,
        dispatch_ep_rank: Optional[int] = None,
        runtime_rank_offset: Optional[int] = None,
        dispatch_rank_offset: Optional[int] = None,
    ):
        if self.is_draft_worker:
            return None

        physical_to_logical_map = (
            physical_to_logical_map or self._default_balloon_physical_to_logical_map()
        )
        if physical_to_logical_map is None:
            return None

        resolved_moe_ep_rank = self._resolve_balloon_rank(
            explicit_rank=moe_ep_rank,
            rank_offset=runtime_rank_offset,
            default_rank=self.moe_ep_rank,
        )
        resolved_dispatch_ep_rank = self._resolve_balloon_rank(
            explicit_rank=dispatch_ep_rank,
            rank_offset=(
                dispatch_rank_offset
                if dispatch_rank_offset is not None
                else runtime_rank_offset
            ),
            default_rank=self.moe_ep_rank,
        )

        return build_expert_location_metadata_from_mapping(
            server_args=self.server_args,
            model_config=self.model_config,
            physical_to_logical_map=physical_to_logical_map,
            moe_ep_rank=resolved_moe_ep_rank,
            ep_size=runtime_ep_size,
            dispatch_ep_rank=resolved_dispatch_ep_rank,
        )

    def register_balloon_global_runtime_bundle(
        self,
        *,
        runtime_ep_size: Optional[int] = None,
        moe_ep_rank: Optional[int] = None,
        dispatch_ep_rank: Optional[int] = None,
        runtime_rank_offset: Optional[int] = None,
        dispatch_rank_offset: Optional[int] = None,
        retained_local_experts: Optional[int] = None,
        active_local_expert_mapping: Optional[List[int]] = None,
        active_local_expert_mapping_by_layer: Optional[Dict[int, List[int]]] = None,
        physical_to_logical_map=None,
        process_group_name: Optional[str] = None,
        kunserve_comm_backend: str = "deepep",
        capture_policy: str = "auto",
        kunserve_pg_names: Optional[Dict[str, str]] = None,
        kunserve_backend_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, torch.Tensor]:
        fused_layers = self._iter_fused_moe_layers()
        if not fused_layers:
            raise ValueError("Balloon runtime requires a model with FusedMoE layers.")

        kunserve_comm_backend = str(kunserve_comm_backend or "deepep").lower()
        capture_policy = str(capture_policy or "auto").lower()
        kunserve_backend_config = dict(kunserve_backend_config or {})
        if kunserve_comm_backend not in ("deepep", "sglang"):
            raise ValueError(
                "Unsupported KunServe GLOBAL communication backend "
                f"{kunserve_comm_backend!r}; expected 'deepep' or 'sglang'."
            )

        # KunServe BALLOON publishes a complementary physical_to_logical_map
        # across replicas (e.g. replica 0 retains [0..31, 64..95] while replica
        # 1 retains [32..63, 96..127]). GLOBAL runtime bundles use physical
        # expert ids as the dispatch domain. Without
        # `ep_dispatch_algorithm="static"` the
        # `logical_to_rank_dispatch_physical_map` table is not built, the
        # model's `topk_ids_logical_to_physical` step is a no-op, and BALLOON
        # forward indexes/routes against the wrong expert rows. The observable
        # symptom is the model collapsing into repeated-character output until
        # it hits max_new_tokens. Fail loudly instead of silently producing
        # garbage.
        if str(getattr(self.server_args, "ep_dispatch_algorithm", None)) != "static":
            raise ValueError(
                "Balloon GLOBAL bundle requires server_args.ep_dispatch_algorithm='static' "
                "so the logical->physical remap is applied at topk time. Pass "
                "--ep-dispatch-algorithm=static (sglang CLI) or "
                "engine_kwargs.sglang.ep_dispatch_algorithm=static (verl). Current value: "
                f"{getattr(self.server_args, 'ep_dispatch_algorithm', None)!r}."
            )
        moe_a2a_backend = get_moe_a2a_backend()
        if kunserve_comm_backend == "deepep":
            if moe_a2a_backend.is_none():
                raise ValueError(
                    "Balloon GLOBAL bundle with kunserve_comm_backend='deepep' "
                    "requires a cross-rank MoE A2A backend. "
                    "moe_a2a_backend='none' would make the GLOBAL bundle's "
                    "auto-registered dispatcher fall back to StandardDispatcher, "
                    "which only does rank-local expert compute + intra-TP "
                    "all-reduce; it cannot route tokens to experts retained by "
                    "the peer KunServe replica. Pass "
                    "engine_kwargs.sglang.moe_a2a_backend=deepep and "
                    "engine_kwargs.sglang.moe_runner_backend=deep_gemm, or set "
                    "kunserve_comm_backend='sglang' to use the new "
                    "CrossReplicaStandardDispatcher. NOTE: the LOCAL bundle is "
                    "intentionally overridden back to Standard by "
                    "_force_local_bundle_to_standard_dispatcher (gated on "
                    "SGLANG_EXPERIMENTAL_VMM_MOE_WEIGHTS), so this flag only "
                    "affects the GLOBAL bundle's default."
                )
            if (
                moe_a2a_backend.is_deepep() or moe_a2a_backend.is_mooncake()
            ) and str(getattr(self.server_args, "moe_runner_backend", None)) != (
                "deep_gemm"
            ):
                raise ValueError(
                    "Balloon GLOBAL bundle with kunserve_comm_backend='deepep' and "
                    "moe_a2a_backend="
                    f"{moe_a2a_backend.value!r} requires "
                    "server_args.moe_runner_backend='deep_gemm'. This sglang build "
                    "only registers DeepEP/Mooncake MoE pre/post permutation paths "
                    "for the deep_gemm runner; leaving the runner as 'auto' or "
                    "'triton' can crash the scheduler during warmup/cuda-graph "
                    "capture. Pass engine_kwargs.sglang.moe_runner_backend=deep_gemm. "
                    f"Current value: {getattr(self.server_args, 'moe_runner_backend', None)!r}."
                )

        active_mappings = self._normalize_balloon_active_mappings(
            retained_local_experts=retained_local_experts,
            active_local_expert_mapping=active_local_expert_mapping,
            active_local_expert_mapping_by_layer=active_local_expert_mapping_by_layer,
        )
        runtime_group = self._resolve_balloon_process_group(process_group_name)
        runtime_group_size = None
        if runtime_group is not None:
            try:
                runtime_group_size = (
                    int(runtime_group.world_size)
                    if hasattr(runtime_group, "world_size")
                    else int(dist.get_world_size(group=runtime_group))
                )
            except Exception:
                runtime_group_size = "unknown"
        if kunserve_comm_backend == "sglang" and runtime_group is None:
            raise ValueError(
                "KunServe comm backend 'sglang' requires process_group_name to "
                "resolve to an initialized global process group."
            )
        self._balloon_global_expert_location_metadata = (
            self._build_balloon_global_metadata(
                physical_to_logical_map=physical_to_logical_map,
                runtime_ep_size=runtime_ep_size,
                moe_ep_rank=moe_ep_rank,
                dispatch_ep_rank=dispatch_ep_rank,
                runtime_rank_offset=runtime_rank_offset,
                dispatch_rank_offset=dispatch_rank_offset,
            )
        )

        # KUNSERVE-DBG: confirm the GLOBAL metadata's
        # logical_to_rank_dispatch_physical_map was actually built and contains
        # the complementary inverse map (logical 32 -> physical 64 etc.). If
        # this is None the static remap can't kick in even with
        # ep_dispatch_algorithm=static set on the CLI.
        gmeta = self._balloon_global_expert_location_metadata
        if gmeta is None:
            _kunserve_ms(
                "[KUNSERVE-DBG] register_balloon_global_runtime_bundle: "
                "global metadata is None tp_rank=%s",
                self.tp_rank,
            )
        else:
            gmap = getattr(gmeta, "logical_to_rank_dispatch_physical_map", None)
            if gmap is None:
                _kunserve_ms(
                    "[KUNSERVE-DBG] register_balloon_global_runtime_bundle: "
                    "GLOBAL metadata.logical_to_rank_dispatch_physical_map is None "
                    "tp_rank=%s ep_dispatch_algorithm=%s -- topk will NOT remap, "
                    "DeepEP will route by raw logical id and corrupt outputs",
                    self.tp_rank,
                    getattr(self.server_args, "ep_dispatch_algorithm", None),
                )
            else:
                _kunserve_ms(
                    "[KUNSERVE-DBG] register_balloon_global_runtime_bundle: "
                    "tp_rank=%s GLOBAL_dispatch_map shape=%s "
                    "layer0[(0,32,64,96)]=(%d,%d,%d,%d) "
                    "p2l_layer0[:8]=%s a2a_backend=%s kunserve_comm_backend=%s "
                    "capture_policy=%s",
                    self.tp_rank,
                    tuple(gmap.shape),
                    int(gmap[0, 0].item()),
                    int(gmap[0, 32].item()) if gmap.shape[1] > 32 else -1,
                    int(gmap[0, 64].item()) if gmap.shape[1] > 64 else -1,
                    int(gmap[0, 96].item()) if gmap.shape[1] > 96 else -1,
                    (
                        gmeta.physical_to_logical_map_cpu[0, :8].tolist()
                        if hasattr(gmeta, "physical_to_logical_map_cpu")
                        else "?"
                    ),
                    get_moe_a2a_backend().value,
                    kunserve_comm_backend,
                    capture_policy,
                )
        self._balloon_prepared_active_mappings = {
            layer_id: mapping.clone() for layer_id, mapping in active_mappings.items()
        }
        self._balloon_process_group_name = process_group_name
        self._balloon_kunserve_comm_backend = kunserve_comm_backend
        self._balloon_capture_policy = capture_policy

        resolved_ep_size = (
            int(runtime_ep_size) if runtime_ep_size is not None else self.moe_ep_size
        )
        resolved_moe_ep_rank = self._resolve_balloon_rank(
            explicit_rank=moe_ep_rank,
            rank_offset=runtime_rank_offset,
            default_rank=self.moe_ep_rank,
        )
        resolved_dispatch_ep_rank = self._resolve_balloon_rank(
            explicit_rank=dispatch_ep_rank,
            rank_offset=(
                dispatch_rank_offset
                if dispatch_rank_offset is not None
                else runtime_rank_offset
            ),
            default_rank=self.moe_ep_rank,
        )
        for layer in fused_layers:
            mapping = active_mappings[layer.layer_id]
            num_dispatch_experts = (
                int(self._balloon_global_expert_location_metadata.num_physical_experts)
                if self._balloon_global_expert_location_metadata is not None
                else int(layer.moe_runner_config.num_experts)
            )
            dispatcher_local_expert_mapping = build_dispatcher_physical_expert_mapping(
                num_physical_experts=num_dispatch_experts,
                runtime_ep_rank=resolved_moe_ep_rank,
                active_local_expert_mapping=mapping,
            )
            if int(layer.layer_id) == 0:
                probe_ids = [0, 32, 64, 96]
                probe_values = {
                    idx: (
                        int(dispatcher_local_expert_mapping[idx].item())
                        if idx < int(dispatcher_local_expert_mapping.numel())
                        else None
                    )
                    for idx in probe_ids
                }
                _kunserve_ms(
                    "[KUNSERVE-DBG] register_balloon_global_runtime_bundle layer0 dispatcher_map: "
                    "tp_rank=%s runtime_ep_rank=%s dispatch_ep_rank=%s active_rows=(%s,%s) "
                    "num_dispatch_experts=%s runtime_reduce_group_size=%s dispatcher_map%s=%s",
                    self.tp_rank,
                    resolved_moe_ep_rank,
                    resolved_dispatch_ep_rank,
                    int(mapping[0].item()),
                    int(mapping[-1].item()),
                    num_dispatch_experts,
                    runtime_group_size,
                    tuple(probe_ids),
                    probe_values,
                )
            global_runner_config = replace(
                layer.local_bundle.moe_runner_config,
                num_local_experts=int(mapping.numel()),
            )

            explicit_global_dispatcher = None
            explicit_global_runner = None

            if kunserve_comm_backend == "sglang":
                from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
                from sglang.srt.layers.moe.token_dispatcher.kunserve_standard import (
                    CrossReplicaStandardDispatcher,
                )
                from sglang.srt.layers.moe.utils import MoeRunnerBackend

                local_ep_size = int(
                    kunserve_backend_config.get("local_ep_size") or self.moe_ep_size
                )
                # Phase D: when capture_policy=fixed_padded, hand the
                # dispatcher capture_max_m so it pre-allocates static
                # buffers and takes the graph-safe path during capture.
                # max_num_token = max(capture_bs) * num_tokens_per_bs covers
                # every batch size we capture.  Leave as None for any other
                # policy so the dispatcher stays purely eager.
                resolved_capture_max_m: Optional[int] = None
                if str(capture_policy or "auto").lower() == "fixed_padded":
                    gr = getattr(self, "graph_runner", None)
                    if gr is not None:
                        resolved_capture_max_m = int(
                            getattr(gr, "max_num_token", 0) or 0
                        ) or None
                    if resolved_capture_max_m is None:
                        _kunserve_ms(
                            "[KUNSERVE-MS] capture_policy=fixed_padded requested "
                            "but graph_runner.max_num_token is unavailable; "
                            "GLOBAL bundle will fall back to eager dispatch."
                        )
                # Phase F: resolve the lane subgroup (cross-replica
                # 2-rank group) and the local TP group.  When both are
                # available the dispatcher will use:
                #   dispatch -> all_gather_into_tensor on lane_group
                #               (no [A,A,B,B] redundancy)
                #   combine  -> reduce_scatter on lane_group +
                #               all_reduce on local_tp_group
                #               (no wasted all_gather half)
                # When unavailable it transparently falls back to the
                # Phase D path on the global runtime_group.
                lane_group = None
                pg_names_lower = {
                    str(k).lower(): v for k, v in (kunserve_pg_names or {}).items()
                }
                lane_key = f"lane_{resolved_moe_ep_rank % local_ep_size}"
                lane_group_name = pg_names_lower.get(lane_key)
                if lane_group_name:
                    try:
                        lane_group = self._resolve_balloon_process_group(
                            lane_group_name
                        )
                    except Exception as exc:
                        _kunserve_ms(
                            "[KUNSERVE-MS] Phase F lane group %r unresolved "
                            "on tp_rank=%s: %r -- falling back to global "
                            "group for dispatch/combine.",
                            lane_group_name,
                            self.tp_rank,
                            exc,
                        )
                        lane_group = None
                local_tp_group = None
                try:
                    from sglang.srt.distributed.parallel_state import get_tp_group

                    tp_coord = get_tp_group()
                    local_tp_group = getattr(tp_coord, "device_group", None)
                except Exception as exc:
                    _kunserve_ms(
                        "[KUNSERVE-MS] Phase F local TP group unavailable: "
                        "%r -- combine optimization disabled.",
                        exc,
                    )
                    local_tp_group = None

                explicit_global_dispatcher = CrossReplicaStandardDispatcher(
                    group=runtime_group,
                    moe_runner_config=global_runner_config,
                    local_expert_mapping=dispatcher_local_expert_mapping,
                    local_ep_size=local_ep_size,
                    replica_rank=resolved_moe_ep_rank // local_ep_size,
                    global_rank=resolved_moe_ep_rank,
                    world_size=resolved_ep_size,
                    capture_max_m=resolved_capture_max_m,
                    lane_group=lane_group,
                    local_tp_group=local_tp_group,
                )
                # Use the normal Standard/Triton MoE core for the correctness
                # backend. This avoids DeepEP/NVSHMEM and DeepGEMM entirely;
                # the dispatcher itself performs global all-gather + all-reduce.
                explicit_global_runner = MoeRunner(
                    MoeRunnerBackend.TRITON,
                    global_runner_config,
                )
                if int(layer.layer_id) == 0:
                    _kunserve_ms(
                        "[KUNSERVE-MS] GLOBAL bundle uses CrossReplicaStandardDispatcher: "
                        "tp_rank=%s global_rank=%s world=%s local_ep_size=%s "
                        "replica_rank=%s capture_policy=%s capture_max_m=%s "
                        "phase_f_lane=%s phase_f_tp=%s backend_config=%s",
                        self.tp_rank,
                        resolved_moe_ep_rank,
                        resolved_ep_size,
                        local_ep_size,
                        resolved_moe_ep_rank // local_ep_size,
                        capture_policy,
                        resolved_capture_max_m,
                        (lane_group_name if lane_group is not None else None),
                        ("ready" if local_tp_group is not None else "unavailable"),
                        kunserve_backend_config,
                    )
            # When SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL is on (default), build
            # the GLOBAL bundle's dispatcher in DeepEP NORMAL mode explicitly
            # (NOT LL). Reason: this host's container does not expose
            # /dev/infiniband/, so NVSHMEM's IBGDA transport fails to init.
            # NVSHMEM falls back to NVL/SHM which is enough for the symmetric
            # heap setup, but LL atomic ops still loop on completion via
            # IBGDA-style signaling and deadlock silently after a few hundred
            # decode steps (see ab_20260507_053135 — BALLOON forward runs 7s
            # then both replicas freeze without any NCCL/Python exception).
            # NORMAL mode for cross-replica intra-node has rdma_ranks=1 and
            # low_latency_mode=False, so DeepEP buffer.py:96 skips NVSHMEM
            # init entirely. Trade-off: dynamic-shape dispatch can't be cuda
            # graph captured, so GLOBAL forward runs eager. LOCAL forward
            # (StandardDispatcher) keeps its cuda graph.
            elif envs.SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL.get():
                from sglang.srt.batch_overlap.two_batch_overlap import (
                    MaybeTboDeepEPDispatcher,
                )
                from sglang.srt.layers.moe.utils import DeepEPMode
                explicit_global_dispatcher = MaybeTboDeepEPDispatcher(
                    group=runtime_group,
                    router_topk=global_runner_config.top_k,
                    permute_fusion=True,
                    num_experts=global_runner_config.num_experts,
                    num_local_experts=int(mapping.numel()),
                    hidden_size=global_runner_config.hidden_size,
                    params_dtype=global_runner_config.params_dtype,
                    deepep_mode=DeepEPMode.NORMAL,  # ← key: force NORMAL, no NVSHMEM
                    async_finish=True,
                    return_recv_hook=True,
                )

            # CrossReplicaStandardDispatcher (sglang backend) uses each
            # rank's OWN local weight tensor rows (0..num_local_experts-1)
            # via the dispatcher's all_gather+reduce pattern — no VMM weight
            # expansion.  Passing mapping=[32..63] here would cause
            # build_dense_expert_runtime_tensors to call
            # tensor.narrow(0, 32, 32) on a 32-row weight tensor, which is
            # an out-of-bounds access → cudaErrorIllegalAddress at warmup.
            # For the sglang path: leave active_local_expert_mapping=None
            # so _active_runtime_tensors stays empty and get_runtime_tensor
            # falls back to the full local weight tensor.
            # The DeepEP/Mooncake paths retain the original mapping because
            # they DO expand the weight tensor via VMM.
            sglang_path = (
                kunserve_comm_backend == "sglang"
                and explicit_global_dispatcher is not None
            )
            layer.register_runtime_bundle(
                variant="global",
                moe_runner_config=global_runner_config,
                dispatcher=explicit_global_dispatcher,  # None ⇒ default DeepEP via create_moe_dispatcher
                runner=explicit_global_runner,
                group=runtime_group,
                moe_ep_size=resolved_ep_size,
                moe_ep_rank=resolved_moe_ep_rank,
                moe_tp_size=layer.moe_tp_size,
                moe_tp_rank=layer.moe_tp_rank,
                num_local_experts=int(mapping.numel()),
                active_local_expert_mapping=(None if sglang_path else mapping),
                dispatcher_local_expert_mapping=dispatcher_local_expert_mapping,
                # GLOBAL bundle combine already aggregates each token's expert
                # outputs across the whole cross-replica EP world:
                #   - DeepEP does it inside DeepEP combine;
                #   - CrossReplicaStandardDispatcher does it with a global
                #     all-reduce then slices back to local tokens.
                # Adding the FusedMoE-level tp_group all-reduce on top would
                # double-reduce within the local TP group. Companion: LOCAL
                # bundle (StandardDispatcher) registered in
                # _force_local_bundle_to_standard_dispatcher passes
                # reduce_results=True because Standard's combine is a no-op and
                # partial sums need a final tp_group all-reduce.
                reduce_results=False,
            )
        return active_mappings

    def ensure_cuda_graph_variant_captured(self, variant: str) -> None:
        if self.graph_runner is None or not hasattr(
            self.graph_runner, "ensure_variant_captured"
        ):
            return
        self.graph_runner.ensure_variant_captured(variant)

    def _get_balloon_kv_cache(self):
        return self.token_to_kv_pool_allocator.get_kvcache()

    def _bytes_per_balloon_kv_slot(self) -> int:
        return int(self._get_balloon_kv_cache().bytes_per_slot())

    def _balloon_kv_vmm_headroom_slots(self) -> int:
        kv_cache = self._get_balloon_kv_cache()
        if not getattr(kv_cache, "_kv_vmm_enabled", False):
            return 0
        reserve_rows = int(
            getattr(kv_cache, "_kv_reserve_rows", int(kv_cache.size) + self.page_size)
        )
        return max(0, reserve_rows - self.page_size - int(kv_cache.size))

    def _moe_weight_vmm_enabled(self) -> bool:
        fused_layers = self._iter_fused_moe_layers()
        if not fused_layers:
            return False
        missing = [
            layer
            for layer in fused_layers
            if not bool(getattr(layer, "_sglang_moe_weight_vmm_allocations", {}))
        ]
        if missing:
            if not getattr(self, "_kunserve_logged_missing_moe_weight_vmm", False):
                sample = missing[:4]
                _kunserve_ms(
                    "[KUNSERVE-MS] MoE weight VMM unavailable on %d/%d layers; "
                    "sample=%s",
                    len(missing),
                    len(fused_layers),
                    [
                        {
                            "layer_id": getattr(layer, "layer_id", None),
                            "quant_method": type(
                                getattr(layer, "quant_method", None)
                            ).__name__,
                            "w13_dtype": str(
                                getattr(getattr(layer, "w13_weight", None), "dtype", None)
                            ),
                            "w2_dtype": str(
                                getattr(getattr(layer, "w2_weight", None), "dtype", None)
                            ),
                        }
                        for layer in sample
                    ],
                )
                self._kunserve_logged_missing_moe_weight_vmm = True
            return False
        return True

    def _sync_balloon_weight_allocations(self) -> None:
        for layer in self._iter_fused_moe_layers():
            for allocation in getattr(
                layer, "_sglang_moe_weight_vmm_allocations", {}
            ).values():
                allocation.sync_from_region()

    def _validate_global_bundle_for_expert_offload(
        self, retained_local_experts: int
    ) -> Dict[int, str]:
        offload_modes: Dict[int, str] = {}
        for layer in self._iter_fused_moe_layers():
            bundle = getattr(layer, "global_bundle", None)
            if bundle is None:
                raise ValueError(
                    f"Layer {layer.layer_id} does not have a registered global runtime bundle."
                )
            if layer.num_fused_shared_experts > 0:
                raise ValueError(
                    "Balloon tail offload is not implemented for fused shared experts."
                )
            mapping = bundle.active_local_expert_mapping
            if mapping is None:
                raise ValueError(
                    f"Layer {layer.layer_id} global bundle is missing active_local_expert_mapping."
                )
            if int(mapping.numel()) != retained_local_experts:
                raise ValueError(
                    f"Layer {layer.layer_id} retains {int(mapping.numel())} experts, expected {retained_local_experts}."
                )
            routed_local_experts = (
                layer.local_bundle.num_local_experts - layer.num_fused_shared_experts
            )
            mapping_start = int(mapping[0].item())
            mapping_end = int(mapping[-1].item())
            if mapping_start == 0 and mapping_end == retained_local_experts - 1:
                offload_modes[layer.layer_id] = "tail"
                continue
            if (
                mapping_start == routed_local_experts - retained_local_experts
                and mapping_end == routed_local_experts - 1
            ):
                offload_modes[layer.layer_id] = "head"
                continue
            raise ValueError(
                "Current balloon offload implementation only supports keeping a contiguous local prefix or suffix."
            )
        return offload_modes

    def _warmup_balloon_global_runtime(
        self,
        *,
        target_variant: str,
        runtime_ep_size: Optional[int],
        moe_ep_rank: Optional[int],
        dispatch_ep_rank: Optional[int],
        runtime_rank_offset: Optional[int],
        dispatch_rank_offset: Optional[int],
        retained_local_experts: Optional[int],
        active_local_expert_mapping: Optional[List[int]],
        active_local_expert_mapping_by_layer: Optional[Dict[int, List[int]]],
        physical_to_logical_map,
        process_group_name: Optional[str],
        capture_cuda_graph: bool,
        kunserve_comm_backend: str = "deepep",
        capture_policy: str = "auto",
        kunserve_pg_names: Optional[Dict[str, str]] = None,
        kunserve_backend_config: Optional[Dict[str, Any]] = None,
    ) -> List[torch.nn.Module]:
        """Idempotent setup that prepares the global runtime bundle and CUDA graph.

        Shared by `prepare_balloon` (state-changing path) and `warmup_balloon`
        (state-preserving path). Does NOT touch `_balloon_state` or
        `_balloon_graph_replay_enabled`. Returns the list of FusedMoE layers
        in case the caller needs them.
        """
        fused_layers = self._iter_fused_moe_layers()

        # Snapshot the live LOCAL expert-location metadata while we're still
        # in `local` state so restore_from_balloon can put it back later.
        # `prepared` state retains this snapshot; we re-take it here so that
        # warmup-only (state stays local) is consistent with prepare's
        # behavior.
        if not self.is_draft_worker and self._balloon_state == "local":
            live_metadata = get_global_expert_location_metadata()
            if live_metadata is not None:
                self._balloon_local_expert_location_metadata = copy.deepcopy(
                    live_metadata
                )

        if target_variant != "global":
            return fused_layers

        if not fused_layers:
            raise ValueError("Global balloon runtime requires FusedMoE layers.")

        # Re-register the global bundle when callers gave us a layout (idempotent
        # — overwrites the prior bundle) or when none has been registered yet.
        if (
            retained_local_experts is not None
            or active_local_expert_mapping is not None
            or active_local_expert_mapping_by_layer is not None
            or getattr(fused_layers[0], "global_bundle", None) is None
        ):
            self.register_balloon_global_runtime_bundle(
                runtime_ep_size=runtime_ep_size,
                moe_ep_rank=moe_ep_rank,
                dispatch_ep_rank=dispatch_ep_rank,
                runtime_rank_offset=runtime_rank_offset,
                dispatch_rank_offset=dispatch_rank_offset,
                retained_local_experts=retained_local_experts,
                active_local_expert_mapping=active_local_expert_mapping,
                active_local_expert_mapping_by_layer=active_local_expert_mapping_by_layer,
                physical_to_logical_map=physical_to_logical_map,
                process_group_name=process_group_name,
                kunserve_comm_backend=kunserve_comm_backend,
                capture_policy=capture_policy,
                kunserve_pg_names=kunserve_pg_names,
                kunserve_backend_config=kunserve_backend_config,
            )

        # ensure_cuda_graph_variant_captured short-circuits when the graph for
        # this variant is already in `self.graphs`.
        if capture_cuda_graph:
            # Skip GLOBAL capture for known dynamic-shape data paths:
            #   - deepep + DEEPEP_NORMAL=True  → DeepEP NORMAL dispatch_a/b
            #     has dynamic shape (no NVSHMEM so we can't go to LL).
            #   - sglang + capture_policy != fixed_padded → the dispatcher
            #     stays in dynamic eager mode (no static buffers, host syncs
            #     present).
            # The sglang+fixed_padded path uses pre-allocated static buffers
            # in CrossReplicaStandardDispatcher and IS graph-safe.
            backend_lower = str(kunserve_comm_backend or "deepep").lower()
            policy_lower = str(capture_policy or "auto").lower()
            if backend_lower == "sglang":
                skip_capture = (
                    target_variant == "global"
                    and policy_lower != "fixed_padded"
                )
            else:
                skip_capture = (
                    target_variant == "global"
                    and envs.SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL.get()
                )
            if not skip_capture:
                # NCCL communicator preheat for the sglang fixed_padded
                # path.  The static CrossReplicaStandardDispatcher uses
                # `all_gather_into_tensor` (dispatch) and `all_reduce`
                # (combine) on `runtime_group`.  The very first time NCCL
                # touches a process group it does communicator
                # bootstrap; if that bootstrap happens *inside* the cuda
                # graph capture context, NCCL either refuses
                # ("init not allowed in graph mode") or bakes
                # per-launch metadata into the graph that goes stale
                # on replay.  Issue one dummy of each collective on the
                # group while we are outside the capture context so the
                # communicator is fully built before capture begins.
                # The deepep path does its own NCCL/NVSHMEM warmup
                # through the DeepEP buffer setup, so we only do this
                # for the sglang backend.
                if (
                    backend_lower == "sglang"
                    and policy_lower == "fixed_padded"
                ):
                    # Collect every NCCL group the static dispatcher
                    # may touch during capture.  Each group's
                    # communicator init must complete OUTSIDE the
                    # graph_capture context.  Order doesn't matter; we
                    # just hit every (group, op) pair the dispatcher
                    # uses.
                    preheat_groups: List[
                        Tuple[str, Optional["dist.ProcessGroup"]]
                    ] = []
                    try:
                        preheat_groups.append(
                            (
                                "runtime",
                                self._resolve_balloon_process_group(
                                    process_group_name
                                ),
                            )
                        )
                    except Exception as exc:
                        _kunserve_ms(
                            "[KUNSERVE-MS] preheat: failed to resolve "
                            "runtime group %r: %r",
                            process_group_name,
                            exc,
                        )
                    # Phase F: also preheat the lane group + local TP
                    # group when present so static dispatch/combine
                    # collectives in the captured graph never trigger
                    # in-graph communicator init.
                    pg_names_map = {
                        str(k).lower(): v
                        for k, v in (kunserve_pg_names or {}).items()
                    }
                    for lane_idx in range(int(self.moe_ep_size)):
                        lane_name = pg_names_map.get(f"lane_{lane_idx}")
                        if lane_name is None:
                            continue
                        if int(lane_idx) != int(self.tp_rank):
                            # Not our lane; skip silently.
                            continue
                        try:
                            preheat_groups.append(
                                (
                                    f"lane_{lane_idx}",
                                    self._resolve_balloon_process_group(
                                        lane_name
                                    ),
                                )
                            )
                        except Exception as exc:
                            _kunserve_ms(
                                "[KUNSERVE-MS] preheat: failed to resolve "
                                "lane group %r: %r",
                                lane_name,
                                exc,
                            )
                    try:
                        from sglang.srt.distributed.parallel_state import (
                            get_tp_group,
                        )

                        preheat_groups.append(
                            ("local_tp", get_tp_group().device_group)
                        )
                    except Exception as exc:
                        _kunserve_ms(
                            "[KUNSERVE-MS] preheat: failed to resolve "
                            "local TP group: %r",
                            exc,
                        )

                    preheat_device = torch.device(
                        "cuda", torch.cuda.current_device()
                    )
                    for label, group in preheat_groups:
                        if group is None:
                            continue
                        try:
                            world = int(dist.get_world_size(group=group))
                            ag_in = torch.zeros(1, device=preheat_device)
                            ag_out = torch.zeros(world, device=preheat_device)
                            dist.all_gather_into_tensor(
                                ag_out, ag_in, group=group
                            )
                            rs_in = torch.zeros(world, device=preheat_device)
                            rs_out = torch.zeros(1, device=preheat_device)
                            try:
                                dist.reduce_scatter_tensor(
                                    rs_out, rs_in, group=group
                                )
                            except Exception:
                                # Some NCCL builds expose
                                # _reduce_scatter_base; ignore failure
                                # for groups that don't support
                                # reduce_scatter (the dispatcher only
                                # uses it on lane group anyway).
                                pass
                            ar_buf = torch.zeros(1, device=preheat_device)
                            dist.all_reduce(
                                ar_buf,
                                op=dist.ReduceOp.SUM,
                                group=group,
                            )
                            torch.cuda.synchronize()
                            _kunserve_ms(
                                "[KUNSERVE-MS] NCCL communicator preheat done "
                                "for %s: world=%d (all_gather_into_tensor + "
                                "reduce_scatter_tensor + all_reduce)",
                                label,
                                world,
                            )
                        except Exception as exc:
                            _kunserve_ms(
                                "[KUNSERVE-MS] NCCL communicator preheat "
                                "FAILED on %s: %r — proceeding into capture "
                                "anyway",
                                label,
                                exc,
                            )
                self.ensure_cuda_graph_variant_captured(target_variant)
            else:
                _kunserve_ms(
                    "[KUNSERVE-MS] skip GLOBAL cuda graph capture: "
                    "GLOBAL bundle uses a dynamic eager communication path "
                    "(comm_backend=%s, capture_policy=%s). BALLOON forward "
                    "will run eager.",
                    kunserve_comm_backend,
                    capture_policy,
                )

        return fused_layers

    def prepare_balloon(
        self,
        *,
        target_variant: str = "global",
        runtime_ep_size: Optional[int] = None,
        moe_ep_rank: Optional[int] = None,
        dispatch_ep_rank: Optional[int] = None,
        runtime_rank_offset: Optional[int] = None,
        dispatch_rank_offset: Optional[int] = None,
        retained_local_experts: Optional[int] = None,
        active_local_expert_mapping: Optional[List[int]] = None,
        active_local_expert_mapping_by_layer: Optional[Dict[int, List[int]]] = None,
        physical_to_logical_map=None,
        process_group_name: Optional[str] = None,
        capture_cuda_graph: bool = True,
        kunserve_comm_backend: str = "deepep",
        capture_policy: str = "auto",
        kunserve_pg_names: Optional[Dict[str, str]] = None,
        kunserve_backend_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        target_variant = self._normalize_balloon_variant(target_variant)
        _kunserve_ms(
            "[KUNSERVE-MS] prepare start: target_variant=%s pg=%s "
            "comm_backend=%s capture_policy=%s capture_cuda_graph=%s "
            "runtime_ep_size=%s rank_offset=%s",
            target_variant,
            process_group_name,
            kunserve_comm_backend,
            capture_policy,
            capture_cuda_graph,
            runtime_ep_size,
            runtime_rank_offset,
        )
        logger.info(
            "Prepare balloon: target_variant=%s runtime_ep_size=%s moe_ep_rank=%s dispatch_ep_rank=%s "
            "runtime_rank_offset=%s dispatch_rank_offset=%s process_group=%s "
            "comm_backend=%s capture_policy=%s capture_cuda_graph=%s",
            target_variant,
            runtime_ep_size,
            moe_ep_rank,
            dispatch_ep_rank,
            runtime_rank_offset,
            dispatch_rank_offset,
            process_group_name,
            kunserve_comm_backend,
            capture_policy,
            capture_cuda_graph,
        )

        self._warmup_balloon_global_runtime(
            target_variant=target_variant,
            runtime_ep_size=runtime_ep_size,
            moe_ep_rank=moe_ep_rank,
            dispatch_ep_rank=dispatch_ep_rank,
            runtime_rank_offset=runtime_rank_offset,
            dispatch_rank_offset=dispatch_rank_offset,
            retained_local_experts=retained_local_experts,
            active_local_expert_mapping=active_local_expert_mapping,
            active_local_expert_mapping_by_layer=active_local_expert_mapping_by_layer,
            physical_to_logical_map=physical_to_logical_map,
            process_group_name=process_group_name,
            capture_cuda_graph=capture_cuda_graph,
            kunserve_comm_backend=kunserve_comm_backend,
            capture_policy=capture_policy,
            kunserve_pg_names=kunserve_pg_names,
            kunserve_backend_config=kunserve_backend_config,
        )

        self._balloon_prepared_variant = target_variant
        self._balloon_state = "prepared"
        self._balloon_graph_replay_enabled = False
        self._balloon_last_error = None
        _kunserve_ms(
            "[KUNSERVE-MS] prepare done: state=prepared target_variant=%s captured_variants=%s graph_replay_enabled=False",
            target_variant,
            self.get_cuda_graph_capture_variants(),
        )
        return self.get_balloon_status()

    def warmup_balloon(
        self,
        *,
        target_variant: str = "global",
        runtime_ep_size: Optional[int] = None,
        moe_ep_rank: Optional[int] = None,
        dispatch_ep_rank: Optional[int] = None,
        runtime_rank_offset: Optional[int] = None,
        dispatch_rank_offset: Optional[int] = None,
        retained_local_experts: Optional[int] = None,
        active_local_expert_mapping: Optional[List[int]] = None,
        active_local_expert_mapping_by_layer: Optional[Dict[int, List[int]]] = None,
        physical_to_logical_map=None,
        process_group_name: Optional[str] = None,
        capture_cuda_graph: bool = True,
        kunserve_comm_backend: str = "deepep",
        capture_policy: str = "auto",
        kunserve_pg_names: Optional[Dict[str, str]] = None,
        kunserve_backend_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Pre-build the GLOBAL runtime bundle and capture its CUDA graph
        without changing balloon state.

        Designed for the KunServe controller to call once at startup, after
        the cross-replica process group is up. After this returns, a
        subsequent prepare_balloon → commit_balloon transition skips the
        ~30s graph capture (it's already cached).

        Idempotency:
        - State must be `local`. Calling from `prepared` or `balloon` is a
          no-op (those paths already did or are doing the work).
        - The underlying `ensure_cuda_graph_variant_captured` is idempotent
          via `has_captured_variant`; calling warmup twice is cheap.
        """
        target_variant = self._normalize_balloon_variant(target_variant)
        if self._balloon_state != "local":
            logger.info(
                "Warmup balloon skipped: current state=%s (not local).",
                self._balloon_state,
            )
            return self.get_balloon_status()

        _kunserve_ms(
            "[KUNSERVE-MS] warmup start: target_variant=%s pg=%s "
            "comm_backend=%s capture_policy=%s capture_cuda_graph=%s "
            "runtime_ep_size=%s rank_offset=%s",
            target_variant,
            process_group_name,
            kunserve_comm_backend,
            capture_policy,
            capture_cuda_graph,
            runtime_ep_size,
            runtime_rank_offset,
        )
        logger.info(
            "Warmup balloon: target_variant=%s runtime_ep_size=%s moe_ep_rank=%s "
            "runtime_rank_offset=%s process_group=%s comm_backend=%s "
            "capture_policy=%s capture_cuda_graph=%s",
            target_variant,
            runtime_ep_size,
            moe_ep_rank,
            runtime_rank_offset,
            process_group_name,
            kunserve_comm_backend,
            capture_policy,
            capture_cuda_graph,
        )

        try:
            self._warmup_balloon_global_runtime(
                target_variant=target_variant,
                runtime_ep_size=runtime_ep_size,
                moe_ep_rank=moe_ep_rank,
                dispatch_ep_rank=dispatch_ep_rank,
                runtime_rank_offset=runtime_rank_offset,
                dispatch_rank_offset=dispatch_rank_offset,
                retained_local_experts=retained_local_experts,
                active_local_expert_mapping=active_local_expert_mapping,
                active_local_expert_mapping_by_layer=active_local_expert_mapping_by_layer,
                physical_to_logical_map=physical_to_logical_map,
                process_group_name=process_group_name,
                capture_cuda_graph=capture_cuda_graph,
                kunserve_comm_backend=kunserve_comm_backend,
                capture_policy=capture_policy,
                kunserve_pg_names=kunserve_pg_names,
                kunserve_backend_config=kunserve_backend_config,
            )
        except Exception as exc:
            self._balloon_last_error = str(exc)
            logger.exception("Warmup balloon failed: target_variant=%s", target_variant)
            raise

        self._balloon_last_error = None
        _kunserve_ms(
            "[KUNSERVE-MS] warmup done: state=local target_variant=%s captured_variants=%s graph_replay_enabled=True",
            target_variant,
            self.get_cuda_graph_capture_variants(),
        )
        return self.get_balloon_status()

    def commit_balloon(
        self,
        *,
        target_variant: str = "global",
        offload_local_experts: int = 0,
        num_slots_to_expand: Optional[int] = None,
        require_prepared: bool = True,
    ) -> Dict[str, Any]:
        from sglang.srt.utils.cuda_vmm import (
            DonorLedger,
            min_granularity_aligned_row_count,
        )

        target_variant = self._normalize_balloon_variant(target_variant)
        offload_local_experts = int(offload_local_experts)
        if self._balloon_state == "balloon":
            current_variant = self.get_cuda_graph_runtime_variant()
            current_offload_local_experts = int(self._balloon_offloaded_local_experts)
            if (
                current_variant == target_variant
                and current_offload_local_experts == offload_local_experts
            ):
                logger.info(
                    "Commit balloon requested for an already-active runtime; returning existing status: "
                    "runtime_variant=%s offloaded_local_experts=%d",
                    current_variant,
                    current_offload_local_experts,
                )
                self._balloon_last_error = None
                return self.get_balloon_status()
            raise ValueError(
                "Model runner is already in balloon mode with "
                f"runtime_variant={current_variant} offloaded_local_experts={current_offload_local_experts}; "
                f"requested target_variant={target_variant} offload_local_experts={offload_local_experts}."
            )
        if require_prepared and self._balloon_prepared_variant != target_variant:
            raise ValueError(
                f"Balloon variant '{target_variant}' must be prepared before commit."
            )

        previous_state = self._balloon_state
        previous_variant = self.get_cuda_graph_runtime_variant()
        previous_replay_enabled = self._balloon_graph_replay_enabled
        borrowed = DonorLedger(label="balloon_commit")
        remaining_donors = None
        added_slots = 0
        self._balloon_unused_donors = None
        self._balloon_graph_replay_enabled = False
        _kunserve_ms(
            "[KUNSERVE-MS] commit start: target_variant=%s offload=%d (graph replay PAUSED)",
            target_variant,
            offload_local_experts,
        )

        try:
            if offload_local_experts > 0:
                retained_local_experts = None
                offload_modes_by_layer: Dict[int, str] = {}
                for layer in self._iter_fused_moe_layers():
                    routed_local_experts = (
                        layer.local_bundle.num_local_experts
                        - layer.num_fused_shared_experts
                    )
                    if offload_local_experts >= routed_local_experts:
                        raise ValueError(
                            f"Cannot offload {offload_local_experts} experts from layer {layer.layer_id}; "
                            f"only {routed_local_experts - 1} routed experts can be removed while keeping at least one active expert."
                        )
                    retained_local_experts = (
                        routed_local_experts - offload_local_experts
                    )

                    allocations = getattr(
                        layer, "_sglang_moe_weight_vmm_allocations", {}
                    )
                    if not allocations:
                        raise ValueError(
                            f"Layer {layer.layer_id} does not expose VMM-backed MoE allocations. "
                            "Enable SGLANG_EXPERIMENTAL_CUDA_VMM=1 and "
                            "SGLANG_EXPERIMENTAL_VMM_MOE_WEIGHTS=1 before launching the server."
                        )

                assert retained_local_experts is not None
                if target_variant == "global":
                    offload_modes_by_layer = (
                        self._validate_global_bundle_for_expert_offload(
                            retained_local_experts
                        )
                    )
                logger.info(
                    "Commit balloon: target_variant=%s offload_local_experts=%d retained_local_experts=%d offload_modes=%s",
                    target_variant,
                    offload_local_experts,
                    retained_local_experts,
                    offload_modes_by_layer,
                )

                for layer in self._iter_fused_moe_layers():
                    allocations = getattr(
                        layer, "_sglang_moe_weight_vmm_allocations", {}
                    )
                    offload_mode = offload_modes_by_layer.get(layer.layer_id, "tail")
                    for name in ("w13_weight", "w2_weight"):
                        allocation = allocations.get(name)
                        if allocation is None:
                            continue
                        rows_per_borrow = min_granularity_aligned_row_count(
                            allocation.row_bytes, allocation.region.granularity
                        )
                        if offload_local_experts % rows_per_borrow != 0:
                            raise ValueError(
                                f"Cannot offload {offload_local_experts} experts from "
                                f"layer {layer.layer_id} param '{name}'; "
                                f"row_bytes={allocation.row_bytes} requires borrow "
                                f"chunks of {rows_per_borrow} rows for "
                                f"granularity={allocation.region.granularity}."
                            )
                        borrow_rows = (
                            allocation.borrow_head_rows
                            if offload_mode == "head"
                            else allocation.borrow_tail_rows
                        )
                        for expert_offset in range(
                            0, offload_local_experts, rows_per_borrow
                        ):
                            borrowed.add(
                                borrow_rows(
                                    rows_per_borrow,
                                    owner={
                                        "layer_id": layer.layer_id,
                                        "param_name": name,
                                        "num_experts": offload_local_experts,
                                        "offload_mode": offload_mode,
                                        "expert_offset": expert_offset,
                                        "chunk_rows": rows_per_borrow,
                                    },
                                )
                            )

                donor_segments = borrowed.pop_all()
                remaining_donors = list(donor_segments)
                total_donor_bytes = sum(
                    segment.size_bytes for segment in donor_segments
                )
                kv_cache = self._get_balloon_kv_cache()
                bytes_per_slot = self._bytes_per_balloon_kv_slot()
                max_slots_from_donor = total_donor_bytes // bytes_per_slot
                if self.page_size > 1:
                    max_slots_from_donor = (
                        max_slots_from_donor // self.page_size * self.page_size
                    )
                kv_vmm_headroom_slots = self._balloon_kv_vmm_headroom_slots()
                if self.page_size > 1:
                    kv_vmm_headroom_slots = (
                        kv_vmm_headroom_slots // self.page_size * self.page_size
                    )
                added_slots = resolve_balloon_kv_slots_to_expand(
                    max_slots_from_donor=max_slots_from_donor,
                    kv_vmm_headroom_slots=kv_vmm_headroom_slots,
                    num_slots_to_expand=num_slots_to_expand,
                )
                donor_compatible_slots = (
                    kv_cache.quantize_slots_to_whole_donor_segments(
                        added_slots, remaining_donors
                    )
                )
                if donor_compatible_slots != added_slots:
                    logger.info(
                        "Quantized balloon KV slots from %d to %d to preserve whole donor segments.",
                        added_slots,
                        donor_compatible_slots,
                    )
                    added_slots = donor_compatible_slots
                logger.info(
                    "Balloon donor summary: donor_segments=%d donor_bytes=%d bytes_per_slot=%d "
                    "max_slots_from_donor=%d kv_vmm_headroom=%d requested_slots=%s final_added_slots=%d",
                    len(donor_segments),
                    total_donor_bytes,
                    bytes_per_slot,
                    max_slots_from_donor,
                    kv_vmm_headroom_slots,
                    num_slots_to_expand,
                    added_slots,
                )

                _kunserve_ms(
                    "[KUNSERVE-MS] donors collected: layers=%d segments=%d donor_bytes=%d "
                    "bytes_per_slot=%d candidate_slots=%d",
                    len(self._iter_fused_moe_layers()),
                    len(donor_segments),
                    total_donor_bytes,
                    bytes_per_slot,
                    added_slots,
                )

                kv_cache_expanded = False
                allocator_expanded = False
                if added_slots > 0:
                    kv_cache.expand_by_slots(
                        added_slots, donor_segments=remaining_donors
                    )
                    kv_cache_expanded = True
                    self.token_to_kv_pool_allocator.expand_by_slots(added_slots)
                    allocator_expanded = True
                    self.sync_kv_capacity(delta_slots=added_slots)
                    _kunserve_ms(
                        "[KUNSERVE-MS] KV expanded: added_slots=%d max_total_num_tokens=%d (gain ~%.2f GiB)",
                        added_slots,
                        self.max_total_num_tokens,
                        added_slots * bytes_per_slot / (1024.0**3),
                    )

                unused_donors = DonorLedger(label="balloon_unused")
                unused_donors.extend(remaining_donors)
                self._balloon_unused_donors = unused_donors
            else:
                self._balloon_unused_donors = DonorLedger(label="balloon_unused")

            if target_variant == "global":
                _kunserve_ms(
                    "[KUNSERVE-DBG] commit_balloon variant=global: "
                    "tp_rank=%s _balloon_global_expert_location_metadata is %s, "
                    "ep_dispatch_algorithm=%s, will_call_update=%s",
                    self.tp_rank,
                    (
                        "None"
                        if self._balloon_global_expert_location_metadata is None
                        else "set"
                    ),
                    getattr(self.server_args, "ep_dispatch_algorithm", None),
                    self._balloon_global_expert_location_metadata is not None,
                )
                if self._balloon_global_expert_location_metadata is not None:
                    self._update_live_expert_location_metadata(
                        self._balloon_global_expert_location_metadata
                    )
                self._switch_all_fused_moe_runtime_bundles("global")
            else:
                if self._balloon_local_expert_location_metadata is not None:
                    self._update_live_expert_location_metadata(
                        self._balloon_local_expert_location_metadata
                    )
                self._switch_all_fused_moe_runtime_bundles("local")
            _kunserve_ms(
                "[KUNSERVE-MS] variant switched to %s (FusedMoE bundles re-bound)",
                target_variant,
            )

            self._balloon_state = "balloon"
            self._balloon_offloaded_local_experts = int(offload_local_experts)
            self._balloon_added_slots = int(added_slots)
            self._balloon_last_error = None
            self._balloon_graph_replay_enabled = True

            # Phase E: reserve a dummy KV slot for keepalive batches.
            # Done after the KV pool has been expanded (so we draw from
            # the new headroom, not from real-request capacity).  Only
            # for the sglang backend's keepalive path; DeepEP uses a
            # different keepalive mechanism.
            if (
                str(self._balloon_kunserve_comm_backend or "").lower()
                == "sglang"
                and self._kunserve_keepalive_dummy_kv_slot is None
            ):
                try:
                    dummy_alloc = self.token_to_kv_pool_allocator.alloc(1)
                    if (
                        dummy_alloc is not None
                        and getattr(dummy_alloc, "numel", lambda: 0)() >= 1
                    ):
                        self._kunserve_keepalive_dummy_kv_slot_tensor = dummy_alloc
                        self._kunserve_keepalive_dummy_kv_slot = int(
                            dummy_alloc[0].item()
                        )
                        _kunserve_ms(
                            "[KUNSERVE-MS] keepalive dummy KV slot reserved: "
                            "slot=%d (allocator size=%d after balloon expand)",
                            self._kunserve_keepalive_dummy_kv_slot,
                            int(self.token_to_kv_pool_allocator.size),
                        )
                    else:
                        _kunserve_ms(
                            "[KUNSERVE-MS] keepalive dummy KV slot alloc returned "
                            "empty -- keepalive will fall back to slot 0 (UNSAFE: "
                            "may corrupt slot 0's KV during graph replay)"
                        )
                except Exception as exc:
                    _kunserve_ms(
                        "[KUNSERVE-MS] keepalive dummy KV slot alloc raised: %r "
                        "-- keepalive will fall back to slot 0 (UNSAFE: may "
                        "corrupt slot 0's KV during graph replay)",
                        exc,
                    )

            _kunserve_ms(
                "[KUNSERVE-MS] commit done: state=balloon variant=%s offloaded=%d added_kv_slots=%d max_total_num_tokens=%d (graph replay RESUMED)",
                self.get_cuda_graph_runtime_variant(),
                self._balloon_offloaded_local_experts,
                self._balloon_added_slots,
                self.max_total_num_tokens,
            )
            logger.info(
                "Committed balloon runtime: state=%s runtime_variant=%s offloaded_local_experts=%d added_kv_slots=%d max_total_num_tokens=%d",
                self._balloon_state,
                self.get_cuda_graph_runtime_variant(),
                self._balloon_offloaded_local_experts,
                self._balloon_added_slots,
                self.max_total_num_tokens,
            )
            return self.get_balloon_status()
        except Exception as exc:
            self._balloon_last_error = str(exc)
            logger.exception(
                "Commit balloon failed: target_variant=%s offload_local_experts=%d",
                target_variant,
                offload_local_experts,
            )
            if (
                added_slots > 0
                and "allocator_expanded" in locals()
                and allocator_expanded
            ):
                try:
                    self.token_to_kv_pool_allocator.shrink_tail(added_slots)
                except Exception:
                    logger.exception(
                        "Failed to roll back balloon KV allocator expansion."
                    )
            if (
                added_slots > 0
                and "kv_cache_expanded" in locals()
                and kv_cache_expanded
            ):
                try:
                    returned = self._get_balloon_kv_cache().shrink_tail(added_slots)
                    returned.restore_all()
                except Exception:
                    logger.exception("Failed to roll back balloon KV expansion.")
            if self._balloon_unused_donors is not None:
                try:
                    self._balloon_unused_donors.restore_all()
                except Exception:
                    logger.exception("Failed to restore unused balloon donor segments.")
            if remaining_donors is not None:
                rollback_donors = DonorLedger(label="balloon_commit_rollback")
                rollback_donors.extend(remaining_donors)
                rollback_donors.restore_all()
            else:
                borrowed.restore_all()
            self._sync_balloon_weight_allocations()
            self._switch_all_fused_moe_runtime_bundles(previous_variant)
            self._balloon_state = previous_state
            self._balloon_graph_replay_enabled = previous_replay_enabled
            self._balloon_offloaded_local_experts = 0
            self._balloon_added_slots = 0
            raise

    def restore_from_balloon(self) -> Dict[str, Any]:
        if self._balloon_state == "local":
            self._balloon_graph_replay_enabled = True
            return self.get_balloon_status()
        logger.info(
            "Restore balloon: current_state=%s added_kv_slots=%d offloaded_local_experts=%d",
            self._balloon_state,
            self._balloon_added_slots,
            self._balloon_offloaded_local_experts,
        )
        self._balloon_graph_replay_enabled = False
        # Phase E: free the keepalive dummy KV slot before shrinking
        # the pool tail; if the slot index is in the tail we're about
        # to remove, freeing it after shrink would be invalid.
        if self._kunserve_keepalive_dummy_kv_slot_tensor is not None:
            try:
                self.token_to_kv_pool_allocator.free(
                    self._kunserve_keepalive_dummy_kv_slot_tensor
                )
            except Exception as exc:
                _kunserve_ms(
                    "[KUNSERVE-MS] keepalive dummy KV slot free failed: %r "
                    "(leaking 1 slot; restore continues)",
                    exc,
                )
            self._kunserve_keepalive_dummy_kv_slot_tensor = None
            self._kunserve_keepalive_dummy_kv_slot = None
        if self._balloon_added_slots > 0:
            self.token_to_kv_pool_allocator.shrink_tail(self._balloon_added_slots)
            returned = self._get_balloon_kv_cache().shrink_tail(
                self._balloon_added_slots
            )
            returned.restore_all()
            self.sync_kv_capacity(
                max_total_num_tokens=self.token_to_kv_pool_allocator.size
            )

        if self._balloon_unused_donors is not None:
            self._balloon_unused_donors.restore_all()
            self._balloon_unused_donors = None

        self._sync_balloon_weight_allocations()
        if self._balloon_local_expert_location_metadata is not None:
            self._update_live_expert_location_metadata(
                self._balloon_local_expert_location_metadata
            )
        self._switch_all_fused_moe_runtime_bundles("local")
        self._balloon_state = "local"
        self._balloon_prepared_variant = None
        self._balloon_offloaded_local_experts = 0
        self._balloon_added_slots = 0
        self._balloon_graph_replay_enabled = True
        self._balloon_last_error = None
        logger.info(
            "Restored local runtime: state=%s runtime_variant=%s max_total_num_tokens=%d",
            self._balloon_state,
            self.get_cuda_graph_runtime_variant(),
            self.max_total_num_tokens,
        )
        return self.get_balloon_status()

    def sync_kv_capacity(
        self,
        *,
        max_total_num_tokens: Optional[int] = None,
        delta_slots: int = 0,
    ) -> Dict[str, Any]:
        if self.is_hybrid_swa:
            raise ValueError("Balloon KV capacity sync does not support hybrid SWA.")

        if max_total_num_tokens is None:
            target = self.max_total_num_tokens + int(delta_slots)
        else:
            target = int(max_total_num_tokens)
        if target < 0:
            raise ValueError(f"max_total_num_tokens must be non-negative, got {target}")
        if target > self.token_to_kv_pool_allocator.size:
            raise ValueError(
                f"Requested token capacity {target} exceeds allocator size {self.token_to_kv_pool_allocator.size}."
            )
        self.max_total_num_tokens = target
        self.full_max_total_num_tokens = target
        return self.get_balloon_status()

    def get_balloon_status(self) -> Dict[str, Any]:
        captured_variants = []
        fused_layers = self._iter_fused_moe_layers()
        live_metadata = (
            self._balloon_local_expert_location_metadata
            or get_global_expert_location_metadata()
        )
        if self.graph_runner is not None and hasattr(
            self.graph_runner, "get_captured_variants"
        ):
            captured_variants = self.graph_runner.get_captured_variants()

        return {
            "state": self._balloon_state,
            "runtime_variant": self.get_cuda_graph_runtime_variant(),
            "prepared_variant": self._balloon_prepared_variant,
            "graph_replay_enabled": self.is_cuda_graph_replay_enabled(),
            "capture_variants": self.get_cuda_graph_capture_variants(),
            "captured_graph_variants": captured_variants,
            "max_total_num_tokens": int(self.max_total_num_tokens),
            "allocator_capacity": int(self.token_to_kv_pool_allocator.size),
            "kv_cache_capacity": int(self._get_balloon_kv_cache().size),
            "kv_cache_vmm_enabled": bool(
                getattr(self._get_balloon_kv_cache(), "_kv_vmm_enabled", False)
            ),
            "kv_cache_vmm_headroom_slots": int(self._balloon_kv_vmm_headroom_slots()),
            "moe_weight_vmm_enabled": bool(self._moe_weight_vmm_enabled()),
            "offloaded_local_experts": int(self._balloon_offloaded_local_experts),
            "added_kv_slots": int(self._balloon_added_slots),
            "fused_moe_layers": len(fused_layers),
            "global_bundle_ready": (
                all(
                    getattr(layer, "global_bundle", None) is not None
                    for layer in fused_layers
                )
                if fused_layers
                else False
            ),
            "tp_size": int(self.tp_size),
            "local_ep_size": int(self.moe_ep_size),
            "balloon_process_group_name": self._balloon_process_group_name,
            "kunserve_comm_backend": self._balloon_kunserve_comm_backend,
            "kunserve_capture_policy": self._balloon_capture_policy,
            "local_num_experts_per_layer": {
                int(layer.layer_id): int(layer.local_bundle.num_local_experts)
                for layer in fused_layers
            },
            "local_routed_experts_per_layer": {
                int(layer.layer_id): int(
                    layer.local_bundle.num_local_experts
                    - layer.num_fused_shared_experts
                )
                for layer in fused_layers
            },
            "num_fused_shared_experts_per_layer": {
                int(layer.layer_id): int(layer.num_fused_shared_experts)
                for layer in fused_layers
            },
            "local_active_local_expert_mapping_by_layer": {
                int(layer.layer_id): (
                    layer.local_bundle.active_local_expert_mapping.tolist()
                    if layer.local_bundle.active_local_expert_mapping is not None
                    else None
                )
                for layer in fused_layers
            },
            "local_physical_to_logical_map": (
                live_metadata.physical_to_logical_map_cpu.tolist()
                if live_metadata is not None
                else None
            ),
            "last_error": self._balloon_last_error,
        }

    def remote_instance_init_transfer_engine(self):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            logger.warning(
                "Please install mooncake for using remote instance transfer engine: pip install mooncake"
            )
            return
        self.remote_instance_transfer_engine = TransferEngine()
        local_ip = get_local_ip_auto()
        self.remote_instance_transfer_engine.initialize(
            local_ip, "P2PHANDSHAKE", "rdma", envs.MOONCAKE_DEVICE.get()
        )
        self.remote_instance_transfer_engine_session_id = (
            f"{local_ip}:{self.remote_instance_transfer_engine.get_rpc_port()}"
        )

    def model_specific_adjustment(self):
        server_args = self.server_args

        if server_args.enable_double_sparsity:
            logger.info(
                "Double sparsity optimization is turned on. Use triton backend without CUDA graph."
            )
            server_args.attention_backend = "triton"
            server_args.disable_cuda_graph = True

        if self.is_multimodal:
            if not self.is_multimodal_chunked_prefill_supported:
                server_args.chunked_prefill_size = -1
                logger.info(
                    f"Automatically turn off --chunked-prefill-size as it is not supported for "
                    f"{self.model_config.hf_config.model_type}"
                )

        if (
            not self.use_mla_backend
            or server_args.attention_backend
            not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
        ):
            server_args.disable_chunked_prefix_cache = True

        if not server_args.disable_chunked_prefix_cache:
            log_info_on_rank0(logger, "Chunked prefix cache is turned on.")

    def check_quantized_moe_compatibility(self):
        if (
            quantization_config := getattr(
                self.model_config.hf_config, "quantization_config", None
            )
        ) is not None and (
            weight_block_size := quantization_config.get("weight_block_size", None)
        ) is not None:
            weight_block_size_n = weight_block_size[0]

            if self.tp_size % self.moe_ep_size != 0:
                raise ValueError(
                    f"tp_size {self.tp_size} must be divisible by ep_size {self.moe_ep_size}"
                )
            moe_tp_size = self.tp_size // self.moe_ep_size

            moe_intermediate_size = getattr(
                self.model_config.hf_text_config, "moe_intermediate_size", None
            )
            if moe_intermediate_size is None:
                return

            if moe_intermediate_size % moe_tp_size != 0:
                raise ValueError(
                    f"moe_intermediate_size {moe_intermediate_size} must be divisible by moe_tp_size ({moe_tp_size}) which is tp_size ({self.tp_size}) divided by moe_ep_size ({self.moe_ep_size})."
                )

            if (moe_intermediate_size // moe_tp_size) % weight_block_size_n != 0:
                raise ValueError(
                    f"For quantized MoE models, please make sure ({moe_intermediate_size=} / {moe_tp_size=}) % {weight_block_size_n=} == 0 "
                    f"where moe_tp_size is equal to tp_size ({self.tp_size}) divided by ep_size ({self.moe_ep_size}). "
                    f"You can fix this by setting arguments `--tp` and `--ep` correctly."
                )

    def init_torch_distributed(self):
        tic = time.perf_counter()
        logger.info("Init torch distributed begin.")

        try:
            torch.get_device_module(self.device).set_device(self.gpu_id)
        except Exception:
            logger.warning(
                f"Context: {self.device=} {self.gpu_id=} {os.environ.get('CUDA_VISIBLE_DEVICES')=} {self.tp_rank=} {self.tp_size=}"
            )
            raise

        if self.device == "cuda":
            if self.server_args.elastic_ep_backend == "mooncake":
                backend = "mooncake"
                if self.server_args.mooncake_ib_device:
                    mooncake_ib_device = self.server_args.mooncake_ib_device.split(",")
                    try:
                        from mooncake import ep as mooncake_ep

                        mooncake_ep.set_device_filter(mooncake_ib_device)
                    except:
                        pass  # A warning will be raised in `init_distributed_environment`
            else:
                backend = "nccl"
        elif self.device == "xpu":
            backend = "xccl"
        elif self.device == "hpu":
            backend = "hccl"
        elif self.device == "cpu":
            backend = "gloo"
        elif self.device == "npu":
            backend = "hccl"
        elif self.device == "musa":
            backend = "mccl"

        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if not self.server_args.enable_p2p_check:
            monkey_patch_p2p_access_check()

        # Allow external orchestrators (e.g. trainpi) to override the distributed
        # init method.  When set to "env://", torch uses MASTER_ADDR/MASTER_PORT
        # env-vars and an externally-created TCPStore, completely avoiding port
        # conflicts with intra-host collocation.
        dist_init_method_override = envs.SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE.get()
        if dist_init_method_override:
            dist_init_method = dist_init_method_override
        elif self.server_args.dist_init_addr:
            dist_init_method = f"tcp://{self.server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{self.dist_port}"
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        set_mscclpp_all_reduce(self.server_args.enable_mscclpp)
        set_torch_symm_mem_all_reduce(self.server_args.enable_torch_symm_mem)

        if not self.is_draft_worker:
            if self.device == "cpu":
                if _is_cpu_amx_available or _is_cpu_arm64:
                    # Bind OpenMP threads to CPU cores
                    torch.ops.sgl_kernel.init_cpu_threads_env(self.local_omp_cpuid)

                    # Set local size to hint SGLang to use shared memory based AllReduce
                    os.environ["LOCAL_SIZE"] = str(self.tp_size)
                    torch.ops.sgl_kernel.initialize(self.tp_size, self.tp_rank)

                    @torch.library.register_fake("sgl_kernel::shm_allgather")
                    def _(data, dim):
                        return torch.cat([data] * self.tp_size, dim=dim)

                else:
                    logger.warning(
                        "init_cpu_threads_env and shared memory based AllReduce is disabled, only intel amx backend and arm64 are supported"
                    )

            # Only initialize the distributed environment on the target model worker.
            init_distributed_environment(
                backend=backend,
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
            )
            initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                attention_data_parallel_size=self.dp_size,
                pipeline_model_parallel_size=self.pp_size,
                expert_model_parallel_size=self.moe_ep_size,
                attention_context_model_parallel_size=self.attn_cp_size,
                moe_data_model_parallel_size=self.moe_dp_size,
                duplicate_tp_group=self.server_args.enable_pdmux,
            )
            initialize_dp_attention(
                server_args=self.server_args,
                model_config=self.model_config,
            )
            if is_npu():
                register_sgl_tp_rank(self.gpu_id)

        min_per_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        self.tp_group = get_tp_group()
        self.pp_group = get_pp_group()
        self.attention_tp_group = get_attention_tp_group()

        # Check memory for tensor parallelism
        local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if self.tp_size > 1 and not self.is_draft_worker:
            if min_per_gpu_memory < local_gpu_memory * 0.9:
                msg = "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                msg += f"{min_per_gpu_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                if envs.SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK.get():
                    raise RuntimeError(msg)
                else:
                    logger.warning(msg)

        logger.info(
            f"Init torch distributed ends. elapsed={time.perf_counter() - tic:.2f} s, "
            f"mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
        )
        return min_per_gpu_memory

    def init_shared_mooncake_transfer_engine(self):
        """
        Need MooncakeTransferEngine when:
        1) PD disaggregation uses mooncake for KV transfer (prefill/decode)
        2) HiCache uses mooncake storage backend
        3) Encoder disaggregation uses mooncake
        """
        use_mooncake_te = (
            (
                self.server_args.disaggregation_mode != "null"
                and self.server_args.disaggregation_transfer_backend == "mooncake"
            )
            or (
                self.server_args.enable_hierarchical_cache
                and self.server_args.hicache_storage_backend == "mooncake"
            )
            or (
                self.server_args.encoder_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                self.server_args.language_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
        )

        if use_mooncake_te:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                init_mooncake_transfer_engine,
            )

            init_mooncake_transfer_engine(
                hostname=get_local_ip_auto(),
                gpu_id=self.gpu_id,
                ib_device=(
                    self.server_args.disaggregation_ib_device
                    or self.server_args.mooncake_ib_device
                ),
            )

    def load_model(self):
        tic_total = time.perf_counter()
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] < 8:
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                self.model_config.dtype = torch.float16
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")

        set_cuda_arch()

        # Prepare the model config
        from sglang.srt.configs.modelopt_config import ModelOptConfig

        modelopt_config = ModelOptConfig(
            quant=self.server_args.modelopt_quant,
            checkpoint_restore_path=self.server_args.modelopt_checkpoint_restore_path,
            checkpoint_save_path=self.server_args.modelopt_checkpoint_save_path,
            export_path=self.server_args.modelopt_export_path,
            quantize_and_serve=self.server_args.quantize_and_serve,
        )

        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
            model_loader_extra_config=self.server_args.model_loader_extra_config,
            tp_rank=self.tp_rank,
            remote_instance_weight_loader_seed_instance_ip=self.server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=self.server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=self.server_args.remote_instance_weight_loader_send_weights_group_ports,
            remote_instance_weight_loader_backend=self.server_args.remote_instance_weight_loader_backend,
            remote_instance_weight_loader_transfer_engine=self.remote_instance_transfer_engine,
            modelopt_config=modelopt_config,
            rl_quant_profile=self.server_args.rl_quant_profile,
            draft_model_idx=self.draft_model_idx,
        )
        if self.device == "cpu":
            self.model_config = adjust_config_with_unaligned_cpu_tp(
                self.model_config, self.load_config, self.tp_size
            )

        if (
            self.server_args.load_format == LoadFormat.REMOTE_INSTANCE
            and self.server_args.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.NCCL
        ):
            if self.tp_rank == 0:
                instance_ip = socket.gethostbyname(socket.gethostname())
                t = threading.Thread(
                    target=trigger_init_weights_send_group_for_remote_instance_request,
                    args=(
                        self.server_args.remote_instance_weight_loader_seed_instance_ip,
                        self.server_args.remote_instance_weight_loader_seed_instance_service_port,
                        self.server_args.remote_instance_weight_loader_send_weights_group_ports,
                        instance_ip,
                    ),
                )
                t.start()

        # Load the model
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
        monkey_patch_vllm_parallel_state()

        enable_cpu_backup = self.server_args.enable_weights_cpu_backup or (
            self.is_draft_worker and self.server_args.enable_draft_weights_cpu_backup
        )
        with self.memory_saver_adapter.region(
            GPU_MEMORY_TYPE_WEIGHTS,
            enable_cpu_backup=enable_cpu_backup,
        ):
            self.loader = get_model_loader(
                load_config=self.load_config,
                model_config=self.model_config,
            )
            self.model = self.loader.load_model(
                model_config=self.model_config,
                device_config=DeviceConfig(self.device, self.gpu_id),
            )
            if hasattr(self.loader, "remote_instance_transfer_engine_weight_info"):
                self.remote_instance_transfer_engine_weight_info = (
                    self.loader.remote_instance_transfer_engine_weight_info
                )
        monkey_patch_vllm_parallel_state(reverse=True)

        get_offloader().post_init()

        # Register model for layerwise NVTX profiling if enabled
        if self.server_args.enable_layerwise_nvtx_marker:
            self.pyt_hooks = PytHooks()
            self.pyt_hooks.register_hooks(self.model, module_prefix="model")

        if self.server_args.kv_cache_dtype == "fp8_e4m3":
            if self.server_args.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(
                        self.server_args.quantization_param_path
                    )
                    logger.info(
                        "Loaded KV cache scaling factors from %s",
                        self.server_args.quantization_param_path,
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__,
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        # Parse other args
        self.sliding_window_size = None
        if hasattr(self.model, "get_attention_sliding_window_size"):
            self.sliding_window_size = self.model.get_attention_sliding_window_size()
        elif (
            self.model_config.is_hybrid_swa
            and self.model_config.sliding_window_size is not None
        ):
            # sliding window field in model config may have different meaning for different kinds of models (e.g., dllm), here we only consider the sliding window in SWA model
            self.sliding_window_size = self.model_config.sliding_window_size
        elif self.model_config.attention_chunk_size is not None:
            self.sliding_window_size = self.model_config.attention_chunk_size
            logger.info(
                f"Setting sliding_window_size to be attention_chunk_size: {self.sliding_window_size}"
            )

        self.dtype = self.model_config.dtype

        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        self.weight_load_mem_usage = before_avail_memory - after_avail_memory
        logger.info(
            f"Load weight end. "
            f"elapsed={time.perf_counter() - tic_total:.2f} s, "
            f"type={type(self.model).__name__}, "
            f"dtype={self.dtype}, "
            f"avail mem={after_avail_memory:.2f} GB, "
            f"mem usage={self.weight_load_mem_usage:.2f} GB."
        )
        if self.server_args.debug_tensor_dump_output_folder is not None:
            register_forward_hook_for_model(
                self.model,
                self.server_args.debug_tensor_dump_output_folder,
                self.server_args.debug_tensor_dump_layers,
                self.tp_size,
                self.tp_rank,
                self.pp_rank,
            )

        # Pre-expand RoPE cache before CUDA Graph capture
        reserve_rope_cache_for_long_sequences(
            self.model,
            self.server_args,
            self.model_config,
            logger,
        )

        if self.server_args.elastic_ep_backend == "mooncake":
            # Mooncake does not support `monitored_barrier`
            dist.barrier(group=get_tp_group().cpu_group)
        else:
            # Handle the case where some ranks do not finish loading.
            try:
                dist.monitored_barrier(
                    group=get_tp_group().cpu_group,
                    timeout=datetime.timedelta(
                        seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S
                    ),
                    wait_all_ranks=True,
                )
            except RuntimeError:
                raise ValueError(
                    f"TP rank {self.tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
                ) from None

    def update_expert_location(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        if ElasticEPStateManager.instance() is not None:
            # TODO: refactor the weights update when elastic ep
            old_expert_location_metadata = get_global_expert_location_metadata()
            assert old_expert_location_metadata is not None
            old_expert_location_metadata.update(
                new_expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )
            self.update_weights_from_disk(
                self.server_args.model_path,
                self.server_args.load_format,
                lambda name: "mlp.experts" in name and "mlp.shared_experts" not in name,
            )
        else:
            self.expert_location_updater.update(
                self.model.routed_experts_weights_of_layer,
                new_expert_location_metadata,
                update_layer_ids=update_layer_ids,
                nnodes=self.server_args.nnodes,
                rank=self.tp_rank,
            )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str,
        weight_name_filter: Optional[Callable[[str], bool]] = None,
        recapture_cuda_graph: bool = False,
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support DefaultModelLoader for now
        loader = get_model_loader(load_config, self.model_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.model)
            )
            if weight_name_filter is not None:
                iter = (
                    (name, weight) for name, weight in iter if weight_name_filter(name)
                )

            return iter

        def model_load_weights(model, iter):
            loader.load_weights_and_postprocess(model, iter, target_device)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.load_config = load_config

        if recapture_cuda_graph and (self.device == "cuda" or self.device == "musa"):
            self.init_device_graphs()

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def init_weights_send_group_for_remote_instance(
        self,
        master_address,
        ports,
        group_rank,
        world_size,
        group_name,
        backend="nccl",
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        logger.info(
            f"init custom process group: tp_rank={self.tp_rank}, gpu_id={self.gpu_id}, master_address={master_address}, master_port={group_port}, "
            f"group_rank={group_rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        torch.cuda.empty_cache()
        success = False
        message = ""
        try:
            self._weights_send_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{group_port}",
                world_size=world_size,
                rank=group_rank,
                group_name=group_name,
                device_id=torch.device("cuda", self.gpu_id),
            )
            dist.barrier(group=self._weights_send_group[group_name])
            success = True
            message = (
                f"Succeeded to init group through {master_address}:{group_port} group."
            )
        except Exception as e:
            message = f"Failed to init group: {e}."
            logger.error(message)

        torch.cuda.empty_cache()
        return success, message

    def send_weights_to_remote_instance(
        self,
        master_address,
        ports,
        group_name,
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        if self._weights_send_group[group_name] is not None:
            send_group = self._weights_send_group[group_name]
        else:
            message = f"Group {group_name} not in _weights_send_group list. Please call `init_weights_send_group_for_remote_instance` first."
            logger.error(message)
            return False, message

        torch.cuda.empty_cache()
        success = False
        message = ""
        try:
            for _, weights in self.model.named_parameters():
                torch.distributed.broadcast(
                    weights,
                    src=0,
                    group=send_group,
                )
            success = True
            message = f"Succeeded to send weights through {master_address}:{group_port} {group_name}."
        except Exception as e:
            message = f"Failed to send weights: {e}."
            logger.error(message)

        # destroy the process group after sending weights
        del self._weights_send_group[group_name]
        torch.distributed.distributed_c10d.destroy_process_group(send_group)
        torch.cuda.empty_cache()
        return success, message

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
        *,
        lane_only_tp_rank: Optional[int] = None,
        explicit_group_rank: Optional[int] = None,
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.

        KunServe Phase F adds the ability to build cross-replica lane
        subgroups (``[rank0, rank2]`` / ``[rank1, rank3]``) which only
        include ONE TP worker per replica.  When ``lane_only_tp_rank``
        is set, every TP worker still receives the RPC but only the one
        with matching ``tp_rank`` actually participates in the NCCL
        rendezvous; the others record a sentinel so they don't try to
        look up a non-existent local handle.  ``explicit_group_rank``
        overrides the ``rank_offset + tp_rank`` formula for the
        participating worker (used because lane subgroup ranks are
        ``replica_idx``, not ``rank_offset + tp_rank``).
        """
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        # Phase F: lane-filtered init.  Non-participating TP workers
        # record None so _resolve_balloon_process_group can detect them
        # and the dispatcher can fall back gracefully.
        if (
            lane_only_tp_rank is not None
            and int(lane_only_tp_rank) != int(self.tp_rank)
        ):
            self._model_update_group[group_name] = None
            return True, (
                f"Skipped non-participating tp_rank={self.tp_rank} for "
                f"lane group {group_name!r} (lane_only_tp_rank="
                f"{lane_only_tp_rank})."
            )

        if explicit_group_rank is not None:
            rank = int(explicit_group_rank)
        else:
            rank = rank_offset + self.tp_rank

        # Idempotent: if a group with the same name is already initialized,
        # return success. Callers (e.g. KunServeController) may retry the same
        # request and we must not double-init or fail the second call.
        if (
            group_name in self._model_update_group
            and self._model_update_group[group_name] is not None
        ):
            logger.info(
                f"init custom process group: group_name={group_name} already initialized, returning success."
            )
            return True, "Process group already initialized."

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            # Best-effort cleanup so a retry with a fresh group_name/port has a
            # clean torch.distributed state. The TCPStore created inside
            # rendezvous() may still be holding the master port until its
            # Python ref is dropped; force gc to release it sooner.
            half_pg = self._model_update_group.pop(group_name, None)
            if half_pg is not None:
                try:
                    torch.distributed.destroy_process_group(half_pg)
                except Exception:
                    logger.exception(
                        "Failed to clean up half-initialized process group %s",
                        group_name,
                    )
            try:
                import gc

                gc.collect()
            except Exception:
                pass
            return False, message

    def destroy_weights_update_group(self, group_name):
        try:
            if group_name in self._model_update_group:
                pg = self._model_update_group.pop(group_name)
                if pg is None:
                    # Phase F sentinel for a non-participating tp_rank.
                    # Nothing to tear down on this side.
                    return True, "Succeeded to destroy custom process group."
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            message = f"Failed to destroy custom process group: {e}."
            logger.error(message)
            return False, message

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
    ):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """

        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        if load_format == "flattened_bucket":
            return self._update_bucketed_weights_from_distributed(
                names, dtypes, shapes, group_name
            )
        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.model.load_weights(weights)
            return True, "Succeeded to update parameter online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def _update_bucketed_weights_from_distributed(
        self, names, dtypes, shapes, group_name
    ):
        try:
            named_tensors = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                named_tensors.append(
                    (name, torch.empty(shape, dtype=target_dtype, device=self.device))
                )
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = bucket.get_flattened_tensor()
            torch.distributed.broadcast(
                flattened_tensor,
                src=0,
                group=self._model_update_group[group_name],
            )
            reconstructed_tensors = bucket.reconstruct_tensors()
            self.model.load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, Union[torch.Tensor, "LocalSerializedTensor"]]],
        load_format: Optional[str] = None,
    ):
        monkey_patch_torch_reductions()
        if load_format == "flattened_bucket":
            # Handle flattened bucket format
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )

        # We need to get device after patch otherwise the device would be wrong
        self.device_module = torch.get_device_module(self.device)
        infered_device = self.device_module.current_device()

        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank, device=infered_device))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            _model_load_weights_direct(self.model, named_tensors)
        elif load_format in self.server_args.custom_weight_loader:
            custom_loader = dynamic_import(load_format)
            custom_loader(self.model, named_tensors)
        elif load_format is None:
            self.model.load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    def _update_weights_from_flattened_bucket(
        self,
        flattened_tensor_bucket_dict,
    ):
        """Handle flattened bucket format for weight updates"""
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # Convert metadata dict to our format
        converted_metadata = []
        for meta in metadata:
            converted_meta = FlattenedTensorMetadata(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                start_idx=meta.start_idx,
                end_idx=meta.end_idx,
                numel=meta.numel,
            )
            converted_metadata.append(converted_meta)

        # Create bucket and reconstruct tensors
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor, metadata=converted_metadata
        )
        reconstructed_tensors = bucket.reconstruct_tensors()

        # Load the reconstructed tensors using the standard method
        self.model.load_weights(reconstructed_tensors)

        return True, "Success"

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        # TODO: (chenyang) Add support for Qwen models.
        try:
            return self.model.get_weights_by_name(
                name, truncate_size, tp_size=self.tp_size
            )
        except Exception as e:
            logger.error(f"Error when getting parameter {name}: {e}")
            return None

    def init_lora_manager(self):
        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            server_args=self.server_args,
            lora_backend=self.server_args.lora_backend,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            max_lora_rank=self.server_args.max_lora_rank,
            target_modules=self.server_args.lora_target_modules,
            lora_paths=self.server_args.lora_paths,
        )

    def load_lora_adapter(self, lora_ref: LoRARef):
        """Load a new lora adapter from disk or huggingface."""

        logger.info(
            f"LoRA adapter loading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.load_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter loading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    def load_lora_adapter_from_tensors(
        self, lora_ref: LoRARef, tensors, config_dict, added_tokens_config=None
    ):
        logger.info(f"LoRA adapter loading from tensors starts: {lora_ref}.")
        result = self.lora_manager.load_lora_adapter_from_tensors(
            lora_ref, tensors, config_dict, added_tokens_config
        )
        logger.info(f"LoRA adapter loading from tensors completes: {lora_ref}.")
        return result

    def unload_lora_adapter(self, lora_ref: LoRARef):
        """Unload a lora adapter that was previously loaded during initialization or dynamic loading."""

        logger.info(
            f"LoRA adapter unloading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.unload_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter unloading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    @property
    def qwen3_next_config(self):
        config = self.model_config.hf_config
        if isinstance(config, Qwen3NextConfig):
            return config
        return None

    @property
    def hybrid_lightning_config(self):
        config = self.model_config.hf_config
        if isinstance(config, BailingHybridConfig):
            return config
        return None

    @property
    def hybrid_gdn_config(self):
        config = self.model_config.hf_config.get_text_config()
        if isinstance(
            config,
            Qwen3NextConfig
            | Qwen3_5Config
            | Qwen3_5MoeConfig
            | JetNemotronConfig
            | JetVLMConfig,
        ):
            return config
        return None

    @property
    def mamba2_config(self):
        config = self.model_config.hf_config
        if isinstance(config, NemotronHConfig) and self.is_draft_worker:
            # NemotronH MTP draft models have no Mamba layers (pattern like "*E")
            # so they shouldn't use HybridLinearAttnBackend
            pattern = getattr(config, "mtp_hybrid_override_pattern", None)
            if pattern is not None and "M" not in pattern:
                return None
        if isinstance(
            config, FalconH1Config | NemotronHConfig | Lfm2Config | Lfm2MoeConfig
        ):
            return config
        if isinstance(config, NemotronH_Nano_VL_V2_Config):
            return config.llm_config

        if isinstance(config, GraniteMoeHybridConfig):
            has_mamba = any(
                layer_type == "mamba"
                for layer_type in getattr(config, "layer_types", [])
            )
            if not has_mamba:
                return None
            else:
                return config

        return None

    @property
    def max_token_pool_size(self):
        """Return the max token pool size considering hybrid swa settings."""
        if self.is_hybrid_swa:
            return min(self.swa_max_total_num_tokens, self.max_total_num_tokens)
        else:
            return self.max_total_num_tokens

    @property
    def kimi_linear_config(self):
        config = self.model_config.hf_config
        if isinstance(config, KimiLinearConfig):
            return config
        return None

    @property
    def mambaish_config(self):
        return (
            self.mamba2_config
            or self.hybrid_gdn_config
            or self.kimi_linear_config
            or self.hybrid_lightning_config
        )

    def can_run_piecewise_cuda_graph(self):
        if self.is_draft_worker:
            return False

        if self.server_args.enable_torch_compile:
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because piecewise_cuda_graph has conflict with torch compile",
            )
            return False
        if self.pp_size > 1:
            # TODO(yuwei): support PP
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because piecewise_cuda_graph does not support PP",
            )
            return False
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            # TODO(yuwei): fix the compilation errors for MOE A2A backend
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph due to existing compilation errors",
            )
            return False
        return True

    def configure_kv_cache_dtype(self):
        if self.server_args.kv_cache_dtype == "auto":
            quant_config = getattr(self.model, "quant_config", None)
            kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
            if (
                isinstance(kv_cache_quant_algo, str)
                and kv_cache_quant_algo.upper() == "FP8"
            ):
                if _is_hip:
                    self.kv_cache_dtype = fp8_dtype
                    self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                        self.kv_cache_dtype
                    ]
                else:
                    self.kv_cache_dtype = torch.float8_e4m3fn
                    self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                        self.kv_cache_dtype
                    ]
            else:
                self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e5m2
        elif self.server_args.kv_cache_dtype == "fp8_e4m3":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e4m3fn
        elif self.server_args.kv_cache_dtype in ("bf16", "bfloat16"):
            self.kv_cache_dtype = torch.bfloat16
        elif self.server_args.kv_cache_dtype == "fp4_e2m1":
            if hasattr(torch, "float4_e2m1fn_x2"):
                self.kv_cache_dtype = torch.float4_e2m1fn_x2
                logger.warning(f"FP4 (E2M1) KV Cache might lead to a accuracy drop!")
            else:
                logger.warning(
                    f"--kv-cache-dtype falls back to 'auto' because this torch version does not support torch.float4_e2m1fn_x2"
                )
                self.kv_cache_dtype = self.dtype
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )

        log_info_on_rank0(logger, f"Using KV cache dtype: {self.kv_cache_dtype}")

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_attention_backend(self):
        """Init attention kernel backend."""
        if self.server_args.enable_pdmux:
            self.attn_backend = self._get_attention_backend(init_new_workspace=True)
            self.decode_attn_backend_group = []
            for _ in range(self.server_args.sm_group_num):
                self.decode_attn_backend_group.append(self._get_attention_backend())
            self.decode_attn_backend = self.decode_attn_backend_group[0]
        elif self.server_args.enable_two_batch_overlap and not self.is_draft_worker:
            self.attn_backend = TboAttnBackend.init_new(self._get_attention_backend)
        else:
            self.attn_backend = self._get_attention_backend()

    def _get_attention_backend(self, init_new_workspace: bool = False):
        """Init attention kernel backend."""
        draft_attn_backend = self.server_args.speculative_draft_attention_backend
        if self.is_draft_worker and draft_attn_backend:
            logger.warning(
                f"Overriding draft attention backend to {draft_attn_backend}."
            )
            return self._get_attention_backend_from_str(
                draft_attn_backend,
                init_new_workspace=init_new_workspace,
            )

        self.prefill_attention_backend_str, self.decode_attention_backend_str = (
            self.server_args.get_attention_backends()
        )

        if self.decode_attention_backend_str != self.prefill_attention_backend_str:
            from sglang.srt.layers.attention.hybrid_attn_backend import (
                HybridAttnBackend,
            )

            attn_backend = HybridAttnBackend(
                self,
                decode_backend=self._get_attention_backend_from_str(
                    self.decode_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
                prefill_backend=self._get_attention_backend_from_str(
                    self.prefill_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
            )
            logger.info(
                f"Using hybrid attention backend for decode and prefill: "
                f"decode_backend={self.decode_attention_backend_str}, "
                f"prefill_backend={self.prefill_attention_backend_str}."
            )
            logger.warning(
                "Warning: Attention backend specified by --attention-backend or default backend might be overridden."
                "The feature of hybrid attention backend is experimental and unstable. Please raise an issue if you encounter any problem."
            )
        else:
            attn_backend = self._get_attention_backend_from_str(
                self.server_args.attention_backend,
                init_new_workspace=init_new_workspace,
            )

        (
            get_global_server_args().prefill_attention_backend,
            get_global_server_args().decode_attention_backend,
        ) = (self.prefill_attention_backend_str, self.decode_attention_backend_str)
        return attn_backend

    def _get_attention_backend_from_str(
        self, backend_str: str, init_new_workspace: bool = False
    ):
        if backend_str not in ATTENTION_BACKENDS:
            raise ValueError(f"Invalid attention backend: {backend_str}")
        self.init_new_workspace = init_new_workspace
        full_attention_backend = ATTENTION_BACKENDS[backend_str](self)
        return attn_backend_wrapper(self, full_attention_backend)

    def init_double_sparsity_channel_config(self, selected_channel):
        selected_channel = "." + selected_channel + "_proj"
        self.sorted_channels = []
        # load channel config
        with open(self.server_args.ds_channel_config_path, "r") as f:
            channel_config = json.load(f)

        for i in range(self.start_layer, self.end_layer):
            key = "model.layers." + str(i) + ".self_attn" + selected_channel
            self.sorted_channels.append(
                torch.tensor(channel_config[key])[
                    :, : self.server_args.ds_heavy_channel_num
                ]
                .contiguous()
                .cuda()
            )

    def kernel_warmup(self):
        """
        Warmup and tune kernels before cuda graph capture.
        Currently only doing FlashInfer autotune.
        """
        if self.device != "cuda":
            return

        if self._should_run_flashinfer_autotune():
            self._flashinfer_autotune()

    def _should_run_flashinfer_autotune(self) -> bool:
        """Check if flashinfer autotune should be run."""
        if self.server_args.disable_flashinfer_autotune:
            return False

        backend_str = self.server_args.moe_runner_backend

        # TODO smor- support other cases for flashinfer autotune, such as, mamba backend

        if backend_str not in [
            "flashinfer_trtllm",
            "flashinfer_mxfp4",
            # TODO: flashinfer_cutlass will cause some flashinfer compilation errors. To be fixed.
            # "flashinfer_cutlass",
        ]:
            return False

        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            return False

        if (
            self.spec_algorithm.is_eagle()
            or self.spec_algorithm.is_standalone()
            or self.spec_algorithm.is_ngram()
        ):
            return not self.is_draft_worker

        return True

    def _flashinfer_autotune(self):
        """Run flashinfer autotune."""
        from flashinfer.autotuner import autotune

        logger.info("Running FlashInfer autotune...")

        # Run warmup on the non-default stream to avoid NCCL 2.29+ cudaMemcpyBatchAsync
        # calls on default stream (unsupported by CUDA) when --enable-symm-mem is used.
        self.forward_stream.wait_stream(torch.cuda.current_stream())
        with torch.get_device_module(self.device).stream(self.forward_stream):
            with torch.inference_mode(), autotune():
                self._dummy_run(
                    batch_size=self.req_to_token_pool.size, run_ctx=autotune()
                )
        torch.cuda.current_stream().wait_stream(self.forward_stream)
        logger.info("FlashInfer autotune completed.")

    def _dummy_run(self, batch_size: int, run_ctx=None):
        """Run a dummy forward pass for warmup/profiling."""
        if self.is_generation:
            capture_forward_mode = ForwardMode.DECODE
        else:
            capture_forward_mode = ForwardMode.EXTEND
        capture_hidden_mode = CaptureHiddenMode.NULL
        num_tokens_per_bs = 1
        if (
            self.spec_algorithm.is_eagle()
            or self.spec_algorithm.is_standalone()
            or self.spec_algorithm.is_ngram()
        ):
            if self.is_draft_worker:
                raise RuntimeError("This should not happen")
            else:
                capture_forward_mode = ForwardMode.TARGET_VERIFY
                num_tokens_per_bs = self.server_args.speculative_num_draft_tokens

        if self.server_args.enable_return_hidden_states:
            capture_hidden_mode = CaptureHiddenMode.FULL

        num_tokens = batch_size * num_tokens_per_bs

        seq_len_fill_value = self.attn_backend.get_cuda_graph_seq_len_fill_value()

        if self.server_args.enable_torch_compile:
            set_torch_compile_config()

        if self.eagle_use_aux_hidden_state:
            self.model.set_eagle3_layers_to_capture()

        require_mlp_tp_gather_ = require_mlp_tp_gather(self.server_args)
        if require_gathered_buffer(self.server_args):
            assert require_mlp_tp_gather_ or require_attn_tp_gather(self.server_args)

        buffers: GraphInputBuffers = GraphInputBuffers.create(
            device=self.device,
            max_bs=batch_size,
            max_num_token=num_tokens,
            hidden_size=self.model_config.hidden_size,
            vocab_size=self.model_config.vocab_size,
            dtype=self.model_config.dtype,
            dp_size=self.server_args.dp_size,
            pp_size=self.server_args.pp_size,
            is_encoder_decoder=self.model_config.is_encoder_decoder,
            require_mlp_tp_gather=require_mlp_tp_gather_,
            seq_len_fill_value=seq_len_fill_value,
            encoder_len_fill_value=0,
            num_tokens_per_bs=num_tokens_per_bs,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
        )
        buffers.num_token_non_padded[...] = num_tokens

        # For extend mode
        if not self.is_generation:
            extend_prefix_lens_cpu = [0] * batch_size
            extend_seq_lens_cpu = [seq_len_fill_value] * batch_size
            extend_num_tokens = num_tokens
            extend_seq_lens = torch.full(
                (batch_size,), seq_len_fill_value, dtype=torch.int32, device=self.device
            )
            extend_prefix_lens = torch.zeros(
                (batch_size,), dtype=torch.int32, device=self.device
            )
            extend_start_loc = torch.arange(
                0, num_tokens, num_tokens_per_bs, dtype=torch.int32, device=self.device
            )
        else:
            extend_prefix_lens_cpu = None
            extend_seq_lens_cpu = None
            extend_num_tokens = None
            extend_seq_lens = None
            extend_prefix_lens = None
            extend_start_loc = None

        if self.server_args.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        if require_mlp_tp_gather_:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.server_args.dp_size,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.server_args.dp_size,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.server_args.dp_size
        elif require_attn_tp_gather(self.server_args):
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            global_dp_buffer_len = num_tokens
        else:
            global_dp_buffer_len = None

        def get_spec_info():
            spec_info = None
            if self.spec_algorithm.is_eagle() or self.spec_algorithm.is_standalone():
                from sglang.srt.speculative.eagle_info import EagleVerifyInput

                if self.is_draft_worker:
                    raise RuntimeError("This should not happen.")
                else:
                    spec_info = EagleVerifyInput(
                        draft_token=None,
                        custom_mask=buffers.custom_mask,
                        positions=None,
                        retrive_index=None,
                        retrive_next_token=None,
                        retrive_next_sibling=None,
                        retrive_cum_len=None,
                        spec_steps=self.server_args.speculative_num_steps,
                        topk=self.server_args.speculative_eagle_topk,
                        draft_token_num=self.server_args.speculative_num_draft_tokens,
                        capture_hidden_mode=CaptureHiddenMode.FULL,
                        seq_lens_sum=None,
                        seq_lens_cpu=None,
                    )

            elif self.spec_algorithm.is_ngram():
                from sglang.srt.speculative.ngram_info import NgramVerifyInput

                spec_info = NgramVerifyInput(
                    draft_token=None,
                    tree_mask=buffers.custom_mask,
                    positions=None,
                    retrive_index=None,
                    retrive_next_token=None,
                    retrive_next_sibling=None,
                    draft_token_num=num_tokens_per_bs,
                )
                spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

            return spec_info

        spec_info = get_spec_info()
        if capture_hidden_mode != CaptureHiddenMode.FULL:
            capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        if self.server_args.enable_lora:
            lora_ids = [None] * batch_size
        else:
            lora_ids = None

        forward_batch = ForwardBatch(
            forward_mode=capture_forward_mode,
            batch_size=batch_size,
            input_ids=buffers.input_ids,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_cpu=buffers.seq_lens_cpu,
            next_token_logits_buffer=buffers.next_token_logits_buffer,
            orig_seq_lens=buffers.seq_lens,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            attn_backend=self.attn_backend,
            out_cache_loc=buffers.out_cache_loc,
            seq_lens_sum=buffers.seq_lens.sum().item(),
            encoder_lens=buffers.encoder_lens,
            return_logprob=False,
            positions=buffers.positions,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            mrope_positions=buffers.mrope_positions,
            spec_algorithm=self.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=capture_hidden_mode,
            num_token_non_padded=buffers.num_token_non_padded,
            global_forward_mode=capture_forward_mode,
            lora_ids=lora_ids,
        )

        if lora_ids is not None:
            self.lora_manager.prepare_lora_batch(forward_batch)

        self.attn_backend.init_forward_metadata(forward_batch)

        def run_once():
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            kwargs = {}
            if (
                self.server_args.pp_size > 1
                and "pp_proxy_tensors"
                in inspect.signature(self.model.forward).parameters
            ):
                kwargs["pp_proxy_tensors"] = PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )
            if not self.is_generation:
                kwargs["get_embedding"] = True

            logits_output_or_pp_proxy_tensors = self.model.forward(
                buffers.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
            return logits_output_or_pp_proxy_tensors

        torch.get_device_module(self.device).synchronize()
        self.tp_group.barrier()
        with torch.inference_mode(), run_ctx or empty_context():
            run_once()

    def init_device_graphs(self):
        """Capture device graphs."""
        self.graph_runner = None
        self.graph_mem_usage = 0

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE:
            return

        if self.device != "cpu" and self.server_args.disable_cuda_graph:
            return

        if self.device == "cpu" and not self.server_args.enable_torch_compile:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        graph_backend = defaultdict(
            lambda: "cuda graph",
            {
                "cpu": "cpu graph",
                "npu": "npu graph",
            },
        )
        logger.info(
            f"Capture {graph_backend[self.device]} begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        graph_runners = defaultdict(
            lambda: CudaGraphRunner,
            {
                "cpu": CPUGraphRunner,
                "npu": NPUGraphRunner,
            },
        )
        self.graph_runner = graph_runners[self.device](self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.graph_mem_usage = before_mem - after_mem
        logger.info(
            f"Capture {graph_backend[self.device]} end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={self.graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def init_piecewise_cuda_graphs(self):
        """Initialize piecewise CUDA graph runner."""
        self.piecewise_cuda_graph_runner = None

        if (
            not self.server_args.enable_piecewise_cuda_graph
            or not self.can_run_piecewise_cuda_graph()
        ):
            return

        # Collect attention layers and moe layers from the model
        self.model.model = resolve_language_model(self.model)
        language_model = getattr(self.model, "language_model", self.model)
        self.attention_layers = []
        self.moe_layers = []
        self.moe_fusions = []
        for layer in language_model.model.layers:
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "attn"):
                    self.attention_layers.append(layer.self_attn.attn)
                elif hasattr(layer.self_attn, "attn_mqa"):
                    # For DeepSeek model
                    self.attention_layers.append(layer.self_attn.attn_mqa)
            # For hybrid model
            elif hasattr(layer, "attn"):
                self.attention_layers.append(layer.attn)
            elif hasattr(layer, "linear_attn"):
                if hasattr(layer.linear_attn, "attn"):
                    self.attention_layers.append(layer.linear_attn.attn)
                else:
                    self.attention_layers.append(layer.linear_attn)
            # For InternVL model
            elif hasattr(layer, "attention"):
                if hasattr(layer.attention, "attn"):
                    self.attention_layers.append(layer.attention.attn)

            moe_block = None
            moe_fusion = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                moe_block = layer.mlp.experts
                moe_fusion = layer.mlp
            if hasattr(layer, "block_sparse_moe") and hasattr(
                layer.block_sparse_moe, "experts"
            ):
                moe_block = layer.block_sparse_moe.experts
                moe_fusion = layer.block_sparse_moe
            if hasattr(layer, "moe") and hasattr(layer.moe, "experts"):
                moe_block = layer.moe.experts
                moe_fusion = layer.moe
            self.moe_layers.append(moe_block)
            self.moe_fusions.append(moe_fusion)

        if len(self.attention_layers) < self.model_config.num_hidden_layers:
            # TODO(yuwei): support Non-Standard GQA
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because some layers do not apply Standard GQA",
            )
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture piecewise CUDA graph begin. avail mem={before_mem:.2f} GB"
        )

        self.piecewise_cuda_graph_runner = PiecewiseCudaGraphRunner(self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        mem_usage = before_mem - after_mem
        logger.info(
            f"Capture piecewise CUDA graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def init_threads_binding(self):
        omp_cpuids = os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", "all")
        cpu_ids_by_node = get_cpu_ids_by_node()
        n_numa_node = len(cpu_ids_by_node)
        if omp_cpuids == "all":
            assert self.tp_size <= n_numa_node, (
                f"SGLANG_CPU_OMP_THREADS_BIND is not set, in this case, "
                f"tp_size {self.tp_size} should be smaller than or equal to number of numa node on the machine {n_numa_node}. "
                f"If you need tp_size to be larger than number of numa node, please set the CPU cores for each tp rank via SGLANG_CPU_OMP_THREADS_BIND explicitly. "
                f"For example, on a machine with 2 numa nodes, where core 0-31 are on numa node 0 and core 32-63 are on numa node 1, "
                f"it is suggested to use -tp 2 and bind tp rank 0 to core 0-31 and tp rank 1 to core 32-63. "
                f"This is the default behavior if SGLANG_CPU_OMP_THREADS_BIND is not set and it is the same as setting SGLANG_CPU_OMP_THREADS_BIND=0-31|32-63. "
                f"If you do need tp_size to be larger than the number of numa nodes, you could set SGLANG_CPU_OMP_THREADS_BIND explicitly for example SGLANG_CPU_OMP_THREADS_BIND=0-15|16-31|32-47|48-63 and run with -tp 4. "
                f"If you don't want each tp rank to use all the cores on one numa node, you could set for example SGLANG_CPU_OMP_THREADS_BIND=0-15|32-47 and run with -tp 2."
            )
            if self.tp_size < n_numa_node:
                logger.warning(
                    f"Detected the current machine has {n_numa_node} numa nodes available, but tp_size is set to {self.tp_size}, so only {self.tp_size} numa nodes are used."
                )
            self.local_omp_cpuid = cpu_ids_by_node[self.tp_rank]
        else:
            threads_bind_list = omp_cpuids.split("|")
            assert self.tp_size == len(threads_bind_list), (
                f"SGLANG_CPU_OMP_THREADS_BIND setting must be aligned with TP size parameter ({self.tp_size}). "
                f"Please double check your settings."
            )
            self.local_omp_cpuid = threads_bind_list[self.tp_rank]
            if self.tp_size > n_numa_node:
                logger.warning(
                    f"TP size ({self.tp_size})is larger than numa node number ({n_numa_node}), "
                    f"in this case the available memory amount of each rank cannot be determined in prior. "
                    f"Please set proper `--max-total-tokens` to avoid the out-of-memory error."
                )

    def apply_torch_tp(self):
        logger.info(f"Enabling torch tensor parallelism on {self.tp_size} devices.")
        from sglang.srt.layers.model_parallel import tensor_parallel

        device_mesh = torch.distributed.init_device_mesh(self.device, (self.tp_size,))
        tensor_parallel(self.model, device_mesh)

    def update_decode_attn_backend(self, stream_idx: int):
        self.decode_attn_backend = self.decode_attn_backend_group[stream_idx]

    def forward_decode(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        if not skip_attn_backend_init:
            if self.server_args.enable_pdmux:
                self.decode_attn_backend.init_forward_metadata(forward_batch)
                forward_batch.attn_backend = self.decode_attn_backend
            else:
                self.attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )

    def forward_extend(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> Tuple[
        Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput], bool
    ]:
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
        if not self.is_generation:
            kwargs["get_embedding"] = True

        can_run_graph = (
            self.piecewise_cuda_graph_runner is not None
            and self.piecewise_cuda_graph_runner.can_run(forward_batch)
        )

        if can_run_graph:
            return (
                self.piecewise_cuda_graph_runner.replay(forward_batch, **kwargs),
                can_run_graph,
            )

        if not skip_attn_backend_init:
            self.attn_backend.init_forward_metadata(forward_batch)

        return (
            self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            ),
            can_run_graph,
        )

    def forward_idle(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        # In DP Attention, IDLE batches are padded (batch_size > 0) for MLP sync.
        # in this case, we need to reinit the forward metadata, otherwise the stale
        # metadata causes batch_size mismatch in attention kernel(e.g. NSA Indexer).
        if forward_batch.batch_size > 0:
            self.attn_backend.init_forward_metadata(forward_batch)

        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )

    def forward_split_prefill(
        self,
        forward_batch: ForwardBatch,
        reinit_attn_backend: bool = False,
        forward_count: int = 1,
    ) -> LogitsProcessorOutput:
        if forward_batch.split_index == 0 or reinit_attn_backend:
            self.attn_backend.init_forward_metadata(forward_batch)
        next_split_index = min(
            forward_batch.split_index + forward_count,
            self.model_config.num_hidden_layers,
        )
        ret = self.model.forward_split_prefill(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            (forward_batch.split_index, next_split_index),
        )
        forward_batch.split_index = next_split_index
        return ret

    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        self.forward_pass_id += 1

        with get_global_expert_distribution_recorder().with_forward_pass(
            self.forward_pass_id,
            forward_batch,
        ) as recorder_outputs:
            output = self._forward_raw(
                forward_batch,
                skip_attn_backend_init,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            elastic_ep_state = ElasticEPStateManager.instance()
            if (
                elastic_ep_state is not None
                and not elastic_ep_state.is_active_equal_last()
            ):
                elastic_ep_state.snapshot_active_to_last()
                elastic_ep_state.sync_active_to_cpu()
                logging.info("EPLB due to rank faults")
                gen = self.eplb_manager.rebalance()
                while True:
                    try:
                        next(gen)
                    except StopIteration:
                        break
                output = self._forward_raw(
                    forward_batch,
                    skip_attn_backend_init,
                    pp_proxy_tensors,
                    reinit_attn_backend,
                    split_forward_count,
                )
        output.expert_distribution_metrics = recorder_outputs.get("metrics")

        # Copy cached routing experts' buffers back to CPU cache
        get_global_experts_capturer().on_forward_end(
            forward_batch=forward_batch,
            can_run_graph=output.can_run_graph,
            cuda_graph_batch=getattr(self.graph_runner, "bs", None),
        )

        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()

        return output

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        mode_check = (
            forward_batch.forward_mode.is_cpu_graph
            if self.device == "cpu"
            else forward_batch.forward_mode.is_cuda_graph
        )
        can_run_graph = bool(
            mode_check()
            and self.graph_runner
            and self.graph_runner.can_run(forward_batch)
        )

        if can_run_graph:
            ret = self.graph_runner.replay(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

        # For MLP sync
        if forward_batch.global_num_tokens_cpu is not None:
            forward_batch.prepare_mlp_sync_batch(self)
        else:
            forward_batch.prepare_attn_tp_scatter_input(self)

        # Normalize num_token_non_padded to be local to this attention TP rank if needed.
        if (
            forward_batch.num_token_non_padded is not None
            and forward_batch.global_num_tokens_gpu is not None
            and require_gathered_buffer(self.server_args)
            and not is_nsa_enable_prefill_cp()
        ):
            forward_batch.adjust_num_token_non_padded_for_attn_tp(
                server_args=self.server_args,
            )

        if forward_batch.forward_mode.is_decode():
            ret = self.forward_decode(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        elif forward_batch.forward_mode.is_split_prefill():
            ret = self.forward_split_prefill(
                forward_batch,
                reinit_attn_backend=reinit_attn_backend,
                forward_count=split_forward_count,
            )
        elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            ret, can_run_graph = self.forward_extend(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        elif forward_batch.forward_mode.is_idle():
            ret = self.forward_idle(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

        if (
            forward_batch.global_num_tokens_cpu is not None
            and self.pp_group.is_last_rank
        ):
            forward_batch.post_forward_mlp_sync_batch(ret)

        return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # NOTE: In overlap mode, the function update_regex_vocab_mask (in sample)
        #       was executed after we processed last batch's results.

        # Calculate logits bias and apply it to next_token_logits.
        sampling_info.update_regex_vocab_mask()
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        # For duplex models with multiple output streams.
        if isinstance(logits_output, tuple):
            return torch.stack(
                [self.sample(values, forward_batch) for values in logits_output],
                axis=-1,
            )

        self._preprocess_logits(logits_output, forward_batch.sampling_info)
        # Sample the next tokens
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
            # For prefill, we only use the position of the last token.
            (
                forward_batch.positions
                if forward_batch.forward_mode.is_decode()
                else forward_batch.seq_lens - 1
            ),
        )
        return next_token_ids

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> None:
        """
        Compute token_ids_logprobs without performing sampling.

        Optimized path for prefill-only requests that need token_ids_logprobs but don't
        require next token generation. Skips expensive sampling operations
        while still providing requested probability information.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output
        """
        if not forward_batch.token_ids_logprobs:
            return

        # Preprocess logits (same as in sample method)
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Delegate to sampler for logprob-only computation
        # This populates logits_output with requested token probabilities
        self.sampler.compute_logprobs_only(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )

    @property
    def model_is_mrope(self) -> bool:
        """Detect if the model has "mrope" rope_scaling type.
        mrope requires keep "rope_deltas" between prompt and decoding phases."""
        rope_scaling = getattr(
            self.model_config.hf_text_config, "rope_parameters", None
        ) or getattr(self.model_config.hf_text_config, "rope_scaling", {})
        if rope_scaling is None:
            return False
        is_mrope_enabled = "mrope_section" in rope_scaling
        return is_mrope_enabled

    def save_remote_model(self, url: str):
        from sglang.srt.model_loader.loader import RemoteModelLoader

        logger.info(f"Saving model to {url}")
        RemoteModelLoader.save_model(self.model, self.model_config.model_path, url)

    def save_sharded_model(
        self, path: str, pattern: Optional[str] = None, max_size: Optional[int] = None
    ):
        from sglang.srt.model_loader.loader import ShardedStateLoader

        logger.info(
            f"Save sharded model to {path} with pattern {pattern} and max_size {max_size}"
        )
        ShardedStateLoader.save_model(self.model, path, pattern, max_size)

    def check_weights(self, action: str):
        self._weight_checker.handle(action=action)

    def update_weights_from_ipc(self, recv_req):
        """Update weights from IPC for checkpoint-engine integration."""
        try:
            from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
                SGLangCheckpointEngineWorkerExtensionImpl,
            )

            # Create a worker extension that integrates with SGLang's model
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self)
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)

    def prealloc_symmetric_memory_pool(self):
        # PyTorch mempools never de-fragment memory in OOM scenarios, so we need to pre-allocate a large chunk of memory to limit fragmentation.
        if (
            self.is_draft_worker
            or not self.server_args.enable_symm_mem
            or envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() <= 0
        ):
            return

        # Memory allocation is tied to a cuda stream, use the forward stream
        with torch.get_device_module(self.device).stream(self.forward_stream):
            logger.info(
                f"Pre-allocating symmetric memory pool with {envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get()} GiB"
            )
            with use_symmetric_memory(get_tp_group()):
                torch.empty(
                    (envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() * 1024 * 1024 * 1024,),
                    dtype=torch.uint8,
                    device=self.device,
                )


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank, device):
    if isinstance(tensor, LocalSerializedTensor):
        tensor = tensor.get(tp_rank)
    return tensor.to(device)


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])
