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
"""A scheduler that manages a tensor parallel GPU worker."""

import faulthandler
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import psutil
import setproctitle
import torch
import torch.distributed
import zmq
from torch.cuda import Stream as CudaStream
from torch.cuda import StreamContext as CudaStreamContext
from torch.distributed import barrier

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.encode_receiver import MMReceiverHTTP
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
    release_req_to_metadata_buffer,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    prepare_abort,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.mamba.ops import (
    initialize_mamba_selective_state_update_backend,
)
from sglang.srt.layers.dp_attention import (
    compute_dp_attention_world_info,
    get_attention_cp_group,
    get_attention_tp_group,
)
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActiveRanksOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    BaseBatchReq,
    BaseReq,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    CheckWeightsReqInput,
    CommitBalloonReqInput,
    CommitBalloonReqOutput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    FreezeGCReq,
    GetBalloonStatusReqInput,
    GetBalloonStatusReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadReqInput,
    GetLoadsReqInput,
    GetWeightsByNameReqInput,
    HealthCheckOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    PauseGenerationReqInput,
    ProfileReq,
    PrepareBalloonReqInput,
    PrepareBalloonReqOutput,
    ReleaseMemoryOccupationReqInput,
    RestoreFromBalloonReqInput,
    RestoreFromBalloonReqOutput,
    WarmupBalloonReqInput,
    WarmupBalloonReqOutput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    SyncKVCapacityReqInput,
    SyncKVCapacityReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.mm_utils import init_mm_embedding_cache, unwrap_shm_features
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
)
from sglang.srt.kunserve_forward_timing import (
    kunserve_scheduler_gap_timing_enabled,
    kunserve_timing_log,
    kunserve_timing_scope,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    ModelWorkerBatch,
    MultimodalInputs,
    Req,
    RequestStage,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_dp_attn_mixin import SchedulerDPAttnMixin
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.scheduler_metrics_mixin import (
    RECORD_STEP_TIME,
    PrefillStats,
    SchedulerMetricsMixin,
)
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin
from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper
from sglang.srt.managers.scheduler_runtime_checker_mixin import (
    SchedulerRuntimeCheckerMixin,
    create_scheduler_watchdog,
)
from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,
)
from sglang.srt.managers.session_controller import Session
from sglang.srt.managers.utils import GenerationBatchResult, validate_input_length
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.server_args import PortArgs, ServerArgs, get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.tracing.trace import (
    process_tracing_init,
    trace_event_batch,
    trace_set_proc_propagate_context,
    trace_set_thread_info,
    trace_slice_batch,
    trace_slice_end,
    trace_slice_start,
)
from sglang.srt.utils import (
    DynamicGradMode,
    broadcast_pyobj,
    configure_gc_logger,
    configure_logger,
    freeze_gc,
    get_available_gpu_memory,
    get_bool_env_var,
    get_int_env_var,
    get_zmq_socket,
    kill_itself_when_parent_died,
    numa_bind_to_node,
    point_to_point_pyobj,
    require_mlp_sync,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


def _kunserve_ms(message: str, *args) -> None:
    """Mirror of model_runner._kunserve_ms for the scheduler subprocess.

    Writes the milestone to logger.warning AND to the file pointed at by
    KUNSERVE_DETAIL_LOG so it survives the multiprocessing fork that hides
    the scheduler's stdout from Ray's actor capture stream.
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
        import datetime as _dt

        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        line = f"[{ts} pid={os.getpid()}] {rendered}\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception:
        pass


def _kun_wd(message: str) -> None:
    """[KUNSERVE-WD] lockstep watchdog probe -> KUNSERVE_DETAIL_LOG only (no
    logger spam). Each line is flushed to disk so it survives a hang. TEMPORARY
    debug instrumentation for the cross-replica negotiate lockstep divergence."""
    path = os.environ.get("KUNSERVE_DETAIL_LOG")
    if not path:
        return
    try:
        import datetime as _dt

        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"[{ts} pid={os.getpid()}] {message}\n")
    except Exception:
        pass


def _kunserve_local_forward_probe_enabled() -> bool:
    return os.environ.get("KUNSERVE_LOCAL_FORWARD_PROBE", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


_BATCH_TIMING_LOG = os.environ.get("SGLANG_BATCH_TIMING_LOG", "").strip()
_REPLICA_RANK = os.environ.get("SGLANG_REPLICA_RANK", "")
_REQ_LIFECYCLE_LOG = os.environ.get("SGLANG_REQ_LIFECYCLE_LOG", "").strip()
if not _REQ_LIFECYCLE_LOG and os.environ.get("SGLANG_KUNSERVE_OUTPUT_DIR"):
    _REQ_LIFECYCLE_LOG = os.path.join(
        os.environ["SGLANG_KUNSERVE_OUTPUT_DIR"],
        f"req_lifecycle_r{_REPLICA_RANK or '0'}.jsonl",
    )
# Test retract decode for debugging purposes
TEST_RETRACT = envs.SGLANG_TEST_RETRACT.get()
TEST_RETRACT_INTERVAL = envs.SGLANG_TEST_RETRACT_INTERVAL.get()
TEST_RETRACT_NO_PREFILL_BS = envs.SGLANG_TEST_RETRACT_NO_PREFILL_BS.get()


@dataclass
class EmbeddingBatchResult:
    embeddings: torch.Tensor
    copy_done: Optional[torch.cuda.Event] = None

    def copy_to_cpu(self):
        """Copy embeddings tensor to CPU in overlap scheduling."""

        if isinstance(self.embeddings, torch.Tensor):
            self.copy_done = torch.get_device_module(self.embeddings.device).Event()
            self.embeddings = self.embeddings.to("cpu", non_blocking=True)
        else:
            assert isinstance(self.embeddings, list)
            if len(self.embeddings) == 0:
                return

            self.copy_done = torch.get_device_module(self.embeddings[0].device).Event()
            self.embeddings = [
                emb.to("cpu", non_blocking=True) for emb in self.embeddings
            ]

        self.copy_done.record()


class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerUpdateWeightsMixin,
    SchedulerProfilerMixin,
    SchedulerMetricsMixin,
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerMultiplexMixin,
    SchedulerRuntimeCheckerMixin,
    SchedulerPPMixin,
    SchedulerDPAttnMixin,
    SchedulerDllmMixin,
):
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        dp_rank: Optional[int],
    ):
        self.is_initializing = True
        self.init_soft_watchdog(server_args)

        # Parse args
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank
        self.attn_cp_rank = attn_cp_rank
        self.attn_cp_size = server_args.attn_cp_size
        self.moe_dp_rank = moe_dp_rank
        self.moe_dp_size = server_args.moe_dp_size
        self.dp_rank = dp_rank
        self.tp_size = server_args.tp_size
        self.moe_ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.dp_size = server_args.dp_size
        self.nccl_port = port_args.nccl_port
        self.schedule_policy = server_args.schedule_policy
        self.enable_priority_scheduling = server_args.enable_priority_scheduling
        self.abort_on_priority_when_disabled = (
            server_args.abort_on_priority_when_disabled
        )
        self.schedule_low_priority_values_first = (
            server_args.schedule_low_priority_values_first
        )
        self.priority_scheduling_preemption_threshold = (
            server_args.priority_scheduling_preemption_threshold
        )
        self.enable_lora = server_args.enable_lora
        self.enable_lora_overlap_loading = server_args.enable_lora_overlap_loading
        self.max_loras_per_batch = server_args.max_loras_per_batch
        self.enable_overlap = not server_args.disable_overlap_schedule
        self.enable_pdmux = server_args.enable_pdmux
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.enable_metrics = server_args.enable_metrics
        self.enable_metrics_for_all_schedulers = (
            server_args.enable_metrics_for_all_schedulers
        )
        self.enable_trace = server_args.enable_trace
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.gpu_id = gpu_id
        self.page_size = server_args.page_size
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
        self.enable_hicache_storage = server_args.hicache_storage_backend is not None
        self.max_recv_per_poll = envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get()
        self.batch_timing_log = _BATCH_TIMING_LOG

        # Distributed rank info
        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
                self.attn_cp_size,
            )
        )

        self.enable_kv_cache_events = bool(
            server_args.kv_events_config and self.attn_tp_rank == 0
        )

        # Init model configs
        self.init_model_config()

        # Init metrics stats
        self.init_metrics(tp_rank, pp_rank, dp_rank)

        # Init inter-process communication
        self.init_ipc_channels(port_args)

        # Init PD-multiplexing context
        if self.enable_pdmux:
            self.init_pdmux()

        # Init tokenizer
        self.init_tokenizer()

        # Init moe config and GEMM config (FP8 GEMM, etc.)
        self.init_moe_gemm_config()

        # Init mamba backend
        self.init_mamba_backend()

        # Launch a model worker and draft model worker if using speculative decoding
        self.init_model_worker()

        if (t := envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.get()) > 0:
            time.sleep(t)

        # Init cache and memory pool
        self.init_cache_with_memory_pool()

        # Init running status
        self.init_running_status()

        # Init chunked prefill
        self.init_chunked_prefill()

        # Init diffusion LLM
        self.init_diffusion_llm()

        # Init schedule policy and new token estimation
        self.init_schedule_policy()

        # Init watchdog, memory saver, input blocker and recv skipper
        self.init_watch_dog_memory_saver_input_blocker()

        # Init profiler
        self.init_profiler()

        # Init prefill-decodedisaggregation
        self.init_disaggregation()

        # Init overlap schedule
        self.init_overlap()

        # Init prefill kv split size when deterministic inference is enabled with various attention backends
        self.init_deterministic_inference_config()

        # Init request dispatcher
        self.init_request_dispatcher()

        # Init LoRA overlap loader
        if self.enable_lora_overlap_loading:
            self.lora_overlap_loader = LoRAOverlapLoader(
                self.tp_worker.model_runner.lora_manager
            )

        # Init the grammar backend for constrained generation
        self.grammar_manager = GrammarManager(self)

        self.is_initializing = False
        self._last_run_batch_end_ts = None
        self.req_lifecycle_log = _REQ_LIFECYCLE_LOG
        self._dumped_rids: set = set()  # 防止同一 rid 重复 dump
        self._queue_wait_ms_cumulative: Dict[str, float] = {}
        self._queue_wait_ms_last_dequeue: Dict[str, float] = {}

    def init_model_config(self):
        self.model_config = ModelConfig.from_server_args(self.server_args)

    def init_ipc_channels(self, port_args: PortArgs):
        context = zmq.Context(2)
        self.idle_sleeper = None

        if self.pp_rank == 0 and self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            self.recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False
            )

            send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            if self.server_args.skip_tokenizer_init:
                # Directly send to the TokenizerManager
                send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # Send to the DetokenizerManager
                send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )

            self.send_to_tokenizer = SenderWrapper(send_to_tokenizer)
            self.send_to_detokenizer = SenderWrapper(send_to_detokenizer)

            if self.server_args.sleep_on_idle:
                self.idle_sleeper = IdleSleeper(
                    [
                        self.recv_from_tokenizer,
                        self.recv_from_rpc,
                    ]
                )
        else:
            self.recv_from_tokenizer = None
            self.recv_from_rpc = None
            self.send_to_tokenizer = SenderWrapper(None)
            self.send_to_detokenizer = SenderWrapper(None)

        if self.current_scheduler_metrics_enabled:
            self.send_metrics_from_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.metrics_ipc_name, False
            )

    def init_tokenizer(self):
        server_args = self.server_args
        self.is_generation = self.model_config.is_generation

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    use_fast=not server_args.disable_fast_image_processor,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )

        # Set reasoning_parser and think_end_id if --reasoning_parser is enabled
        if self.server_args.reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=self.server_args.reasoning_parser, stream_reasoning=False
            )
            self.tokenizer.think_end_id = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )[0]

    def init_mamba_backend(self) -> None:
        initialize_mamba_selective_state_update_backend(self.server_args)

    def init_moe_gemm_config(self):
        # For the MM models, check the text_config for MoE settings
        config_to_check = getattr(
            self.model_config.hf_config, "text_config", self.model_config.hf_config
        )

        if hasattr(config_to_check, "num_experts_per_tok"):
            initialize_moe_config(self.server_args)

        # Initialize GEMM-related configuration for FP8 and FP4 backends.
        initialize_fp8_gemm_config(self.server_args)
        initialize_fp4_gemm_config(self.server_args)

        # This must be called after initialize_moe_config
        self.require_mlp_sync = require_mlp_sync(self.server_args)

    def init_tp_model_worker(self):
        from sglang.srt.managers.tp_worker import TpModelWorker

        self.tp_worker = TpModelWorker(
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=self.moe_ep_rank,
            pp_rank=self.pp_rank,
            attn_cp_rank=self.attn_cp_rank,
            moe_dp_rank=self.moe_dp_rank,
            dp_rank=self.dp_rank,
            nccl_port=self.nccl_port,
        )

    def maybe_init_draft_worker(self):
        if self.spec_algorithm.is_none():
            self.draft_worker = None
            return

        # Launch a draft worker for speculative decoding
        draft_worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=self.moe_ep_rank,
            nccl_port=self.nccl_port,
            target_worker=self.tp_worker,
            dp_rank=self.dp_rank,
            attn_cp_rank=self.attn_cp_rank,
            moe_dp_rank=self.moe_dp_rank,
        )

        if self.server_args.speculative_draft_load_format is not None:
            self.server_args.load_format = (
                self.server_args.speculative_draft_load_format
            )
            logger.info(
                f"Using draft model load_format: '{self.server_args.speculative_draft_load_format}'"
            )

        DraftWorkerClass = self.spec_algorithm.create_worker(self.server_args)
        self.draft_worker = DraftWorkerClass(**draft_worker_kwargs)

    def init_model_worker(self):
        self.init_tp_model_worker()
        self.maybe_init_draft_worker()

        # Dispatch the model worker
        if self.spec_algorithm.is_none():
            self.model_worker = self.tp_worker
        else:
            self.model_worker = self.draft_worker

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            self.forward_stream,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        if get_global_server_args().pp_max_micro_batch_size is None:
            get_global_server_args().pp_max_micro_batch_size = max(
                self.max_running_requests // self.pp_size, 1
            )

        self.tp_group = get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = get_attention_tp_group()
        self.attn_tp_cpu_group = self.attn_tp_group.cpu_group
        self.attn_cp_group = get_attention_cp_group()
        self.attn_cp_cpu_group = self.attn_cp_group.cpu_group
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # NOTE: dp_tp_* are request/data-plane coordination groups (not tensor collectives).
        # When DP attention is enabled, scope to the attention-TP group; otherwise use
        # the base TP group. Entry rank is the local rank 0 in that group.
        # Use the CPU (gloo) group to broadcast VLM Python objects and avoid CUDA
        # stream/device coupling (#11910).
        self.dp_tp_group = (
            self.attn_tp_group
            if self.server_args.enable_dp_attention
            else self.tp_group
        )
        self.dp_tp_cpu_group = self.dp_tp_group.cpu_group

        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        set_random_seed(self.random_seed)

        # Print debug info
        if self.tp_rank == 0:
            avail_mem = get_available_gpu_memory(
                self.device, self.gpu_id, empty_cache=False
            )
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={self.server_args.chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}={avail_mem:.2f} GB"
            )

        if self.enable_metrics and hasattr(self, "metrics_collector"):
            self.metrics_collector.emit_cache_config_info(
                self.page_size, self.max_total_num_tokens // self.page_size
            )

    def init_cache_with_memory_pool(self):
        server_args = self.server_args

        # Hybrid memory pool
        self.is_hybrid_swa = self.tp_worker.is_hybrid_swa
        self.is_hybrid_ssm = (
            self.tp_worker.model_runner.hybrid_gdn_config is not None
            or self.tp_worker.model_runner.mamba2_config is not None
        )

        self.sliding_window_size = None
        if self.is_hybrid_swa:
            self.sliding_window_size = self.tp_worker.sliding_window_size
            self.full_tokens_per_layer, self.swa_tokens_per_layer = (
                self.tp_worker.get_tokens_per_layer_info()
            )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.tp_worker.get_memory_pool()
        )

        # Create cache
        params = CacheInitParams(
            disable=server_args.disable_radix_cache,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            page_size=self.page_size,
            is_eagle=self.spec_algorithm.is_eagle(),
            tp_cache_group=(
                self.attn_tp_cpu_group
                if self.server_args.enable_dp_attention
                else self.tp_cpu_group
            ),
            eviction_policy=server_args.radix_eviction_policy,
            enable_metrics=self.enable_metrics,
            enable_kv_cache_events=self.enable_kv_cache_events,
            enable_mamba_extra_buffer=server_args.enable_mamba_extra_buffer(),
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            chunked_prefill_size=server_args.chunked_prefill_size,
            sliding_window_size=self.sliding_window_size,
        )

        if (
            server_args.chunked_prefill_size is not None
            and server_args.disable_radix_cache
        ):
            if not self.is_hybrid_swa:
                from sglang.srt.mem_cache.chunk_cache import ChunkCache

                self.tree_cache = ChunkCache(params)
            else:
                from sglang.srt.mem_cache.chunk_cache import SWAChunkCache

                self.tree_cache = SWAChunkCache(params)
        else:

            if envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get():
                # lazy import to avoid JIT overhead
                from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp

                logger.info("Using experimental C++ radix tree implementation.")
                self.tree_cache = RadixCacheCpp(params=params, server_args=server_args)
            elif self.enable_hierarchical_cache:
                from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

                self.tree_cache = HiRadixCache(params=params, server_args=server_args)
                self.tp_worker.register_hicache_layer_transfer_counter(
                    self.tree_cache.cache_controller.layer_done_counter
                )
            elif self.is_hybrid_swa:
                from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache

                self.tree_cache = SWARadixCache(params=params)
            elif self.is_hybrid_ssm:
                from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

                self.tree_cache = MambaRadixCache(params)
            elif server_args.enable_lmcache:
                from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import (
                    LMCRadixCache,
                )

                self.tree_cache = LMCRadixCache(
                    params=params,
                    model_config=self.model_config,
                    tp_size=self.tp_size,
                    rank=self.tp_rank,
                    tp_group=self.tp_group,
                )
            else:
                self.tree_cache = RadixCache(params)

        if (
            server_args.disaggregation_mode == "decode"
            and server_args.disaggregation_decode_enable_offload_kvcache
        ):
            self.decode_offload_manager = DecodeKVCacheOffloadManager(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tp_group=params.tp_cache_group,
                tree_cache=self.tree_cache,
                server_args=self.server_args,
            )
        else:
            self.decode_offload_manager = None

        embedding_cache_size = envs.SGLANG_VLM_CACHE_SIZE_MB.get()
        init_mm_embedding_cache(embedding_cache_size * 1024 * 1024)

    def init_running_status(self):
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.return_health_check_ct = 0
        self.num_retracted_reqs: int = 0
        self.num_paused_reqs: int = 0
        self.sessions: Dict[str, Session] = {}
        self.forward_sleep_time = None
        self._engine_paused = False
        self.expand_requested = False
        self.expand_request_reason: Optional[str] = None
        self.balloon_keepalive_step_ct: int = 0
        self._balloon_keepalive_active = False
        self._kunserve_prefill_blocked_full_log_ct: int = 0
        self._kunserve_graph_prefill_defer_log_ct: int = 0
        self._kunserve_phase_e_prefill_defer_log_ct: int = 0
        self._kunserve_balloon_prefill_mem_defer_log_ct: int = 0
        self._kunserve_balloon_prefill_budget_log_ct: int = 0
        # Phase E negotiation is a lockstep collective.  A rank cannot safely
        # decide to negotiate only because its own local batch changed; peers
        # that reuse a cached decision would deadlock.  The cache therefore
        # behaves event-driven for steady decode/shrink steps by reusing the
        # last shared bucket, with a low-frequency deterministic refresh to
        # admit prefill/growth and to eventually lower the shared bucket after
        # all ranks shrink.  The old default (16) made refresh cost visible in
        # every profile; 256 keeps the safety valve while removing most steady
        # decode negotiations.
        self._phase_e_negotiate_interval: int = max(
            1, get_int_env_var("KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL", 256)
        )
        self._phase_e_cache_guard_enabled: bool = os.environ.get(
            "KUNSERVE_PHASE_E_CACHE_GUARD", "1"
        ) not in ("0", "false", "False", "no", "NO")
        self._phase_e_cache_valid: bool = False
        self._phase_e_cached_max_bs: int = 0
        self._phase_e_cached_min_bs: int = 0
        self._phase_e_cached_raw_max_bs: int = 0
        self._phase_e_cached_raw_min_bs: int = 0
        self._phase_e_cached_any_force_eager: bool = False
        self._phase_e_cached_steps_left: int = 0
        self._phase_e_cached_state_fingerprint: Optional[int] = None

    def init_chunked_prefill(self):
        # Init chunked prefill
        self.chunked_prefill_size = self.server_args.chunked_prefill_size
        if self.chunked_prefill_size <= 0:  # -1 means disable
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None
            and self.server_args.enable_mixed_chunk
        )

        # Init the dynamic chunking predictor for PP
        self.enable_dynamic_chunking = (
            self.server_args.enable_dynamic_chunking and self.pp_size > 1
        )
        if self.enable_dynamic_chunking:
            try:
                self.profile_and_init_predictor()
            except Exception as e:
                logger.warning(
                    f"[PP Dynamic Chunk] Failed to profile prefill latency: {e}. "
                    "Dynamic chunking will be disabled."
                )
                self.enable_dynamic_chunking = False

    def init_schedule_policy(self):
        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            self.enable_hierarchical_cache,
            self.enable_priority_scheduling,
            self.schedule_low_priority_values_first,
        )
        self.prefill_delayer: Optional[PrefillDelayer] = None
        if self.server_args.enable_prefill_delayer:
            self.prefill_delayer = PrefillDelayer(
                dp_size=self.dp_size,
                attn_tp_size=self.attn_tp_size,
                cpu_group=self.tp_cpu_group,
                server_args=self.server_args,
                metrics_collector=(
                    self.metrics_collector if self.enable_metrics else None
                ),
                max_delay_passes=self.server_args.prefill_delayer_max_delay_passes,
                token_usage_low_watermark=self.server_args.prefill_delayer_token_usage_low_watermark,
            )
        # Enable preemption for priority scheduling.
        self.try_preemption = self.enable_priority_scheduling
        self.init_new_token_ratio = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * self.server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(),
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
        self.new_token_ratio = self.init_new_token_ratio

    def init_soft_watchdog(self, server_args: ServerArgs):
        if (x := server_args.soft_watchdog_timeout) is not None:
            self.soft_watchdog = create_scheduler_watchdog(
                self, watchdog_timeout=x, soft=True
            )

    def init_watch_dog_memory_saver_input_blocker(self):
        # Start watchdog thread
        self.watchdog = create_scheduler_watchdog(
            self, watchdog_timeout=self.server_args.watchdog_timeout
        )

        # Init memory saver, profiler and metric stats
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )
        self.offload_tags = set()

        # Init recv skipper and input blocker
        self.recv_skipper = SchedulerRecvSkipper.maybe_create(self.server_args)
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # Configure GC logger
        if envs.SGLANG_LOG_GC.get():
            configure_gc_logger()

    def init_disaggregation(self):
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )

        if self.draft_worker is None or self.spec_algorithm.is_ngram():
            draft_token_to_kv_pool = None
        elif self.spec_algorithm.supports_spec_v2() and self.enable_overlap:
            if self.server_args.enable_multi_layer_eagle:
                draft_runner = self.draft_worker.draft_worker.draft_runner_list[0]
            else:
                draft_runner = self.draft_worker.draft_worker.draft_runner
            draft_token_to_kv_pool = draft_runner.token_to_kv_pool
            model_config = draft_runner.model_config
        else:
            # todo: should we fix this when enabling mtp or it doesn't matter since we only enable mtp in decode node thus we don't transfer draft kvs between P and D?
            draft_token_to_kv_pool = self.draft_worker.model_runner.token_to_kv_pool
            model_config = self.draft_worker.model_config

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
        ):  # *2 for the headroom.
            buffer_size = (self.req_to_token_pool.size) * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=(
                    model_config.hidden_size
                    if self.spec_algorithm.is_eagle()
                    else 16  # minimal padding size for RDMA
                ),
                hidden_states_dtype=(
                    model_config.dtype
                    if self.spec_algorithm.is_eagle()
                    else torch.float32
                ),
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            # The decode requests polling kv cache
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.attn_tp_cpu_group,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                tp_rank=self.tp_rank,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                tree_cache=self.tree_cache,
            )

            # The decode requests pending for pre-allocation
            self.disagg_decode_prealloc_queue = DecodePreallocQueue(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                transfer_queue=self.disagg_decode_transfer_queue,
                tree_cache=self.tree_cache,
                gloo_group=self.attn_tp_cpu_group,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                dp_size=self.server_args.dp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                max_total_num_tokens=self.max_total_num_tokens,
                prefill_pp_size=self.server_args.disaggregation_prefill_pp,
                pp_rank=self.pp_rank,
                num_reserved_decode_tokens=self.server_args.num_reserved_decode_tokens,
                transfer_backend=self.transfer_backend,
            )

        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # *2 for the headroom.
            buffer_size = self.max_running_requests * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=(
                    model_config.hidden_size
                    if self.spec_algorithm.is_eagle()
                    or self.spec_algorithm.is_standalone()
                    else 16  # minimal padding size for RDMA
                ),
                hidden_states_dtype=(
                    model_config.dtype
                    if self.spec_algorithm.is_eagle()
                    or self.spec_algorithm.is_standalone()
                    else torch.float32
                ),
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                gloo_group=self.attn_tp_cpu_group,
                max_total_num_tokens=self.max_total_num_tokens,
                decode_tp_size=self.server_args.disaggregation_decode_tp,
                decode_dp_size=self.server_args.disaggregation_decode_dp,
                scheduler=self,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                transfer_backend=self.transfer_backend,
            )
            # The prefill requests that are in the middle of kv sending
            self.disagg_prefill_inflight_queue: List[Req] = []

        # Init mm receiver for EPD disaggregation mode
        if (
            self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            self.mm_receiver = MMReceiverHTTP(
                self.server_args,
                hf_config=self.model_config.hf_config,
                pp_rank=self.pp_rank,
                tp_rank=self.tp_rank,
                tp_group=self.tp_group,
                scheduler=self,
            )

    def init_overlap(self):
        self.device_module = torch.get_device_module(self.device)
        self.default_stream: CudaStream = self.device_module.current_stream()
        if self.device == "cpu":
            self.default_stream.synchronize = lambda: None  # No-op for CPU

        self.forward_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.forward_stream
        )
        self.copy_stream: CudaStream = self.device_module.Stream()
        self.copy_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.copy_stream
        )

        if not self.enable_overlap:
            self.future_map = None
            return

        self.future_map = FutureMap(
            self.max_running_requests,
            self.chunked_prefill_size,
            self.model_config.context_len,
            self.device,
            self.spec_algorithm,
        )
        self.batch_record_buf = [None] * 2
        self.batch_record_ct = 0

    def init_deterministic_inference_config(self):
        """Initialize deterministic inference configuration for different attention backends."""
        if not self.server_args.enable_deterministic_inference:
            self.truncation_align_size = None
            return

        backend_sizes = {
            "flashinfer": ("SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096),
            "triton": ("SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE", 4096),
        }
        env_var, default_size = backend_sizes.get(
            self.server_args.attention_backend, (None, None)
        )
        self.truncation_align_size = (
            get_int_env_var(env_var, default_size) if env_var else None
        )

    def init_request_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
                (FlushCacheReqInput, self.flush_cache_wrapped),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AttachHiCacheStorageReqInput, self.attach_hicache_storage_wrapped),
                (DetachHiCacheStorageReqInput, self.detach_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),
                (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),
                (DestroyWeightsUpdateGroupReqInput, self.destroy_weights_update_group),
                (
                    InitWeightsSendGroupForRemoteInstanceReqInput,
                    self.init_weights_send_group_for_remote_instance,
                ),
                (
                    SendWeightsToRemoteInstanceReqInput,
                    self.send_weights_to_remote_instance,
                ),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.update_weights_from_distributed,
                ),
                (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),
                (UpdateWeightsFromIPCReqInput, self.update_weights_from_ipc),
                (GetWeightsByNameReqInput, self.get_weights_by_name),
                (ReleaseMemoryOccupationReqInput, self.release_memory_occupation),
                (ResumeMemoryOccupationReqInput, self.resume_memory_occupation),
                (CheckWeightsReqInput, self.check_weights),
                (SlowDownReqInput, self.slow_down),
                (ProfileReq, self.profile),
                (FreezeGCReq, self.handle_freeze_gc),
                (GetBalloonStatusReqInput, self.get_balloon_status),
                (PrepareBalloonReqInput, self.prepare_balloon),
                (WarmupBalloonReqInput, self.warmup_balloon),
                (CommitBalloonReqInput, self.commit_balloon),
                (RestoreFromBalloonReqInput, self.restore_from_balloon),
                (SyncKVCapacityReqInput, self.sync_kv_capacity),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),
                (
                    LoadLoRAAdapterFromTensorsReqInput,
                    self.load_lora_adapter_from_tensors,
                ),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
                (GetLoadReqInput, self.get_load),
                (GetLoadsReqInput, self.get_loads),
                (PauseGenerationReqInput, self.pause_generation),
                (ContinueGenerationReqInput, self.continue_generation),
                (DumperControlReqInput, self.handle_dumper_control),
            ]
        )

    def _abort_on_running_timeout(self):
        # NOTE: this should be called before a batch is launched,
        # as current spec-v1 still filters batch inside verify stage.
        timeout_s = envs.SGLANG_REQ_RUNNING_TIMEOUT.get()
        if timeout_s <= 0:
            return
        if self.running_batch.is_empty():
            return

        deadline = time.perf_counter() - timeout_s
        for req in self.running_batch.reqs:
            if not req.finished() and 0 < req.time_stats.forward_entry_time < deadline:
                req.to_finish = FINISH_ABORT(
                    "Request running timeout reached.", HTTPStatus.SERVICE_UNAVAILABLE
                )

    def _kunserve_scheduler_gap_log(
        self,
        event: str,
        start_ns: int,
        *,
        batch: Optional[ScheduleBatch] = None,
        **fields: Any,
    ) -> None:
        if not kunserve_scheduler_gap_timing_enabled():
            return

        now_ns = time.perf_counter_ns()
        payload: Dict[str, Any] = {
            "elapsed_ms": round((now_ns - start_ns) / 1_000_000.0, 3),
            "forward_ct": int(self.forward_ct),
            "running": len(self.running_batch.reqs),
            "waiting": len(self.waiting_queue),
            "last_batch_mode": (
                str(self.last_batch.forward_mode) if self.last_batch is not None else "None"
            ),
            "last_batch_size": (
                int(self.last_batch.batch_size()) if self.last_batch is not None else 0
            ),
        }
        if self._last_run_batch_end_ts is not None:
            payload["since_last_run_end_ms"] = round(
                (time.perf_counter() - self._last_run_batch_end_ts) * 1000.0,
                3,
            )
        result_queue = getattr(self, "result_queue", None)
        if result_queue is not None:
            payload["result_queue_len"] = len(result_queue)
        if batch is not None:
            payload.update(
                {
                    "mode": str(batch.forward_mode),
                    "batch_size": int(batch.batch_size()),
                    "is_decode": bool(batch.forward_mode.is_decode()),
                    "is_extend": bool(batch.forward_mode.is_extend()),
                    "is_idle": bool(batch.forward_mode.is_idle()),
                    "is_extend_in_batch": bool(
                        getattr(batch, "is_extend_in_batch", False)
                    ),
                }
            )
        else:
            payload.update({"mode": "None", "batch_size": 0})
        payload.update(fields)
        kunserve_timing_log(event, **payload)

    @DynamicGradMode()
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            # Receive requests
            stage_ns = time.perf_counter_ns()
            recv_reqs = self.recv_requests()
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_recv_requests_end",
                stage_ns,
                recv_count=len(recv_reqs),
                loop="normal",
            )
            stage_ns = time.perf_counter_ns()
            self.process_input_requests(recv_reqs)
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_process_input_end",
                stage_ns,
                recv_count=len(recv_reqs),
                loop="normal",
            )
            if self._engine_paused:
                continue

            # Get the next batch to run
            stage_ns = time.perf_counter_ns()
            batch = self.get_next_batch_to_run()
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_get_next_batch_end",
                stage_ns,
                batch=batch,
                loop="normal",
            )

            # Phase E: negotiate the per-step bs/mode across the
            # cross-replica runtime_group so all 4 ranks enter
            # forward with the same ``local_m``.  Without this, the peer
            # replica's captured GLOBAL graph would replay with a fixed
            # bs while the idle local rank tries to all_gather an empty
            # tensor -> NCCL shape mismatch / silent hang.
            phase_e_force_eager_set = False
            if self._kunserve_phase_e_active():
                stage_ns = time.perf_counter_ns()
                local_status = self._local_balloon_status_or_stop()
                self._kunserve_scheduler_gap_log(
                    "scheduler_gap_phase_e_status_end",
                    stage_ns,
                    batch=batch,
                    loop="normal",
                    status_state=(
                        str(local_status.get("state"))
                        if isinstance(local_status, dict)
                        else "None"
                    ),
                )
                if local_status is not None:
                    (
                        negotiated_max_bs,
                        negotiated_min_bs,
                        negotiated_any_force_eager,
                        phase_e_from_cache,
                        phase_e_state_fingerprint,
                        negotiated_raw_max_bs,
                        negotiated_raw_min_bs,
                    ) = self._phase_e_get_step_decision(
                        batch=batch,
                        local_status=local_status,
                        loop="normal",
                    )
                    if batch is None and negotiated_max_bs > 0:
                        # Peer has real work; build a matching keepalive batch.
                        stage_ns = time.perf_counter_ns()
                        batch = self._build_balloon_keepalive_batch(
                            negotiated_max_bs, local_status
                        )
                        self._kunserve_scheduler_gap_log(
                            "scheduler_gap_build_keepalive_end",
                            stage_ns,
                            batch=batch,
                            loop="normal",
                            target_bs=int(negotiated_max_bs),
                        )
                    elif batch is None and negotiated_max_bs == 0:
                        # All replicas idle -> stop keepalive, skip this step.
                        self._stop_balloon_keepalive("all replicas idle")
                    # Symmetric force-eager decision: every rank
                    # observed the same (max, min) so we all agree.
                    stage_ns = time.perf_counter_ns()
                    self._phase_e_apply_step_decision(
                        negotiated_max_bs=negotiated_max_bs,
                        negotiated_min_bs=negotiated_min_bs,
                        negotiated_raw_max_bs=negotiated_raw_max_bs,
                        negotiated_raw_min_bs=negotiated_raw_min_bs,
                        negotiated_any_force_eager=negotiated_any_force_eager,
                        graph_bs_override_hint=(
                            negotiated_max_bs if phase_e_from_cache else None
                        ),
                        state_fingerprint=phase_e_state_fingerprint,
                    )
                    self._kunserve_scheduler_gap_log(
                        "scheduler_gap_phase_e_apply_decision_end",
                        stage_ns,
                        batch=batch,
                        loop="normal",
                        max_bs=int(negotiated_max_bs),
                        min_bs=int(negotiated_min_bs),
                        raw_max_bs=int(negotiated_raw_max_bs),
                        raw_min_bs=int(negotiated_raw_min_bs),
                        any_force_eager=bool(negotiated_any_force_eager),
                    )
                    phase_e_force_eager_set = True
            elif batch is None:
                # Non-sglang backends (DeepEP NORMAL): keep the legacy
                # local-only keepalive behavior.
                stage_ns = time.perf_counter_ns()
                batch = self._maybe_get_balloon_keepalive_batch()
                self._kunserve_scheduler_gap_log(
                    "scheduler_gap_maybe_keepalive_end",
                    stage_ns,
                    batch=batch,
                    loop="normal",
                )
            else:
                self._stop_balloon_keepalive("scheduled real batch")
            self.cur_batch = batch

            # Launch the current batch.  The Phase E per-step
            # force-eager flag is set above; it must be cleared even
            # if run_batch raises so the next iteration starts from a
            # known state.
            try:
                if batch:
                    result = self.run_batch(batch)
                    stage_ns = time.perf_counter_ns()
                    self.process_batch_result(batch, result)
                    self._kunserve_scheduler_gap_log(
                        "scheduler_gap_process_batch_result_end",
                        stage_ns,
                        batch=batch,
                        loop="normal",
                    )
                else:
                    # When the server is idle, do self-check and re-init some states
                    stage_ns = time.perf_counter_ns()
                    self.self_check_during_idle()
                    self._kunserve_scheduler_gap_log(
                        "scheduler_gap_self_check_idle_end",
                        stage_ns,
                        loop="normal",
                    )
            finally:
                if phase_e_force_eager_set:
                    stage_ns = time.perf_counter_ns()
                    mr = getattr(self.tp_worker, "model_runner", None)
                    if mr is not None:
                        mr.set_balloon_step_force_eager(False)
                        deepep_setter = getattr(
                            mr, "set_balloon_deepep_step_any_extend", None
                        )
                        if callable(deepep_setter):
                            deepep_setter(None)
                    self._kunserve_scheduler_gap_log(
                        "scheduler_gap_phase_e_clear_decision_end",
                        stage_ns,
                        batch=batch,
                        loop="normal",
                    )

            # Update last_batch
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                stage_ns = time.perf_counter_ns()
                self.self_check_during_busy()
                self._kunserve_scheduler_gap_log(
                    "scheduler_gap_self_check_busy_end",
                    stage_ns,
                    batch=batch,
                    loop="normal",
                )

    @DynamicGradMode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        self.result_queue: Deque[
            Tuple[ScheduleBatch, Union[GenerationBatchResult, EmbeddingBatchResult]]
        ] = deque()

        def pop_and_process():
            # Process the results of the last batch
            stage_ns = time.perf_counter_ns()
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_overlap_pop_process_end",
                stage_ns,
                batch=tmp_batch,
                loop="overlap",
                result_queue_len_after=len(self.result_queue),
            )

        while True:
            # Receive requests
            stage_ns = time.perf_counter_ns()
            recv_reqs = self.recv_requests()
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_recv_requests_end",
                stage_ns,
                recv_count=len(recv_reqs),
                loop="overlap",
            )
            stage_ns = time.perf_counter_ns()
            self.process_input_requests(recv_reqs)
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_process_input_end",
                stage_ns,
                recv_count=len(recv_reqs),
                loop="overlap",
            )
            if self._engine_paused:
                continue

            # Get the next batch to run
            stage_ns = time.perf_counter_ns()
            batch = self.get_next_batch_to_run()
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_get_next_batch_end",
                stage_ns,
                batch=batch,
                loop="overlap",
            )

            # Phase E: negotiate per-step bs/mode with peer replicas BEFORE
            # deciding overlap & launching.  See the
            # non-overlap loop above for the rationale.  We do the sync
            # here even when ``batch is not None`` so the active replica
            # advertises its bs to peers, which lets an idle peer build a
            # matching keepalive on the same step.
            phase_e_negotiated_max: Optional[int] = None
            phase_e_negotiated_min: Optional[int] = None
            phase_e_negotiated_any_force_eager: bool = False
            phase_e_negotiated_raw_max: int = 0
            phase_e_negotiated_raw_min: int = 0
            phase_e_from_cache: bool = False
            phase_e_status: Optional[Dict[str, Any]] = None
            phase_e_state_fingerprint: Optional[int] = None
            if self._kunserve_phase_e_active():
                stage_ns = time.perf_counter_ns()
                phase_e_status = self._local_balloon_status_or_stop()
                self._kunserve_scheduler_gap_log(
                    "scheduler_gap_phase_e_status_end",
                    stage_ns,
                    batch=batch,
                    loop="overlap",
                    status_state=(
                        str(phase_e_status.get("state"))
                        if isinstance(phase_e_status, dict)
                        else "None"
                    ),
                )
                if phase_e_status is not None:
                    (
                        phase_e_negotiated_max,
                        phase_e_negotiated_min,
                        phase_e_negotiated_any_force_eager,
                        phase_e_from_cache,
                        phase_e_state_fingerprint,
                        phase_e_negotiated_raw_max,
                        phase_e_negotiated_raw_min,
                    ) = self._phase_e_get_step_decision(
                        batch=batch,
                        local_status=phase_e_status,
                        loop="overlap",
                    )

            # Phase E EARLY keepalive build (overlap loop).  The original
            # elif at the bottom of this loop only fires when ``last_batch``
            # is also None, leaving a one-step gap on the transition
            # "last real batch still in result_queue" → "queue empty".
            # During that gap the peer replica's lane_group.all_gather has
            # no participant on this rank → deadlock until SILENCE_THRESHOLD
            # or NCCL timeout kills the run.  Building the keepalive HERE
            # (before run_batch) closes that gap.  Safe to do alongside an
            # in-flight last_batch because (a) the phantom req_pool entry
            # is distinct from any real request's req_pool slot, (b) the
            # dummy_kv_slot is reserved permanently, and (c) the keepalive
            # batch goes through the same overlap pipeline (result queued,
            # popped on next iter).
            real_batch_this_step = batch is not None
            if (
                batch is None
                and phase_e_negotiated_max is not None
                and phase_e_negotiated_max > 0
                and phase_e_status is not None
            ):
                stage_ns = time.perf_counter_ns()
                batch = self._build_balloon_keepalive_batch(
                    int(phase_e_negotiated_max), phase_e_status
                )
                self._kunserve_scheduler_gap_log(
                    "scheduler_gap_build_keepalive_end",
                    stage_ns,
                    batch=batch,
                    loop="overlap",
                    target_bs=int(phase_e_negotiated_max),
                )
            elif (
                batch is None
                and phase_e_negotiated_max == 0
            ):
                self._stop_balloon_keepalive("all replicas idle")

            if real_batch_this_step:
                self._stop_balloon_keepalive("scheduled real batch")
            self.cur_batch = batch
            stage_ns = time.perf_counter_ns()
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_disable_overlap_check_end",
                stage_ns,
                batch=batch,
                loop="overlap",
                disable_overlap_for_batch=bool(disable_overlap_for_batch),
            )

            # Phase E force-eager decision for the current batch (if any).
            # Done after we know the launched bs; will be applied before
            # run_batch and cleared right after.  The (max, min) pair is
            # symmetric across ranks so every rank reaches the same
            # decision.
            phase_e_force_eager_set = False
            if (
                phase_e_negotiated_max is not None
                and phase_e_negotiated_min is not None
                and batch is not None
            ):
                stage_ns = time.perf_counter_ns()
                self._phase_e_apply_step_decision(
                    negotiated_max_bs=int(phase_e_negotiated_max),
                    negotiated_min_bs=int(phase_e_negotiated_min),
                    negotiated_raw_max_bs=int(phase_e_negotiated_raw_max),
                    negotiated_raw_min_bs=int(phase_e_negotiated_raw_min),
                    negotiated_any_force_eager=phase_e_negotiated_any_force_eager,
                    graph_bs_override_hint=(
                        int(phase_e_negotiated_max) if phase_e_from_cache else None
                    ),
                    state_fingerprint=phase_e_state_fingerprint,
                )
                self._kunserve_scheduler_gap_log(
                    "scheduler_gap_phase_e_apply_decision_end",
                    stage_ns,
                    batch=batch,
                    loop="overlap",
                    max_bs=int(phase_e_negotiated_max),
                    min_bs=int(phase_e_negotiated_min),
                    raw_max_bs=int(phase_e_negotiated_raw_max),
                    raw_min_bs=int(phase_e_negotiated_raw_min),
                    any_force_eager=bool(phase_e_negotiated_any_force_eager),
                )
                phase_e_force_eager_set = True

            # If we do not need to overlap the current batch with the last batch,
            # we can process the last batch immediately.
            if disable_overlap_for_batch:
                pop_and_process()

            # Launch the current batch (try/finally so the per-step
            # force-eager flag is always cleared even on exception).
            try:
                if batch:
                    batch_result = self.run_batch(batch)
                    stage_ns = time.perf_counter_ns()
                    self.result_queue.append((batch.copy(), batch_result))
                    self._kunserve_scheduler_gap_log(
                        "scheduler_gap_overlap_result_queue_append_end",
                        stage_ns,
                        batch=batch,
                        loop="overlap",
                        result_queue_len_after=len(self.result_queue),
                    )
                else:
                    batch_result = None
            finally:
                if phase_e_force_eager_set:
                    stage_ns = time.perf_counter_ns()
                    mr = getattr(self.tp_worker, "model_runner", None)
                    if mr is not None:
                        mr.set_balloon_step_force_eager(False)
                        deepep_setter = getattr(
                            mr, "set_balloon_deepep_step_any_extend", None
                        )
                        if callable(deepep_setter):
                            deepep_setter(None)
                    phase_e_force_eager_set = False
                    self._kunserve_scheduler_gap_log(
                        "scheduler_gap_phase_e_clear_decision_end",
                        stage_ns,
                        batch=batch,
                        loop="overlap",
                    )

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                if phase_e_negotiated_max is not None:
                    # Phase E branch: only enter keepalive if peer is busy.
                    if phase_e_negotiated_max > 0 and phase_e_status is not None:
                        stage_ns = time.perf_counter_ns()
                        keepalive_batch = self._build_balloon_keepalive_batch(
                            int(phase_e_negotiated_max), phase_e_status
                        )
                        self._kunserve_scheduler_gap_log(
                            "scheduler_gap_build_keepalive_end",
                            stage_ns,
                            batch=keepalive_batch,
                            loop="overlap_tail",
                            target_bs=int(phase_e_negotiated_max),
                        )
                    else:
                        self._stop_balloon_keepalive("all replicas idle")
                        keepalive_batch = None
                else:
                    stage_ns = time.perf_counter_ns()
                    keepalive_batch = self._maybe_get_balloon_keepalive_batch()
                    self._kunserve_scheduler_gap_log(
                        "scheduler_gap_maybe_keepalive_end",
                        stage_ns,
                        batch=keepalive_batch,
                        loop="overlap_tail",
                    )

                if keepalive_batch is not None:
                    batch = keepalive_batch
                    self.cur_batch = batch
                    # Phase E force-eager decision for the keepalive
                    # case (now that batch has its final bs).  Then
                    # run_batch with try/finally cleanup.
                    if (
                        phase_e_negotiated_max is not None
                        and phase_e_negotiated_min is not None
                    ):
                        stage_ns = time.perf_counter_ns()
                        self._phase_e_apply_step_decision(
                            negotiated_max_bs=int(phase_e_negotiated_max),
                            negotiated_min_bs=int(phase_e_negotiated_min),
                            negotiated_raw_max_bs=int(phase_e_negotiated_raw_max),
                            negotiated_raw_min_bs=int(phase_e_negotiated_raw_min),
                            negotiated_any_force_eager=phase_e_negotiated_any_force_eager,
                            graph_bs_override_hint=(
                                int(phase_e_negotiated_max)
                                if phase_e_from_cache
                                else None
                            ),
                            state_fingerprint=phase_e_state_fingerprint,
                        )
                        self._kunserve_scheduler_gap_log(
                            "scheduler_gap_phase_e_apply_decision_end",
                            stage_ns,
                            batch=batch,
                            loop="overlap_keepalive",
                            max_bs=int(phase_e_negotiated_max),
                            min_bs=int(phase_e_negotiated_min),
                            raw_max_bs=int(phase_e_negotiated_raw_max),
                            raw_min_bs=int(phase_e_negotiated_raw_min),
                            any_force_eager=bool(phase_e_negotiated_any_force_eager),
                        )
                        phase_e_force_eager_set = True
                    try:
                        batch_result = self.run_batch(batch)
                        stage_ns = time.perf_counter_ns()
                        self.result_queue.append((batch.copy(), batch_result))
                        self._kunserve_scheduler_gap_log(
                            "scheduler_gap_overlap_result_queue_append_end",
                            stage_ns,
                            batch=batch,
                            loop="overlap_keepalive",
                            result_queue_len_after=len(self.result_queue),
                        )
                    finally:
                        if phase_e_force_eager_set:
                            stage_ns = time.perf_counter_ns()
                            mr2 = getattr(self.tp_worker, "model_runner", None)
                            if mr2 is not None:
                                mr2.set_balloon_step_force_eager(False)
                                deepep_setter = getattr(
                                    mr2, "set_balloon_deepep_step_any_extend", None
                                )
                                if callable(deepep_setter):
                                    deepep_setter(None)
                            self._kunserve_scheduler_gap_log(
                                "scheduler_gap_phase_e_clear_decision_end",
                                stage_ns,
                                batch=batch,
                                loop="overlap_keepalive",
                            )
                else:
                    # When the server is idle, do self-check and re-init some states
                    stage_ns = time.perf_counter_ns()
                    self.self_check_during_idle()
                    self._kunserve_scheduler_gap_log(
                        "scheduler_gap_self_check_idle_end",
                        stage_ns,
                        loop="overlap",
                    )

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            if self.is_generation:
                stage_ns = time.perf_counter_ns()
                self.launch_batch_sample_if_needed(batch_result)
                self._kunserve_scheduler_gap_log(
                    "scheduler_gap_launch_sample_end",
                    stage_ns,
                    batch=batch,
                    loop="overlap",
                    has_batch_result=batch_result is not None,
                    delayed_sample=(
                        batch_result is not None
                        and batch_result.delay_sample_func is not None
                    ),
                )

            # Update last_batch
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                stage_ns = time.perf_counter_ns()
                self.self_check_during_busy()
                self._kunserve_scheduler_gap_log(
                    "scheduler_gap_self_check_busy_end",
                    stage_ns,
                    batch=batch,
                    loop="overlap",
                )

    def is_disable_overlap_for_batch(self, batch: ScheduleBatch) -> bool:
        # For two consecutive prefill batches, we disable overlap to improve the TTFT of the first batch.
        # This might slightly hurt the throughput, so we use an environment variable to control it.
        disable_overlap_for_batch = (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
            and batch
            and batch.forward_mode.is_extend()
            and self.last_batch
            and self.last_batch.forward_mode.is_extend()
        )

        # We do not support overlap + spec + grammar yet,
        # so we need to turn off overlap for this batch.
        # TODO(lsyin): support overlap + spec + grammar
        need_grammar_sync = (
            batch
            and batch.is_spec_v2
            and batch.has_grammar
            and batch.forward_mode.is_decode()
            and len(self.result_queue) > 0
        )

        return disable_overlap_for_batch or need_grammar_sync

    def recv_limit_reached(self, num_recv_reqs: int) -> bool:
        if self.max_recv_per_poll < 0:
            return False
        return num_recv_reqs >= self.max_recv_per_poll

    def recv_requests(
        self,
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""

        if self.recv_skipper is not None:
            last_forward_mode = (
                self.last_batch.forward_mode if self.last_batch is not None else None
            )
            if not self.recv_skipper.handle(last_forward_mode):
                return []

        if self.pp_rank == 0:
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                recv_reqs = []

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                        recv_req = unwrap_shm_features(recv_req)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_rpc = self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_rpc)
            else:
                recv_reqs = None
        else:
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                dp_offset = self.attn_dp_rank * self.attn_tp_size
                recv_reqs = point_to_point_pyobj(
                    [],
                    self.pp_rank * self.tp_size + dp_offset,
                    self.world_group.cpu_group,
                    (self.pp_rank - 1) * self.tp_size + dp_offset,
                    self.pp_rank * self.tp_size + dp_offset,
                )
            else:
                recv_reqs = None

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        if self.server_args.enable_dp_attention:
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
            else:
                work_reqs = None
                control_reqs = None

            if self.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )

            if self.attn_cp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )

            if self.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        elif self.tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )

        # Process MM requests under EPD-disaggregation mode
        if (
            self.pp_rank == 0
            and self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            recv_reqs, abort_reqs = self.mm_receiver.process_waiting_requests(recv_reqs)
            for req, error_msg, error_code in abort_reqs:

                status_code = (
                    HTTPStatus.BAD_REQUEST
                    if error_code == 400
                    else HTTPStatus.INTERNAL_SERVER_ERROR
                )
                prepare_abort(req, error_msg, status_code=status_code)
                self.stream_output([req], req.return_logprob)

        if self.enable_trace:
            for req in recv_reqs:
                if isinstance(
                    req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
                ):
                    trace_set_proc_propagate_context(req.rid, req.trace_context)
                    trace_slice_start("", req.rid, anonymous=True)

        return recv_reqs

    def _split_work_and_control_reqs(self, recv_reqs: List):
        work_reqs = [
            req
            for req in recv_reqs
            if isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        control_reqs = [
            req
            for req in recv_reqs
            if not isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        return work_reqs, control_reqs

    def process_input_requests(self, recv_reqs: List):

        for recv_req in recv_reqs:
            # If it is a health check generation request and there are running requests, ignore it.
            if is_health_check_generate_req(recv_req) and (
                self.chunked_req is not None
                or self.dllm_manager.any_staging_reqs()
                or not self.running_batch.is_empty()
                or len(self.offload_tags) > 0
            ):
                self.return_health_check_ct += 1
                continue

            output = self._request_dispatcher(recv_req)
            if output is not None:
                if not isinstance(output, RpcReqOutput):
                    self.send_to_tokenizer.send_output(output, recv_req)
                else:
                    if self.recv_from_rpc is not None:
                        self.recv_from_rpc.send_pyobj(output)

    def init_req_max_new_tokens(self, req):
        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )

    def _process_and_broadcast_mm_inputs(
        self,
        raw_mm_inputs: Optional[dict],
    ):
        """Materialize MultimodalInputs once on the entry rank and broadcast to others.

        Entry rank:
        - constructs MultimodalInputs.from_dict(raw_mm_inputs) once
        - broadcasts to other ranks in self.cpu_group (if world_size > 1)

        Non-entry ranks:
        - receive the object via broadcast (if world_size > 1)
        - otherwise (single-rank / no group) fall back to local from_dict

        Returns:
            MultimodalInputs | None
        """
        if raw_mm_inputs is None:
            return None

        group_world_size = 1
        try:
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and self.dp_tp_cpu_group is not None
            ):
                group_world_size = torch.distributed.get_world_size(
                    group=self.dp_tp_cpu_group
                )
        except Exception as e:
            logger.warning(
                f"Failed to get world size in mm_inputs handling with {e}, fallback to 1."
            )

        # In case tp size > 1, all the Scheduler TP ranks runs the duplicated computing
        # process in CPU which occupies the main thread CPU cycle. This computing logic
        # merely needs to be run on TP0 and be broadcast to other TP ranks.
        # Since the Scheduler is single-threaded, any large CPU cost will impact
        # handling of other messages. For example, CPU hits 99.9% can significantly
        # increase the CUDA kernel launch time.
        if self.dp_tp_group.rank_in_group == 0:
            # Only the entry rank materializes once from dict.
            image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
            # Broadcast to other TP ranks (use src=0 within the group).
            if group_world_size > 1:
                obj_list = [image_inputs]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
        else:
            # Non-entry ranks: receive if group size > 1; otherwise materialize locally.
            if group_world_size > 1:
                obj_list = [None]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
            else:
                image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)

        return image_inputs

    def _get_multimodal_inputs(self, mm_inputs_dict: dict):
        if self.server_args.enable_broadcast_mm_inputs_process:
            return self._process_and_broadcast_mm_inputs(mm_inputs_dict)
        else:
            return MultimodalInputs.from_dict(mm_inputs_dict)

    def _maybe_clear_mm_inputs(self, batch: ScheduleBatch) -> None:
        for req in batch.reqs:
            if not req.finished() or not (mm_inputs := req.multimodal_inputs):
                continue
            # For session requests, keep mm_inputs for the next request
            if req.session_id:
                continue
            # For non-session requests, clear features and mm_inputs
            for item in mm_inputs.mm_items:
                item.feature = None
            req.multimodal_inputs = None

    def _set_request_ingress_time(self, req: Req, recv_req: Any) -> None:
        """Set a stable per-request ingress timestamp for scheduler-side e2e."""
        if req.time_stats.lb_entry_time > 0:
            return

        recv_perf = getattr(recv_req, "received_time_perf", None)
        if isinstance(recv_perf, (int, float)) and recv_perf > 0:
            req.time_stats.lb_entry_time = recv_perf
        else:
            req.time_stats.lb_entry_time = time.perf_counter()

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        # Create a new request
        if (
            recv_req.session_params is None
            or recv_req.session_params.id is None
            or recv_req.session_params.id not in self.sessions
        ):
            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                seq_length = len(recv_req.input_embeds)
                fake_input_ids = [1] * seq_length
                recv_req.input_ids = fake_input_ids

            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port

            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                stream=recv_req.stream,
                lora_id=recv_req.lora_id,
                input_embeds=recv_req.input_embeds,
                custom_logit_processor=recv_req.custom_logit_processor,
                require_reasoning=recv_req.require_reasoning,
                return_hidden_states=recv_req.return_hidden_states,
                return_routed_experts=recv_req.return_routed_experts,
                eos_token_ids=self.model_config.hf_eos_token_id,
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_port=recv_req.bootstrap_port,
                bootstrap_room=recv_req.bootstrap_room,
                disagg_mode=self.disaggregation_mode,
                data_parallel_rank=recv_req.data_parallel_rank,
                vocab_size=self.model_config.vocab_size,
                priority=recv_req.priority,
                metrics_collector=(
                    self.metrics_collector if self.enable_metrics else None
                ),
                routing_key=recv_req.routing_key,
                http_worker_ipc=recv_req.http_worker_ipc,
                dllm_config=self.dllm_config,
            )
            req.tokenizer = self.tokenizer
            self._set_request_ingress_time(req, recv_req)

            if self.disaggregation_mode != DisaggregationMode.NULL:
                # Invalid request for disaggregated mode
                if recv_req.bootstrap_room is None:
                    error_msg = (
                        f"Invalid request: Disaggregated request received without "
                        f"bootstrap room id. {req.rid=}"
                    )
                    logger.error(error_msg)
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.stream_output([req], req.return_logprob)
                    return

            if (
                recv_req.session_params is not None
                and recv_req.session_params.id is not None
            ):
                req.set_finish_with_abort(
                    f"Invalid request: session id {recv_req.session_params.id} does not exist"
                )
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return
        else:
            # Create a new request from a previous session
            session = self.sessions[recv_req.session_params.id]
            req = session.create_req(
                recv_req, self.tokenizer, self.model_config.vocab_size
            )
            self._set_request_ingress_time(req, recv_req)
            if isinstance(req.finished_reason, FINISH_ABORT):
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)

            # For session requests, adjust mm_inputs offsets by the prefix length.
            # Session.create_req prepends previous context to origin_input_ids,
            # so offsets from the new prompt need to be shifted.
            if len(recv_req.input_ids) < len(req.origin_input_ids):
                assert recv_req.session_params.id in self.sessions
                prefix_len = len(req.origin_input_ids) - len(recv_req.input_ids)
                for mm_item in image_inputs.mm_items:
                    if mm_item.offsets:
                        mm_item.offsets = [
                            (start + prefix_len, end + prefix_len)
                            for start, end in mm_item.offsets
                        ]

            # The following steps are already fast, execute locally on each rank.
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            req.origin_input_ids = self.pad_input_ids_func(
                req.origin_input_ids, image_inputs
            )
            req.extend_image_inputs(image_inputs)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        # initialize before returning
        self.init_req_max_new_tokens(req)

        # Validate prompt length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        if not recv_req.return_logprob and recv_req.logprob_start_len != -1:
            # When return_logprob is False, logprob_start_len should be ignored
            recv_req.logprob_start_len = -1

        if recv_req.logprob_start_len == -1:
            if recv_req.return_logprob and recv_req.token_ids_logprob is None:
                # If logprob is required but neither token_ids_logprob nor logprob_start_len is
                # set, return the logprobs for output tokens by default
                req.logprob_start_len = len(req.origin_input_ids) - 1
            elif req.is_prefill_only:
                # For prefill-only requests with logprob_start_len == -1, set logprob_start_len
                # beyond input sequence to skip input logprob computation entirely
                req.logprob_start_len = len(req.origin_input_ids)
            else:
                # If return_logprob is False, only the last token requires logprob computation
                req.logprob_start_len = -1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        if req.logprob_start_len > len(req.origin_input_ids):
            error_msg = f"{req.logprob_start_len=} is higher than the number of input tokens {len(req.origin_input_ids)=}. Please use a smaller logprob_start_len."
            req.logprob_start_len = -1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
        if not added_to_grammar_queue:
            self._add_request_to_queue(req)

    def handle_batch_generate_request(
        self,
        recv_req: BatchTokenizedGenerateReqInput,
    ):
        """Handle optimized batch generate request."""
        logger.debug(f"Processing batch generate request with {len(recv_req)} requests")

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_generate_request(tokenized_req)

    def _prefetch_kvcache(self, req: Req):
        if self.enable_hicache_storage:
            req.init_next_round_input(self.tree_cache)
            if req.last_node.backuped:
                # only to initiate the prefetch if the last node is backuped
                # otherwise, the allocated GPU memory must be locked for integrity
                last_hash = req.last_host_node.get_last_hash_value()
                matched_len = len(req.prefix_indices) + req.host_hit_length
                new_input_tokens = req.fill_ids[matched_len:]

                prefix_keys = (
                    req.last_node.get_prefix_hash_values(req.last_node.parent)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                self.tree_cache.prefetch_from_storage(
                    req.rid,
                    req.last_host_node,
                    new_input_tokens,
                    last_hash,
                    prefix_keys,
                )

    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
        if self.disaggregation_mode == DisaggregationMode.NULL:
            if not self._set_or_validate_priority(req):
                return
            if self._abort_on_queued_limit(req):
                return
            self._prefetch_kvcache(req)
            self.waiting_queue.append(req)
            req.time_stats.wait_queue_entry_time = time.perf_counter()
            trace_slice_end(RequestStage.REQUEST_PROCESS, req.rid, auto_next_anon=True)
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prefetch_kvcache(req)
            self.disagg_prefill_bootstrap_queue.add(
                req, self.model_config.num_key_value_heads
            )
            req.time_stats.prefill_bootstrap_queue_entry_time = time.perf_counter()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
            if not is_retracted:
                req.time_stats.decode_prealloc_queue_entry_time = time.perf_counter()
        else:
            raise ValueError(f"Invalid {self.disaggregation_mode=}")

    def _set_or_validate_priority(self, req: Req) -> bool:
        """Set the default priority value, or abort the request based on the priority scheduling mode."""
        if self.enable_priority_scheduling and req.priority is None:
            if self.schedule_low_priority_values_first:
                req.priority = sys.maxsize
            else:
                req.priority = -sys.maxsize - 1
        elif (
            not self.enable_priority_scheduling
            and req.priority is not None
            and self.abort_on_priority_when_disabled
        ):
            abort_req = AbortReq(
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": "Using priority is disabled for this server. Please send a new request without a priority.",
                },
                rid=req.rid,
            )
            self.send_to_tokenizer.send_output(abort_req, req)
            return False
        return True

    def _abort_on_queued_limit(self, recv_req: Req) -> bool:
        """Abort an incoming or existing request if the waiting queue is full. Returns True if the incoming request is aborted."""
        if (
            self.max_queued_requests is None
            or len(self.waiting_queue) + 1 <= self.max_queued_requests
        ):
            return False

        # Reject the incoming request by default.
        req_to_abort = recv_req
        message = "The request queue is full."
        if self.enable_priority_scheduling:
            # With priority scheduling, consider aboritng an existing request based on the priority.
            # direction = 1  => smaller number = higher priority; -1 => larger number = higher priority.
            # max(...) + (direction * priority, queue_time_start) picks the least-preferred request.
            # Tie: later queue_time_start (newer) is evicted first. Preempt only if strictly better.
            direction = 1 if self.schedule_low_priority_values_first else -1
            key_fn = lambda item: (
                direction * item[1].priority,
                item[1].time_stats.wait_queue_entry_time,
            )
            idx, candidate_req = max(enumerate(self.waiting_queue), key=key_fn)
            abort_existing_req = (
                direction * recv_req.priority < direction * candidate_req.priority
            )
            if abort_existing_req:
                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(candidate_req.rid)
                elif self.enable_hierarchical_cache:
                    self.tree_cache.terminate_prefetch(candidate_req.rid)
                self.waiting_queue.pop(idx)
                req_to_abort = candidate_req
                message = "The request is aborted by a higher priority request."

        self.send_to_tokenizer.send_output(
            AbortReq(
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": message,
                },
                rid=req_to_abort.rid,
            ),
            req_to_abort,
        )
        return req_to_abort.rid == recv_req.rid

    def _abort_on_waiting_timeout(self):
        if (timeout_s := envs.SGLANG_REQ_WAITING_TIMEOUT.get()) <= 0:
            return

        deleted_reqs = set()
        deadline = time.perf_counter() - timeout_s
        for req in self.waiting_queue:
            entry_time = req.time_stats.wait_queue_entry_time
            if 0 < entry_time < deadline:
                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(req.rid)
                self.send_to_tokenizer.send_output(
                    AbortReq(
                        finished_reason={
                            "type": "abort",
                            "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                            "message": "Request waiting timeout reached.",
                        },
                        rid=req.rid,
                    ),
                    req,
                )
                deleted_reqs.add(req)

        if deleted_reqs:
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in deleted_reqs
            ]

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            token_type_ids=recv_req.token_type_ids,
            priority=recv_req.priority,
            dimensions=recv_req.dimensions,
            lora_id=recv_req.lora_id,
            http_worker_ipc=recv_req.http_worker_ipc,
        )
        req.tokenizer = self.tokenizer
        self._set_request_ingress_time(req, recv_req)

        # Handle multimodal inputs
        if recv_req.image_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.image_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            # The `pad_input_ids_func` is model-specific and may be None for
            # embedding models or models not requiring special padding.
            # If None, `req.origin_input_ids` is expected to be correctly populated already.
            if self.pad_input_ids_func:
                req.origin_input_ids = self.pad_input_ids_func(
                    req.origin_input_ids, image_inputs
                )

            req.extend_image_inputs(image_inputs)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self._add_request_to_queue(req)
                return

        # Validate prompts length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        req.logprob_start_len = -1
        self._add_request_to_queue(req)

    def handle_batch_embedding_request(
        self,
        recv_req: BatchTokenizedEmbeddingReqInput,
    ):
        """Handle optimized batch embedding request."""
        logger.debug(
            f"Processing batch embedding request with {len(recv_req)} requests"
        )

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_embedding_request(tokenized_req)

    def stash_chunked_request(self, req: Req):
        self.tree_cache.cache_unfinished_req(req, chunked=True)

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        stage_ns = time.perf_counter_ns()
        self._abort_on_waiting_timeout()
        self._abort_on_running_timeout()
        if self.dllm_config is not None:
            self.dllm_manager.filter_finished_reqs()
        self._kunserve_scheduler_gap_log(
            "scheduler_gap_get_next_precheck_end",
            stage_ns,
            loop="get_next",
        )

        # Merge the prefill batch into the running batch
        stage_ns = time.perf_counter_ns()
        chunked_req_to_exclude = set()

        if self.dllm_config is not None and self.dllm_manager.any_staging_reqs():
            chunked_req_to_exclude.update(self.dllm_manager.staging_queue)
            for req in self.dllm_manager.staging_queue:
                self.stash_chunked_request(req)

        if self.chunked_req is not None:
            # Move the chunked request out of the batch so that we can merge
            # only finished requests to running_batch.
            chunked_req_to_exclude.add(self.chunked_req)
            self.stash_chunked_request(self.chunked_req)

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req is not None:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            if self.dllm_config is not None and self.last_batch.reqs:
                chunked_req_to_exclude.update(self.last_batch.reqs)

            # Filter batch
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            # Merge the new batch into the running batch.
            # For prefill-only batch, we can avoid going through decoding step.
            if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    # Merge running_batch with prefill batch
                    self.running_batch.merge_batch(self.last_batch)
        self._kunserve_scheduler_gap_log(
            "scheduler_gap_get_next_merge_last_end",
            stage_ns,
            loop="get_next",
            chunked_exclude=len(chunked_req_to_exclude),
        )

        stage_ns = time.perf_counter_ns()
        if self.dllm_config is not None:
            new_batch = self.get_new_batch_dllm()
        else:
            new_batch = self.get_new_batch_prefill()
        self._kunserve_scheduler_gap_log(
            "scheduler_gap_get_next_prefill_select_end",
            stage_ns,
            batch=new_batch,
            loop="get_next",
            got_new_batch=new_batch is not None,
        )

        need_mlp_sync = self.require_mlp_sync
        if need_mlp_sync and not self.spec_algorithm.is_none():
            # NOTE: This branch makes sure prefill and decode batches will not be mixed when spec and dp-attn is enabled.
            # Before merging the new batch into running batch:
            # 1. All new batches are none -> need_mlp_sync remains true (sync is needed for decode batch).
            # 2. All new batches are some (prefill / idle) -> we do not need prepare mlp sync one more time.
            new_batch = self.maybe_prepare_mlp_sync_batch(new_batch)
            need_mlp_sync = new_batch is None

        if new_batch is not None:
            # Run prefill first if possible
            ret = new_batch
        else:
            # Run decode
            stage_ns = time.perf_counter_ns()
            if not self.running_batch.is_empty():
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None
            else:
                ret = None
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_get_next_decode_select_end",
                stage_ns,
                batch=ret,
                loop="get_next",
            )

        # Handle DP attention and log stats
        stage_ns = time.perf_counter_ns()
        ret = self.maybe_prepare_mlp_sync_batch(ret, need_sync=need_mlp_sync)
        self._kunserve_scheduler_gap_log(
            "scheduler_gap_get_next_mlp_sync_end",
            stage_ns,
            batch=ret,
            loop="get_next",
            need_mlp_sync=bool(need_mlp_sync),
        )

        if ret:
            trace_event_batch("schedule", ret.reqs)

        return ret

    def get_num_allocatable_reqs(self, running_bs):
        res = get_global_server_args().pp_max_micro_batch_size - running_bs
        if self.pp_size > 1:
            res = min(res, self.req_to_token_pool.available_size())
        return res

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        prefill_delayer_single_pass = None
        if self.prefill_delayer:
            _, token_usage, _, _ = self._get_token_info()
            prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
                self.prefill_delayer, token_usage=token_usage
            )

        stage_ns = time.perf_counter_ns()
        ret = self._get_new_batch_prefill_raw(
            prefill_delayer_single_pass=prefill_delayer_single_pass
        )
        self._kunserve_scheduler_gap_log(
            "scheduler_gap_get_new_batch_prefill_raw_end",
            stage_ns,
            batch=ret,
            loop="get_new_batch_prefill",
            has_prefill_delayer=self.prefill_delayer is not None,
        )

        if self.prefill_delayer:
            stage_ns = time.perf_counter_ns()
            prefill_delayer_single_pass.finalize(actual_prefill=ret is not None)
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_prefill_delayer_finalize_end",
                stage_ns,
                batch=ret,
                loop="get_new_batch_prefill",
            )

        return ret

    def _get_new_batch_prefill_raw(
        self, prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor]
    ) -> Optional[ScheduleBatch]:
        # Check if the grammar is ready in the grammar queue
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if self.try_preemption:
            # Reset batch_is_full to try preemption with a prefill adder.
            self.running_batch.batch_is_full = False

        if self._phase_e_may_defer_prefill_for_cache():
            self._kunserve_phase_e_prefill_defer_log_ct += 1
            log_ct = self._kunserve_phase_e_prefill_defer_log_ct
            try:
                available_tokens = self.token_to_kv_pool_allocator.available_size()
            except Exception:
                available_tokens = -1
            if log_ct in (1, 10, 100) or log_ct % 1000 == 0:
                _kunserve_ms(
                    "[KUNSERVE-MS] defer prefill under phase-e cached bucket: "
                    "count=%d running=%d waiting=%d chunked=%s cached_bs=%d "
                    "steps_left=%d available_tokens=%d max_total=%d",
                    log_ct,
                    len(self.running_batch.reqs),
                    len(self.waiting_queue),
                    self.chunked_req is not None,
                    int(self._phase_e_cached_max_bs),
                    int(self._phase_e_cached_steps_left),
                    int(available_tokens),
                    int(self.max_total_num_tokens),
                )
            kunserve_timing_log(
                "scheduler_prefill_deferred_phase_e_cache",
                count=int(self._kunserve_phase_e_prefill_defer_log_ct),
                running=len(self.running_batch.reqs),
                waiting=len(self.waiting_queue),
                chunked=self.chunked_req is not None,
                cached_max_bs=int(self._phase_e_cached_max_bs),
                cached_steps_left=int(self._phase_e_cached_steps_left),
                available_tokens=int(available_tokens),
                max_total=int(self.max_total_num_tokens),
            )
            return None

        if self._kunserve_should_defer_prefill_for_balloon_memory():
            return None

        if (
            self.chunked_req is None
            and self._kunserve_should_defer_prefill_for_global_graph()
        ):
            self._kunserve_graph_prefill_defer_log_ct += 1
            log_ct = self._kunserve_graph_prefill_defer_log_ct
            try:
                available_tokens = self.token_to_kv_pool_allocator.available_size()
            except Exception:
                available_tokens = -1
            if log_ct in (1, 10, 100) or log_ct % 1000 == 0:
                retracted_waiting = sum(
                    1
                    for req in self.waiting_queue
                    if bool(getattr(req, "is_retracted", False))
                    or bool(getattr(req, "retracted_stain", False))
                )
                _kunserve_ms(
                    "[KUNSERVE-MS] defer prefill under global graph: "
                    "count=%d running=%d waiting=%d retracted_waiting=%d "
                    "batch_is_full=%s available_tokens=%d max_total=%d",
                    log_ct,
                    len(self.running_batch.reqs),
                    len(self.waiting_queue),
                    int(retracted_waiting),
                    bool(getattr(self.running_batch, "batch_is_full", False)),
                    int(available_tokens),
                    int(self.max_total_num_tokens),
                )
            kunserve_timing_log(
                "scheduler_prefill_deferred_global_graph",
                count=int(self._kunserve_graph_prefill_defer_log_ct),
                running=len(self.running_batch.reqs),
                waiting=len(self.waiting_queue),
                available_tokens=int(available_tokens),
                max_total=int(self.max_total_num_tokens),
            )
            return None

        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            if self.running_batch.batch_is_full and len(self.waiting_queue) > 0:
                self._kunserve_prefill_blocked_full_log_ct += 1
                log_ct = self._kunserve_prefill_blocked_full_log_ct
                if log_ct in (1, 10, 100) or log_ct % 1000 == 0:
                    try:
                        available_tokens = (
                            self.token_to_kv_pool_allocator.available_size()
                        )
                    except Exception:
                        available_tokens = -1
                    mr = getattr(self.tp_worker, "model_runner", None)
                    _kunserve_ms(
                        "[KUNSERVE-MS] prefill blocked by batch_is_full: "
                        "count=%d state=%s variant=%s running=%d waiting=%d "
                        "available_tokens=%d max_total=%d expand_requested=%s",
                        log_ct,
                        getattr(mr, "_balloon_state", "local"),
                        getattr(mr, "_balloon_runtime_variant", "local"),
                        len(self.running_batch.reqs),
                        len(self.waiting_queue),
                        int(available_tokens),
                        int(self.max_total_num_tokens),
                        bool(self.expand_requested),
                    )
            if self.running_batch.batch_is_full and len(self.waiting_queue) > 0:
                kunserve_timing_log(
                    "scheduler_prefill_blocked_batch_full",
                    count=int(self._kunserve_prefill_blocked_full_log_ct),
                    running=len(self.running_batch.reqs),
                    waiting=len(self.waiting_queue),
                    max_total=int(self.max_total_num_tokens),
                )
            return None

        running_bs = len(self.running_batch.reqs)
        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked requests has just been released.
        # In PP case, chunked requests (or dllm requests) can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked requests to be added, otherwise, there will be a memory leak.
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.chunked_req is not None
            and not self.try_preemption
        ):
            self.running_batch.batch_is_full = True
            return None

        if self.enable_hierarchical_cache:
            self.tree_cache.check_hicache_events()

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue, self.running_batch)

        if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
            # If we are testing retraction and the running batch size exceeds
            # TEST_RETRACT_NO_PREFILL_BS, we skip the prefill to keep the requests
            # in the waiting queue.
            return None

        # Determine chunked_prefill_size for this batch
        chunked_prefill_size = self.chunked_prefill_size
        if self.chunked_req is not None and self.enable_dynamic_chunking:
            history_len = len(self.chunked_req.prefix_indices)
            dynamic_size = self.predict_next_chunk_size(history_len)
            if dynamic_size is not None:
                chunked_prefill_size = dynamic_size
        max_prefill_tokens, chunked_prefill_size = (
            self._kunserve_effective_prefill_budget_for_balloon(
                self.max_prefill_tokens, chunked_prefill_size
            )
        )

        # Prefill policy
        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            max_prefill_tokens,
            chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=self.server_args.prefill_max_requests,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            dllm_config=self.dllm_config,
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.enable_lora:
            running_loras = {req.lora_id for req in self.running_batch.reqs}

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            if self.enable_lora and req.lora_id not in running_loras:
                if self.enable_lora_overlap_loading:
                    # For overlapping loading of LoRA weights with computation, we will load each adapter one at a time,
                    # as opposed to loading them in one batch
                    res = self.lora_overlap_loader.try_overlap_load_lora(
                        req.lora_id, running_loras
                    )
                    if not res:
                        continue
                else:
                    new_lora_set = {req.lora_id} | running_loras
                    if not self.tp_worker.model_runner.lora_manager.validate_lora_batch(
                        new_lora_set
                    ):
                        continue

            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    self.running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if not self.try_preemption or not adder.preempt_to_schedule(
                    req, self.server_args
                ):
                    break

            if self.enable_hicache_storage:
                prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    continue
                # Pop the number of tokens loaded from storage (L3 hits)
                req.storage_hit_length = self.tree_cache.pop_prefetch_loaded_tokens(
                    req.rid
                )

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=(self.chunked_req is not None),
                truncation_align_size=self.truncation_align_size,
            )

            if self.enable_lora:
                running_loras.add(req.lora_id)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (not self.running_batch.is_empty())
                    else:
                        self.running_batch.batch_is_full = True
                break

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None
        # [RESUME-PREFILL] mark when this prefill batch is re-prefilling RETRACTED
        # requests (the suspected garbling trigger). Pairs with [RETRACT] by rid.
        try:
            import os as _os, datetime as _dt
            _dbg = _os.environ.get("KUNSERVE_DETAIL_LOG")
            if _dbg:
                _res = [
                    (getattr(r, "rid", "?")[:8], len(r.output_ids), int(getattr(r, "retraction_count", 0)))
                    for r in can_run_list
                    if bool(getattr(r, "retracted_stain", False)) or int(getattr(r, "retraction_count", 0)) > 0
                ]
                if _res:
                    with open(_dbg, "a", encoding="utf-8") as _f:
                        _f.write(
                            f"[{_dt.datetime.now()} pid={_os.getpid()}] [KUNSERVE-DBG] "
                            f"[RESUME-PREFILL] n_resumed={len(_res)} n_total={len(can_run_list)} "
                            f"reqs(rid,outlen,retractcnt)={_res[:8]}\n"
                        )
        except Exception:
            pass
        kunserve_timing_log(
            "scheduler_prefill_scheduled",
            new_reqs=len(can_run_list),
            running=len(self.running_batch.reqs),
            waiting_before=len(self.waiting_queue),
            chunked_req=self.chunked_req is not None,
            log_input_tokens=int(adder.log_input_tokens),
            max_prefill_tokens=int(max_prefill_tokens),
            chunked_prefill_size=(
                int(chunked_prefill_size) if chunked_prefill_size is not None else None
            ),
        )
        self._kunserve_prefill_blocked_full_log_ct = 0
        self._kunserve_graph_prefill_defer_log_ct = 0

        if self.enable_metrics:
            # only record queue time when enable_metrics is True to avoid overhead
            for req in can_run_list:
                req.add_latency(RequestStage.PREFILL_WAITING)

        # Snapshot before dequeue: only requests actually dequeued from waiting_queue
        # should contribute to queue_wait_ms_cumulative.
        waiting_req_ids_before = {id(x) for x in self.waiting_queue}
        can_run_set = set(can_run_list)
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]
        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if adder.new_chunked_req is not None:
            # Update chunked prefill
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req is not None:
            self.chunked_req.is_chunked += 1

        # Record for logging prefill stats after forward
        self.adder = adder
        self.can_run_list = can_run_list
        self.running_bs = len(self.running_batch.reqs)

        # Record metrics
        now_for_queue = time.perf_counter()
        for req in can_run_list:
            was_dequeued_from_waiting = id(req) in waiting_req_ids_before
            if self.req_lifecycle_log and was_dequeued_from_waiting:
                wait_enter = req.time_stats.wait_queue_entry_time
                if isinstance(wait_enter, (int, float)) and wait_enter > 0:
                    waited_ms = max(0.0, (now_for_queue - wait_enter) * 1000.0)
                    self._queue_wait_ms_cumulative[req.rid] = (
                        self._queue_wait_ms_cumulative.get(req.rid, 0.0) + waited_ms
                    )
                    self._queue_wait_ms_last_dequeue[req.rid] = waited_ms

            if req.time_stats.forward_entry_time == 0:
                req.time_stats.forward_entry_time = now_for_queue
                if self.enable_metrics:
                    self.metrics_collector.observe_queue_time(
                        req.time_stats.get_queueing_time(),
                    )
            req._prefill_count = int(getattr(req, "_prefill_count", 0)) + 1

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            chunked_req=self.chunked_req,
        )
        if self.enable_hierarchical_cache:
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        new_batch.prepare_for_extend()

        # Record prefill stats for logging after forward
        new_batch.prefill_stats = PrefillStats(
            log_input_tokens=adder.log_input_tokens,
            log_hit_tokens=adder.log_hit_tokens,
            new_token_ratio=adder.new_token_ratio,
            running_bs=len(self.running_batch.reqs),
            num_new_seqs=len(can_run_list),
        )

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
        ):
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            self.running_batch.filter_batch(v1_spec_info_filtered=True)
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs
            self.running_batch = ScheduleBatch(
                reqs=[], batch_is_full=self.running_batch.batch_is_full
            )
        else:
            new_batch.decoding_reqs = None

        return new_batch

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch."""
        initial_bs = batch.batch_size()

        batch.filter_batch(v1_spec_info_filtered=True)
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # Check if decode out of memory
        if (kv_full_retract_flag := not batch.check_decode_mem()) or (
            TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            old_available_tokens = self.token_to_kv_pool_allocator.available_size()
            old_ratio = self.new_token_ratio
            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
                self.server_args
            )
            new_available_tokens = self.token_to_kv_pool_allocator.available_size()
            new_token_gained = new_available_tokens - old_available_tokens
            kunserve_timing_log(
                "scheduler_retract_decode",
                retracted=len(retracted_reqs),
                running_after=batch.batch_size(),
                waiting=len(self.waiting_queue),
                old_available=int(old_available_tokens),
                new_available=int(new_available_tokens),
                gained=int(new_token_gained),
                max_total=int(self.max_total_num_tokens),
                kv_full=bool(kv_full_retract_flag),
            )

            # [RETRACT] KunServe detail-log marker (same file as [MOE-IO]/[MASKED-*])
            # so we can correlate a retract event with the subsequent re-prefill and
            # any garbling onset (suspected: re-prefill in balloon GLOBAL corrupts).
            try:
                import os as _os, datetime as _dt
                _dbg = _os.environ.get("KUNSERVE_DETAIL_LOG")
                if _dbg and len(retracted_reqs) > 0:
                    _rids = [(getattr(r, "rid", "?")[:8], len(r.output_ids)) for r in retracted_reqs[:8]]
                    with open(_dbg, "a", encoding="utf-8") as _f:
                        _f.write(
                            f"[{_dt.datetime.now()} pid={_os.getpid()}] [KUNSERVE-DBG] "
                            f"[RETRACT] n={len(retracted_reqs)} running_after={batch.batch_size()} "
                            f"waiting={len(self.waiting_queue)} kv_full={bool(kv_full_retract_flag)} "
                            f"gained={int(new_token_gained)} reqs(rid,outlen)={_rids}\n"
                        )
            except Exception:
                pass

            self.num_retracted_reqs = len(retracted_reqs)
            if self.enable_metrics and len(retracted_reqs) > 0:
                self.metrics_collector.increment_retracted_reqs(
                    num_retracted_reqs=len(retracted_reqs),
                    num_retracted_input_tokens=sum(
                        len(r.origin_input_ids) for r in retracted_reqs
                    ),
                    num_retracted_output_tokens=sum(
                        len(r.output_ids) for r in retracted_reqs
                    ),
                )
            self.new_token_ratio = new_token_ratio
            for req in reqs_to_abort:
                abort_reason: FINISH_ABORT = req.to_finish
                self.send_to_tokenizer.send_output(
                    AbortReq(abort_message=abort_reason.message, rid=req.rid), req
                )

            msg_prefix = (
                "KV cache pool is full. Retract requests. "
                if kv_full_retract_flag
                else "Testing retraction. "
            )
            if kv_full_retract_flag:
                # Once we have already entered (or are entering) the BALLOON
                # runtime, there is no second expansion available — additional
                # retracts mean the workload genuinely exceeds the enlarged
                # KV pool. Suppress further expand_requested fires so the
                # controller stops re-polling and the metric counts reflect
                # the actual one-shot transition.
                mr = getattr(self.tp_worker, "model_runner", None)
                balloon_state = (
                    str(getattr(mr, "_balloon_state", "local"))
                    if mr is not None
                    else "local"
                )
                if balloon_state == "local":
                    prev_expand_requested = self.expand_requested
                    self.expand_requested = True
                    self.expand_request_reason = "retract_decode"
                    if not prev_expand_requested:
                        _kunserve_ms(
                            "[KUNSERVE-MS] expand_requested fired by retract_decode: "
                            "available_tokens=%d gained_tokens=%d running=%d waiting=%d max_total=%d",
                            new_available_tokens,
                            new_token_gained,
                            len(batch.reqs),
                            len(self.waiting_queue),
                            int(self.max_total_num_tokens),
                        )
                    else:
                        logger.info(
                            "[KunServeScheduler] expand requested by retract_decode: "
                            "prev_expand=%s available_tokens=%d gained_tokens=%d "
                            "running=%d waiting=%d max_total_num_tokens=%d",
                            prev_expand_requested,
                            new_available_tokens,
                            new_token_gained,
                            len(batch.reqs),
                            len(self.waiting_queue),
                            int(self.max_total_num_tokens),
                        )
                else:
                    logger.info(
                        "[KunServeScheduler] retract_decode under BALLOON state=%s; "
                        "not re-firing expand_requested. available_tokens=%d gained=%d "
                        "running=%d waiting=%d max_total=%d",
                        balloon_state,
                        new_available_tokens,
                        new_token_gained,
                        len(batch.reqs),
                        len(self.waiting_queue),
                        int(self.max_total_num_tokens),
                    )
            msg_details = f"#retracted_reqs: {len(retracted_reqs)}, #new_tokens_gained: {new_token_gained}"
            if kv_full_retract_flag:
                msg_details += (
                    f", #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
                )
            logger.warning(msg_prefix + msg_details)

            for req in retracted_reqs:
                self._accumulate_retract_wasted_ms(req)
                self._add_request_to_queue(req, is_retracted=True)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        # Update batch tensors
        batch.prepare_for_decode()
        return batch

    def record_batch_in_overlap(self, model_worker_batch: ModelWorkerBatch):
        # FIXME(lsyin): hacky way to keep a reference to avoid GPU tensors being freed by torch GC
        # NOTE: More Reliable: record all tensors into the forward stream
        # NOTE: - for all future tensors, we shall always read from future map
        #       - for all non-future tensors (produced only by schedule stream),
        #       we shall keep its reference not being release during all the forwarding pass
        self.batch_record_ct = (self.batch_record_ct + 1) % 2
        self.batch_record_buf[self.batch_record_ct] = model_worker_batch

    def run_batch(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        self.forward_ct += 1
        now = time.perf_counter()
        gap_ms = 0.0
        if self._last_run_batch_end_ts is not None:
            gap_ms = (now - self._last_run_batch_end_ts) * 1000
        step_start = now
        run_timing_fields = {
            "forward_ct": int(self.forward_ct),
            "mode": str(batch.forward_mode),
            "batch_size": int(batch.batch_size()),
            "running": len(self.running_batch.reqs),
            "waiting": len(self.waiting_queue),
            "gap_ms": round(gap_ms, 3),
        }
        kunserve_timing_log("scheduler_run_batch_begin", **run_timing_fields)
        local_forward_probe = _kunserve_local_forward_probe_enabled()
        if local_forward_probe:
            _kun_wd(
                "[KUNSERVE-LOCAL] scheduler_run_batch_begin fwd_ct=%d mode=%s "
                "bs=%d running=%d waiting=%d gap_ms=%.3f"
                % (
                    int(self.forward_ct),
                    batch.forward_mode,
                    int(batch.batch_size()),
                    len(self.running_batch.reqs),
                    len(self.waiting_queue),
                    float(gap_ms),
                )
            )
        # Whether to run the profiler
        self._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        # Capture prefill start time for EXTEND mode
        if batch.forward_mode == ForwardMode.EXTEND:
            current_time = time.perf_counter()
            for req in batch.reqs:
                req.time_stats.prefill_start_time_host = current_time

        # Place holder handling for pd-disagg decode event loop
        if batch.forward_mode.is_prebuilt():
            return self._run_batch_prebuilt(batch)

        # Run forward
        if self.is_generation:
            if self.spec_algorithm.is_none() or self.enable_overlap:
                # In most cases, we use the model worker batch to run the forward.
                worker_batch_or_batch = batch.get_model_worker_batch()
            else:
                # In speculative decoding v1 (non-overlap) case, we use the batch directly.
                # TODO(lsyin): delete this branch after unifying the abstraction.
                worker_batch_or_batch = batch

            if self.enable_overlap:
                model_worker_batch = worker_batch_or_batch
                self.record_batch_in_overlap(model_worker_batch)

                # Sampling info will be modified during forward, so we store a copy.
                model_worker_batch.sampling_info = (
                    model_worker_batch.sampling_info.copy_for_forward()
                )

                bs = len(model_worker_batch.seq_lens)
                future_indices = self.future_map.alloc_future_indices(bs)

                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.default_stream)
                    self.future_map.resolve_future(model_worker_batch)
                    with self.record_forward_metrics(batch):
                        batch_result = self.model_worker.forward_batch_generation(
                            model_worker_batch
                            # here pp is not compatible with overlap
                        )
                    # FIXME(lsyin): maybe move this to forward_batch_generation
                    batch_result.copy_done = self.device_module.Event()
                    if batch_result.delay_sample_func is None:
                        self.future_map.store_to_map(future_indices, batch_result)
                        batch_result.copy_to_cpu(return_logprob=batch.return_logprob)
                    else:
                        batch_result.future_indices = future_indices

                # FIXME(lsyin): move this assignment elsewhere
                future_indices_or_next_token_ids = -future_indices.indices

                if batch.is_spec_v2:
                    # FIXME(lsyin): tmp code for spec v2
                    # We only keep future indices for next draft input

                    batch.spec_info = batch_result.next_draft_input
                    batch.spec_info.future_indices = future_indices

                    # batch.spec_info = EagleDraftInput(
                    #     future_indices=future_indices,
                    #     verify_done=batch_result.next_draft_input.verify_done,
                    # )

                    # The future value, usually for next batch preparation
                    # Current implementation strictly synchronizes the seq_lens
                    batch.seq_lens = batch_result.next_draft_input.new_seq_lens
            elif self.enable_pdmux and batch.forward_mode.is_split_prefill():
                batch_result = self.tp_worker.forward_batch_split_prefill(batch)
                future_indices_or_next_token_ids = batch_result.next_token_ids
            else:
                kwargs = (
                    {"pp_proxy_tensors": pp_proxy_tensors}
                    if self.spec_algorithm.is_none()
                    else {}
                )
                if local_forward_probe:
                    _kun_wd(
                        "[KUNSERVE-LOCAL] scheduler_forward_enter fwd_ct=%d mode=%s bs=%d"
                        % (int(self.forward_ct), batch.forward_mode, int(batch.batch_size()))
                    )
                with self.record_forward_metrics(batch):
                    batch_result = self.model_worker.forward_batch_generation(
                        worker_batch_or_batch, **kwargs
                    )
                if local_forward_probe:
                    _kun_wd(
                        "[KUNSERVE-LOCAL] scheduler_forward_exit fwd_ct=%d mode=%s bs=%d"
                        % (int(self.forward_ct), batch.forward_mode, int(batch.batch_size()))
                    )
                future_indices_or_next_token_ids = batch_result.next_token_ids
                if local_forward_probe:
                    _kun_wd(
                        "[KUNSERVE-LOCAL] scheduler_update_cache_enter fwd_ct=%d mode=%s bs=%d"
                        % (int(self.forward_ct), batch.forward_mode, int(batch.batch_size()))
                    )
                self.update_cache_from_scheduler(batch, batch_result)
                if local_forward_probe:
                    _kun_wd(
                        "[KUNSERVE-LOCAL] scheduler_update_cache_exit fwd_ct=%d mode=%s bs=%d"
                        % (int(self.forward_ct), batch.forward_mode, int(batch.batch_size()))
                    )

            # NOTE: future_indices_or_next_token_ids is used in ScheduleBatch,
            #       which can probably be replaced by future_indices later [TODO(lsyin)].
            #       we shall still keep the original outputs, e.g. next_token_ids
            #       in the GenerationBatchOutput for processing after copy_done.
            batch.output_ids = future_indices_or_next_token_ids

            # These 2 values are needed for processing the output, but the values can be
            # modified by overlap schedule. So we have to copy them here so that
            # we can use the correct values in output processing.
            if batch.return_logprob:
                batch_result.extend_input_len_per_req = [
                    req.extend_input_len for req in batch.reqs
                ]
                batch_result.extend_logprob_start_len_per_req = [
                    req.extend_logprob_start_len for req in batch.reqs
                ]
            else:
                batch_result.extend_input_len_per_req = None
                batch_result.extend_logprob_start_len_per_req = None

            ret = batch_result
        else:  # embedding or reward model
            model_worker_batch = batch.get_model_worker_batch()

            if self.enable_overlap:
                self.record_batch_in_overlap(model_worker_batch)
                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.default_stream)
                    embeddings = self.tp_worker.forward_batch_embedding(
                        model_worker_batch
                    )
                    ret = EmbeddingBatchResult(embeddings=embeddings)
                    ret.copy_to_cpu()
            else:
                embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)
                ret = EmbeddingBatchResult(embeddings=embeddings)

        # Capture prefill end time for EXTEND mode
        if batch.forward_mode == ForwardMode.EXTEND:
            current_time = time.perf_counter()
            for req in batch.reqs:
                req.time_stats.prefill_end_time_host = current_time

        if (
            self.server_args.enable_dp_attention
            and self.server_args.elastic_ep_backend == "mooncake"
        ):
            # Get the tensors indicating rank activeness
            tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
            tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
            tp_active_ranks &= tp_active_ranks_cpu
            dp_active_ranks = tp_active_ranks.reshape(self.dp_size, -1).prod(axis=1)
            self.send_to_tokenizer.send_output(
                ActiveRanksOutput(status=dp_active_ranks.tolist())
            )

        self._log_decode_step_timing(batch, ret, step_start, gap_ms)
        self._last_run_batch_end_ts = time.perf_counter()
        kunserve_timing_log(
            "scheduler_run_batch_end",
            elapsed_ms=round((self._last_run_batch_end_ts - step_start) * 1000.0, 3),
            **run_timing_fields,
        )
        return ret

    def _accumulate_retract_wasted_ms(self, req: Req) -> None:
        """Accumulate wasted decode time after prefill due to retract."""
        prefill_end = getattr(req.time_stats, "prefill_end_time_host", 0.0)
        if not isinstance(prefill_end, (int, float)) or prefill_end <= 0:
            return
        wasted_ms = (time.perf_counter() - prefill_end) * 1000.0
        if wasted_ms <= 0:
            return
        req._retract_wasted_ms = (
            float(getattr(req, "_retract_wasted_ms", 0.0)) + wasted_ms
        )

    def _maybe_dump_finished_reqs(self, batch: ScheduleBatch) -> None:
        """Dump one JSONL record per finished request in the current batch."""
        if not self.req_lifecycle_log:
            return
        if self.pp_rank != 0 or self.attn_tp_rank != 0 or self.attn_cp_rank != 0:
            return
        if batch is None or not hasattr(batch, "reqs") or not batch.reqs:
            return

        now_host = time.perf_counter()
        wall_now = time.time()
        records = []

        def norm_ts(v):
            return v if isinstance(v, (int, float)) and v > 0 else None

        def sub_ms(a, b):
            if a is None or b is None:
                return None
            return round((a - b) * 1000.0, 3)

        for req in batch.reqs:
            if not req.finished():
                continue
            if req.rid in self._dumped_rids:
                continue
            self._dumped_rids.add(req.rid)

            ts = req.time_stats
            lb_enter = norm_ts(getattr(ts, "lb_entry_time", None))
            wait_enter = norm_ts(getattr(ts, "wait_queue_entry_time", None))
            forward_enter = norm_ts(getattr(ts, "forward_entry_time", None))
            prefill_start = norm_ts(getattr(ts, "prefill_start_time_host", None))
            prefill_end = norm_ts(getattr(ts, "prefill_end_time_host", None))
            completion = norm_ts(getattr(ts, "completion_time", None))
            finish_host = completion if completion is not None else now_host

            try:
                finish_json = (
                    req.finished_reason.to_json() if req.finished_reason else {}
                )
            except Exception:
                finish_json = {}

            queue_wait_ms_cumulative = self._queue_wait_ms_cumulative.get(req.rid)
            prefill_ms = sub_ms(prefill_end, prefill_start)
            decode_ms = sub_ms(finish_host, prefill_end)
            e2e_ms = sub_ms(
                finish_host, lb_enter if lb_enter is not None else wait_enter
            )
            prefill_count = max(1, int(getattr(req, "_prefill_count", 1)))
            retract_wasted_ms = round(float(getattr(req, "_retract_wasted_ms", 0.0)), 3)
            extra_prefill_ms = (
                round(prefill_ms * max(0, prefill_count - 1), 3)
                if isinstance(prefill_ms, (int, float))
                else None
            )

            ideal_e2e_ms_estimate = None
            if isinstance(e2e_ms, (int, float)):
                ideal = e2e_ms
                if isinstance(queue_wait_ms_cumulative, (int, float)):
                    ideal -= queue_wait_ms_cumulative
                ideal -= retract_wasted_ms
                if isinstance(extra_prefill_ms, (int, float)):
                    ideal -= extra_prefill_ms
                ideal_e2e_ms_estimate = round(max(0.0, ideal), 3)

            e2e_saved_ms_estimate = (
                round(e2e_ms - ideal_e2e_ms_estimate, 3)
                if isinstance(e2e_ms, (int, float))
                and isinstance(ideal_e2e_ms_estimate, (int, float))
                else None
            )

            queue_wait_ms_last = self._queue_wait_ms_last_dequeue.get(req.rid)
            record = {
                "ts": wall_now,
                "replica_rank": _REPLICA_RANK,
                "rid": req.rid,
                "prompt_len": (
                    len(req.origin_input_ids_unpadded)
                    if hasattr(req, "origin_input_ids_unpadded")
                    else None
                ),
                "prompt_len_padded": (
                    len(req.origin_input_ids)
                    if hasattr(req, "origin_input_ids")
                    else None
                ),
                "output_len": (
                    len(req.output_ids) if hasattr(req, "output_ids") else None
                ),
                "finish_type": finish_json.get("type", "unknown"),
                "finish_reason": finish_json,
                "t_lb_entry": lb_enter,
                "t_wait_enter": wait_enter,
                "t_forward_enter": forward_enter,
                "t_prefill_start": prefill_start,
                "t_prefill_end": prefill_end,
                "t_finish": finish_host,
                "queue_wait_ms": (
                    round(queue_wait_ms_last, 3)
                    if isinstance(queue_wait_ms_last, (int, float))
                    else sub_ms(forward_enter, wait_enter)
                ),
                "queue_wait_ms_cumulative": (
                    round(queue_wait_ms_cumulative, 3)
                    if queue_wait_ms_cumulative is not None
                    else None
                ),
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "e2e_ms": e2e_ms,
                "prefill_count": prefill_count,
                "retract_wasted_ms": retract_wasted_ms,
                "extra_prefill_ms": extra_prefill_ms,
                "ideal_e2e_ms_estimate": ideal_e2e_ms_estimate,
                "e2e_saved_ms_estimate": e2e_saved_ms_estimate,
                # KunServe per-request decode-step accounting. Counters are
                # incremented in _log_decode_step_timing each time the request
                # appears in a decode batch.
                "decode_steps_local": int(
                    getattr(req, "_kunserve_decode_steps_local", 0)
                ),
                "decode_steps_balloon": int(
                    getattr(req, "_kunserve_decode_steps_balloon", 0)
                ),
            }
            records.append(record)
            self._queue_wait_ms_cumulative.pop(req.rid, None)
            self._queue_wait_ms_last_dequeue.pop(req.rid, None)

        if not records:
            return

        try:
            os.makedirs(os.path.dirname(self.req_lifecycle_log) or ".", exist_ok=True)
            with open(self.req_lifecycle_log, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("Failed to write req lifecycle log: %s", e)

    def launch_batch_sample_if_needed(
        self, batch_result: GenerationBatchResult
    ) -> Union[GenerationBatchResult]:
        # TODO(lsyin): make the delayed sample a default behavior after
        # unifying the forward_batch_generation interface (related to spec V2).
        if batch_result is None or batch_result.delay_sample_func is None:
            return

        stage_ns = time.perf_counter_ns()
        with self.forward_stream_ctx:
            self.forward_stream.wait_stream(self.default_stream)
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_launch_sample_wait_stream_end",
                stage_ns,
                batch=self.cur_batch,
                loop="launch_sample",
            )
            stage_ns = time.perf_counter_ns()
            _batch_result = batch_result.delay_sample_func()
            assert _batch_result is batch_result
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_launch_sample_func_end",
                stage_ns,
                batch=self.cur_batch,
                loop="launch_sample",
            )
            stage_ns = time.perf_counter_ns()
            self.future_map.store_to_map(batch_result.future_indices, batch_result)
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_launch_sample_store_future_end",
                stage_ns,
                batch=self.cur_batch,
                loop="launch_sample",
            )
            stage_ns = time.perf_counter_ns()
            batch_result.copy_to_cpu(return_logprob=self.cur_batch.return_logprob)
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_launch_sample_copy_to_cpu_end",
                stage_ns,
                batch=self.cur_batch,
                loop="launch_sample",
            )

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        stage_ns = time.perf_counter_ns()
        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result)
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_process_result_core_end",
                stage_ns,
                batch=batch,
                loop="process_batch_result",
                core="decode",
            )
            stage_ns = time.perf_counter_ns()
            trace_slice_batch(RequestStage.DECODE_LOOP, batch.reqs)
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_process_result_trace_end",
                stage_ns,
                batch=batch,
                loop="process_batch_result",
            )
        elif batch.forward_mode.is_extend():
            if batch.is_dllm():
                self.process_batch_result_dllm(batch, result)
                core = "dllm"
            else:
                self.process_batch_result_prefill(batch, result)
                core = "prefill"
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_process_result_core_end",
                stage_ns,
                batch=batch,
                loop="process_batch_result",
                core=core,
            )
        elif batch.forward_mode.is_prebuilt():
            self.process_batch_result_prebuilt(batch)
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_process_result_core_end",
                stage_ns,
                batch=batch,
                loop="process_batch_result",
                core="prebuilt",
            )
        elif batch.forward_mode.is_idle():
            self.process_batch_result_idle(batch, result)
            self._kunserve_scheduler_gap_log(
                "scheduler_gap_process_result_core_end",
                stage_ns,
                batch=batch,
                loop="process_batch_result",
                core="idle",
            )

        stage_ns = time.perf_counter_ns()
        self.log_batch_result_stats(batch, result)
        self._kunserve_scheduler_gap_log(
            "scheduler_gap_process_result_stats_end",
            stage_ns,
            batch=batch,
            loop="process_batch_result",
        )
        stage_ns = time.perf_counter_ns()
        self._maybe_clear_mm_inputs(batch)
        self._kunserve_scheduler_gap_log(
            "scheduler_gap_process_result_clear_mm_end",
            stage_ns,
            batch=batch,
            loop="process_batch_result",
        )
        stage_ns = time.perf_counter_ns()
        self._maybe_dump_finished_reqs(batch)
        self._kunserve_scheduler_gap_log(
            "scheduler_gap_process_result_dump_finished_end",
            stage_ns,
            batch=batch,
            loop="process_batch_result",
        )
        stage_ns = time.perf_counter_ns()
        self.maybe_send_health_check_signal()
        self._kunserve_scheduler_gap_log(
            "scheduler_gap_process_result_health_end",
            stage_ns,
            batch=batch,
            loop="process_batch_result",
        )

    def maybe_send_health_check_signal(self):
        if self.return_health_check_ct:
            # Return some signal for the health check.
            # This is used to prevent the health check signal being blocked by long context prefill.
            # However, one minor issue is that this code path does not check the status of detokenizer manager.
            self.return_health_check_ct -= 1
            self.send_to_tokenizer.send_output(HealthCheckOutput())

    def flush_cache_wrapped(self, recv_req: FlushCacheReqInput):
        success = self.flush_cache()
        return FlushCacheReqOutput(success=success)

    def clear_hicache_storage_wrapped(self, recv_req: ClearHiCacheReqInput):
        if self.enable_hierarchical_cache:
            self.tree_cache.clear_storage_backend()
            logger.info("Hierarchical cache cleared successfully!")
            if_success = True
        else:
            logging.warning("Hierarchical cache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

    def _is_idle_for_hicache_storage_op(self) -> bool:
        """Stricter idle check for storage attach/detach.

        We require:
        - no running batches (including overlap/pp/disagg paths) via `_is_no_request()`
        - no queued requests in scheduler queues (waiting/grammar/disagg queues)
        """
        if not self._is_no_request():
            return False
        if len(self.waiting_queue) != 0:
            return False
        if len(self.grammar_manager.grammar_queue) != 0:
            return False
        return True

    def attach_hicache_storage_wrapped(
        self, recv_req: AttachHiCacheStorageReqInput
    ) -> AttachHiCacheStorageReqOutput:
        if not self.enable_hierarchical_cache:
            return AttachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self._is_idle_for_hicache_storage_op():
            return AttachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject attach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
                ),
            )

        if not hasattr(self.tree_cache, "attach_storage_backend"):
            return AttachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic attach.",
            )

        try:
            ok, msg = self.tree_cache.attach_storage_backend(
                storage_backend=recv_req.hicache_storage_backend,
                storage_backend_extra_config_json=recv_req.hicache_storage_backend_extra_config_json,
                served_model_name=self.server_args.served_model_name,
                hicache_storage_prefetch_policy=recv_req.hicache_storage_prefetch_policy,
                hicache_write_policy=recv_req.hicache_write_policy,
            )
        except Exception as e:
            logger.exception("Attach HiCache storage backend failed with exception.")
            return AttachHiCacheStorageReqOutput(success=False, message=str(e))
        if ok:
            self.enable_hicache_storage = True
            self.server_args.hicache_storage_backend = recv_req.hicache_storage_backend
            if recv_req.hicache_storage_backend_extra_config_json is not None:
                self.server_args.hicache_storage_backend_extra_config = (
                    recv_req.hicache_storage_backend_extra_config_json
                )
            if recv_req.hicache_storage_prefetch_policy is not None:
                self.server_args.hicache_storage_prefetch_policy = (
                    recv_req.hicache_storage_prefetch_policy
                )
            if recv_req.hicache_write_policy is not None:
                self.server_args.hicache_write_policy = recv_req.hicache_write_policy
            logger.info(
                f"Attached HiCache storage backend: {recv_req.hicache_storage_backend}"
            )
        return AttachHiCacheStorageReqOutput(success=ok, message=msg)

    def detach_hicache_storage_wrapped(
        self, recv_req: DetachHiCacheStorageReqInput
    ) -> DetachHiCacheStorageReqOutput:
        if not self.enable_hierarchical_cache:
            return DetachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self._is_idle_for_hicache_storage_op():
            return DetachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject detach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
                ),
            )

        if not hasattr(self.tree_cache, "detach_storage_backend"):
            return DetachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic detach.",
            )

        # Idempotent detach: even if scheduler thinks storage is disabled, we still
        # attempt best-effort cleanup in tree_cache (it may have leftover state).
        try:
            ok, msg = self.tree_cache.detach_storage_backend()
        except Exception as e:
            logger.exception("Detach HiCache storage backend failed with exception.")
            return DetachHiCacheStorageReqOutput(success=False, message=str(e))

        if ok or (not self.enable_hicache_storage):
            # Treat "already disabled / nothing to do" as success for idempotence.
            self.enable_hicache_storage = False
            self.server_args.hicache_storage_backend = None
            self.server_args.hicache_storage_backend_extra_config = None
            logger.info("Detached HiCache storage backend.")
            return DetachHiCacheStorageReqOutput(
                success=True, message=msg or "HiCache storage backend is detached."
            )

        return DetachHiCacheStorageReqOutput(success=False, message=msg)

    def _is_no_request(self):
        no_request = (
            self.running_batch.is_empty()
            and (self.last_batch is None or self.last_batch.is_empty())
            and (self.cur_batch is None or self.cur_batch.is_empty())
            and (not self.enable_overlap or len(self.result_queue) == 0)
            and (self.pp_size == 1 or all(x.is_empty() for x in self.running_mbs))
        )
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            no_request &= (
                len(self.disagg_prefill_bootstrap_queue.queue) == 0
                and len(self.disagg_prefill_inflight_queue) == 0
            )
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            no_request &= (
                len(self.disagg_decode_prealloc_queue.queue) == 0
                and len(self.disagg_decode_transfer_queue.queue) == 0
            )
        return no_request

    def _kunserve_prepare_for_memory_release(self) -> None:
        """Drain harmless KunServe overlap state before SGLang memory release.

        ``release_memory_occupation`` is issued as a control request at the top
        of the overlap scheduler loop.  With Phase E cached GLOBAL graph replay
        the previous iteration can leave an IDLE keepalive result in
        ``result_queue`` even though all real requests are already finished.
        That queued IDLE result makes the original SGLang no-request assertion
        fail.  We process queued results first, then let the original assertion
        continue to reject any real ongoing request.
        """
        result_queue = getattr(self, "result_queue", None)
        result_queue_len_before = len(result_queue) if result_queue is not None else 0
        drained = 0
        if self.enable_overlap and result_queue is not None:
            while len(result_queue) > 0:
                tmp_batch, tmp_result = result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
                drained += 1

        self._stop_balloon_keepalive("memory release")
        self._phase_e_reset_cached_decision("memory release")

        model_runner = getattr(self.tp_worker, "model_runner", None)
        if model_runner is not None:
            try:
                model_runner.set_balloon_step_force_eager(False)
            except Exception:
                pass
            try:
                model_runner.set_balloon_step_graph_bs_override(None)
            except Exception:
                pass

        if drained > 0 and self.running_batch.is_empty():
            self.last_batch = None
            self.cur_batch = None
        elif self.last_batch is not None and self.last_batch.is_empty():
            self.last_batch = None
            if self.cur_batch is not None and self.cur_batch.is_empty():
                self.cur_batch = None

        logger.info(
            "[KunServeScheduler] memory release cleanup: drained_overlap=%d "
            "result_queue_before=%d result_queue_after=%d running=%d waiting=%d "
            "no_request=%s",
            drained,
            result_queue_len_before,
            len(result_queue) if result_queue is not None else 0,
            len(self.running_batch.reqs),
            len(self.waiting_queue),
            self._is_no_request(),
        )
        kunserve_timing_log(
            "kunserve_memory_release_cleanup",
            drained_overlap=int(drained),
            result_queue_before=int(result_queue_len_before),
            result_queue_after=int(len(result_queue) if result_queue is not None else 0),
            running=len(self.running_batch.reqs),
            waiting=len(self.waiting_queue),
            no_request=bool(self._is_no_request()),
        )

    def _stop_balloon_keepalive(self, reason: str) -> None:
        if self._balloon_keepalive_active:
            logger.info(
                "[KunServeScheduler] stop balloon keepalive: reason=%s steps=%d",
                reason,
                self.balloon_keepalive_step_ct,
            )
            self._balloon_keepalive_active = False

    def _kunserve_phase_e_active(self) -> bool:
        """Whether Phase E cross-replica bs sync should run this step.

        For GLOBAL backends with cross-replica MoE collectives every rank must
        enter the MoE path on every decode step.  Fixed-padded CUDA graph replay additionally
        needs matching padded bs values, but eager GLOBAL still needs a
        non-empty keepalive when a peer replica is busy.  Otherwise an idle
        replica sends an empty IDLE batch through Qwen3 prepare_mlp / global
        collectives while its peer is decoding, which can crash or hang.
        """
        model_runner = getattr(self.tp_worker, "model_runner", None)
        if model_runner is None:
            return False
        if str(getattr(model_runner, "_balloon_state", "local")) != "balloon":
            return False
        backend = str(
            getattr(model_runner, "_balloon_kunserve_comm_backend", "deepep") or ""
        ).lower()
        if backend not in ("sglang", "deepep"):
            return False
        try:
            variant_getter = getattr(model_runner, "get_cuda_graph_runtime_variant")
            runtime_variant = str(variant_getter())
        except Exception:
            runtime_variant = str(
                getattr(model_runner, "_balloon_runtime_variant", "local")
            )
        if runtime_variant != "global":
            return False
        # Skip the sync until a runtime_group has actually been resolved.
        return getattr(model_runner, "_balloon_process_group_name", None) is not None

    def _kunserve_global_graph_replay_active(self) -> bool:
        """True only for the fixed-padded GLOBAL CUDA graph steady state."""
        if not self._kunserve_phase_e_active():
            return False
        model_runner = getattr(self.tp_worker, "model_runner", None)
        if model_runner is None:
            return False
        capture_policy = str(
            getattr(model_runner, "_balloon_capture_policy", "auto") or "auto"
        ).lower()
        if capture_policy != "fixed_padded":
            return False
        try:
            if not bool(model_runner.is_cuda_graph_replay_enabled()):
                return False
        except Exception:
            if not bool(getattr(model_runner, "_balloon_graph_replay_enabled", False)):
                return False
        return True

    def _kunserve_balloon_global_runtime_active(self) -> bool:
        model_runner = getattr(self.tp_worker, "model_runner", None)
        if model_runner is None:
            return False
        if str(getattr(model_runner, "_balloon_state", "local")) != "balloon":
            return False
        try:
            variant_getter = getattr(model_runner, "get_cuda_graph_runtime_variant")
            runtime_variant = str(variant_getter())
        except Exception:
            runtime_variant = str(
                getattr(model_runner, "_balloon_runtime_variant", "local")
            )
        return runtime_variant == "global"

    def _kunserve_align_prefill_budget(self, value: int) -> int:
        value = max(int(value), int(self.page_size))
        return max(int(self.page_size), (value // int(self.page_size)) * int(self.page_size))

    def _kunserve_effective_prefill_budget_for_balloon(
        self,
        max_prefill_tokens: int,
        chunked_prefill_size: Optional[int],
    ) -> Tuple[int, Optional[int]]:
        """Limit eager EXTEND token count after GLOBAL balloon.

        Decode replay is graph-captured, but a waiting/retracted request still
        enters as EXTEND and therefore runs eager.  Under GLOBAL dispatch that
        eager step pads/gathers by token count across replicas, so a very large
        re-prefill can allocate hundreds of MiB of temporary MoE/dispatch
        buffers while graph private pools and the expanded KV pool are resident.
        Keep the cap local to balloon/global so baseline and local KunServe
        startup prefill keep the configured large prefill budget.
        """
        if not self._kunserve_balloon_global_runtime_active():
            return int(max_prefill_tokens), chunked_prefill_size

        max_cap = get_int_env_var("KUNSERVE_BALLOON_MAX_PREFILL_TOKENS", 8192)
        chunk_cap = get_int_env_var(
            "KUNSERVE_BALLOON_CHUNKED_PREFILL_SIZE",
            max_cap if max_cap > 0 else 8192,
        )

        effective_max = int(max_prefill_tokens)
        if max_cap > 0:
            effective_max = min(
                effective_max, self._kunserve_align_prefill_budget(max_cap)
            )

        effective_chunk = chunked_prefill_size
        if chunk_cap > 0:
            aligned_chunk = self._kunserve_align_prefill_budget(chunk_cap)
            effective_chunk = (
                aligned_chunk
                if effective_chunk is None
                else min(int(effective_chunk), aligned_chunk)
            )

        changed = (
            int(effective_max) != int(max_prefill_tokens)
            or effective_chunk != chunked_prefill_size
        )
        if changed:
            self._kunserve_balloon_prefill_budget_log_ct += 1
            log_ct = self._kunserve_balloon_prefill_budget_log_ct
            if log_ct in (1, 10, 100) or log_ct % 1000 == 0:
                _kunserve_ms(
                    "[KUNSERVE-MS] balloon prefill budget cap: count=%d "
                    "max_prefill=%d->%d chunk=%s->%s running=%d waiting=%d",
                    log_ct,
                    int(max_prefill_tokens),
                    int(effective_max),
                    str(chunked_prefill_size),
                    str(effective_chunk),
                    len(self.running_batch.reqs),
                    len(self.waiting_queue),
                )
            kunserve_timing_log(
                "scheduler_balloon_prefill_budget_cap",
                count=int(log_ct),
                max_prefill_tokens=int(max_prefill_tokens),
                effective_max_prefill_tokens=int(effective_max),
                chunked_prefill_size=(
                    int(chunked_prefill_size)
                    if chunked_prefill_size is not None
                    else None
                ),
                effective_chunked_prefill_size=(
                    int(effective_chunk) if effective_chunk is not None else None
                ),
                running=len(self.running_batch.reqs),
                waiting=len(self.waiting_queue),
            )

        return int(effective_max), effective_chunk

    def _kunserve_should_defer_prefill_for_balloon_memory(self) -> bool:
        if not self._kunserve_balloon_global_runtime_active():
            return False
        if len(self.waiting_queue) == 0 and self.chunked_req is None:
            return False
        running_batch = getattr(self, "running_batch", None)
        if running_batch is None or running_batch.is_empty():
            return False

        try:
            min_free_gb = float(
                os.environ.get("KUNSERVE_BALLOON_PREFILL_MIN_FREE_GB", "1.0")
            )
        except Exception:
            min_free_gb = 1.0
        if min_free_gb <= 0:
            return False

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except Exception:
            return False
        min_free_bytes = int(min_free_gb * (1024**3))
        if int(free_bytes) >= min_free_bytes:
            self._kunserve_balloon_prefill_mem_defer_log_ct = 0
            return False

        self._kunserve_balloon_prefill_mem_defer_log_ct += 1
        log_ct = self._kunserve_balloon_prefill_mem_defer_log_ct
        try:
            available_tokens = self.token_to_kv_pool_allocator.available_size()
        except Exception:
            available_tokens = -1
        if log_ct in (1, 10, 100) or log_ct % 1000 == 0:
            _kunserve_ms(
                "[KUNSERVE-MS] defer balloon prefill for cuda headroom: "
                "count=%d free_gb=%.3f threshold_gb=%.3f running=%d waiting=%d "
                "chunked=%s available_tokens=%d max_total=%d",
                log_ct,
                float(free_bytes) / (1024**3),
                float(min_free_gb),
                len(self.running_batch.reqs),
                len(self.waiting_queue),
                self.chunked_req is not None,
                int(available_tokens),
                int(self.max_total_num_tokens),
            )
        kunserve_timing_log(
            "scheduler_prefill_deferred_balloon_memory",
            count=int(log_ct),
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
            threshold_bytes=int(min_free_bytes),
            running=len(self.running_batch.reqs),
            waiting=len(self.waiting_queue),
            chunked=self.chunked_req is not None,
            available_tokens=int(available_tokens),
            max_total=int(self.max_total_num_tokens),
        )
        return True

    def _kunserve_should_defer_prefill_for_global_graph(self) -> bool:
        """Legacy conservative gate for keeping graph replay decode-only.

        This is now disabled by default.  Phase E separately gathers a
        per-step ``force_eager`` bit, so an EXTEND/re-prefill step can run
        immediately after BALLOON without letting any peer replay a mismatched
        GLOBAL CUDA graph.  Re-enable only for reproducing old graph issues.
        """
        if os.environ.get("KUNSERVE_DEFER_PREFILL_UNDER_GLOBAL_GRAPH", "0") not in (
            "1",
            "true",
            "True",
            "yes",
        ):
            return False

        running_batch = getattr(self, "running_batch", None)
        if (
            running_batch is None
            or running_batch.is_empty()
            or len(self.waiting_queue) == 0
        ):
            return False

        if not self._kunserve_balloon_global_runtime_active():
            return False

        return self._kunserve_global_graph_replay_active()

    def _phase_e_reset_cached_decision(self, reason: str) -> None:
        if self._phase_e_cache_valid:
            kunserve_timing_log(
                "phase_e_cache_reset",
                reason=str(reason),
                cached_max_bs=int(self._phase_e_cached_max_bs),
                cached_min_bs=int(self._phase_e_cached_min_bs),
                cached_raw_max_bs=int(self._phase_e_cached_raw_max_bs),
                cached_raw_min_bs=int(self._phase_e_cached_raw_min_bs),
                cached_steps_left=int(self._phase_e_cached_steps_left),
                cached_state_fingerprint=self._phase_e_cached_state_fingerprint,
            )
        self._phase_e_cache_valid = False
        self._phase_e_cached_max_bs = 0
        self._phase_e_cached_min_bs = 0
        self._phase_e_cached_raw_max_bs = 0
        self._phase_e_cached_raw_min_bs = 0
        self._phase_e_cached_any_force_eager = False
        self._phase_e_cached_steps_left = 0
        self._phase_e_cached_state_fingerprint = None

    def _phase_e_cache_enabled(self) -> bool:
        # Reusing a cached graph bucket without the guard collective cannot
        # observe peer raw-batch shrink/growth inside the cache window.  Keep
        # cache disabled when the guard is disabled so Phase E falls back to
        # lockstep negotiation every step instead of risking stale graph replay.
        return (
            self._phase_e_negotiate_interval > 1
            and self._phase_e_cache_guard_enabled
            and self._kunserve_global_graph_replay_active()
        )

    def _phase_e_cache_due(self) -> bool:
        if not self._phase_e_cache_enabled():
            return True
        if not self._phase_e_cache_valid:
            return True
        return self._phase_e_cached_steps_left <= 0

    def _phase_e_may_defer_prefill_for_cache(self) -> bool:
        """Hold new prefill until the next lockstep Phase E refresh.

        Skipping Phase E negotiation is safe only while every rank keeps
        replaying the same cached decode bucket.  A local EXTEND/mixed step
        would need all peers to force eager on that same scheduler step.  The
        deterministic refresh interval is the lockstep event boundary that
        lets a local waiting request become globally visible without letting
        one rank enter the collective alone.
        """
        if not self._phase_e_cache_enabled():
            return False
        if not self._phase_e_cache_valid:
            return False
        if self._phase_e_cached_steps_left <= 0:
            return False
        if len(self.waiting_queue) == 0 and self.chunked_req is None:
            return False
        return True

    def _phase_e_batch_mode_code(self, batch: Optional[ScheduleBatch]) -> int:
        if batch is None:
            return 0
        try:
            if batch.forward_mode.is_idle():
                return 1
            if batch.forward_mode.is_decode():
                return 2
            if batch.forward_mode.is_extend(include_draft_extend_v2=True):
                return 3
        except Exception:
            pass
        return 4

    def _phase_e_local_state_signature(
        self,
        *,
        batch: Optional[ScheduleBatch],
        local_bs: int,
        local_padded: int,
        local_force_eager: bool,
    ) -> Tuple[int, ...]:
        """Small per-rank state vector used to validate cached Phase E buckets.

        The cached GLOBAL graph bucket is safe only while every rank keeps the
        same scheduler shape/control state that produced the cache. A local
        shrink, an arriving waiting request, or a chunked-prefill transition is
        enough reason to refresh/force eager together on a synchronized step.
        """
        result_queue_len = 0
        result_queue = getattr(self, "result_queue", None)
        if result_queue is not None:
            try:
                result_queue_len = len(result_queue)
            except Exception:
                result_queue_len = 0
        last_batch = getattr(self, "last_batch", None)
        try:
            last_batch_size = last_batch.batch_size() if last_batch is not None else 0
        except Exception:
            last_batch_size = 0
        running_batch = getattr(self, "running_batch", None)
        running_reqs = getattr(running_batch, "reqs", []) if running_batch else []
        batch_is_full = bool(getattr(running_batch, "batch_is_full", False))
        return (
            int(local_bs),
            int(local_padded),
            1 if local_force_eager else 0,
            int(self._phase_e_batch_mode_code(batch)),
            int(len(running_reqs)),
            int(len(self.waiting_queue)),
            1 if batch_is_full else 0,
            1 if self.chunked_req is not None else 0,
            int(result_queue_len),
            int(last_batch_size),
        )

    def _phase_e_update_cached_decision(
        self,
        *,
        negotiated_max_bs: int,
        negotiated_min_bs: int,
        negotiated_raw_max_bs: int,
        negotiated_raw_min_bs: int,
        negotiated_any_force_eager: bool,
        state_fingerprint: Optional[int],
    ) -> None:
        if not self._phase_e_cache_enabled():
            self._phase_e_reset_cached_decision("cache disabled")
            return

        negotiated_max_bs = int(negotiated_max_bs)
        negotiated_min_bs = int(negotiated_min_bs)
        negotiated_raw_max_bs = int(negotiated_raw_max_bs)
        negotiated_raw_min_bs = int(negotiated_raw_min_bs)
        cacheable = (
            negotiated_max_bs > 0
            and not bool(negotiated_any_force_eager)
            and self._capture_bs_supported(negotiated_max_bs)
        )
        if not cacheable:
            self._phase_e_reset_cached_decision("uncacheable negotiated step")
            return

        self._phase_e_cache_valid = True
        self._phase_e_cached_max_bs = negotiated_max_bs
        self._phase_e_cached_min_bs = negotiated_min_bs
        self._phase_e_cached_raw_max_bs = negotiated_raw_max_bs
        self._phase_e_cached_raw_min_bs = negotiated_raw_min_bs
        self._phase_e_cached_any_force_eager = bool(negotiated_any_force_eager)
        self._phase_e_cached_steps_left = max(
            0, int(self._phase_e_negotiate_interval) - 1
        )
        self._phase_e_cached_state_fingerprint = state_fingerprint
        kunserve_timing_log(
            "phase_e_cache_update",
            interval=int(self._phase_e_negotiate_interval),
            cached_max_bs=int(self._phase_e_cached_max_bs),
            cached_min_bs=int(self._phase_e_cached_min_bs),
            cached_raw_max_bs=int(self._phase_e_cached_raw_max_bs),
            cached_raw_min_bs=int(self._phase_e_cached_raw_min_bs),
            cached_any_force_eager=bool(self._phase_e_cached_any_force_eager),
            cached_steps_left=int(self._phase_e_cached_steps_left),
            cached_state_fingerprint=self._phase_e_cached_state_fingerprint,
        )

    def _phase_e_try_reuse_cached_decision(
        self,
        *,
        local_raw_bs: int,
        local_padded: int,
        local_force_eager: bool,
        local_signature: Tuple[int, ...],
    ) -> Optional[Tuple[int, int, bool, bool, Optional[int], int, int]]:
        if not self._phase_e_cache_enabled() or not self._phase_e_cache_valid:
            return None
        if self._phase_e_cached_steps_left <= 0:
            return None
        if local_force_eager:
            # This should normally be prevented by prefill deferral until a
            # deterministic refresh step where every rank negotiates together.
            # Do not replay a decode graph for a local EXTEND/mixed batch.
            self._phase_e_reset_cached_decision("local force eager")
            return None
        if int(local_padded) > int(self._phase_e_cached_max_bs):
            # A local growth beyond the cached bucket means the cache is no
            # longer safe.  With prefill deferral this should be rare, but a
            # refresh is the only safe way to make every rank switch together.
            self._phase_e_reset_cached_decision("local padded exceeds cache")
            return None

        if (
            self._phase_e_cache_guard_enabled
            and self._phase_e_cached_state_fingerprint is not None
        ):
            previous_state_fingerprint = self._phase_e_cached_state_fingerprint
            (
                guard_max_bs,
                guard_min_bs,
                guard_any_force_eager,
                guard_state_fingerprint,
                guard_raw_max_bs,
                guard_raw_min_bs,
            ) = self.negotiate_balloon_step_state(
                int(local_raw_bs),
                local_force_eager=bool(local_force_eager),
                local_padded_bs=int(local_padded),
                local_state_signature=local_signature,
            )
            cache_changed = (
                guard_state_fingerprint != previous_state_fingerprint
                or int(guard_max_bs) > int(self._phase_e_cached_max_bs)
                or bool(guard_any_force_eager)
            )
            if cache_changed:
                self._phase_e_reset_cached_decision("guard state changed")
                kunserve_timing_log(
                    "phase_e_cache_guard_event",
                    local_padded=int(local_padded),
                    guard_max_bs=int(guard_max_bs),
                    guard_min_bs=int(guard_min_bs),
                    guard_any_force_eager=bool(guard_any_force_eager),
                    guard_raw_max_bs=int(guard_raw_max_bs),
                    guard_raw_min_bs=int(guard_raw_min_bs),
                    guard_raw_busy_mismatch=bool(
                        int(guard_raw_min_bs) > 0
                        and int(guard_raw_min_bs) != int(guard_raw_max_bs)
                    ),
                    previous_state_fingerprint=previous_state_fingerprint,
                    guard_state_fingerprint=guard_state_fingerprint,
                )
                self._phase_e_update_cached_decision(
                    negotiated_max_bs=int(guard_max_bs),
                    negotiated_min_bs=int(guard_min_bs),
                    negotiated_raw_max_bs=int(guard_raw_max_bs),
                    negotiated_raw_min_bs=int(guard_raw_min_bs),
                    negotiated_any_force_eager=bool(guard_any_force_eager),
                    state_fingerprint=guard_state_fingerprint,
                )
                # All ranks observed the same guard event. Reuse the freshly
                # negotiated bucket on this step; only a real force-eager signal
                # (EXTEND/mixed prefill) disables graph replay.
                return (
                    int(guard_max_bs),
                    int(guard_min_bs),
                    bool(guard_any_force_eager),
                    False,
                    guard_state_fingerprint,
                    int(guard_raw_max_bs),
                    int(guard_raw_min_bs),
                )

        self._phase_e_cached_steps_left -= 1
        kunserve_timing_log(
            "phase_e_cache_reuse",
            interval=int(self._phase_e_negotiate_interval),
            local_padded=int(local_padded),
            local_force_eager=bool(local_force_eager),
            cached_max_bs=int(self._phase_e_cached_max_bs),
            cached_min_bs=int(self._phase_e_cached_min_bs),
            cached_raw_max_bs=int(self._phase_e_cached_raw_max_bs),
            cached_raw_min_bs=int(self._phase_e_cached_raw_min_bs),
            cached_any_force_eager=bool(self._phase_e_cached_any_force_eager),
            cached_steps_left=int(self._phase_e_cached_steps_left),
            cached_state_fingerprint=self._phase_e_cached_state_fingerprint,
        )
        return (
            int(self._phase_e_cached_max_bs),
            int(self._phase_e_cached_min_bs),
            bool(self._phase_e_cached_any_force_eager),
            True,
            self._phase_e_cached_state_fingerprint,
            int(self._phase_e_cached_raw_max_bs),
            int(self._phase_e_cached_raw_min_bs),
        )

    def _phase_e_get_step_decision(
        self,
        *,
        batch: Optional[ScheduleBatch],
        local_status: Dict[str, Any],
        loop: str,
    ) -> Tuple[int, int, bool, bool, Optional[int], int, int]:
        local_bs = batch.batch_size() if batch is not None else 0
        stage_ns = time.perf_counter_ns()
        local_force_eager = self._kunserve_batch_requires_eager_for_phase_e(batch)
        local_padded = self._padded_capture_bs(local_bs)
        _kun_wd(
            "[KUNSERVE-WD] phase_e_decide fwd_ct=%d local_bs=%d padded=%d fe=%d loop=%s"
            % (
                int(self.forward_ct),
                int(local_bs),
                int(local_padded),
                1 if local_force_eager else 0,
                str(loop),
            )
        )
        local_signature = self._phase_e_local_state_signature(
            batch=batch,
            local_bs=int(local_bs),
            local_padded=int(local_padded),
            local_force_eager=bool(local_force_eager),
        )
        self._kunserve_scheduler_gap_log(
            "scheduler_gap_phase_e_local_shape_end",
            stage_ns,
            batch=batch,
            loop=loop,
            local_bs=int(local_bs),
            local_padded=int(local_padded),
            local_force_eager=bool(local_force_eager),
            local_signature=str(local_signature),
        )

        cached = self._phase_e_try_reuse_cached_decision(
            local_raw_bs=int(local_bs),
            local_padded=int(local_padded),
            local_force_eager=bool(local_force_eager),
            local_signature=local_signature,
        )
        if cached is not None:
            (
                negotiated_max_bs,
                negotiated_min_bs,
                negotiated_any_force_eager,
                phase_e_from_cache,
                state_fingerprint,
                negotiated_raw_max_bs,
                negotiated_raw_min_bs,
            ) = cached
            kunserve_timing_log(
                "phase_e_negotiated",
                source="cache" if phase_e_from_cache else "guard_event",
                forward_ct=self.forward_ct,
                mode=str(batch.forward_mode) if batch is not None else "None",
                local_bs=int(local_bs),
                local_padded=int(local_padded),
                local_force_eager=bool(local_force_eager),
                max_bs=int(negotiated_max_bs),
                min_bs=int(negotiated_min_bs),
                raw_max_bs=int(negotiated_raw_max_bs),
                raw_min_bs=int(negotiated_raw_min_bs),
                any_force_eager=bool(negotiated_any_force_eager),
                running=len(self.running_batch.reqs),
                waiting=len(self.waiting_queue),
                cached_steps_left=int(self._phase_e_cached_steps_left),
                state_fingerprint=state_fingerprint,
            )
            return (
                int(negotiated_max_bs),
                int(negotiated_min_bs),
                bool(negotiated_any_force_eager),
                bool(phase_e_from_cache),
                state_fingerprint,
                int(negotiated_raw_max_bs),
                int(negotiated_raw_min_bs),
            )

        (
            negotiated_max_bs,
            negotiated_min_bs,
            negotiated_any_force_eager,
            state_fingerprint,
            negotiated_raw_max_bs,
            negotiated_raw_min_bs,
        ) = self.negotiate_balloon_step_state(
            local_bs,
            local_force_eager=local_force_eager,
            local_padded_bs=local_padded,
            local_state_signature=local_signature,
        )
        raw_busy_mismatch = (
            int(negotiated_raw_min_bs) > 0
            and int(negotiated_raw_min_bs) != int(negotiated_raw_max_bs)
        )
        kunserve_timing_log(
            "phase_e_negotiated",
            source="collective",
            forward_ct=self.forward_ct,
            mode=str(batch.forward_mode) if batch is not None else "None",
            local_bs=int(local_bs),
            local_padded=int(local_padded),
            local_force_eager=bool(local_force_eager),
            max_bs=int(negotiated_max_bs),
            min_bs=int(negotiated_min_bs),
            raw_max_bs=int(negotiated_raw_max_bs),
            raw_min_bs=int(negotiated_raw_min_bs),
            raw_busy_mismatch=bool(raw_busy_mismatch),
            any_force_eager=bool(negotiated_any_force_eager),
            running=len(self.running_batch.reqs),
            waiting=len(self.waiting_queue),
            state_fingerprint=state_fingerprint,
        )
        self._phase_e_update_cached_decision(
            negotiated_max_bs=int(negotiated_max_bs),
            negotiated_min_bs=int(negotiated_min_bs),
            negotiated_raw_max_bs=int(negotiated_raw_max_bs),
            negotiated_raw_min_bs=int(negotiated_raw_min_bs),
            negotiated_any_force_eager=bool(negotiated_any_force_eager),
            state_fingerprint=state_fingerprint,
        )
        return (
            int(negotiated_max_bs),
            int(negotiated_min_bs),
            bool(negotiated_any_force_eager),
            False,
            state_fingerprint,
            int(negotiated_raw_max_bs),
            int(negotiated_raw_min_bs),
        )

    def _padded_capture_bs(self, local_bs: int) -> int:
        """Mirror cuda_graph_runner.can_run's bisect-up rule.

        Returns the bs that the captured graph would actually replay
        with, or ``local_bs`` if no graph would be selected.  Phase E
        feeds this into the cross-replica negotiation so the agreed
        value matches the real per-rank graph that would fire.
        """
        if local_bs <= 0:
            return 0
        model_runner = getattr(self.tp_worker, "model_runner", None)
        if model_runner is None:
            return int(local_bs)
        capture_policy = str(
            getattr(model_runner, "_balloon_capture_policy", "auto") or "auto"
        ).lower()
        if capture_policy != "fixed_padded":
            return int(local_bs)
        try:
            replay_enabled = bool(model_runner.is_cuda_graph_replay_enabled())
        except Exception:
            replay_enabled = bool(
                getattr(model_runner, "_balloon_graph_replay_enabled", False)
            )
        if not replay_enabled:
            return int(local_bs)
        graph_runner = getattr(model_runner, "graph_runner", None) if model_runner else None
        if graph_runner is None:
            return int(local_bs)
        capture_bs = getattr(graph_runner, "capture_bs", None) or []
        if not capture_bs:
            return int(local_bs)
        import bisect

        idx = bisect.bisect_left(capture_bs, int(local_bs))
        if idx >= len(capture_bs):
            # Too large for any captured graph -> would go eager anyway.
            return int(local_bs)
        return int(capture_bs[idx])

    def _capture_bs_supported(self, target_bs: int) -> bool:
        if int(target_bs) <= 0:
            return False
        model_runner = getattr(self.tp_worker, "model_runner", None)
        graph_runner = getattr(model_runner, "graph_runner", None) if model_runner else None
        capture_bs = getattr(graph_runner, "capture_bs", None) or []
        return int(target_bs) in {int(bs) for bs in capture_bs}

    def _kunserve_batch_requires_eager_for_phase_e(
        self, batch: Optional[ScheduleBatch]
    ) -> bool:
        if batch is None:
            return False
        try:
            if batch.forward_mode.is_extend(include_draft_extend_v2=True):
                return True
        except Exception:
            pass
        return bool(getattr(batch, "is_extend_in_batch", False))

    def _phase_e_apply_step_decision(
        self,
        *,
        negotiated_max_bs: int,
        negotiated_min_bs: int,
        negotiated_raw_max_bs: int = 0,
        negotiated_raw_min_bs: int = 0,
        negotiated_any_force_eager: bool = False,
        graph_bs_override_hint: Optional[int] = None,
        state_fingerprint: Optional[int] = None,
    ) -> None:
        """Set ``_balloon_step_force_eager`` based on the negotiation.

        Symmetric rule (every rank sees the same ``(max, min)`` and
        reaches the same decision):

        * ``any_force_eager`` -- at least one rank is running EXTEND/mixed
          prefill.  All ranks must skip graph replay because GLOBAL graph
          collectives are fixed to decode-shaped token counts.
        * ``min_bs == 0 and max_bs > 0`` -- idle/busy decode split.  Idle
          replicas build a keepalive batch of size ``max_bs`` and graph replay
          is safe only when ``any_force_eager`` is false.
        * ``raw_min_bs > 0 and raw_min_bs != raw_max_bs`` -- busy/busy raw
          mismatch.  This is safe for GLOBAL graph replay as long as every rank
          uses the same negotiated graph bucket; smaller local batches pad extra
          rows to dummy req/KV state in ``CudaGraphRunner``.
        * ``min_bs > 0 and min_bs != max_bs`` -- busy/busy with mismatched
          padded bs.  All ranks replay the ``max_bs`` GLOBAL graph bucket.
        * ``min_bs == max_bs`` -- uniform decode graph bucket.
        """
        model_runner = getattr(self.tp_worker, "model_runner", None)
        if model_runner is None:
            return
        negotiated_max_bs = int(negotiated_max_bs)
        negotiated_min_bs = int(negotiated_min_bs)
        negotiated_raw_max_bs = int(negotiated_raw_max_bs)
        negotiated_raw_min_bs = int(negotiated_raw_min_bs)
        raw_busy_mismatch = (
            negotiated_raw_min_bs > 0
            and negotiated_raw_min_bs != negotiated_raw_max_bs
        )
        graph_bs_override: Optional[int] = None
        force_eager = bool(negotiated_any_force_eager)
        mismatch_decode = (
            negotiated_min_bs > 0 and negotiated_min_bs != negotiated_max_bs
        )
        if graph_bs_override_hint is not None and not force_eager:
            graph_bs_override_hint = int(graph_bs_override_hint)
            if self._capture_bs_supported(graph_bs_override_hint):
                graph_bs_override = graph_bs_override_hint
            else:
                force_eager = True
        elif negotiated_max_bs > 0 and not force_eager:
            if self._capture_bs_supported(negotiated_max_bs):
                graph_bs_override = negotiated_max_bs
            else:
                force_eager = True

        expected_graph_bs: Optional[int] = None
        if not force_eager and negotiated_max_bs > 0:
            expected_graph_bs = (
                graph_bs_override if graph_bs_override is not None else negotiated_max_bs
            )
        try:
            model_runner.set_balloon_step_force_eager(force_eager)
            deepep_setter = getattr(
                model_runner, "set_balloon_deepep_step_any_extend", None
            )
            if callable(deepep_setter):
                backend = str(
                    getattr(model_runner, "_balloon_kunserve_comm_backend", "") or ""
                ).lower()
                deepep_setter(
                    bool(negotiated_any_force_eager) if backend == "deepep" else None
                )
            if not force_eager:
                model_runner.set_balloon_step_graph_bs_override(graph_bs_override)
            guard_setter = getattr(model_runner, "set_balloon_step_graph_guard", None)
            if guard_setter is not None:
                guard_setter(
                    expected_graph_bs,
                    max_bs=negotiated_max_bs,
                    min_bs=negotiated_min_bs,
                    raw_max_bs=negotiated_raw_max_bs,
                    raw_min_bs=negotiated_raw_min_bs,
                    state_fingerprint=state_fingerprint,
                )
        except Exception:
            pass
        kunserve_timing_log(
            "phase_e_step_decision",
            max_bs=int(negotiated_max_bs),
            min_bs=int(negotiated_min_bs),
            raw_max_bs=int(negotiated_raw_max_bs),
            raw_min_bs=int(negotiated_raw_min_bs),
            raw_busy_mismatch=bool(raw_busy_mismatch),
            any_force_eager=bool(negotiated_any_force_eager),
            mismatch_decode=bool(mismatch_decode),
            force_eager=bool(force_eager),
            graph_bs_override=graph_bs_override,
            graph_bs_override_hint=graph_bs_override_hint,
            expected_graph_bs=expected_graph_bs,
            state_fingerprint=state_fingerprint,
        )

    def _local_balloon_status_or_stop(self) -> Optional[Dict[str, Any]]:
        try:
            local_status = self.tp_worker.get_balloon_status(GetBalloonStatusReqInput())
        except Exception:
            self._stop_balloon_keepalive("balloon status query failed")
            logger.exception(
                "[KunServeScheduler] failed to query balloon status before keepalive"
            )
            return None
        if str(local_status.get("state")) != "balloon":
            self._stop_balloon_keepalive("runtime is not balloon")
            return None
        return local_status

    def negotiate_balloon_step_state(
        self,
        local_bs: int,
        local_force_eager: bool = False,
        local_state_signature: Optional[Tuple[int, ...]] = None,
        local_padded_bs: Optional[int] = None,
    ) -> Tuple[int, int, bool, Optional[int], int, int]:
        """Phase E synchronized shape/state negotiation.

        The first three return values are the historical batch-size decision.
        ``state_fingerprint`` additionally fingerprints the tiny per-rank
        scheduler signatures gathered in the same collective, so cached GLOBAL
        graph buckets can be invalidated symmetrically on all ranks.
        """
        stage_ns = time.perf_counter_ns()
        local_padded_for_timing = (
            int(local_padded_bs) if local_padded_bs is not None else int(local_bs)
        )
        with kunserve_timing_scope(
            "phase_e_negotiate",
            loop="phase_e",
            local_bs=int(local_bs),
            local_padded_bs=int(local_padded_for_timing),
            local_force_eager=bool(local_force_eager),
        ):
            max_bs, min_bs, any_force_eager, state_fingerprint, raw_max_bs, raw_min_bs = (
                self.tp_worker.model_runner.negotiate_balloon_step_state(
                    int(local_bs),
                    bool(local_force_eager),
                    local_padded_bs=local_padded_bs,
                    local_state_signature=local_state_signature,
                )
            )
        self._kunserve_scheduler_gap_log(
            "scheduler_gap_phase_e_negotiate_end",
            stage_ns,
            loop="phase_e",
            local_bs=int(local_bs),
            local_padded_bs=(
                int(local_padded_bs) if local_padded_bs is not None else int(local_bs)
            ),
            local_force_eager=bool(local_force_eager),
            max_bs=int(max_bs),
            min_bs=int(min_bs),
            raw_max_bs=int(raw_max_bs),
            raw_min_bs=int(raw_min_bs),
            any_force_eager=bool(any_force_eager),
            state_fingerprint=state_fingerprint,
        )
        return (
            int(max_bs),
            int(min_bs),
            bool(any_force_eager),
            state_fingerprint,
            int(raw_max_bs),
            int(raw_min_bs),
        )

    def negotiate_balloon_step_bs(
        self, local_bs: int, local_force_eager: bool = False
    ) -> Tuple[int, int, bool]:
        """Phase E entry point in the scheduler.

        Returns ``(max_bs, min_bs, any_force_eager)`` across the
        cross-replica runtime group.  Callers use ``max_bs`` for idle
        keepalive size and the other two fields for the symmetric graph/eager
        decision.
        """
        max_bs, min_bs, any_force_eager, _, _, _ = self.negotiate_balloon_step_state(
            int(local_bs), bool(local_force_eager)
        )
        return int(max_bs), int(min_bs), bool(any_force_eager)

    def _build_balloon_keepalive_batch(
        self, target_bs: int, local_status: Dict[str, Any]
    ) -> ScheduleBatch:
        """Phase E: build an IDLE batch shaped to a peer-negotiated bs.

        ``target_bs == 0`` is the historical empty-batch behavior used when
        every replica is idle.  ``target_bs > 0`` is the Phase E flow: the
        keepalive batch carries that many dummy decode tokens so the
        cross-replica all_gather_into_tensor / all_reduce inside the
        captured GLOBAL graph see the matching ``local_m`` on every rank.

        The dummy tokens' ``out_cache_loc`` is redirected to the
        keepalive KV scratch slot reserved by
        ``ModelRunner.commit_balloon`` so the captured graph's KV
        write cannot collide with real-request slot 0.  See Phase E
        BUG #2 in the kunserve docs.
        """
        self.balloon_keepalive_step_ct += 1
        if not self._balloon_keepalive_active:
            logger.warning(
                "[KunServeScheduler] start balloon keepalive: variant=%s "
                "offloaded=%s added_slots=%s target_bs=%d",
                local_status.get("runtime_variant"),
                local_status.get("offloaded_local_experts"),
                local_status.get("added_kv_slots"),
                int(target_bs),
            )
            self._balloon_keepalive_active = True
        elif self.balloon_keepalive_step_ct % 128 == 0:
            logger.info(
                "[KunServeScheduler] balloon keepalive progress: steps=%d "
                "variant=%s target_bs=%d",
                self.balloon_keepalive_step_ct,
                local_status.get("runtime_variant"),
                int(target_bs),
            )

        # Pull the keepalive KV scratch slot and phantom req_pool entry
        # from model_runner.  None means commit_balloon didn't reserve them
        # (legacy DeepEP path, or alloc failed); prepare_for_idle will
        # then fall back to slot 0 / req_pool[0] with a warning.
        dummy_kv_slot: Optional[int] = None
        phantom_req_idx: Optional[int] = None
        model_runner = getattr(self.tp_worker, "model_runner", None)
        if model_runner is not None and int(target_bs) > 0:
            dummy_kv_slot = getattr(
                model_runner, "_kunserve_keepalive_dummy_kv_slot", None
            )
            phantom_req_idx = getattr(
                model_runner, "_kunserve_keepalive_phantom_req_idx", None
            )

        keepalive_batch = self.get_idle_batch(
            target_bs=int(max(target_bs, 0)),
            dummy_kv_slot=dummy_kv_slot,
            phantom_req_idx=phantom_req_idx,
        )
        # Same shape-fixup as before: force attn_tp_size dummy tokens for
        # mlp_sync paths so the DP-attention scatter doesn't see [0,0]
        # for an IDLE batch.  When target_bs > 0 the batch already has
        # the right shape from prepare_for_idle(), but the DP-sync count
        # fields still need to reflect that.  See the original repro
        # ab_20260507_113241 R1 crash for details.
        if self.require_mlp_sync:
            attn_tp_size = max(self.attn_tp_size, 1)
            effective = (
                int(target_bs) if target_bs > 0 else attn_tp_size
            )
            keepalive_batch.global_num_tokens = [effective]
            keepalive_batch.global_num_tokens_for_logprob = [effective]
            keepalive_batch.global_forward_mode = ForwardMode.IDLE
        keepalive_batch.is_extend_in_batch = False
        keepalive_batch.can_run_dp_cuda_graph = False
        return keepalive_batch

    def _maybe_get_balloon_keepalive_batch(
        self, *, target_bs: int = 0
    ) -> Optional[ScheduleBatch]:
        """Legacy entry point preserved for DeepEP backend.

        For the sglang Phase D + Phase E flow callers should use
        ``negotiate_balloon_step_bs`` + ``_build_balloon_keepalive_batch``
        explicitly so the cross-replica sync happens once per step rather
        than only on the idle branch.
        """
        local_status = self._local_balloon_status_or_stop()
        if local_status is None:
            return None
        return self._build_balloon_keepalive_batch(int(target_bs), local_status)

    def flush_cache(self):
        """Flush the memory pool and cache."""
        if self._is_no_request():
            self.cur_batch = None
            self.last_batch = None
            self.tree_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()
            self.grammar_manager.clear()
            self.reset_metrics()

            if self.draft_worker:
                self.draft_worker.clear_cache_pool()

            # TODO: allow optional empty cache
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
            success = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {len(self.running_batch.reqs)}"
            )
            success = False
        return success

    def _sync_runtime_capacity_cache(self, max_total_num_tokens: int) -> None:
        self.max_total_num_tokens = int(max_total_num_tokens)
        self.tp_worker.max_total_num_tokens = int(max_total_num_tokens)
        if self.model_worker is not self.tp_worker and hasattr(
            self.model_worker, "max_total_num_tokens"
        ):
            self.model_worker.max_total_num_tokens = int(max_total_num_tokens)

    def _clear_batch_full_after_capacity_growth(
        self, source: str, old_max_total: int, new_max_total: int
    ) -> None:
        if int(new_max_total) <= int(old_max_total):
            return
        running_batch = getattr(self, "running_batch", None)
        if running_batch is None:
            return

        was_full = bool(getattr(running_batch, "batch_is_full", False))
        running_batch.batch_is_full = False
        self._kunserve_prefill_blocked_full_log_ct = 0
        self._kunserve_graph_prefill_defer_log_ct = 0
        defer_prefill = self._kunserve_should_defer_prefill_for_global_graph()
        try:
            available_tokens = self.token_to_kv_pool_allocator.available_size()
        except Exception:
            available_tokens = -1
        _kunserve_ms(
            "[KUNSERVE-MS] %s capacity grew; cleared batch_is_full: "
            "old_max_total=%d new_max_total=%d was_full=%s running=%d "
            "waiting=%d available_tokens=%d defer_prefill_global_graph=%s",
            source,
            int(old_max_total),
            int(new_max_total),
            was_full,
            len(running_batch.reqs),
            len(self.waiting_queue),
            int(available_tokens),
            bool(defer_prefill),
        )

    def _format_balloon_status(self, status: Dict[str, Any]) -> str:
        if not status:
            return "status=<empty>"
        return (
            "state={state} variant={variant} max_tokens={max_tokens} "
            "offloaded={offloaded} added_slots={added_slots} expand={expand} "
            "running={running} waiting={waiting}"
        ).format(
            state=status.get("state"),
            variant=status.get("runtime_variant"),
            max_tokens=status.get("max_total_num_tokens"),
            offloaded=status.get("offloaded_local_experts"),
            added_slots=status.get("added_kv_slots"),
            expand=status.get("expand_requested"),
            running=status.get("num_running_requests"),
            waiting=status.get("num_waiting_requests"),
        )

    def get_balloon_status(self, recv_req: GetBalloonStatusReqInput):
        status = self.tp_worker.get_balloon_status(recv_req)
        try:
            _, token_usage, _, _ = self._get_token_info()
        except Exception:
            token_usage = getattr(self.stats, "token_usage", 0.0)
        cur_batch = getattr(self, "cur_batch", None)
        last_batch = getattr(self, "last_batch", None)

        def _batch_mode(batch):
            return str(batch.forward_mode) if batch is not None else "None"

        def _batch_size(batch):
            return int(batch.batch_size()) if batch is not None else 0

        status.update(
            {
                "scheduler_status_ts": float(time.time()),
                "scheduler_forward_ct": int(getattr(self, "forward_ct", 0)),
                "scheduler_cur_batch_mode": _batch_mode(cur_batch),
                "scheduler_cur_batch_size": _batch_size(cur_batch),
                "scheduler_last_batch_mode": _batch_mode(last_batch),
                "scheduler_last_batch_size": _batch_size(last_batch),
                "expand_requested": bool(self.expand_requested),
                "expand_request_reason": self.expand_request_reason,
                "num_waiting_requests": len(self.waiting_queue),
                "num_running_requests": len(self.running_batch.reqs),
                "token_usage": float(token_usage or 0.0),
                "gen_throughput": float(
                    getattr(self, "last_gen_throughput", 0.0) or 0.0
                ),
                "scheduler_max_total_num_tokens": int(self.max_total_num_tokens),
                "balloon_keepalive_steps": int(self.balloon_keepalive_step_ct),
                "balloon_keepalive_active": bool(self._balloon_keepalive_active),
                # Phase E telemetry.  Useful for verifying lockstep
                # behavior in failure investigations: if
                # ``phase_e_force_eager_step`` flips True frequently the
                # workload is busy/busy with mismatched padded bs and
                # graph replay is not paying off.
                "phase_e_active": bool(self._kunserve_phase_e_active()),
                "phase_e_force_eager_step": bool(
                    getattr(
                        getattr(self.tp_worker, "model_runner", None),
                        "_balloon_step_force_eager",
                        False,
                    )
                ),
                "phase_e_graph_bs_override": getattr(
                    getattr(self.tp_worker, "model_runner", None),
                    "_balloon_step_graph_bs_override",
                    None,
                ),
                "phase_e_negotiate_interval": int(
                    self._phase_e_negotiate_interval
                ),
                "phase_e_cache_valid": bool(self._phase_e_cache_valid),
                "phase_e_cached_max_bs": int(self._phase_e_cached_max_bs),
                "phase_e_cached_raw_max_bs": int(self._phase_e_cached_raw_max_bs),
                "phase_e_cached_raw_min_bs": int(self._phase_e_cached_raw_min_bs),
                "phase_e_cached_steps_left": int(
                    self._phase_e_cached_steps_left
                ),
            }
        )
        return GetBalloonStatusReqOutput(status=status)

    def prepare_balloon(self, recv_req: PrepareBalloonReqInput):
        logger.info(
            "[KunServeScheduler] prepare_balloon request: target=%s runtime_ep_size=%s "
            "runtime_rank_offset=%s dispatch_rank_offset=%s process_group=%s "
            "comm_backend=%s capture_policy=%s capture_graph=%s",
            recv_req.target_variant,
            recv_req.runtime_ep_size,
            recv_req.runtime_rank_offset,
            recv_req.dispatch_rank_offset,
            recv_req.process_group_name,
            recv_req.kunserve_comm_backend,
            recv_req.capture_policy,
            recv_req.capture_cuda_graph,
        )
        try:
            self._phase_e_reset_cached_decision("prepare_balloon")
            status = self.tp_worker.prepare_balloon(recv_req)
            logger.info(
                "[KunServeScheduler] prepare_balloon success: %s",
                self._format_balloon_status(status),
            )
            return PrepareBalloonReqOutput(
                success=True,
                message="Prepared balloon runtime.",
                status=status,
            )
        except Exception as exc:
            logger.exception(
                "[KunServeScheduler] prepare_balloon failed: target=%s process_group=%s",
                recv_req.target_variant,
                recv_req.process_group_name,
            )
            return PrepareBalloonReqOutput(
                success=False,
                message=str(exc),
                status=self.get_balloon_status(GetBalloonStatusReqInput()).status,
            )

    def warmup_balloon(self, recv_req: WarmupBalloonReqInput):
        logger.info(
            "[KunServeScheduler] warmup_balloon request: target=%s runtime_ep_size=%s "
            "runtime_rank_offset=%s dispatch_rank_offset=%s process_group=%s "
            "comm_backend=%s capture_policy=%s capture_graph=%s",
            recv_req.target_variant,
            recv_req.runtime_ep_size,
            recv_req.runtime_rank_offset,
            recv_req.dispatch_rank_offset,
            recv_req.process_group_name,
            recv_req.kunserve_comm_backend,
            recv_req.capture_policy,
            recv_req.capture_cuda_graph,
        )
        try:
            status = self.tp_worker.warmup_balloon(recv_req)
            logger.info(
                "[KunServeScheduler] warmup_balloon success: %s",
                self._format_balloon_status(status),
            )
            return WarmupBalloonReqOutput(
                success=True,
                message="Warmed up balloon runtime.",
                status=status,
            )
        except Exception as exc:
            logger.exception(
                "[KunServeScheduler] warmup_balloon failed: target=%s process_group=%s",
                recv_req.target_variant,
                recv_req.process_group_name,
            )
            return WarmupBalloonReqOutput(
                success=False,
                message=str(exc),
                status=self.get_balloon_status(GetBalloonStatusReqInput()).status,
            )

    def commit_balloon(self, recv_req: CommitBalloonReqInput):
        logger.info(
            "[KunServeScheduler] commit_balloon request: target=%s offload_local_experts=%s "
            "num_slots_to_expand=%s require_prepared=%s",
            recv_req.target_variant,
            recv_req.offload_local_experts,
            recv_req.num_slots_to_expand,
            recv_req.require_prepared,
        )
        try:
            old_max_total = int(self.max_total_num_tokens)
            status = self.tp_worker.commit_balloon(recv_req)
            self._phase_e_reset_cached_decision("commit_balloon")
            new_max_total = int(status["max_total_num_tokens"])
            self._sync_runtime_capacity_cache(new_max_total)
            self._clear_batch_full_after_capacity_growth(
                "commit_balloon", old_max_total, new_max_total
            )
            self.expand_requested = False
            self.expand_request_reason = None
            logger.info(
                "[KunServeScheduler] commit_balloon success: %s",
                self._format_balloon_status(status),
            )
            return CommitBalloonReqOutput(
                success=True,
                message="Committed balloon runtime.",
                status=status,
            )
        except Exception as exc:
            logger.exception(
                "[KunServeScheduler] commit_balloon failed: target=%s offload_local_experts=%s",
                recv_req.target_variant,
                recv_req.offload_local_experts,
            )
            return CommitBalloonReqOutput(
                success=False,
                message=str(exc),
                status=self.get_balloon_status(GetBalloonStatusReqInput()).status,
            )

    def restore_from_balloon(self, recv_req: RestoreFromBalloonReqInput):
        logger.info(
            "[KunServeScheduler] restore_from_balloon request: require_idle=%s "
            "running=%d waiting=%d",
            recv_req.require_idle,
            len(self.running_batch.reqs),
            len(self.waiting_queue),
        )
        if recv_req.require_idle and not self._is_no_request():
            logger.warning(
                "[KunServeScheduler] restore_from_balloon rejected because scheduler is not idle: "
                "running=%d waiting=%d",
                len(self.running_batch.reqs),
                len(self.waiting_queue),
            )
            return RestoreFromBalloonReqOutput(
                success=False,
                message="restore_from_balloon requires the scheduler to be idle.",
                status=self.get_balloon_status(GetBalloonStatusReqInput()).status,
            )

        try:
            status = self.tp_worker.restore_from_balloon(recv_req)
            self._phase_e_reset_cached_decision("restore_from_balloon")
            self._sync_runtime_capacity_cache(status["max_total_num_tokens"])
            self.expand_requested = False
            self.expand_request_reason = None
            logger.info(
                "[KunServeScheduler] restore_from_balloon success: %s",
                self._format_balloon_status(status),
            )
            return RestoreFromBalloonReqOutput(
                success=True,
                message="Restored from balloon runtime.",
                status=status,
            )
        except Exception as exc:
            logger.exception(
                "[KunServeScheduler] restore_from_balloon failed: require_idle=%s",
                recv_req.require_idle,
            )
            return RestoreFromBalloonReqOutput(
                success=False,
                message=str(exc),
                status=self.get_balloon_status(GetBalloonStatusReqInput()).status,
            )

    def sync_kv_capacity(self, recv_req: SyncKVCapacityReqInput):
        logger.info(
            "[KunServeScheduler] sync_kv_capacity request: max_total_num_tokens=%s delta_slots=%s",
            recv_req.max_total_num_tokens,
            recv_req.delta_slots,
        )
        try:
            old_max_total = int(self.max_total_num_tokens)
            status = self.tp_worker.sync_kv_capacity(recv_req)
            new_max_total = int(status["max_total_num_tokens"])
            self._sync_runtime_capacity_cache(new_max_total)
            self._clear_batch_full_after_capacity_growth(
                "sync_kv_capacity", old_max_total, new_max_total
            )
            logger.info(
                "[KunServeScheduler] sync_kv_capacity success: %s",
                self._format_balloon_status(status),
            )
            return SyncKVCapacityReqOutput(
                success=True,
                message="Synchronized KV capacity.",
                status=status,
            )
        except Exception as exc:
            logger.exception(
                "[KunServeScheduler] sync_kv_capacity failed: max_total_num_tokens=%s delta_slots=%s",
                recv_req.max_total_num_tokens,
                recv_req.delta_slots,
            )
            return SyncKVCapacityReqOutput(
                success=False,
                message=str(exc),
                status=self.get_balloon_status(GetBalloonStatusReqInput()).status,
            )

    def get_internal_state(self, recv_req: GetInternalStateReq):
        ret = vars(get_global_server_args())
        ret["last_gen_throughput"] = self.last_gen_throughput
        ret["memory_usage"] = {
            "weight": round(self.tp_worker.model_runner.weight_load_mem_usage, 2),
            "kvcache": round(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2
            ),
            "token_capacity": int(self.max_total_num_tokens),
            "graph": round(self.tp_worker.model_runner.graph_mem_usage, 2),
        }
        ret["balloon_status"] = self.get_balloon_status(
            GetBalloonStatusReqInput()
        ).status
        ret["effective_max_running_requests_per_dp"] = self.max_running_requests

        if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
            ret["avg_spec_accept_length"] = (
                self.spec_total_num_accepted_tokens / self.spec_total_num_forward_ct
            )

        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.step_time_dict

        # This field is not serializable.
        ret.pop("model_config", None)

        return GetInternalStateReqOutput(internal_state=ret)

    def set_internal_state(self, recv_req: SetInternalStateReq):
        server_args_dict = recv_req.server_args
        args_allow_update = set(
            [
                "pp_max_micro_batch_size",
                "speculative_accept_threshold_single",
                "speculative_accept_threshold_acc",
            ]
        )

        if_success = True
        for k, v in server_args_dict.items():
            if k not in args_allow_update:
                logging.warning(f"Updating {k} is not supported.")
                if_success = False
                break
            elif k == "pp_max_micro_batch_size" and (
                v > self.max_running_requests // self.pp_size or v < 1
            ):
                logging.warning(
                    f"Updating {k} to {v} is rejected because it is out of the valid range [1, {self.max_running_requests // self.pp_size}]."
                )
                if_success = False
                break

        if if_success:
            if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
                avg_spec_accept_length = (
                    self.spec_total_num_accepted_tokens / self.spec_total_num_forward_ct
                )
                logger.info(f"{avg_spec_accept_length=}")
            self.spec_total_num_accepted_tokens = self.spec_total_num_forward_ct = 0
            for k, v in server_args_dict.items():
                setattr(get_global_server_args(), k, v)
            logger.info(f"Global server args updated! {get_global_server_args()=}")
        return SetInternalStateReqOutput(
            updated=True,
            server_args=vars(get_global_server_args()),
        )

    def handle_rpc_request(self, recv_req: RpcReqInput):
        # Handle RPC requests
        logger.info(
            f"handle_rpc_request: {recv_req.method}, param: {recv_req.parameters}"
        )

        success = True
        exec = None
        try:
            func = getattr(self, recv_req.method)
            if recv_req.parameters is not None:
                func(**recv_req.parameters)
            else:
                func()
        except Exception as e:
            success = False
            exec = e
            logger.error(f"Failed to call rpc {recv_req.method}: {str(e)}")

        barrier()
        return RpcReqOutput(success, "" if not exec else str(exec))

    def abort_request(self, recv_req: AbortReq):
        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in reversed(to_del):
            # Abort method 1: directly pop from the queue
            # This only works for requests that have not started anything.
            # We still need to send something back to TokenizerManager to clean up the state.
            req = self.waiting_queue.pop(i)
            if self.enable_hicache_storage:
                # to release prefetch events associated with the request
                self.tree_cache.release_aborted_request(req.rid)
            self.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
            # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                release_kv_cache(req, self.tree_cache)
            # For disaggregation prefill mode, free the metadata buffer index
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                release_req_to_metadata_buffer(
                    req, self.req_to_metadata_buffer_idx_allocator
                )

            # For mamba radix cache
            if (
                req.mamba_pool_idx is not None
                and self.disaggregation_mode != DisaggregationMode.DECODE
            ):
                release_kv_cache(req, self.tree_cache, is_insert=False)
            logger.debug(f"Abort queued request. {req.rid=}")

        # Delete the requests in the grammar queue
        # Abort method 2: call `set_finish_with_abort`
        # The request will still run one prefill forward pass.
        # In this case, we change the input_ids to be only one token to make this prefill cheap.
        self.grammar_manager.abort_requests(recv_req)

        # Delete requests not in the waiting queue when PD disaggregation is enabled
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Abort requests that have not yet been bootstrapped
            for req in self.disagg_prefill_bootstrap_queue.queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort bootstrap queue request. {req.rid=}")
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

            # Abort in-flight requests
            for req in self.disagg_prefill_inflight_queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort inflight queue request. {req.rid=}")
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # Abort requests that have not yet finished preallocation
            for decode_req in self.disagg_decode_prealloc_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort prealloc queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()

            # Abort requests waiting for kvcache to release tree cache
            for decode_req in self.disagg_decode_transfer_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort transfer queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()

            # Abort requests already retracted to CPU cache
            if self.disagg_decode_prealloc_queue.retracted_queue:
                remaining_retracted = []
                for decode_req in self.disagg_decode_prealloc_queue.retracted_queue:
                    if recv_req.abort_all or decode_req.rid.startswith(recv_req.rid):
                        assert hasattr(decode_req, "kv_cache_cpu")
                        del decode_req.kv_cache_cpu
                        self.send_to_tokenizer.send_output(
                            AbortReq(rid=decode_req.rid), decode_req
                        )
                    else:
                        remaining_retracted.append(decode_req)
                self.disagg_decode_prealloc_queue.retracted_queue = remaining_retracted

        # Delete requests in the running batch
        if self.cur_batch is self.running_batch or self.cur_batch is None:
            reqs = self.running_batch.reqs
        else:
            reqs = self.running_batch.reqs + self.cur_batch.reqs

        for req in reqs:
            if not req.finished() and (
                recv_req.abort_all or req.rid.startswith(recv_req.rid)
            ):
                # Abort method 3: set `to_finish`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug(f"Abort running request. {req.rid=}")
                req.to_finish = FINISH_ABORT()

    def _pause_engine(self) -> Tuple[List[Req], int]:
        raise NotImplementedError()

    def pause_generation(self, recv_req: PauseGenerationReqInput):
        self._engine_paused = True

        if self.enable_overlap and self.last_batch:
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            chunked_req_to_exclude = set()
            if recv_req.mode == "in_place":
                if self.chunked_req is not None:
                    chunked_req_to_exclude.add(self.chunked_req)
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            self.running_batch.merge_batch(self.last_batch)

        self.last_batch = None
        self.cur_batch = None

        if recv_req.mode == "retract":
            self.running_batch.filter_batch(v1_spec_info_filtered=True)
            if len(self.running_batch.reqs) != 0:
                retracted_reqs = self.running_batch.retract_all(self.server_args)
                for req in retracted_reqs:
                    self._accumulate_retract_wasted_ms(req)
                    self._add_request_to_queue(req)

            self.running_batch.batch_is_full = False
            self.chunked_req = None

    def continue_generation(self, recv_req: ContinueGenerationReqInput):
        self._engine_paused = False

    def load_lora_adapter(
        self, recv_req: LoadLoRAAdapterReqInput
    ) -> LoadLoRAAdapterReqOutput:
        """In-place loading a new lora adapter from disk or huggingface."""

        result = self.tp_worker.load_lora_adapter(recv_req)
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        """In-place loading a new lora adapter from serialized tensors."""

        result = self.tp_worker.load_lora_adapter_from_tensors(recv_req)
        return result

    def unload_lora_adapter(
        self, recv_req: UnloadLoRAAdapterReqInput
    ) -> UnloadLoRAAdapterReqOutput:
        """Unload the lora adapter."""

        result = self.tp_worker.unload_lora_adapter(recv_req)
        return result

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        """Init the seed and client instance communication group."""
        success, message = self.tp_worker.init_weights_send_group_for_remote_instance(
            recv_req
        )
        return InitWeightsSendGroupForRemoteInstanceReqOutput(success, message)

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        """Send the seed instance weights to the destination instance."""
        success, message = self.tp_worker.send_weights_to_remote_instance(recv_req)
        return SendWeightsToRemoteInstanceReqOutput(success, message)

    def slow_down(self, recv_req: SlowDownReqInput):
        t = recv_req.forward_sleep_time
        if t is not None and t <= 0:
            t = None
        self.forward_sleep_time = t
        return SlowDownReqOutput()

    def expert_distribution_handle(self, recv_req: ExpertDistributionReq):
        action = recv_req.action
        if action == ExpertDistributionReqType.START_RECORD:
            get_global_expert_distribution_recorder().start_record()
        elif action == ExpertDistributionReqType.STOP_RECORD:
            get_global_expert_distribution_recorder().stop_record()
        elif action == ExpertDistributionReqType.DUMP_RECORD:
            get_global_expert_distribution_recorder().dump_record()
        else:
            raise ValueError(f"Unrecognized ExpertDistributionReq value: {recv_req=}")
        return ExpertDistributionReqOutput()

    def open_session(self, recv_req: OpenSessionReqInput):
        # handle error
        session_id = recv_req.session_id
        if session_id in self.sessions:
            logger.warning(f"session id {session_id} already exist, cannot open.")
            return OpenSessionReqOutput(session_id, False)
        elif session_id is None:
            logger.warning("session id is None, cannot open.")
            return OpenSessionReqOutput(session_id, False)
        else:
            self.sessions[session_id] = Session(
                recv_req.capacity_of_str_len, session_id
            )
            return OpenSessionReqOutput(session_id, True)

    def close_session(self, recv_req: CloseSessionReqInput):
        # handle error
        session_id = recv_req.session_id
        if session_id not in self.sessions:
            logger.warning(f"session id {session_id} does not exist, cannot delete.")
        else:
            del self.sessions[session_id]

    def maybe_sleep_on_idle(self):
        if self.idle_sleeper is not None:
            self.idle_sleeper.maybe_sleep()

    def handle_freeze_gc(self, recv_req: FreezeGCReq):
        """Handle freeze_gc request: freeze scheduler's GC and forward to detokenizer."""
        freeze_gc("Scheduler")
        self.send_to_detokenizer.send_output(recv_req, recv_req)
        return None

    def handle_dumper_control(self, recv_req: DumperControlReqInput):
        from sglang.srt.debug_utils.dumper import dumper

        try:
            response: list = []
            if (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                response = dumper._handle_http_control_request(
                    method=recv_req.method, body=recv_req.body
                )
            self.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=True, response=response), recv_req
            )
        except Exception as e:
            print(f"[Scheduler] handle_dumper_control error: {e}", flush=True)
            self.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=False, response=[], error=str(e)),
                recv_req,
            )

    def _log_decode_step_timing(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        step_start: float,
        gap_ms: float,
    ) -> None:
        if not self.batch_timing_log:
            return
        if not batch.forward_mode.is_decode():
            return
        if not self.is_generation:
            return
        if self.pp_rank != 0 or self.attn_tp_rank != 0 or self.attn_cp_rank != 0:
            return

        batch_size = len(batch.reqs)
        if batch_size <= 0:
            return

        step_ms = (time.perf_counter() - step_start) * 1000.0
        effective_tok_per_s_per_req = 1000.0 / (step_ms + gap_ms)
        step_tokens = batch_size  # decode 每 req 通常输出 1 token
        tok_per_s = step_tokens / max(step_ms / 1000.0, 1e-9)
        total_kv_tokens = sum(
            len(req.origin_input_ids) + len(req.output_ids) for req in batch.reqs
        )

        # KunServe BALLOON-state tags. Reading these is cheap (plain attribute
        # access). Tagging per-step lets the analyzer split the step_ms
        # distribution by (balloon_state, batch_size) and compute the
        # cross-replica EP overhead by comparing matched buckets.
        balloon_state = "unknown"
        runtime_variant = "unknown"
        num_offloaded_local_experts = 0
        try:
            mr = getattr(self.tp_worker, "model_runner", None)
            if mr is not None:
                balloon_state = str(getattr(mr, "_balloon_state", "local"))
                num_offloaded_local_experts = int(
                    getattr(mr, "_balloon_offloaded_local_experts", 0) or 0
                )
                # get_cuda_graph_runtime_variant() returns "local"/"global"
                # but is only present if cuda graph capture is enabled.
                getter = getattr(mr, "get_cuda_graph_runtime_variant", None)
                if callable(getter):
                    runtime_variant = str(getter())
        except Exception:
            pass

        # Per-request decode-step counters for the lifecycle log. Cheap, and
        # enables computing per-request BALLOON overhead in post-processing
        # (decode_steps_balloon × avg(step_ms_balloon − step_ms_local @ matched batch_size)).
        for req in batch.reqs:
            if balloon_state == "balloon":
                req._kunserve_decode_steps_balloon = (
                    int(getattr(req, "_kunserve_decode_steps_balloon", 0)) + 1
                )
            else:
                req._kunserve_decode_steps_local = (
                    int(getattr(req, "_kunserve_decode_steps_local", 0)) + 1
                )

        record = {
            "ts": time.time(),
            "replica_rank": _REPLICA_RANK,
            "batch_size": batch_size,
            "total_kv_tokens": int(total_kv_tokens),
            "avg_kv_tokens_per_req": round(total_kv_tokens / max(batch_size, 1), 3),
            "step_ms": round(step_ms, 3),
            "launch_ms": round(step_ms, 3),
            "gap_ms": round(gap_ms, 3),  # ← 新增（很有用，见 Q3）
            "iter_ms": round(step_ms + gap_ms, 3),  # ← 新增（真正的墙钟时间）
            "tok_per_s": round(tok_per_s, 3),
            "tok_per_s_per_req": round(tok_per_s / max(batch_size, 1), 3),
            "forward_mode": str(batch.forward_mode),
            "effective_tok_per_s_per_req": round(effective_tok_per_s_per_req, 3),
            "balloon_state": balloon_state,
            "runtime_variant": runtime_variant,
            "num_offloaded_local_experts": num_offloaded_local_experts,
        }

        try:
            with open(self.batch_timing_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("Failed to write batch timing log: %s", e)

    # placeholder for override
    def update_cache_from_scheduler(
        self, schedule_batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        pass

    def get_remote_instance_transfer_engine_info(self):
        return self.tp_worker.get_remote_instance_transfer_engine_info()


class IdleSleeper:
    """
    In setups which have long inactivity periods it is desirable to reduce
    system power consumption when sglang does nothing. This would lead not only
    to power savings, but also to more CPU thermal headroom when a request
    eventually comes. This is important in cases when multiple GPUs are connected
    as each GPU would otherwise pin one thread at 100% CPU usage.

    The simplest solution is to use zmq.Poller on all sockets that may receive
    data that needs handling immediately.
    """

    def __init__(self, sockets):
        self.poller = zmq.Poller()
        self.last_empty_time = time.time()
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)

        self.empty_cache_interval = envs.SGLANG_EMPTY_CACHE_INTERVAL.get()

    def maybe_sleep(self):
        self.poller.poll(1000)
        if (
            self.empty_cache_interval > 0
            and time.time() - self.last_empty_time > self.empty_cache_interval
        ):
            self.last_empty_time = time.time()
            torch.cuda.empty_cache()


def is_health_check_generate_req(recv_req):
    rid = getattr(recv_req, "rid", None)
    return rid is not None and rid.startswith("HEALTH_CHECK")


def is_work_request(recv_req):
    return isinstance(
        recv_req,
        (
            TokenizedGenerateReqInput,
            TokenizedEmbeddingReqInput,
            BatchTokenizedGenerateReqInput,
            BatchTokenizedEmbeddingReqInput,
        ),
    )


class SenderWrapper:
    def __init__(self, socket: zmq.Socket):
        self.socket = socket

    def send_output(
        self,
        output: Union[BaseReq, BaseBatchReq],
        recv_obj: Optional[Union[BaseReq, BaseBatchReq]] = None,
    ):
        if self.socket is None:
            return

        if (
            isinstance(recv_obj, BaseReq)
            and recv_obj.http_worker_ipc is not None
            and output.http_worker_ipc is None
        ):
            # handle communicator reqs for multi-http worker case
            output.http_worker_ipc = recv_obj.http_worker_ipc

        self.socket.send_pyobj(output)


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    # Generate the logger prefix
    prefix = ""
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
        dp_rank = int(os.environ["SGLANG_DP_RANK"])
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"
    if server_args.attn_cp_size > 1:
        prefix += f" ATTN_CP{attn_cp_rank}"
    if server_args.moe_dp_size > 1:
        prefix += f" MOE_DP{moe_dp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
        )
    if (
        numa_node := server_args.numa_node
    ) is not None and not envs.SGLANG_NUMA_BIND_V2.get():
        numa_bind_to_node(numa_node[gpu_id])

    # Set up tracing
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "Scheduler"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank)

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )
        result_dict = {
            "status": "ready",
            "max_total_num_tokens": scheduler.max_total_num_tokens,
            "max_req_input_len": scheduler.max_req_input_len,
        }
        if server_args.remote_instance_weight_loader_use_transfer_engine():
            (
                remote_instance_transfer_engine_session_id,
                remote_instance_transfer_engine_weights_info_dict,
            ) = scheduler.get_remote_instance_transfer_engine_info()
            result_dict.update(
                {
                    "tp_rank": tp_rank,
                    "remote_instance_transfer_engine_session_id": remote_instance_transfer_engine_session_id,
                    "remote_instance_transfer_engine_weights_info_dict": remote_instance_transfer_engine_weights_info_dict,
                }
            )

        pipe_writer.send(result_dict)

        # Dispatch to the appropriate event loop based on the disaggregation mode
        disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
        if disaggregation_mode == DisaggregationMode.NULL:
            if scheduler.enable_pdmux:
                scheduler.event_loop_pdmux()
            elif server_args.pp_size > 1:
                scheduler.event_loop_pp()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap()
            else:
                scheduler.event_loop_normal()
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            if server_args.pp_size > 1:
                scheduler.event_loop_pp_disagg_prefill()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_prefill()
            else:
                scheduler.event_loop_normal_disagg_prefill()

        elif disaggregation_mode == DisaggregationMode.DECODE:
            if server_args.pp_size > 1:
                scheduler.event_loop_pp_disagg_decode()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_decode()
            else:
                scheduler.event_loop_normal_disagg_decode()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
