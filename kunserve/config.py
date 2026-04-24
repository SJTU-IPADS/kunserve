from typing import Optional, List

import torch
import sys
from transformers import AutoConfig

from kunserve.utils import GB


class CacheConfig:
    """Configuration for the key-value cache.

    Args:
        block_size: Number of tokens in a block.
        gpu_memory_utilization: The maximum percentage of GPU memory that can be used.
        cpu_swap_space: The maximum CPU swap space in bytes that can be used.
    """

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: int = 0.9,
        decoding_gpu_memory_utilization: int = 0.9,
        cpu_swap_space: int = 0,
        kv_cache_ratio: float = 1.0,
        enable_mock: bool = False,
        max_preemption_cpu_blocks_usage: float = 0.8,
        reserved_blocks_ratio: float = 0.1,
        reserved_blocks_ratio_for_resharding: float = 0.05,
        max_allocation_times: int = 1,
    ):
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.decoding_gpu_memory_utilization = decoding_gpu_memory_utilization
        self.cpu_swap_space = cpu_swap_space * GB
        self.kv_cache_ratio = kv_cache_ratio
        self.enable_mock = enable_mock
        self.max_preemption_cpu_blocks_usage = max_preemption_cpu_blocks_usage
        self.reserved_blocks_ratio = reserved_blocks_ratio
        self.reserved_blocks_ratio_for_resharding = reserved_blocks_ratio_for_resharding
        self.max_allocation_times = max_allocation_times


class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        tensor_parallel_size: number of tensor parallel groups.
        tensor_parallel_rank: rank in the tensor parallel group.
        pipeline_parallel_size: number of pipeline parallel groups.
        pipeline_parallel_rank: rank in the pipeline parallel group.
        replica_size: number of replicas
        replica_rank: rank of this replica
        transfer_layer_by_layer: whether to enable this feature
    """

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        pipeline_parallel_size: int = 1,
        pipeline_parallel_rank: int = 0,
        group_parallel_size: int = 1,
        group_parallel_rank: int = 0,
        replica_size: int = 1,
        replica_rank: int = 0,
        transfer_layer_by_layer: int = 0,
        num_virtual_engine: int = 1,
        global_rank: int = 0,
        global_size: int = 1,
    ) -> None:
        self.tensor_parallel_size = tensor_parallel_size
        self.tensor_parallel_rank = tensor_parallel_rank
        self.pipeline_parallel_size = pipeline_parallel_size
        self.pipeline_parallel_rank = pipeline_parallel_rank
        self.group_parallel_size = group_parallel_size
        self.group_parallel_rank = group_parallel_rank
        self.replica_size = replica_size
        self.replica_rank = replica_rank
        self.transfer_layer_by_layer = transfer_layer_by_layer
        self.num_virtual_engine = num_virtual_engine

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        self.use_parallel = self.world_size > 1

        self.global_rank = global_rank
        self.global_size = global_size

    def to_list(self) -> List[int]:
        return [
            self.tensor_parallel_size,
            self.tensor_parallel_rank,
            self.pipeline_parallel_size,
            self.pipeline_parallel_rank,
            self.group_parallel_size,
            self.group_parallel_rank,
            self.replica_size,
            self.replica_rank,
        ]

    def is_last_stage(self) -> bool:
        return self.pipeline_parallel_rank == self.pipeline_parallel_size - 1


class DisaggParallelConfig:
    """Configuration for disaggregated execution.

    Args:
        context: Context stage parallel config
        decoding: Decoding stage parallel config
    """

    def __init__(
            self,
            context: ParallelConfig = ParallelConfig(),
            decoding: ParallelConfig = ParallelConfig(),
    ) -> None:
        self.context = context
        self.decoding = decoding

    def get_num_workers(self) -> int:
        """Get the total number of workers (GPUs) needed."""
        return self.context.world_size + self.decoding.world_size


class ContextStageSchedConfig:
    """Configuration for the context stage scheduler.

    Args:
        policy: The scheduling policy.
        max_batch_size: The maximum number of requests in a batch.
        max_tokens_per_batch: The maximum number of input tokens in a batch.
    """

    def __init__(
            self,
            policy: str,
            max_batch_size: int,
            max_tokens_per_batch: int,
            enable_chunked_prefill: bool,
            chunked_prefill_size: int,
            first_token_slo: int,
            enable_reject: bool,
            parallel_config: ParallelConfig = None,
    ):
        assert policy in [
            "fcfs"
        ], f"policy {policy} not supported"
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.enable_chunked_prefill = enable_chunked_prefill
        self.chunked_prefill_size = chunked_prefill_size
        self.first_token_slo = first_token_slo
        self.enable_reject = enable_reject
        self.parallel_config = parallel_config


class DecodingStageSchedConfig:
    """Configuration for the decoding stage scheduler.

    Args:
        policy: The scheduling policy.
        max_batch_size: The maximum number of requests in a batch.
        max_tokens_per_batch: The maximum number of input tokens in a batch.
    """

    def __init__(
            self,
            policy: str,
            preempt_method: str,
            max_batch_size: int,
            max_tokens_per_batch: int,
            max_preempted_times: int,
            enable_reject: bool,
            model_name: str = None,
            waiting_block_prop_threshold: float = 0.05,
            parallel_config: ParallelConfig = None,
    ):
        assert policy in [
            "fcfs",
            "srpt",
            "mlfq",
            "sj-mlfq",
        ], f"policy {policy} not supported"
        assert preempt_method in [
            "swap", "recompute",
        ], f"preemption method {preempt_method} not supported"
        self.policy = policy
        self.preempt_method = preempt_method
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_preempted_times = max_preempted_times
        self.enable_reject = enable_reject
        self.model_name = model_name
        self.waiting_block_prop_threshold = waiting_block_prop_threshold
        self.parallel_config = parallel_config


class DisaggSchedConfig:
    """Configuration for the disaggregated scheduler.

    Args:
        context_sched_config: The context stage scheduling configuration.
        decoding_sched_config: The decoding stage scheduling configuration.
    """

    def __init__(
            self,
            context: ContextStageSchedConfig,
            decoding: DecodingStageSchedConfig,
    ):
        self.context = context
        self.decoding = decoding


class ColocatedSchedConfig:
    """Configuration for the colocation scheduler.

        Args:
            policy: The scheduling policy.
            max_batch_size: The maximum number of requests in a batch.
            max_tokens_per_batch: The maximum number of input tokens in a batch.
        """

    def __init__(
            self,
            policy: str,
            preempt_method: str,
            max_batch_size: int,
            max_tokens_per_batch: int,
            enable_chunked_prefill: bool = False,
            chunked_prefill_size: int = 0,
            chunked_prefill_budget: int = 0,
            model_name: str = None,
            use_tensor_cores: bool = False,
            parallel_config: ParallelConfig = None,
    ):
        assert preempt_method in [
            "swap", "recompute",
        ], f"preemption method {preempt_method} not supported"
        self.policy = policy
        self.preempt_method = preempt_method
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.enable_chunked_prefill = enable_chunked_prefill
        self.chunked_prefill_size = chunked_prefill_size
        self.chunked_prefill_budget = chunked_prefill_budget 
        self.model_name = model_name
        self.use_tensor_cores = use_tensor_cores
        self.parallel_config = parallel_config


_TORCH_DTYPE_MAP = {"fp16": torch.half, "bf16": torch.bfloat16, "fp32": torch.float32}


class ModelConfig:
    """Configuration for the model.

    Args:
        model: Model name or path.
        tokenizer: Tokenizer name or path.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, and "slow" will always use the slow tokenizer.
            Default to "auto".
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        dtype: Data type of the model. Default to "fp16".
        seed: Random seed for reproducing.
    """

    def __init__(
            self,
            model: str,
            tokenizer: Optional[str],
            tokenizer_mode: str = "auto",
            trust_remote_code: bool = False,
            dtype: str = "fp16",
            seed: int = 1,
            use_dummy_weights: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer if tokenizer else model
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.seed = seed
        self.dtype = dtype
        self.hf_config = self._get_hf_config()
        self._verify_args()
        self.use_dummy_weights = use_dummy_weights

    def _verify_args(self):
        assert self.dtype in [
            "fp16", "bf16", "fp32"
        ], f"dtype must be 'fp16', 'bf16' or 'fp32', current {self.dtype}."
    
    def abandon_hf_config(self):
        """some hf_config is not serialiable, we abandon it to prepare for ray init"""
        self.hf_config = None
        
    def load_hf_config(self):
        self.hf_config = self._get_hf_config()

    def _get_hf_config(self):
        try:
            config = AutoConfig.from_pretrained(
                self.model, trust_remote_code=self.trust_remote_code
            )
        except:
            raise ValueError(
                f"Failed to load the model config, please check the model name or path: {self.model}"
            )
        if getattr(config, "text_config", None) is not None: # multimodal model
            config = getattr(config, "text_config", None)
        return config

    def get_dtype_size(self) -> int:
        if self.dtype == "fp16" or self.dtype == "bf16":
            return 2
        elif self.dtype == "fp32":
            return 4
        else:
            raise NotImplementedError(f"dtype {self.dtype} not supported")

    def get_torch_dtype(self) -> torch.dtype:
        return _TORCH_DTYPE_MAP[self.dtype]

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_head_size(self) -> int:
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_ffn_inter_dim(self) -> int:
        # For LLaMA-2:
        return self.hf_config.intermediate_size
    
    def get_layernorm_eps(self) -> float:
        return self.hf_config.rms_norm_eps

    def get_q_heads(self, parallel_config: ParallelConfig = ParallelConfig()) -> int:
        # For LLaMA-2:
        return (
                self.hf_config.num_attention_heads
                // parallel_config.tensor_parallel_size
        )

    def get_num_heads(self, parallel_config: ParallelConfig = ParallelConfig()) -> int:
        # For GPTBigCode & Falcon:
        # Note: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        new_decoder_arch_falcon = self.hf_config.model_type == "falcon" and getattr(
            self.hf_config, "new_decoder_architecture", False
        )
        if not new_decoder_arch_falcon and getattr(
                self.hf_config, "multi_query", False
        ):
            # Multi-query attention, only one KV head.
            return 1

        # For Falcon:
        if getattr(self.hf_config, "n_head_kv", None) is not None:
            return self.hf_config.n_head_kv // parallel_config.tensor_parallel_size

        # For LLaMA-2:
        if getattr(self.hf_config, "num_key_value_heads", None) is not None:
            return (
                self.hf_config.num_key_value_heads // parallel_config.tensor_parallel_size
            )
        
        # For ChatGLM (w/ MQA):
        if getattr(self.hf_config, "multi_query_group_num", None) is not None:
            return (
                self.hf_config.multi_query_group_num // parallel_config.tensor_parallel_size
            )

        # Normal case:
        total_num_attention_heads = self.hf_config.num_attention_heads
        assert total_num_attention_heads % parallel_config.tensor_parallel_size == 0, (
            f"Total number of attention heads ({total_num_attention_heads}) "
            f"must be divisible by the size of tensor parallel group "
            f"({parallel_config.tensor_parallel_size})."
        )
        return total_num_attention_heads // parallel_config.tensor_parallel_size

    def get_max_model_len(self) -> int:
        max_model_len = float("inf")
        possible_keys = [
            # OPT
            "max_position_embeddings",
            # GPT-2
            "n_positions",
            # MPT
            "max_seq_len",
            # ChatGLM
            "seq_length",
            # Others
            "max_sequence_length",
            "max_seq_length",
            "seq_len",
        ]
        for key in possible_keys:
            max_len_key = getattr(self.hf_config, key, None)
            if max_len_key is not None:
                max_model_len = min(max_model_len, max_len_key)
        return max_model_len

    def get_num_layers(self, parallel_config: ParallelConfig = ParallelConfig()) -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        assert total_num_hidden_layers % parallel_config.pipeline_parallel_size == 0, (
            f"Number of layers ({total_num_hidden_layers}) must be divisible "
            f"by the size of pipeline parallel group "
            f"({parallel_config.pipeline_parallel_size})."
        )
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size

    def get_total_params(self, parallel_config: ParallelConfig = ParallelConfig()) -> int:
        return (
                self.hf_config.vocab_size * self.get_hidden_size()  # vocab embed
                + self.get_max_model_len() * self.get_hidden_size()  # position embed
                + 4
                * self.get_num_layers(parallel_config)
                * (self.get_hidden_size() ** 2)  # attention
                / parallel_config.tensor_parallel_size  # attention is divided by tp
                + 8
                * self.get_num_layers(parallel_config)
                * (self.get_hidden_size() ** 2)  # FFN
                / parallel_config.tensor_parallel_size  # FFN is divided by tp
                + 5 * self.get_num_layers(parallel_config) * self.get_hidden_size()  # bias
        )

    def get_model_size_in_bytes(
            self, parallel_config: ParallelConfig = ParallelConfig()
    ) -> int:
        return self.get_total_params(parallel_config) * self.get_dtype_size()

    def get_layer_size_in_bytes(
            self, parallel_config: ParallelConfig = ParallelConfig()
    ) -> int:
    # (2 * num_qo_head + 2 * num_kv_head) * head_size * hidden_size // QKVO
    #                                     + 2 * hidden_size                                                                // RMSNorm
    #                                     + 3 * hidden_size * inter_size
        return (
            (2 * self.get_q_heads(parallel_config) + 2 * self.get_num_heads(parallel_config)) * self.get_head_size() * self.get_hidden_size()
            + 2 * self.get_hidden_size()
            + 3 * self.get_hidden_size() * self.get_ffn_inter_dim()
        ) // parallel_config.tensor_parallel_size * self.get_dtype_size()