import torch

from kunserve.config import ColocatedSchedConfig, ContextStageSchedConfig, DecodingStageSchedConfig, ModelConfig, ParallelConfig, CacheConfig
from kunserve.logger import init_logger

logger = init_logger(__name__)

def get_model_op(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
    sched_config: (
      ContextStageSchedConfig | DecodingStageSchedConfig | ColocatedSchedConfig
    ),
    num_gpu_blocks,
):
    model_name = model_config.model
    model_type = model_config.hf_config.model_type
    
    logger.info(f"Model dtype is {model_config.get_torch_dtype()}, model type is {model_type}, model config is\n{model_config.hf_config}")
    
    # match model_type:
      # case 'opt':
      #   logger.info(f"Using OPT model, max model length is {model_config.get_max_model_len()}")
      #   return torch.classes.gpt_ops.OptOp(
      #       model_config.hf_config.vocab_size,
      #       model_config.get_max_model_len(),
      #       model_config.get_hidden_size(),
      #       model_config.get_num_layers(),
      #       model_config.get_num_heads(),
      #       model_config.get_head_size(),
      #       model_config.dtype,
      #       cache_config.block_size,
      #       cache_config.max_num_blocks_per_req,
      #       parallel_config.to_list(),
      #   )
      # case 'gpt2':
      #   return torch.classes.gpt_ops.Gpt2Op(
      #       model_config.hf_config.vocab_size,
      #       model_config.get_max_model_len(),
      #       model_config.get_hidden_size(),
      #       model_config.get_num_layers(),
      #       model_config.get_num_heads(),
      #       model_config.get_head_size(),
      #       model_config.dtype,
      #       cache_config.block_size,
      #       cache_config.max_num_blocks_per_req,
      #       parallel_config.to_list(),
      #   )
      # case _ if 'llama' in model_type:
    logger.info(f"Using LlamaOp, max model length is {model_config.get_max_model_len()}, head_size is {model_config.get_head_size()}, inter size is {model_config.get_ffn_inter_dim()}")  
    return torch.classes.gpt_ops.LlamaOp(
      num_gpu_blocks,                       # max_num_pages
      model_config.get_num_layers(parallel_config),        # layers
      model_config.get_q_heads(parallel_config),           # qo_head
      model_config.get_num_heads(parallel_config),         # kv_head
      cache_config.block_size,              # page_size
      model_config.get_head_size(),         # head_size
      model_config.get_ffn_inter_dim(),     # inter_size
      model_config.hf_config.vocab_size,    # vocab_size
      sched_config.max_batch_size,          # max_batch_size
      sched_config.max_tokens_per_batch,    # max_batch_tokens
      model_config.get_max_model_len(),     # max_position_embeddings
      parallel_config.tensor_parallel_size, # tensor_parallel_size
    )
      # case 'qwen2':
      #   logger.info(f"Using Qwen2Op, max model length is {model_config.get_max_model_len()}")
        
      #   return torch.classes.gpt_ops.Qwen2Op(
      #       model_config.hf_config.vocab_size,
      #       model_config.get_max_model_len(),
      #       model_config.get_hidden_size(),
      #       model_config.get_num_layers(),
      #       model_config.get_q_heads(),
      #       model_config.get_num_heads(),
      #       model_config.get_head_size(),
      #       model_config.get_ffn_inter_dim(),
      #       model_config.get_layernorm_eps(),
      #       model_config.dtype,
      #       cache_config.block_size,
      #       cache_config.max_num_blocks_per_req,
      #       parallel_config.to_list(),
      #   )
      # case 'chatglm':
      #   logger.info(f"Using Llama2Op, max model length is {model_config.get_max_model_len()}")
        
      #   return torch.classes.gpt_ops.GlmOp(
      #     model_config.hf_config.vocab_size,
      #       model_config.get_max_model_len(),
      #       model_config.get_hidden_size(),
      #       model_config.get_num_layers(),
      #       model_config.get_q_heads(),
      #       model_config.get_num_heads(),
      #       model_config.get_head_size(),
      #       model_config.get_ffn_inter_dim(),
      #       model_config.dtype,
      #       cache_config.block_size,
      #       cache_config.max_num_blocks_per_req,
      #       parallel_config.to_list(),
      #   )
    raise NotImplementedError(f"model {model_name} not supported")
