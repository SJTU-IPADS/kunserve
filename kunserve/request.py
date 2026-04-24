"""Sampling parameters for text generation."""
from typing import List, Optional, Union, Tuple, Deque
from collections import deque
import time
from enum import Enum
import math

from kunserve.logger import init_logger
from kunserve.utils import Counter
from kunserve.config import (
    ParallelConfig
)
from kunserve.ray_queue import ServerInfo

logger = init_logger(__name__)

class SamplingParams:
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).

    Args:
        n: Number of output sequences to return for the given prompt.
        best_of: Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. This is treated as
            the beam width when `use_beam_search` is True. By default, `best_of`
            is set to `n`.
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        use_beam_search: Whether to use beam search instead of sampling.
        stop: List of strings that stop the generation when they are generated.
            The returned output will not contain the stop strings.
        ignore_eos: Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
        max_tokens: Maximum number of tokens to generate per output sequence.
        logprobs: Number of log probabilities to return per output token.
    """

    _SAMPLING_EPS = 1e-5

    def __init__(
        self,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_beam_search: bool = False,
        stop: Union[None, str, List[str]] = None,
        ignore_eos: bool = False,
        max_tokens: int = 16,
        logprobs: Optional[int] = None,
    ) -> None:
        self.n = n
        self.best_of = best_of if best_of is not None else n
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_beam_search = use_beam_search
        if stop is None:
            self.stop = []
        elif isinstance(stop, str):
            self.stop = [stop]
        else:
            self.stop = list(stop)
        self.ignore_eos = ignore_eos
        self.max_tokens = max_tokens
        self.logprobs = logprobs

        self._verify_args()
        if self.use_beam_search:
            self._verity_beam_search()
        elif self.temperature < self._SAMPLING_EPS:
            # Zero temperature means greedy sampling.
            self._verify_greedy_sampling()

    def _verify_args(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        if self.best_of < self.n:
            raise ValueError(
                f"best_of must be greater than or equal to n, "
                f"got n={self.n} and best_of={self.best_of}."
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                "presence_penalty must be in [-2, 2], got " f"{self.presence_penalty}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                "frequency_penalty must be in [-2, 2], got "
                f"{self.frequency_penalty}."
            )
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(
                f"top_k must be -1 (disable), or at least 1, " f"got {self.top_k}."
            )
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {self.max_tokens}.")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(f"logprobs must be non-negative, got {self.logprobs}.")

    def _verity_beam_search(self) -> None:
        if self.best_of == 1:
            raise ValueError(
                "best_of must be greater than 1 when using beam "
                f"search. Got {self.best_of}."
            )
        if self.temperature > self._SAMPLING_EPS:
            raise ValueError("temperature must be 0 when using beam search.")
        if self.top_p < 1.0 - self._SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using beam search.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using beam search.")

    def _verify_greedy_sampling(self) -> None:
        if self.best_of > 1:
            raise ValueError(
                "best_of must be 1 when using greedy sampling." f"Got {self.best_of}."
            )
        if self.top_p < 1.0 - self._SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using greedy sampling.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using greedy sampling.")

    def __repr__(self) -> str:
        return (
            f"SamplingParams(n={self.n}, "
            f"best_of={self.best_of}, "
            f"presence_penalty={self.presence_penalty}, "
            f"frequency_penalty={self.frequency_penalty}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"use_beam_search={self.use_beam_search}, "
            f"stop={self.stop}, "
            f"ignore_eos={self.ignore_eos}, "
            f"max_tokens={self.max_tokens}, "
            f"logprobs={self.logprobs})"
        )


class TokenOutput:
    """The output of request in one step of inference.
    It contains the information of corresponding request and the generated tokens until this step.
    """

    def __init__(
        self,
        new_token: str,
        new_token_id: int,
        new_token_time: float,
    ):
        self.new_token = new_token
        self.new_token_id = new_token_id
        self.new_token_time = new_token_time

    def __repr__(self) -> str:
        return (
            f"TokenOutput(new_token={self.new_token}, "
            f"new_token_id={self.new_token_id}), "
            f"new_token_time={self.new_token_time}). "
        )
    
class RequestOutput:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: Optional[List[int]],
        outputs: List[TokenOutput],
        start_time: float,
        server_info: ServerInfo,
        finished: bool,
        text: str = None,
    ):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs
        self.start_time = start_time
        self.server_info = server_info
        self.finished = finished
        self.text = text


class Request:
    """A request contains the user's prompt, generated tokens and related information.
    Args:
        arrival_time: the absolute or relative time when the request arrives.
        request_id: the unique identifier for the request.
        prompt: the prompt provided by the user.
        prompt_token_ids: the token ids of the prompt.
        sampling_params: sampling parameters for the request.
        priority: the priority of this request, default is 0.
    """

    def __init__(
        self,
        arrival_time: float,
        request_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams = SamplingParams(),
        server_info: ServerInfo = None,
        priority: int = 0,
        fusion_exec: bool = False,
    ):
        # static states
        self.arrival_time = arrival_time
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.sampling_params = sampling_params
        self.server_info = server_info

        # dynamic states
        self.sched_timestamp = 0.0
        self.last_token_timestamp = 0.0
        self.outputs = [] # token outputs (updated every step)
        self.generated_tokens = []
        self.generated_token_ids = []
        self.is_finished = False
        self.is_running = False
        self.output_capacity = 0        # number of tokens for generated tokens

        # Used in chunked-prefill
        self.prefill_begin_index = 0
        self.prefill_end_index = len(self.prompt_token_ids)
        self.chunk_size = 0

        # Used in layer-by-layer kvcache transfer
        self.target_decoding_engine_rank = -1
        self.target_block_indexes = []
        self.kvcache_ready = True

        # the timestamps of when a request starts its first iteration
        self.start_time = None
        # self.end_time = 0.0

        self.process_time = 0.0
        self.last_step_time = 0.0
        self.execution_time = 0.0

        self.priority = priority
        self.use_mock_block = False
        self.preempted_times = 0
        self.preempting = False
        self.fusion_exec = fusion_exec
        self.once_fusion_exec = False
        self.kv_fusion_rank = 0
        
        self.resched_times = 0

        # Used in migration
        self.is_migrating_out = False
        self.prev_migrated_blocks = 0
        self.migration_dst_engine = 0
        self.allocated_blocks = 0

        # Used in resharding
        self.is_resharding = False
        self.prev_resharded_blocks = 0

        # used in chunked prefill
        self.is_last_chunk = False
        self.is_assigned = False

    def clear_migration_info(self):
        self.is_migrating_out = False
        self.prev_migrated_blocks = 0
        self.migration_dst_engine = 0
        self.allocated_blocks = 0

    def clear_resharding_info(self):
        self.is_resharding = False
        self.prev_resharded_blocks = 0

    def get_priority(self) -> int:
        return self.priority
    
    def set_priority(self, priority: int) -> None:
        self.priority = priority
        
    def set_mock(self):
        if self.use_mock_block:
            # we dont mock a request twice
            return
        self.use_mock_block = True
        
    def prepare_for_recompute(self, fusion_exec: bool = False):
        # self.prompt = self.prompt + ''.join(self.generated_tokens) # only token id is useful
        self.prompt_token_ids.extend(self.generated_token_ids)
        self.sampling_params.max_tokens -= len(self.generated_tokens)
        self.generated_tokens = []
        self.generated_token_ids = []
        self.prefill_begin_index = 0
        self.prefill_end_index = len(self.prompt_token_ids)
        self.fusion_exec = True
        self.kv_fusion_rank = 0
    
    def _check_finish_condition(self):
        if self.get_output_len() >= self.sampling_params.max_tokens:
            self.is_finished = True

        if not self.sampling_params.ignore_eos:
            if self.get_output_len() and (
                self.generated_tokens[-1] in self.sampling_params.stop
            ):
                self.is_finished = True
                
    def used_up_blocks(self) -> bool:
        return self.get_output_len() >= self.output_capacity

    def add_generated_token(self, token: str, token_id: int):
        if self.get_output_len() > self.sampling_params.max_tokens:
            raise ValueError(
                f"The generated tokens ({self.get_output_len()}) exceed the maximum output length {self.sampling_params.max_tokens} for request {self.request_id}."
            )
        self.generated_tokens.append(token)
        self.generated_token_ids.append(token_id)
        self.outputs.append(
            TokenOutput(
                new_token=token,
                new_token_id=token_id,
                new_token_time=time.time(),
            )
        )
        self._check_finish_condition()

    def need_to_responsed(self) -> bool:
        return self.is_finished
        # or TODO: streaming requests

    def is_context_stage(self) -> bool:
        return len(self.generated_tokens) == 0

    def get_input_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        assert len(self.generated_tokens) == len(self.generated_token_ids)
        return len(self.generated_token_ids)
    
    def allocate_next_chunk(self, additional_tokens: int = 0) -> int:
        self.chunk_size += additional_tokens
        self.prefill_begin_index = self.prefill_end_index
        input_len = self.get_input_len()
        self.prefill_end_index = min(
            self.prefill_begin_index + self.chunk_size, 
            input_len
        )
        return self.chunk_size - (self.prefill_end_index - self.prefill_begin_index)
        # TODO: optimze the last chunk to avoid chunk fragementation?
    
    def have_follow_up_chunk(self) -> bool:
        return len(self.generated_tokens) == 0 and self.prefill_end_index < len(self.prompt_token_ids)

    def set_chunk(self, chunk_size: int):
        self.chunk_size = chunk_size
        if chunk_size > 0:
            self.prefill_end_index = min(
                len(self.prompt_token_ids),
                self.prefill_begin_index + chunk_size
            )
        else:
            assert False, f"Fail to set a valid chunk size(current: {chunk_size=}) in chunked prefill!"
        return chunk_size - (self.prefill_end_index - self.prefill_begin_index)

    def get_optimal_chunk_size(self, budget):
        a = 1
        b = 2 * self.prefill_begin_index + 1
        c = -2 * budget
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return 0
        sqrt_d = math.sqrt(discriminant)
        n_max = (-b + sqrt_d) / 2
        cur_chunk_size = math.floor(n_max)
        return max(cur_chunk_size, 1024)
    
    def set_next_chunk(self, chunk_size):
        self.prefill_begin_index = self.prefill_end_index
        return self.set_chunk(chunk_size)
        # logger.info(f"set chunksize of {chunk_size=} for request {self.request_id}")

    def prepare_for_sequence_parallel_chunk(self):
        self.prefill_begin_index = self.prefill_end_index
        self.prefill_end_index = len(self.prompt_token_ids)
            
    def get_num_tokens(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """
        if self.is_context_stage():
            return len(self.prompt_token_ids)
        
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    def get_response(self) -> str:
        return "".join(self.generated_tokens)

    def get_input_tokens_ids(self) -> List[int]:
        """The token ids of the input tokens for the next iteration.
        For request in the context stage, this is equal to the prompt_token_ids.
        For request in the decoding stage, this is equal to the newly generated token.
        """
        if self.is_context_stage():
            return self.prompt_token_ids[self.prefill_begin_index:self.prefill_end_index]
        else:
            # this is generally not true, if speculative decoding is used
            return [self.generated_token_ids[-1]]

    def get_num_input_tokens(self) -> int:
        return len(self.get_input_tokens_ids())
    
    def get_unfinished_prefill_tokens(self) -> int:
        if not self.is_context_stage():
            return 0
        return len(self.prompt_token_ids) - self.prefill_end_index

    def get_first_new_token_index(self) -> int:
        """The index of the first newly generated tokens.
        In the decoding phase, only the input tokens need to compute QKV. The index
        of the first token in the input tokens of next round is needed to do positional
        embedding and decoding phase kernel correctly.

        Note: Currently, the last token in self.output_tokens is the first newly generated token.
        This might not be true if speculative decoding is used.
        """
        # assert self.get_input_len() + self.get_output_len() - 1 < 4096, f"context forward a too long request, p times: {self.preempted_times}, input_len: {self.get_input_len()}, output_len: {self.get_output_len()}"
        
        return (
            self.prefill_begin_index
            if self.is_context_stage()
            else self.get_input_len() + self.get_output_len()
        )

    def get_process_time(self) -> float:
        return self.process_time

    def reset_process_time(self) -> None:
        self.process_time = 0.0

    def add_process_time(self, running_time: float) -> None:
        self.process_time += running_time

    def get_execution_time(self) -> float:
        return self.execution_time

    def reset_execution_time(self) -> None:
        self.execution_time = 0.0

    def add_execution_time(self, running_time: float) -> None:
        self.execution_time += running_time

    def get_kvcache_slots(self) -> float:
        """The number of kvcache slots needed for the request.
        The number of kvcache slots is the total number of tokens in the request.
        """
        return self.get_input_len() + self.get_output_len()

    def __repr__(self) -> str:
        return (
            f"Request(arrival_time = {self.arrival_time}, "
            f"request_id={self.request_id}, "
            f"prompt={self.prompt}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"generated_tokens={self.generated_tokens}, "
            f"generated_token_ids={self.generated_token_ids}, "
            f"is_context_stage={self.is_context_stage()}, "
            f"is_finished={self.is_finished})"
        )

    def __str__(self) -> str:
        return f"Request {self.request_id}: {self.prompt} {self.get_response()}"


class BatchedRequests:
    def __init__(
        self,
        requests: Optional[List[Request]] = None,
        enable_chunked_prefill: bool = False,
    ) -> None:
        if requests is None:
            self.requests = []
        else:
            self.requests = requests
        self.start_time = None
        self.step_method = "step"
        if enable_chunked_prefill:
            for request in self.requests:
                request.set_chunk()

    def __len__(self):
        return len(self.requests)
    
    def __iter__(self):
        return iter(self.requests)

    def __str__(self) -> str:
        return f"BatchedRequests: {self.requests}"

    def __repr__(self) -> str:
        return f"BatchedRequests: {self.requests}"

    def add_request(self, request: Request):
        assert (
            request.request_id not in self.get_request_ids()
        ), f"request {request.request_id} already exists in {self.get_request_ids()}"
        self.requests.append(request)

    def add_requests(self, requests: List[Request]):
        for request in requests:
            self.add_request(request)

    def pop_finished_requests(self) -> List[Request]:
        finished_requests, unfinished_requests = [], []
        for request in self.requests:
            if request.is_finished:
                finished_requests.append(request)
            else:
                unfinished_requests.append(request)
        self.requests = unfinished_requests
        
        return finished_requests

    def pop_finished_remote_attn_requests(self) -> List[Request]:
        finished_requests, unfinished_requests = [], []
        for request in self.requests:
            if request.is_finished and request.fusion_exec == False:
                finished_requests.append(request)
            else:
                unfinished_requests.append(request)
        self.requests = unfinished_requests
        return finished_requests

    def pop_unassigned_requests(self) -> List[Request]:
        unassigned_requests, assigned_requests = [], []
        for request in self.requests:
            # shall be called when request has just finished prefill
            if not request.is_assigned:
                unassigned_requests.append(request)
            else:
                assigned_requests.append(request)
        self.requests = assigned_requests
        return unassigned_requests
    
    def is_running(self) -> bool:
        return self.start_time is not None
    
    def set_sched_time(self):
        cur_time = time.time()
        for request in self.requests:
            request.sched_timestamp = cur_time

    def start_one_iteration(self, start_time):
        """Update the start time of the batch before its execution of iteration."""
        assert self.start_time is None, "the batch has already started one iteration"
        
        if len(self.requests) == 0:
            return
        # assert len(self.requests), "the batch does not have any requests"
        if self.start_time is None:
            self.start_time = start_time
            
        for request in self.requests:
            # set start only for those new requests
            if request.start_time is None:
                request.start_time = start_time

            
    def finish_one_stage():
        """This method is used to finish one stage in fusion"""
        pass

    def finish_one_iteration(
        self,
        generated_tokens: List[str],
        generated_tokens_ids: List[int],
        end_time: float,
    ):
        """Update the requests in the batch after it finishes one iteration
        Note: the order of generated tokens should align with self.requests.
        """
        if len(self.requests) == 0:
            return
        assert self.start_time is not None, "the batch has not been started"
        for request, generated_token, generated_token_id in zip(
            self.requests, generated_tokens, generated_tokens_ids
        ):
            request.last_step_time = end_time
            request.add_process_time(end_time - self.start_time)
            if not request.have_follow_up_chunk():  # if chunked prefill has not finished, should not add generate token
                request.add_generated_token(generated_token, generated_token_id)
            # else:
                # logger.info(f"\trefuse to add request for chunked prefill, {request.prefill_begin_index=}, {request.prefill_end_index=}")
        self.start_time = None

    #### General Getters
    def get_request_ids(self) -> List[int]:
        return [request.request_id for request in self.requests]

    def get_kvcache_slots(self) -> int:
        return sum([request.get_kvcache_slots() for request in self.requests])

    def get_unfinished_prefill_tokens(self) -> int:
        return sum([request.get_unfinished_prefill_tokens() for request in self.requests])
    
    def get_context_requests(self) -> List[Request]:
        return [request for request in self.requests if request.is_context_stage()]
    
    def get_decode_requests(self) -> List[Request]:
        return [request for request in self.requests if not request.is_context_stage()]
    
    def split_requests(self) -> Tuple[List[Request], List[Request]]:
        prefill_requests, decode_requests = [], []
        for request in self.requests:
            if request.is_context_stage():
                prefill_requests.append(request)
            else:
                decode_requests.append(request)
        return prefill_requests, decode_requests

    def get_fusion_requests(self) -> Tuple[List[Request], List[Request]]:
        return ([request for request in self.requests if request.fusion_exec], 
                [request for request in self.requests if not request.fusion_exec])

    #### Getters for the GPT operator parameters ####
    def get_input_tokens_batched(self) -> List[List[int]]:
        return [request.get_input_tokens_ids() for request in self.requests]
    
    def get_num_input_tokens_batched(self) -> List[int]:
        return [request.get_num_input_tokens() for request in self.requests]
    
    def get_num_input_tokens(self) -> int:
        return sum([request.get_num_input_tokens() for request in self.requests])

    def get_first_token_indexes(self) -> List[int]:
        return [request.get_first_new_token_index() for request in self.requests]

    def get_is_context_stage(self, use_tensor_cores: bool=False) -> List[int]:
        if use_tensor_cores:
            return [1] * len(self.requests)
        return [int(request.is_context_stage()) for request in self.requests]


def create_request(
    prompt: Optional[str],
    prompt_token_ids: Optional[List[str]],
    sampling_params: SamplingParams,
    server_info: Optional[ServerInfo],
    request_counter: Counter,
    tokenizer,
    max_model_len: int = 16384,
    arrival_time: Optional[float] = None,
    request_id: Optional[int] = None,
) -> Request:
    if request_id is None:
        request_id = next(request_counter)
    if prompt_token_ids is None:
        assert prompt is not None
        prompt_token_ids = tokenizer.encode(prompt)
    if prompt is None:
        assert prompt_token_ids is not None
        prompt = tokenizer.decode(prompt_token_ids)

    # truncate prompt if it is too long for the model
    if len(prompt_token_ids) + sampling_params.max_tokens > max_model_len:
        prompt_token_ids = prompt_token_ids[: max_model_len - sampling_params.max_tokens]

    if arrival_time is None:
        arrival_time = time.time()
    
    return Request(
        arrival_time,
        request_id,
        prompt,
        prompt_token_ids,
        sampling_params,
        server_info=server_info,
    )


class MigratingRequest:
    """
    MigratingRequest: elements in the "bridge" queue.
    
    Each MigratingRequest represents a request that:
      - Has finished the context stage
      - Has not yet acceptted by the decoding stage
      - Its block is still on context stage's GPU memory (i.e. migration needed)
      
    Those requests are produced by ContextStageLLMEngine, queued in the "bridge"
    queue, and finally consumed by DecodingStageLLMEngine, which forms a
    producer-consumer pattern.
    
    For more information about the design & implementation of disaggregation,
    please refer to engine.py.
    """
    
    def __init__(
        self,
        req: Request,
        block_indexes: List[int],
        context_parallel_config: ParallelConfig,
    ):
        self.req = req
        self.block_indexes = block_indexes
        self.context_parallel_config = context_parallel_config


class AttentionRequest:
    def __init__(
        self,
        req: Request,
        block_indexes: List[int],
        context_parallel_config: ParallelConfig,
    ):
        self.req = req
        self.block_indexes = block_indexes
        self.context_parallel_config = context_parallel_config
        
        