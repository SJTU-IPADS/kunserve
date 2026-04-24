from dataclasses import asdict

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser
from vllm.inputs import TokensPrompt

import os
import pandas as pd
import random
import time

import math

def get_prompts(dataset_path, num_prompts: int, offset: int, client_id: int = 0):
    dataset = pd.read_parquet(dataset_path)
    start = offset + client_id * num_prompts

    slice_df = dataset[dataset["prompt_len"] < 16384].iloc[start : start + num_prompts]

    if len(slice_df) < num_prompts:
        repeat_times = int(math.ceil(num_prompts / len(slice_df)))
        slice_df = pd.concat([slice_df] * repeat_times, ignore_index=True)
        slice_df = slice_df.iloc[:num_prompts]
    
    return list(slice_df["prompt"])

def main(args):
    # Create prompts
    prompts = get_prompts(args.dataset, args.num_prompts, offset=0)
    # prompts = ["what is large language model? please introduce briefly"]

    # # Create a sampling params object.
    # sampling_params = SamplingParams(n=args.n,
    #                                  temperature=args.temperature,
    #                                  top_p=args.top_p,
    #                                  top_k=args.top_k,
    #                                  max_tokens=args.max_tokens,
    #                                  ignore_eos=True)

    # Create an LLM.
    # The default model is 'facebook/opt-125m'
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**asdict(engine_args))

    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    start = time.time()
    outputs = llm.generate(prompts, SamplingParams(max_tokens=8192))
    end = time.time()
    generated_time = end - start

    # Print the outputs.
    final_dataset = []
    total_prompt_tokens = 0
    total_output_tokens = 0
    for output in outputs:

        prompt_tokens = output.prompt_token_ids
        total_prompt_tokens += len(prompt_tokens)

        generated_tokens = output.outputs[0].token_ids
        total_output_tokens += len(generated_tokens)

        # print(f"prompt: {output.prompt}, {output.outputs[0].text}, {len(generated_tokens)} tokens")

        final_dataset.append({
            "prompt": output.prompt,
            "prompt_token_ids": prompt_tokens,
            "prompt_len": len(prompt_tokens),
            "response_len": len(generated_tokens),
        })
    
    final_report = f"**Evaluation report**\nIt cost {generated_time:.3f}s to generate {total_output_tokens} tokens " \
        + f"for {total_prompt_tokens} input tokens (from {len(outputs)} requests), " \
        + f"i.e., {total_output_tokens / generated_time:.0f} tokens/s." \
        + "\n**End of report**" 
    print(final_report)

    final_df = pd.DataFrame(final_dataset)
    final_df.to_parquet(args.save_path)
    print(f"Save {len(final_dataset)} results to {args.save_path}.")


if __name__ == '__main__':
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument("--dataset",
                       type=str,
                       default="/mnt/bn/crx-lq/data/longbenchcode-llama/train.parquet",
                       help="Source of prompts used for inference")
    parser.add_argument("--num-prompts",
                       type=int,
                       default=800,
                       help="Number of prompts used for inference")
    parser.add_argument("--save-path",
                       type=str,
                       default="/mnt/bn/crx-lq/data/longbenchcode-llama/gen.parquet",
                       help="Save path for the generated results")

    args = parser.parse_args()
    main(args)