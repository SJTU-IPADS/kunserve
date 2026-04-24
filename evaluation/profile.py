import ray
import copy

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')

import argparse

from kunserve.setup import connect_to_ray_cluster, setup_kunserve
from kunserve.utils import get_log_name

from kunserve.kunserve_config import *
import asyncio
import pandas as pd

from evaluation.trace import *

from kunserve.logger import init_logger
logger = init_logger(__name__)

    
def benchmark(bench_config: BenchConfig):
    # TODO: read from dataset
    requests = list(get_prompts(
        bench_config.dataset, 
        num_prompts=bench_config.nlimit, 
        offset=bench_config.dataset_offset, 
        client_id=0,
        shuffle=bench_config.shuffle_dataset
    ))
    
    processed_requests = []
    model_names = {}
    for req in requests:
        prompt_len, response_len = req.prompt_len, req.response_len
        if req.prompt_len + req.response_len > 16384:
            prompt_len = 16384 - response_len

        # if req.model_name == "ExpertQA":
        processed_requests.append((prompt_len, response_len))

        # if req.model_name not in model_names:
        #     model_names[req.model_name] = 0
        # model_names[req.model_name] += 1

    prompt_lens = np.array([req[0] for req in processed_requests])
    output_lens = np.array([req[1] for req in processed_requests])
    print(f"{model_names=}")

    print(
        f"******************Profiler report begin**********************\n"
        f"Number of prompt is {len(prompt_lens)}.\n"
        f"Average prompt length is {prompt_lens.mean()} tokens, minimal is {prompt_lens.min()}, maximal is {prompt_lens.max()}.\n"
        f"Average output length is {output_lens.mean()} tokens, minimal is {output_lens.min()}, maximal is {output_lens.max()}.\n"
        f"******************Profiler report end**********************"
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    cli_args = parser.parse_args()
    kunserve_config: KunServeConfig = get_cli_args(cli_args)
    benchmark(kunserve_config.bench_config)