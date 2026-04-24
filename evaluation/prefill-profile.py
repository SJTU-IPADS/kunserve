import random
import ray

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

    
async def benchmark(llm_server, bench_config: BenchConfig, cfg_file: str):
    # TODO: read from dataset

    decode_length=1

    # add a batch of prefill requests (to simulate queuing)
    length_tuples = [
        [512] * 16, # warmup
        # [256] * 32,
        [512] * 16,
        [1024] * 8,
        [512, 512, 512, 512, 1024, 1024, 2048, 2048],
        [2048] * 4,
        [1024, 1024, 1024, 1024, 4096],
        [2048, 2048, 4096],
        # [4096] * 2,
        # [1024] * 1,
        # [2048] * 1,
        # [4096] * 1,
        # [6144] * 1,
        [8192] * 1,
    ]

    prefix_lens = [0, 100, 500, 1000, 2000, 3000, 4000, 8000]

    print(f"Start benchmark for {length_tuples}")

    ttft_results = {}

    for prefix_len in prefix_lens:
        ttft_results[prefix_len] = []

    for length_tuple in length_tuples:
        for prefix_len in prefix_lens:
            
            if len(length_tuple) > 1 and prefix_len > 0:
                continue
            
            if prefix_len >= max(length_tuple):
                continue

            start_time = time.perf_counter()
            batch_prompts = [""] * len(length_tuple)
            batch_prompt_token_ids = [
                [random.randint(0, 16384) for i in range(context_len)]
                for context_len in length_tuple
            ]
            new_task = llm_server.batch_generate.remote(
                batch_prompts=batch_prompts,
                batch_prompt_token_ids=batch_prompt_token_ids,
                prefix_len=prefix_len,
                max_tokens=decode_length,
            )
            responses = await new_task 
            dur_s = time.perf_counter() - start_time
            print(f"Benchmark for {length_tuple=} terminates after {dur_s}s.")
        
            outputs = [resp["outputs"] for resp in responses]
            output_lens = np.array([len(output) for output in outputs])
            metrics = [resp["metrics"] for resp in responses]
            queues = np.array([metric["queue"] for metric in metrics])
            ttfts = np.array([metric["ttft"] for metric in metrics])
            tbts = np.array([metric["tbt"] for metric in metrics])

            print(
                f"******************Benchmark report for {length_tuple} begin**********************\n"
                f"The configuration file is {cfg_file}.\n"
                f"Finish {len(responses)} generation tasks in {dur_s}s.\n"
                f"Mean queue time is {queues.mean()}s, P50 is {np.percentile(queues, 50):.3f} s, P90 is {np.percentile(queues, 90):.3f} s, P99 is {np.percentile(queues, 99):.3f} s.\n"
                f"Mean TTFT is {ttfts.mean()}s, P50 is {np.percentile(ttfts, 50):.3f} s, P90 is {np.percentile(ttfts, 90):.3f} s, P99 is {np.percentile(ttfts, 99):.3f} s.\n"
                f"Mean TBT is {tbts.mean()}s, P50 is {np.percentile(tbts, 50):.3f} s, P90 is {np.percentile(tbts, 90):.3f} s, P99 is {np.percentile(tbts, 99):.3f} s.\n"
                f"Average generation throughput is {output_lens.sum() / dur_s:.2f} tokens/s, {len(responses) / dur_s:.2f} req/s.\n"
                f"******************Benchmark report for {length_tuple} end**********************"
            )

            ttft_results[prefix_len].append(ttfts.mean())

    for prefix_len in prefix_lens:
        for i, ttft_result in enumerate(ttft_results[prefix_len]):
            print(f"Under prefix {prefix_len}, for tuple {i}, ttft result is {ttft_result * 1000:.0f} ms")
    try:
        await llm_server.terminate.remote()
    except Exception as e:
        print(f"Error occurred when terminating the server: {e}")
    ray.kill(llm_server)

    # metrics_df = pd.DataFrame(metrics)
    # metric_log_name = get_log_name(bench_config.log_path, bench_config.qps, bench_config.cv, bench_config.dist) + ".parquet"
    # metrics_df.to_parquet(metric_log_name)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    cli_args = parser.parse_args()
    """ create LLM servers equal to the number of instance groups """
    connect_to_ray_cluster()
    kunserve_config: KunServeConfig = get_cli_args(cli_args)
    llm_servers = setup_kunserve(kunserve_config)
    asyncio.run(benchmark(llm_servers[0], kunserve_config.bench_config, cli_args.config))