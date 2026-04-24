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

    decode_length=200

    num_ongoing = 40
    ongoing_context_length = 1024
    
    # prepare some dummy decode requests
    # ongoing_requests = []
    # for _ in range(num_ongoing):
    #     ongoing_requests.append(
    #         llm_server.generate.remote(
    #             prompt="",
    #             prompt_token_ids=[random.randint(0, 16384) for i in range(ongoing_context_length)],
    #             max_tokens=decode_length,
    #         )
    #     )
    # assume only on instance in this micro bench

    check_pending_num = ray.get(llm_server.check_pending_num.remote(0, 0))
    while check_pending_num > 0:
        print(f"Waiting for {check_pending_num} requests to enter decode phase.")
        await asyncio.sleep(1)
        check_pending_num = ray.get(llm_server.check_pending_num.remote(0, 0))

    print(f"Benchmark start after {num_ongoing} requests are in decode phase.")
    start_time = time.perf_counter()
    # add a batch of prefill requests (to simulate queuing)
    new_request_lengths = [16284, 14230, 4936, 4096] * 4 + [2048]
    batch_prompts = [""] * len(new_request_lengths)
    batch_prompt_token_ids = [
        [random.randint(0, 16384) for i in range(context_len)]
        for context_len in new_request_lengths
    ]
    new_task = llm_server.batch_generate.remote(
        batch_prompts=batch_prompts,
        batch_prompt_token_ids=batch_prompt_token_ids,
        max_tokens=decode_length,
    )

    responses = await new_task 
    # ongoing_responses = await asyncio.gather(*ongoing_requests)
    dur_s = time.perf_counter() - start_time
    print(f"Benchmark terminates after {dur_s}s.")
    
    outputs = [resp["outputs"] for resp in responses]
    prompt_lens = np.array([resp["prompt_len"] for resp in responses])
    output_lens = np.array([len(output) for output in outputs])
    metrics = [resp["metrics"] for resp in responses]
    queues = np.array([metric["queue"] for metric in metrics])
    ttfts = np.array([metric["ttft"] for metric in metrics])
    tbts = np.array([metric["tbt"] for metric in metrics])

    # ongoing_metrics = [resp["metrics"] for resp in ongoing_responses]
    # ongoing_tbts = np.array([metric["tbt"] for metric in ongoing_metrics])

    # overall_tbts = np.concatenate([tbts, ongoing_tbts])

    try:
        await llm_server.terminate.remote()
    except Exception as e:
        print(f"Error occurred when terminating the server: {e}")
    ray.kill(llm_server)

    print(
        f"******************Benchmark report begin**********************\n"
        f"The configuration file is {cfg_file}.\n"
        f"Finish {len(responses)} generation tasks in {dur_s}s.\n"
        f"Mean queue time is {queues.mean()}s, P50 is {np.percentile(queues, 50):.3f} s, P90 is {np.percentile(queues, 90):.3f} s, P99 is {np.percentile(queues, 99):.3f} s.\n"
        f"Mean TTFT is {ttfts.mean()}s, P50 is {np.percentile(ttfts, 50):.3f} s, P90 is {np.percentile(ttfts, 90):.3f} s, P99 is {np.percentile(ttfts, 99):.3f} s.\n"
        f"For new, mean TBT is {tbts.mean()}s, P50 is {np.percentile(tbts, 50):.3f} s, P90 is {np.percentile(tbts, 90):.3f} s, P99 is {np.percentile(tbts, 99):.3f} s.\n"
        # f"For ongoing, mean TBT is {ongoing_tbts.mean()}s, P50 is {np.percentile(ongoing_tbts, 50):.3f} s, P90 is {np.percentile(ongoing_tbts, 90):.3f} s, P99 is {np.percentile(ongoing_tbts, 99):.3f} s.\n"
        # f"For overall, mean TBT is {overall_tbts.mean()}s, P50 is {np.percentile(overall_tbts, 50):.3f} s, P90 is {np.percentile(overall_tbts, 90):.3f} s, P99 is {np.percentile(overall_tbts, 99):.3f} s.\n"
        # f"Average prompt length is {prompt_lens.mean()} tokens, minimal is {prompt_lens.min()}, maximal is {prompt_lens.max()}.\n"
        # f"Average output length is {output_lens.mean()} tokens, minimal is {output_lens.min()}, maximal is {output_lens.max()}.\n"
        f"Average generation throughput is {output_lens.sum() / dur_s:.2f} tokens/s, {len(responses) / dur_s:.2f} req/s.\n"
        f"******************Benchmark report end**********************"
    )

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