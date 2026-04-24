import ray
import os

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')

import argparse

from kunserve.setup import connect_to_ray_cluster, setup_kunserve
from kunserve.utils import get_log_name

from kunserve.kunserve_config import *
import asyncio
import pandas as pd

from evaluation.trace import *

import matplotlib.pyplot as plt

from kunserve.logger import init_logger
logger = init_logger(__name__)


def plot_request_trace(profile_results, time_window=5):
    timestamps = np.array([result["time"] for result in profile_results])
    timestamps.sort()

    if len(timestamps) == 0:
        print("No data to plot")
        return
    
    min_time = np.min(timestamps)
    max_time = np.max(timestamps)
    bins = np.arange(start=min_time, stop=max_time + time_window, step=time_window)

    counts, bin_edges = np.histogram(timestamps, bins=bins)
    window_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure(figsize=(12, 6))
    plt.plot(window_centers, counts, linestyle='-', linewidth=2)

    plt.xlabel(f"Time (s)", fontsize=12)
    plt.ylabel("Requests Count", fontsize=12)
    plt.title(f"Request Rate Over Time ({time_window}-second intervals)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(
        os.path.splitext(__file__)[0] + ".jpg",
        dpi=1000,
        format="jpg",
        bbox_inches="tight",
    )

async def benchmark(bench_config: BenchConfig, conf_path):
    # TODO: read from dataset
    requests = get_prompts(
        bench_config.dataset, 
        num_prompts=bench_config.nlimit, 
        offset=bench_config.dataset_offset, 
        client_id=0,
        trace_repeat_times=bench_config.trace_repeat_times,
    )

    intervals = None
    if bench_config.dist == "real":
        intervals = get_trace_intervals(bench_config.trace, bench_config.trace_offset,)
                                        # trace_repeat_start=bench_config.trace_repeat_start,
                                        # trace_repeat_end=bench_config.trace_repeat_end,
                                        # trace_repeat_times=bench_config.trace_repeat_times,)

    acum_time = 0
    for i, interval in enumerate(intervals):
        acum_time += interval[1]
        if acum_time >= 2000:
            print(f"current request interval: {i}, {acum_time=}")
            break
    
    # async_requests = async_request_gen(
    #     requests,
    #     qps=bench_config.qps, 
    #     distribution=bench_config.dist, 
    #     coefficient_variation=bench_config.cv,
    #     interval=intervals,
    # )

    # requests = []
    # start_time = time.perf_counter()
    # async for prompt in async_requests:
    #     requests.append({
    #         "time": time.perf_counter() - start_time,
    #     })
    # print(f"num of requests: {len(requests)}")
    # plot_request_trace(requests)
    # pd.DataFrame(requests).to_csv(os.path.splitext(conf_path)[0] + ".csv",)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    cli_args = parser.parse_args()
    """ create LLM servers equal to the number of instance groups """
    kunserve_config: KunServeConfig = get_cli_args(cli_args)
    asyncio.run(benchmark(kunserve_config.bench_config, cli_args.config))