import math
import pandas as pd
import numpy as np
import time
import asyncio

from kunserve.logger import init_logger
logger = init_logger(__name__)


def get_prompts(
    dataset_path, num_prompts: int, offset: int, client_id: int = 0,
    trace_repeat_start: int = None, trace_repeat_end: int = None, trace_repeat_times: int = None,
    shuffle: bool = False
):
    dataset = pd.read_parquet(dataset_path)
    print(f"Total {len(dataset)} rows in {dataset_path}.")
    start = offset + client_id * num_prompts
    slice_df = dataset[dataset["response_len"] <= 8192]
    if shuffle:
        slice_df = slice_df.sample(frac=1, random_state=42)
        
    slice_df = slice_df.iloc[start : start + num_prompts]
    print(f"Total {len(slice_df)} rows after shuffling, demand: {num_prompts}.")

    if len(slice_df) < num_prompts:
        data_repeat_times = int(math.ceil(num_prompts / len(slice_df)))
        slice_df = pd.concat([slice_df] * data_repeat_times, ignore_index=True)
        slice_df = slice_df.iloc[:num_prompts]

    if trace_repeat_start is not None and trace_repeat_end is not None:
        num_repeat_entries = trace_repeat_end - trace_repeat_start
        repeat_slice = slice_df.iloc[-num_repeat_entries:]
        for _ in range(trace_repeat_times):
            slice_df = pd.concat([slice_df, repeat_slice], ignore_index=True)
    
    print(f"Total {len(slice_df)} rows after repeat.")
    return slice_df.itertuples(index=False, name='Dataset')

def get_trace_intervals(
    trace_path: str, trace_offset: int, 
    time_scale_factor: float = 1.0,
    num_timestamps = None,
    trace_repeat_start: int = None,
    trace_repeat_end: int = None,
    trace_repeat_times: int = None,
):
    trace = pd.read_parquet(trace_path)
    trace = trace.iloc[trace_offset:]
    intervals = trace["timestamp"].diff().dropna()
    intervals /= time_scale_factor
    if num_timestamps is not None:
        intervals = intervals.iloc[:num_timestamps]
    if trace_repeat_start is not None and trace_repeat_end is not None:
        num_repeat_entries = trace_repeat_end - trace_repeat_start
        repeat_intervals = intervals.iloc[-num_repeat_entries:]
        for _ in range(trace_repeat_times):
            intervals = pd.concat([intervals, repeat_intervals], ignore_index=True)
    print(f"number of intervals after concat: {len(intervals)}")
    return intervals.items()

def get_wait_time(mean_time_between_requests: float, distribution: str, coefficient_variation: float = 0.0) -> float:
    if distribution == "uniform":
        return mean_time_between_requests
    elif distribution == "gamma":
        variance = (coefficient_variation * mean_time_between_requests) ** 2
        shape = mean_time_between_requests ** 2 / variance
        scale = variance / mean_time_between_requests
        return np.random.gamma(shape, scale)
    else:
        # poisson distribution
        return np.random.exponential(mean_time_between_requests)

def request_gen(generator, qps: float, distribution="uniform"):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                time.sleep(get_wait_time(1.0 / qps, distribution))
        except StopIteration:
            return

async def async_request_gen(request_generator, qps: float=0.0, distribution="uniform", coefficient_variation: float=0.0, interval=None):
    np.random.seed(42)
    while True:
        try:
            item = next(request_generator)
            yield item
            if distribution == "real":
                next_interval = next(interval)[1]
                await asyncio.sleep(next_interval)
            elif distribution != "burst":
                await asyncio.sleep(get_wait_time(1.0 / qps, distribution, coefficient_variation))
        except StopIteration:
            return
        


def get_intervals(n, qps: float, distribution="uniform", coefficient_variation: float = 0.0):
    intervals = []
    for _ in range(n):
        if distribution != "burst":
            intervals.append(get_wait_time(1.0 / qps, distribution, coefficient_variation))
        else:
            intervals.append(0)
    return intervals