import random
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

from kunserve.logger import init_logger
logger = init_logger(__name__)

llm_server_loads = []
active = True

SERVER_REPORT_EPOCH = 0.5
SERVER_REPORT_INTERVAL = 1

async def collect_llm_server_loads(llm_servers):
    global active
    global llm_server_loads
    while active:
        tasks = []
        for llm_server in llm_servers:
            tasks.append(llm_server.get_unfinised_num.remote())
        llm_server_loads = await asyncio.gather(*tasks)
        await asyncio.sleep(SERVER_REPORT_EPOCH)

async def collect_server_metrics(servers, log_name):
    epoch_time = []
    thpt_per_epoch = []
    recv_reqs_per_epoch = []
    finish_reqs_per_epoch = []
    mean_ttft_per_epoch = []
    mean_tbt_per_epoch = []
    mem_demand_per_epoch = []
    states_per_epoch = []

    prev_recv_requests = 0
    prev_finish_requests = 0
    prev_output_tokens = 0

    epoch_start = time.perf_counter()
    global_start = epoch_start
    while active:
        start_time = time.perf_counter()
        
        total_output_tokens = 0
        total_received_requests = 0
        total_finished_requests = 0
        avg_loads = []
        states = []
        for server in servers:
            output_tokens, received_requests, finished_requests, avg_load, state = await server.report_server_metrics.remote()
            total_output_tokens += output_tokens
            total_received_requests += received_requests
            total_finished_requests += finished_requests
            avg_loads.append(avg_load)
            states.append(state)

        ttfts, tbts = [], []
        for server in servers:
            ttft, tbt = await server.pop_metrics.remote()
            ttfts.extend(ttft)
            tbts.extend(tbt)

        epoch_end = time.perf_counter()
        
        epoch_time.append(epoch_end - global_start)

        output_tokens_in_cur_epoch = total_output_tokens - prev_output_tokens
        thpt_per_epoch.append(output_tokens_in_cur_epoch / (epoch_end - epoch_start))
        prev_output_tokens = total_output_tokens
    
        recv_reqs_cur_epoch = total_received_requests - prev_recv_requests
        recv_reqs_per_epoch.append(recv_reqs_cur_epoch)
        prev_recv_requests = total_received_requests

        finish_reqs_cur_epoch = total_finished_requests - prev_finish_requests
        finish_reqs_per_epoch.append(finish_reqs_cur_epoch)
        prev_finish_requests = total_finished_requests

        mean_ttft_in_cur_epoch = np.mean(ttfts) if len(ttfts) > 0 else 0
        mean_ttft_per_epoch.append(mean_ttft_in_cur_epoch)

        mean_tbt_in_cur_epoch = np.mean(tbts) if len(tbts) > 0 else 0
        mean_tbt_per_epoch.append(mean_tbt_in_cur_epoch)

        mem_demand_cur_epoch = np.mean(avg_loads)
        mem_demand_per_epoch.append(mem_demand_cur_epoch)

        states_per_epoch.append(states)

        print(f"[Metrics epoch {epoch_time[-1]:.0f}] outputs: {output_tokens_in_cur_epoch}, received: {recv_reqs_cur_epoch}, finished: {finish_reqs_cur_epoch}, mean ttft: {mean_ttft_in_cur_epoch:.3f}s, avg load: {mem_demand_cur_epoch:.2f}")

        epoch_start = epoch_end

        elapsed_time = time.perf_counter() - start_time
        # await asyncio.sleep(max(0, SERVER_REPORT_INTERVAL - elapsed_time))
        await asyncio.sleep(SERVER_REPORT_INTERVAL)
    
    data = {
        "time": epoch_time,
        "thpt": thpt_per_epoch,
        "received requests": recv_reqs_per_epoch,
        "finished requests": finish_reqs_per_epoch,
        "memory demand": mem_demand_per_epoch,
        "mean ttft": mean_ttft_per_epoch,
        "mean tbt": mean_tbt_per_epoch,
        "states": states_per_epoch,
    }
    df = pd.DataFrame(data)

    mem_demand = df["memory demand"]
    avg_mem = mem_demand.mean()
    print(f"Average memory demand is {avg_mem:.2f}%.")

    df.to_csv(log_name + '.csv', index=False)

async def benchmark(llm_servers, bench_config: BenchConfig, cfg_file: str):
    global active
    global llm_server_loads

    if not os.path.exists(bench_config.log_path):
        logger.info(f"Create log path {bench_config.log_path}.")
        os.makedirs(bench_config.log_path, exist_ok=True)
    # log_name = get_log_name(bench_config.log_path, bench_config.qps, bench_config.cv, bench_config.dist, bench_config.dataset_scale_factor)
    log_name = os.path.join(bench_config.log_path, bench_config.log_name)

    if bench_config.dataset_scale_factor >= 1.0:
        num_prompts = int(math.ceil(bench_config.dataset_scale_factor)) * 5 * bench_config.nlimit
    else:
        num_prompts = bench_config.nlimit
    print(f"{num_prompts=}")

    if bench_config.trace_repeat_times:
        print(f"The trace will be repeated for {bench_config.trace_repeat_times} times.")
    
    requests = get_prompts(
        bench_config.dataset, 
        num_prompts=num_prompts, 
        offset=bench_config.dataset_offset, 
        client_id=0,
        shuffle=bench_config.shuffle_dataset,
        trace_repeat_start=bench_config.trace_repeat_start,
        trace_repeat_end=bench_config.trace_repeat_end,
        trace_repeat_times=bench_config.trace_repeat_times,
    )
    requests = list(requests)
    print(f"number of requests after repeat is {len(requests)}, trace_repeat_times: {bench_config.trace_repeat_times}")
    
    intervals = None
    if bench_config.dist == "real":
        intervals = get_trace_intervals(
            bench_config.trace, bench_config.trace_offset,
            time_scale_factor=bench_config.time_scale_factor,
            num_timestamps=num_prompts,
            trace_repeat_start=bench_config.trace_repeat_start, 
            trace_repeat_end=bench_config.trace_repeat_end, 
            trace_repeat_times=bench_config.trace_repeat_times,
        )

    async_requests = async_request_gen(
        iter(range(len(requests))),
        qps=bench_config.qps, 
        distribution=bench_config.dist, 
        coefficient_variation=bench_config.cv,
        interval=intervals
    )
    report_tasks = [
        asyncio.create_task(collect_llm_server_loads(llm_servers)),
        asyncio.create_task(collect_server_metrics(llm_servers, log_name))
    ]

    random.seed(42)
    tasks = []
    
    start_time = time.time()

    if True:
        request_id = 0
        nrepeat = int(math.ceil(bench_config.dataset_scale_factor))
        select_ratio = bench_config.dataset_scale_factor / nrepeat
        
        random_nums = []
        if bench_config.trace_repeat_times is not None:
            num_repeats = bench_config.trace_repeat_end - bench_config.trace_repeat_start
        i = 0
        num_replay_reqs = (
            bench_config.trace_repeat_end - bench_config.trace_repeat_start
            if bench_config.trace_repeat_end and bench_config.trace_repeat_start else 0
        )
        replay_point = num_prompts - num_replay_reqs
        async for _ in async_requests:
            # random_nums.append([])

            if (bench_config.trace_repeat_end 
                and bench_config.trace_repeat_start 
                and i == replay_point):
                # no drop for peak wave evaluation (with trace replay enabled)
                random.seed(0xdeadbeef)
                if bench_config.trace_repeat_times > 0:
                    select_ratio *= 2
            
            if num_replay_reqs > 0 and (i - replay_point) % num_replay_reqs == 0:
                # reset random seed for replay requests routinely
                random.seed(0xdeadbeef)

            # the repeat means we send multiple requests in one timestamp using TraceUpscaler's method
            for j in range(nrepeat):
                # if bench_config.trace_repeat_end is None or i < bench_config.trace_repeat_end:
                random_num = random.random()
                    # random_nums[-1].append(random_num)
                # else:
                #     index = bench_config.trace_repeat_start + (i - bench_config.trace_repeat_end) % num_repeats
                #     random_num = random_nums[index][j]
                
                if random_num <= select_ratio:
                    assert request_id < len(requests), f"request_id {request_id} exceeds the number of requests {len(requests)}."

                    prompt = requests[request_id]
                    select_server = llm_server_loads.index(min(llm_server_loads))
                    tasks.append(llm_servers[select_server].generate.remote(
                        request_id,
                        prompt.prompt,
                        max_tokens=prompt.response_len,
                    ))
                    llm_server_loads[select_server] += 1
            
                request_id += 1
            # insert interval between repeated requests
            if (bench_config.trace_repeat_end is not None
                and i >= bench_config.trace_repeat_end - 1
                and (i - bench_config.trace_repeat_end + 1) % num_repeats == 0
            ):
                if bench_config.trace_repeat_interval is not None:
                    await asyncio.sleep(bench_config.trace_repeat_interval)
            i += 1
    
    request_flow_dur_s = time.time() - start_time
    qps = len(tasks) / request_flow_dur_s
    responses = await asyncio.gather(*tasks)
    dur_s = time.time() - start_time
    print(f"Benchmark terminates after {dur_s}s.")
    
    active = False
    await asyncio.gather(*report_tasks)
    
    outputs = [resp["outputs"] for resp in responses]
    prompt_lens = np.array([resp["prompt_len"] for resp in responses])
    output_lens = np.array([len(output) for output in outputs])
    metrics = [resp["metrics"] for resp in responses]
    queues = np.array([metric["queue"] for metric in metrics])
    ttfts = np.array([metric["ttft"] for metric in metrics])
    tbts = np.array([metric["tbt"] for metric in metrics])

    for llm_server in llm_servers:
        try:
            await llm_server.terminate.remote()
        except Exception as e:
            print(f"Error occurred when terminating the server: {e}")
        ray.kill(llm_server)

    print(
        f"******************Benchmark report begin**********************\n"
        f"The configuration file is {cfg_file}.\n"
        f"Finish {len(responses)} generation tasks in {dur_s}s, send QPS is {qps:.3f}.\n"
        f"Mean queue time is {queues.mean()}s, P50 is {np.percentile(queues, 50):.3f} s, P90 is {np.percentile(queues, 90):.3f} s, P99 is {np.percentile(queues, 99):.3f} s, P999 is {np.percentile(queues, 99.9):.3f} s.\n"
        f"Mean TTFT is {ttfts.mean()}s, P50 is {np.percentile(ttfts, 50):.3f} s, P90 is {np.percentile(ttfts, 90):.3f} s, P99 is {np.percentile(ttfts, 99):.3f} s, P999 is {np.percentile(ttfts, 99.9):.3f} s.\n"
        f"Mean TBT is {tbts.mean()}s, P50 is {np.percentile(tbts, 50):.3f} s, P90 is {np.percentile(tbts, 90):.3f} s, P99 is {np.percentile(tbts, 99):.3f} s, P999 is {np.percentile(tbts, 99.9):.3f} s.\n"
        f"Average prompt length is {prompt_lens.mean()} tokens, minimal is {prompt_lens.min()}, maximal is {prompt_lens.max()}.\n"
        f"Average output length is {output_lens.mean()} tokens, minimal is {output_lens.min()}, maximal is {output_lens.max()}.\n"
        f"Average generation throughput is {output_lens.sum() / dur_s:.2f} tokens/s, {len(responses) / dur_s:.2f} req/s.\n"
        f"******************Benchmark report end**********************"
    )

    metrics_df = pd.DataFrame(metrics)
    metrics_df["arrival_time"] = metrics_df["arrival_time"] - start_time
    # for i in range(len(metrics_df)):
    #     if metrics_df.at[i, "ttft"] > 5 or metrics_df.at[i, "input_len"] > 8000:
    #         print(f"request {metrics_df.at[i, 'request_id']} find a large ttft: {metrics_df.at[i, 'ttft']:.2f}, prompt len: {metrics_df.at[i, 'input_len']}")
    metric_log_name = log_name + ".parquet"
    metrics_df.to_parquet(metric_log_name)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    cli_args = parser.parse_args()
    """ create LLM servers equal to the number of instance groups """
    connect_to_ray_cluster()
    try:
        kunserve_config: KunServeConfig = get_cli_args(cli_args)
    except Exception as e:
        print(f"Error occurred when parsing the configuration file: {e}")
        exit(1)
    print(f"The evaluated model is {kunserve_config.bench_config.model}.")
    llm_servers = setup_kunserve(kunserve_config)
    llm_server_loads = [0] * len(llm_servers)

    asyncio.run(benchmark(llm_servers, kunserve_config.bench_config, cli_args.config))


    