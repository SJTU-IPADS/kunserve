import argparse
from datetime import datetime
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def process_azure_trace(trace_file):
    trace = pd.read_csv(trace_file)
    time_format = '%Y-%m-%d %H:%M:%S.%f'
    timestamps = []
    prompt_lengths = []
    output_lengths = []
    for _, row in trace.iterrows():
        timestamps.append(
            time.mktime(
                datetime.strptime(row["TIMESTAMP"][:-1], time_format).timetuple()
            )
        )
        prompt_lengths.append(row["ContextTokens"])
        output_lengths.append(row["GeneratedTokens"])

    print(f"prompt length: mean: {np.mean(prompt_lengths)}, p50: {np.percentile(prompt_lengths, 50)}, p90: {np.percentile(prompt_lengths, 90)}, p99: {np.percentile(prompt_lengths, 99)}")
    print(f"output length: mean: {np.mean(output_lengths)}, p50: {np.percentile(output_lengths, 50)}, p90: {np.percentile(output_lengths, 90)}, p99: {np.percentile(output_lengths, 99)}")

    return timestamps

def process_azure_v2_trace(trace_file):
    trace = pd.read_csv(trace_file)
    time_format = '%Y-%m-%d %H:%M:%S.%f+00:00'
    timestamps = []
    for _, row in trace.iterrows():
        timestamps.append(
            time.mktime(
                datetime.strptime(row["TIMESTAMP"], time_format).timetuple()
            )
        )
    return timestamps

def plot_trace(timestamps: pd.DataFrame, output_path: str, offset=0, limit=None):
    timestamps = timestamps.iloc[offset:]
    if limit is not None:
        timestamps = timestamps.iloc[:limit]

    timestamps = (timestamps - timestamps.iloc[0]).astype(int)
    
    data = timestamps.value_counts().sort_index().reset_index()
    data.columns = ['timestamp', 'count']
    # print(data)

    plt.figure()
    
    plt.title("AzureCode Trace")
    plt.bar(data["timestamp"], data["count"], 
        width=1.0, 
        edgecolor='black',
        linewidth=0.5,
        facecolor='white')
    plt.ylim(0, 70)
    
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-path', default='/data/trace/azurecode.csv')
    parser.add_argument('--save-path', default='/data/trace/azurecode.parquet')
    parser.add_argument('--output-dir', default='/app/KunServe/evaluation/trace')
    parser.add_argument('--use-v2', action='store_true')
    parser.add_argument('--offset', "-o", type=int, default=5100)
    parser.add_argument('--limit', "-l", type=int, default=927)
    parser.add_argument('--gen', '-g', action='store_true', default=False)
    parser.add_argument('--plot', '-p', action='store_true', default=False)

    args = parser.parse_args()

    if args.use_v2:
        timestamps = process_azure_v2_trace(args.trace_path)
    else:
        timestamps = process_azure_trace(args.trace_path)
    
    trace_df = pd.DataFrame({"timestamp": timestamps})

    if args.gen:
        trace_df.to_parquet(args.save_path, index=False)

    if args.plot:
        trace_df.to_csv(os.path.join(args.output_dir, "azurecode.csv"), index=False)
        plot_trace(trace_df, os.path.join(args.output_dir, f"azurecode-{args.offset}-{args.limit}.pdf"), args.offset, args.limit)

    print(f"Total {len(timestamps)} timestamps in the trace, saved to {args.save_path}")