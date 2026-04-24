import argparse
from datetime import datetime
import os
import time

import pandas as pd
import matplotlib.pyplot as plt

def process_burstgpt_trace(trace_file, offset, limit, scale=1.0):
    trace = pd.read_csv(trace_file)
    timestamps = []
    for i, row in trace.iterrows():
        if limit is not None and i >= offset + limit:
            break
        timestamps.append(int(row['Timestamp'] / scale))
    return timestamps

def plot_trace(timestamps: pd.DataFrame, output_path: str, offset=0, limit=None):
    timestamps = timestamps.iloc[offset:]
    if limit is not None:
        timestamps = timestamps.iloc[:limit]

    timestamps = (timestamps - timestamps.iloc[0]).astype(int)
    num_requests = len(timestamps)
    
    data = timestamps.value_counts().sort_index().reset_index()
    data.columns = ['timestamp', 'count']
    # print(data)

    max_timestamp = data['timestamp'].max()
    rate = num_requests / max_timestamp
    print(f"{num_requests} requests, {max_timestamp} s, request rate: {rate:.2f} req/s")

    plt.figure()
    
    plt.title("BurstGPT Trace")
    plt.bar(data["timestamp"], data["count"], 
        width=1.0, 
        edgecolor='black',
        linewidth=0.5,
        facecolor='white')
    
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-path', default='/data/trace/burstgpt.csv')
    parser.add_argument('--save-path', default='/data/trace/burstgpt.parquet')
    parser.add_argument('--output-dir', default='/app/KunServe/evaluation/trace')
    parser.add_argument('--offset', "-o", type=int, default=474438)
    parser.add_argument('--limit', "-l", type=int, default=775)
    parser.add_argument('--scale', '-s', type=float, default=10.0)
    parser.add_argument('--gen', '-g', action='store_true', default=False)
    parser.add_argument('--plot', '-p', action='store_true', default=False)

    args = parser.parse_args()

    timestamps = process_burstgpt_trace(args.trace_path, args.offset, args.limit, args.scale)
    trace_df = pd.DataFrame({"timestamp": timestamps})

    if args.gen:
        trace_df.to_parquet(args.save_path, index=False)
    
    if args.plot:
        trace_df.to_csv(os.path.join(args.output_dir, "burstgpt.csv"), index=False)
        plot_trace(trace_df, os.path.join(args.output_dir, f"burstgpt-{args.scale}-{args.offset}-{args.limit}.pdf"), args.offset, args.limit)

    print(f"Total {len(timestamps)} timestamps in the trace, saved to {args.save_path}")