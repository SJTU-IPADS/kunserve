import argparse
from datetime import datetime
import os
import time

import pandas as pd
import matplotlib.pyplot as plt


def plot_trace(timestamps: pd.DataFrame, output_path: str, offset=0, limit=None):
    timestamps = timestamps.iloc[offset:]
    if limit is not None:
        timestamps = timestamps.iloc[:limit]

    timestamps = (timestamps - timestamps.iloc[0]).astype(int)
    
    data = timestamps.value_counts().sort_index().reset_index()
    data.columns = ['timestamp', 'count']
    # print(data)

    plt.figure()
    
    plt.title("maf Trace")
    plt.bar(data["timestamp"], data["count"], 
        width=1.0, 
        edgecolor='black',
        linewidth=0.5,
        facecolor='white')
    
    plt.savefig(output_path)
    plt.close()

def process_azure_trace(trace_file, timex):
    trace = pd.read_csv(trace_file)
    trace["timestamp"] = (trace["end_timestamp"] - trace["duration"]) / timex
    return trace["timestamp"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-path', default='/data/trace/maf.csv')
    parser.add_argument('--save-path', default='/data/trace/maf-{}x.parquet')
    parser.add_argument('--output-dir', default='/app/KunServe/evaluation/trace')
    parser.add_argument('--offset', "-o", type=int, default=319091)
    parser.add_argument('--limit', "-l", type=int, default=None)
    parser.add_argument('--timex', type=int, default=1000)

    args = parser.parse_args()
    trace_df = pd.DataFrame(process_azure_trace(args.trace_path, args.timex))
    plot_trace(trace_df, os.path.join(args.output_dir, "maf.pdf"), args.offset, args.limit)
    
    save_path = args.save_path.format(args.timex)
    trace_df = trace_df.iloc[args.offset:]
    if args.limit:
        trace_df = trace_df.iloc[:args.limit]
    trace_df = trace_df.sample(frac=1/args.timex, random_state=42).sort_values("timestamp").reset_index(drop=True)
    print(trace_df)
    trace_df.to_parquet(save_path, index=False)
    trace_df.to_csv(os.path.join(args.output_dir, "maf.csv"), index=False)

    print(f"Total {len(trace_df)} timestamps in the trace, saved to {save_path}")