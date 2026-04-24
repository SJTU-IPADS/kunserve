import argparse
from datetime import datetime
import os
import time

import pandas as pd
import matplotlib.pyplot as plt

def process_azure_trace(trace_file):
    trace = pd.read_csv(trace_file)
    trace["timestamp"] = trace["end_timestamp"] - trace["duration"]
    return trace["timestamp"]


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', default='/ssd/trace/maf.parquet')
    parser.add_argument('--output-dir', default='evaluation/trace')
    parser.add_argument('--offset', "-o", type=int, default=0)
    parser.add_argument('--limit', "-l", type=int, default=None)

    args = parser.parse_args()
    trace_df = pd.read_parquet(args.save_path)
    
    prev_slice = trace_df[(trace_df["timestamp"] < 0.53635 * 10 ** 6)]
    slice_df = trace_df[(trace_df["timestamp"] >= 0.53635 * 10 ** 6) & (trace_df["timestamp"] < 0.53655 * 10 ** 6)]
    print(f"prev: {len(prev_slice)}, {len(slice_df)}")

    plot_trace(slice_df, os.path.join(args.output_dir, "maf.pdf"), args.offset, args.limit)

    print(f"Total {len(slice_df)} timestamps in the trace, saved to {args.save_path}")