import argparse
from datetime import datetime
import os
import time

import pandas as pd
import matplotlib.pyplot as plt

def process_azure_trace(trace_file):
    trace = pd.read_csv(trace_file)
    time_format = '%Y-%m-%d %H:%M:%S.%f'
    timestamps = []
    for _, row in trace.iterrows():
        timestamps.append(
            time.mktime(
                datetime.strptime(row["TIMESTAMP"][:-1], time_format).timetuple()
            )
        )
    return timestamps

def process_azure_v2_trace(trace_file):
    def parse_timestamp(ts):
        try:
            return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f+00:00')
        except ValueError:
            return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S+00:00')

    trace = pd.read_csv(trace_file)
    timestamps = []
    for _, row in trace.iterrows():
        timestamps.append(
            time.mktime(parse_timestamp(row["TIMESTAMP"]).timetuple())
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
    # start = data[data["timestamp"] < 1700]["count"].sum()
    # end = data[data["timestamp"] < 1800]["count"].sum()
    # print(f"{start=} {end=}")

    plt.figure()
    
    plt.title("azureconv Trace")
    plt.bar(data["timestamp"], data["count"], 
        width=1.0, 
        edgecolor='black',
        linewidth=0.5,
        facecolor='white')
    
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-path', default='/ssd/trace/azureconv.csv')
    parser.add_argument('--save-path', default='/ssd/trace/azureconv.parquet')
    parser.add_argument('--output-dir', default='/app/KunServe/evaluation/trace')
    parser.add_argument('--scale-factor', type=int, default=35)
    parser.add_argument('--use-v2', action='store_true')
    parser.add_argument('--offset', "-o", type=int, default=9342)  # we timestamps in 1700~1800s of AzureConv trace
    parser.add_argument('--limit', "-l", type=int, default=759)

    args = parser.parse_args()

    if args.use_v2:
        timestamps = process_azure_v2_trace(args.trace_path)
    else:
        timestamps = process_azure_trace(args.trace_path)
    trace_df = pd.DataFrame({"timestamp": timestamps})

    trace_df.to_parquet(args.save_path, index=False)
    
    trace_df.to_csv(os.path.join(args.output_dir, "azureconv.csv"), index=False)
    plot_trace(trace_df, os.path.join(args.output_dir, "azureconv.pdf"), args.offset, args.limit)

    print(f"Total {len(timestamps)} timestamps in the trace, saved to {args.save_path}")