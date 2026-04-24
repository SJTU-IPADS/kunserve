"""
This dataset is called LongBench, it does not have a response, 
we can use DeepSeek-R1-Distill-Llama-70B to generate the response.
"""

import re
import os
import datasets
from datasets import concatenate_datasets, load_dataset

from hdfs_io import copy, makedirs
import argparse

from kunserve.tokenizer import get_tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-dir', default='~/data/longdata-llama')
    parser.add_argument('--hdfs-dir', default=None)
    parser.add_argument('--tokenizer', default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

    args = parser.parse_args()

    tokenizer = get_tokenizer(
        args.tokenizer, 
        trust_remote_code=True,
    ) if args.tokenizer else None

    train_dataset = concatenate_datasets([
        load_dataset('togethercomputer/Long-Data-Collections', split='train', trust_remote_code=True),
    ])
    
    # add a row to each data item that represents a unique id

    def make_map_fn(split):

        def process_fn(sample, idx):
            prompt = sample.pop('prompt')
            response = sample.pop('completion')

            prompt_token_ids = tokenizer.encode(prompt)
            response_token_ids = tokenizer.encode(response)

            if len(prompt_token_ids) + len(response_token_ids) > 16384:
                prompt_len = 16384 - len(response_token_ids)
                prompt_token_ids = prompt_token_ids[:prompt_len]
                prompt = tokenizer.decode(prompt_token_ids)

            data = {
                "prompt": prompt,
                "prompt_token_ids": prompt_token_ids,
                "prompt_len": len(prompt_token_ids),
                "response_len": len(response_token_ids),
                "split": split,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)