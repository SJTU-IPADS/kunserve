"""
This dataset is called LiveCodeBench, it does not have a response, 
we can use DeepSeek-R1-Distill-Llama-70B to generate the response.
"""

import re
import os
import datasets

from hdfs_io import copy, makedirs
import argparse

from datasets import Features, Value

from kunserve.tokenizer import get_tokenizer
import pyarrow as pa

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-dir', default='~/data/codebench-llama')
    parser.add_argument('--hdfs-dir', default=None)
    parser.add_argument('--tokenizer', default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

    args = parser.parse_args()

    tokenizer = get_tokenizer(
        args.tokenizer, 
    ) if args.tokenizer else None

    data_source = "livecodebench/code_generation_lite"
    dataset = datasets.load_dataset(data_source)
    train_dataset = dataset['test']
    table = train_dataset.data.table
    new_schema = pa.schema([
        (field.name, pa.large_string()) if str(field.type) == 'string' else (field.name, field.type)
        for field in table.schema
    ])
    converted_table = table.cast(new_schema)
    train_dataset = datasets.Dataset(converted_table)

    # add a row to each data item that represents a unique id

    system_prompt = "You will be given a competitive programming problem. Please reason step by step about the solution, then provide a complete implementation in C++17.\n" \
        + "Your solution must read input from standard input (cin), write output to standard output (cout).Do not include any debug prints or additional output.\n" \
        + "Put your final solution within a single code block:\n```cpp\n<your code here>```\n"

    def make_map_fn(split):

        def process_fn(sample, idx):
            title = sample.pop('question_title')
            question = sample.pop('question_content')
            cases = sample.pop('public_test_cases')

            prompt = system_prompt + "\nTitle\n" + title + "\nDescription\n" + question + "\nTest cases\n" + cases
            prompt_token_ids = tokenizer.encode(prompt)

            data = {
                "data_source": data_source,
                "prompt_len": len(prompt_token_ids),
                "output_len": 0,
                "split": split,
            }
            return data

        return process_fn
    
    train_dataset = train_dataset.map(
        function=make_map_fn('test'), 
        with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)