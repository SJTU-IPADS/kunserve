import re
import os
import datasets
from datasets import concatenate_datasets, load_dataset

# from hdfs_io import copy, makedirs
import argparse
import numpy as np

from kunserve.tokenizer import get_tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-dir', default='/data/dataset/leval-llama')
    parser.add_argument('--hdfs-dir', default=None)
    parser.add_argument('--tokenizer', default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

    args = parser.parse_args()

    tokenizer = get_tokenizer(
        args.tokenizer, 
        trust_remote_code=True,
    ) if args.tokenizer else None

    data_source = "L4NLP/LEval"
    # add a row to each data item that represents a unique id
    train_dataset = concatenate_datasets([
        load_dataset(data_source, name='financial_qa', split='test', trust_remote_code=True),
        load_dataset(data_source, name='legal_contract_qa', split='test', trust_remote_code=True),
        load_dataset(data_source, name='multidoc_qa', split='test', trust_remote_code=True),
        load_dataset(data_source, name='narrative_qa', split='test', trust_remote_code=True),
        load_dataset(data_source, name='natural_question', split='test', trust_remote_code=True),
        load_dataset(data_source, name='scientific_qa', split='test', trust_remote_code=True)
    ])

    def make_map_fn(split):

        def process_fn(sample, idx):
            question = sample.pop('input')
            # context = sample.pop('documents')
            response = sample.pop('outputs')[0]

            prompt = question
            prompt_token_ids = tokenizer.encode(prompt)
            response_token_ids = tokenizer.encode(response)

            if len(prompt_token_ids) > 16384:
                prompt_len = 16384
                prompt_token_ids = prompt_token_ids[:prompt_len]
                prompt = tokenizer.decode(prompt_token_ids)

            data = {
                "data_source": data_source,
                "prompt": prompt,
                "prompt_len": len(prompt_token_ids),
                "response_len": len(response_token_ids),
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    # train_dataset = train_dataset.shuffle(seed=42)
    # train_dataset = train_dataset.sort('prompt_len', reverse=True)
    train_df = train_dataset.to_pandas()
    print(train_df)

    avg_prompt_len = np.array(train_dataset["prompt_len"]).mean()
    avg_output_len = np.array(train_dataset["response_len"]).mean()
    print(f"{avg_prompt_len=}, {avg_output_len=}")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)