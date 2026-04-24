import re
import os
import datasets

from hdfs_io import copy, makedirs
import argparse

from kunserve.tokenizer import get_tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-dir', default='~/data/chinese-llama')
    parser.add_argument('--hdfs-dir', default=None)
    parser.add_argument('--tokenizer', default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

    args = parser.parse_args()

    tokenizer = get_tokenizer(
        args.tokenizer, 
    ) if args.tokenizer else None

    data_source = "Congliu/Chinese-DeepSeek-R1-Distill-data-110k"
    dataset = datasets.load_dataset(data_source, 'default')
    train_dataset = dataset['train']
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(sample, idx):
            question = sample.pop('input')
            response = sample.pop('reasoning_content') + " " + sample.pop('content')
            prompt_token_ids = tokenizer.encode(question)
            response_token_ids = tokenizer.encode(response)

            data = {
                "data_source": data_source,
                "prompt": question,
                "response": response,
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
