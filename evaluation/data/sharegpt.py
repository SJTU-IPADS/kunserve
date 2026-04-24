import re
import os
import datasets
import numpy as np
import pandas as pd
import argparse

from kunserve.tokenizer import get_tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', default='/data/dataset/sharegpt_json')
    parser.add_argument('--local-dir', default='/data/dataset/sharegpt-llama')
    parser.add_argument('--hdfs-dir', default=None)
    parser.add_argument('--tokenizer', default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

    args = parser.parse_args()

    tokenizer = get_tokenizer(
        args.tokenizer, 
    ) if args.tokenizer else None

    processed_requests = []
    for file in os.listdir(args.src_dir):
        if file.endswith('.jsonl'):
            json_dir = os.path.join(args.src_dir, file)
            dataset = pd.read_json(json_dir, lines=True)["conversations"]
            # add a row to each data item that represents a unique id 
            for entry in dataset:
                prompts = [item['value'] for item in entry[:-1]]
                
                prompt = ''.join(prompts)
                response = entry[-1]['value']

                prompt_token_ids = tokenizer.encode(prompt)
                response_token_ids = tokenizer.encode(response)

                # Truncate the prompt due to GPT-4's limited context
                if len(prompt_token_ids) + len(response_token_ids) > 4096:
                    prompt_len = 4096 - len(response_token_ids)
                    prompt_token_ids = prompt_token_ids[:prompt_len]
                    prompt = tokenizer.decode(prompt_token_ids)

                data = {
                    "prompt": prompt,
                    "response": response,
                    "prompt_len": len(prompt_token_ids),
                    "response_len": len(response_token_ids),
                }
                processed_requests.append(data)
    
    train_dataset = pd.DataFrame(processed_requests)

    prune_ratio = 1
    head_prompt_len = np.percentile(train_dataset['prompt_len'], prune_ratio)
    tail_prompt_len = np.percentile(train_dataset['prompt_len'], 100 - prune_ratio)
    head_response_len = np.percentile(train_dataset['response_len'], prune_ratio)
    tail_response_len = np.percentile(train_dataset['response_len'], 100 - prune_ratio)
    pruned_dataset = train_dataset[(train_dataset['prompt_len'] > head_prompt_len) & (train_dataset['prompt_len'] < tail_prompt_len)]
    train_dataset = pruned_dataset[(pruned_dataset['response_len'] > head_response_len) & (pruned_dataset['response_len'] < tail_response_len)]

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)