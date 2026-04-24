import re
import os
from typing import List, Tuple
import datasets

import pandas as pd
import argparse
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from kunserve.tokenizer import get_tokenizer

from datetime import datetime

def parse_timestamp(timestamp):
    # 预处理时区冒号
    if timestamp[-3] == ":":
        timestamp = timestamp[:-3] + timestamp[-2:]
    
    # 尝试不同格式解析
    for fmt in [
        "%Y-%m-%d %H:%M:%S.%f%z",  # 带微秒
        "%Y-%m-%d %H:%M:%S%z"      # 不带微秒
    ]:
        try:
            return datetime.strptime(timestamp, fmt)
        except ValueError:
            continue
    raise ValueError("无法解析时间戳")

def generate_prompt(
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        length: int) -> Tuple[str, List[int]]:
    LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

    lorem_ipsum_length = len(tokenizer.encode(LOREM_IPSUM))
    prompt = LOREM_IPSUM * ((length + lorem_ipsum_length - 1) // lorem_ipsum_length)

    prompt_token_ids = tokenizer.encode(prompt)
    prompt_token_ids = prompt_token_ids[:length]
    prompt = tokenizer.decode(prompt_token_ids)
    
    return prompt, prompt_token_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', default='/data/dataset/azureconv.csv')
    parser.add_argument('--local-dir', default='/data/dataset/azureconv-llama')
    parser.add_argument('--tokenizer', default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
    parser.add_argument('--limit', "-l", type=int, default=None)

    args = parser.parse_args()

    tokenizer = get_tokenizer(
        args.tokenizer, 
    ) if args.tokenizer else None

    dataset = pd.read_csv(args.src_dir)
    # add a row to each data item that represents a unique id
    processed_requests = []
    
    total = min(args.limit, len(dataset)) if args.limit is not None else len(dataset)
    global_start_ts = None
    for i, entry in tqdm(dataset.iterrows(), total=total):
        if args.limit is not None and i >= args.limit:
            break
        # prompt, prompt_token_ids = generate_prompt(tokenizer, entry['ContextTokens'])
        prompt_len = entry['ContextTokens']
        response_len = entry['GeneratedTokens']
        timestamp = parse_timestamp(entry['TIMESTAMP'])
        if global_start_ts is None:
            global_start_ts = timestamp

        data = {
            "time": timestamp,
            "prompt": "",
            "prompt_len": prompt_len,
            "response_len": response_len,
        }
        processed_requests.append(data)

    train_dataset = pd.DataFrame(processed_requests)
    train_dataset["time"] -= global_start_ts
    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))