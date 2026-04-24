import re
import os
from typing import List, Tuple
import datasets

import pandas as pd
import argparse
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from kunserve.tokenizer import get_tokenizer

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
    parser.add_argument('--src-dir', default='/data/dataset/burstgpt.csv')
    parser.add_argument('--local-dir', default='/data/dataset/burstgpt-new-llama')
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
    for i, entry in tqdm(dataset.iterrows(), total=total):
        if args.limit is not None and i >= args.limit:
            break
        prompt, prompt_token_ids = generate_prompt(tokenizer, entry['Request tokens'])
        response_len = entry['Response tokens']

        if len(prompt_token_ids) + response_len > 4096:
            prompt_len = 4096 - response_len
            prompt_token_ids = prompt_token_ids[:prompt_len]
            prompt = tokenizer.decode(prompt_token_ids)

        data = {
            "model": entry['Model'],
            "type": entry['Log Type'],
            "prompt": prompt,
            "prompt_len": len(prompt_token_ids),
            "response_len": response_len,
        }
        processed_requests.append(data)

    train_dataset = pd.DataFrame(processed_requests)
    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))