from vllm import LLM
from vllm import SamplingParams
from transformers import AutoTokenizer
import random
import json
from collections import Counter
import time
import numpy as np
import os
import argparse

def main():
    with open("bad_conversations.jsonl", "r", encoding="utf-8") as file:
        rate_conversations = [json.loads(line)["conversation"] + json.loads(line)["response_A"] for line in file]

    model_name =  "allenai/Llama-3.1-Tulu-3-8B-SFT" 
    # model_name = "unsloth/Meta-Llama-3.1-8b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.save_pretrained("tokenizer_tmp")

    # llm = LLM(model="unsloth/Meta-Llama-3.1-8b-instruct")
    llm = LLM(model=model_name, tokenizer="tokenizer_tmp")
    conversations = []
    
    for j in range(len(rate_conversations)):
        conversation = [
            {
                "role": "system",
                "content": "Rate the provided conversation on a scale of 1 to 10 based on the provided criteria where 1 is bad and 10 is good"
            },
            {
                "role": "user",
                "content": f"""Using the criteria of the conversation being coherent and avoiding repetition, what would you rate the following conversation and why: {rate_conversations[j]}
        """,
            },
        ]
        conversations.append(conversation)

    rand_seed = np.random.randint(1, 1000)
    sampling_params = SamplingParams(n=1, min_tokens=10, max_tokens=2048, temperature=1, seed=rand_seed)
    outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=True)
    time.sleep(1)
    
    with open("rate_convos2.txt", "a", encoding="utf-8") as f:
        for output in outputs:
            f.write(output.outputs[0].text + "\n\n")

    
if __name__ == "__main__":
    main()