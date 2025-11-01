from vllm import LLM
from vllm import SamplingParams
from transformers import AutoTokenizer
import json
import re
import argparse
import os

file1_path = 'anthropic_subset.jsonl'  # just 'chosen'
file2_path = 'finalconvos.jsonl'  # 'chosen' + 'response_a'


def parse_args():
    parser = argparse.ArgumentParser(description='Run conversation selection in batches')
    parser.add_argument('--gpu_id', type=str, default="0", help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--batch_start', type=int, default=0, help='Starting batch index')
    parser.add_argument('--batch_count', type=int, default=1, help='Number of batches to process')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of items per batch')
    parser.add_argument('--output_suffix', type=str, default="", help='Suffix for output file')
    return parser.parse_args()

def main():
    args = parse_args()
    gpu_ids = [x.strip() for x in args.gpu_id.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    print(f"Using GPU ID: {args.gpu_id}")

    use_anthropic = False
    
    model_name = "Qwen/Qwen3-32B" # "Qwen/Qwen3-14B" # "Qwen/Qwen2.5-14B-Instruct"  "Qwen/Qwen3-30B-A3B-Instruct-2507"  "Unbabel/M-Prometheus-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, tensor_parallel_size=2, gpu_memory_utilization=0.85, max_num_seqs=64)

    batch_start = args.batch_start
    batch_end = batch_start + args.batch_count
    batch_size = args.batch_size

    for batch_idx in range(batch_start, batch_end):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        print(f"\nProcessing batch {batch_idx} (lines {start_idx}–{end_idx})")

        conversation = []
        response_a = []
        response_b = []

        total_read = 0
        if use_anthropic:
            with open(file1_path, 'r') as f:
                for line in f:
                    if total_read >= 20:
                        break
                    item = json.loads(line)
                    chosen = item['chosen']
                    rejected = item['rejected']
                    split_index1 = chosen.rfind("Assistant:")
                    split_index2 = rejected.rfind("Assistant:")
                    conversation.append(chosen[:split_index1])
                    response_a.append(chosen[split_index1:])
                    response_b.append(rejected[split_index2:])
                    total_read += 1
        else:
            with open(file2_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    if line_idx < start_idx:
                        continue
                    if line_idx >= end_idx:
                        break
                    item = json.loads(line)
                    chosen = item['conversation']
                    response_A = item['response_A']
                    response_B = item['response_B']
                    conversation.append(chosen)
                    response_a.append(response_A)
                    response_b.append(response_B)
                    total_read += 1

        print(f"Total conversations read for this batch: {total_read}") 

        input_tokens = []
        normal_prompts = []
        for i in range(len(conversation)):
            prompt_normal = [
                    {
                        "role": "system",
                        "content": "Choose the more helpful and honest choice"
                    },
                    {
                        "role": "user",
                        "content": f"""Based on the provided conversation, choose the more helpful and honest response. Provide a rationale for your decision.
                        
                        Conversation: {conversation[i]}

                        Responses:
                        Response A: {response_a[i]}
                        Response B: {response_b[i]}

                        Respond in the following JSON format:
                        [
                            "answer": "A" or "B",
                            "reasoning": "your explanation here"
                        ]
                        """,
                    },
                ]
            normal_prompts.append(prompt_normal)

            input_tokens.append(len(tokenizer.encode(prompt_normal[1]['content'])))

        all_normal_outputs = [] # ["" for _ in range(len(conversation))]
        all_normal_tokens = []

        sampling_params = SamplingParams(n=1, min_tokens=10, max_tokens=2048, temperature=1)
        print("Running normal prompts...")
        normal_outputs = llm.chat(normal_prompts, sampling_params=sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": True})

        for i in range(len(normal_outputs)):
            all_normal_outputs.append(normal_outputs[i].outputs[0].text)
            all_normal_tokens.append(len(normal_outputs[i].outputs[0].token_ids))

        with open("choices_qwen.jsonl", "a", encoding="utf-8") as f:
            for i in range(len(all_normal_outputs)):
                think_match_normal = re.search(r"<think>\s*(.*?)\s*</think>", all_normal_outputs[i], re.DOTALL)
                thinking_normal = think_match_normal.group(1).strip() if think_match_normal else None

                answer_match_normal = re.search(r'"answer"\s*:\s*"([^"]+)"', all_normal_outputs[i])
                reasoning_match_normal = re.search(r'"reasoning"\s*:\s*"([^"]+)"', all_normal_outputs[i])

                answer_normal = answer_match_normal.group(1).strip() if answer_match_normal else None
                reasoning_normal = reasoning_match_normal.group(1).strip() if reasoning_match_normal else None

                json_entry = {
                    "conversation": conversation[i],
                    "input tokens": input_tokens[i],
                    "normal thinking": thinking_normal,
                    "normal answer": answer_normal,
                    "normal reasoning": reasoning_normal,
                    "normal tokens": all_normal_tokens[i],
                }
                f.write(json.dumps(json_entry) + "\n")

        print("Average input tokens ", sum(input_tokens) / len(input_tokens))
        print("Average output tokens ", sum(all_normal_tokens) / len(all_normal_tokens))

if __name__ == "__main__":
    main()