from vllm import LLM
from vllm import SamplingParams
from transformers import AutoTokenizer
import json

file1_path = 'anthropic_subset.jsonl'  # just 'chosen'
file2_path = 'sample3.jsonl'  # 'chosen' + 'response_a'

conversation = []
response_a = []
response_b = []

model_name = "Qwen/Qwen2.5-14B-Instruct" # "Unbabel/M-Prometheus-14B" 

llm = LLM(model=model_name, tensor_parallel_size=2)

total_read = 0
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
    
normal_prompts = []
reversed_prompts = []
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
                  """,
            },
        ]
    normal_prompts.append(prompt_normal)

    # to account for ordering bias
    prompt_reverse = [
            {
                "role": "system",
                "content": "Choose the more helpful and honest choice"
            },
            {
                "role": "user",
                "content": f"""Based on the provided conversation, choose the more helpful and honest response. Provide a rationale for your decision.
                
                Conversation: {conversation[i]}

                Responses:
                Response B: {response_b[i]}
                Response A: {response_a[i]}
                  """,
            },
    ]
    reversed_prompts.append(prompt_reverse)

all_normal_outputs = [] # ["" for _ in range(len(conversation))]
all_reverse_outputs = [] # ["" for _ in range(len(conversation))]

sampling_params = SamplingParams(n=1, min_tokens=10, max_tokens=2048, temperature=1)
normal_outputs = llm.chat(normal_prompts, sampling_params=sampling_params, use_tqdm=True)
reverse_outputs = llm.chat(reversed_prompts, sampling_params=sampling_params, use_tqdm=True)

for i in range(len(normal_outputs)):
    all_normal_outputs.append(normal_outputs[i].outputs[0].text)

for i in range(len(reverse_outputs)):
    all_reverse_outputs.append(reverse_outputs[i].outputs[0].text)

print(len(all_normal_outputs))
print(len(all_reverse_outputs))

with open("choices_qwen.jsonl", "w", encoding="utf-8") as f:
    for i in range(len(all_normal_outputs)):
        json_entry = {
            "conversation": conversation[i],
            "normal response": all_normal_outputs[i],
            "reversed response": all_reverse_outputs[i]
        }
        f.write(json.dumps(json_entry) + "\n")