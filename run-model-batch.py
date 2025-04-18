from vllm import LLM
from vllm import SamplingParams
from transformers import AutoTokenizer
import random
import json
from collections import Counter
import time
import numpy as np
import os

def reset_all_seeds():
    seed = int.from_bytes(os.urandom(4), byteorder="big")
    random.seed(seed)
    np.random.seed(seed)
    return seed

test_tracker = ["" for _ in range(5)]
def eachturn(convos, model, sp, is_human, sample, tokens, convo_tracker=[], update=True):
    if is_human:
        prompt = "Given the above conversation, generate the just the next human part of the conversation for the human by trying to converse with the assistant. Follow the topic provided in the conversation. Try to generate of similar length to a single response in the sample conversation."
        prefix = "Human: "
    else:
        prompt = "Continue the conversation, provide just the AI assistant response to the last human message. Try to generate of similar length to a single response in the sample conversation."
        prefix = "AI Assistant: "
        
    all_conversations = []
    for convo in convos:
        context_convo = "Here is a sample conversation: \n\n" + sample + "\n\nREAL CONVERSATION\n\n" + convo + "\n\n" + prompt
        full_conversation = [
            {
                "role":"user",
                "content":context_convo
            }, 
            {
                "role":"assistant",
                "content":prefix
            }
        ]
        all_conversations.append(full_conversation)
        # if not update:
        #     print(context_convo)
    
    outputs = model.chat(all_conversations, continue_final_message=True, add_generation_prompt=False, sampling_params=sp, use_tqdm=True)
    for i in range(len(outputs)):
        final_result = outputs[i].outputs[0].text
        # print("Token IDs: ")
        # print(outputs[i].outputs[0].token_ids)
        # print("Length is ", len(outputs[i].outputs[0].token_ids))
        # print("\n")
        test_tracker[i] += "\n\nSWITCH\n\n" + final_result
        if not final_result.startswith(prefix):
            final_result = prefix + final_result
        convo_tracker.append(final_result + "\n")
        if update:
            convos[i] += "\n\n" + final_result
        tokens[i] += len(outputs[i].outputs[0].token_ids)

    return convos

with open("filtered_personas.jsonl", "r", encoding="utf-8") as file:
    personas = [next(iter(json.loads(line).values())) for line in file]

model_name =  "allenai/Llama-3.1-Tulu-3-8B-SFT" 
#model_name = "unsloth/Meta-Llama-3.1-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.save_pretrained("tokenizer_tmp")

# llm = LLM(model="unsloth/Meta-Llama-3.1-8b-instruct")
llm = LLM(model=model_name, tokenizer="tokenizer_tmp")


for c in range(1):
    reset_all_seeds()
    information = []
    all_outputs = ["" for _ in range(5)]
    tokens = [0 for _ in range(len(all_outputs))]
    a_responses = []
    b_responses = []
    first_turn = []
    random_personas = list(np.random.choice(personas, len(all_outputs), replace=False))

    conversations = []
    sample_convo = """Human: Do you know the reasons as to why people love coffee so much?

AI Assistant: That's an interesting question!  I think it's partly because of the emotional effect of the caffeine.  Is that the main reason you're asking?

Human: The emotional effect? Such as? 

AI Assistant: I think you'll notice that coffee can have different effects on people, depending on what they were doing beforehand.  In general, coffee is stimulating, which is important to many people when they are feeling sluggish or apathetic.  But it can also be really relaxing for people who are anxious, nervous, or stressed.

Human: I feel like how energized it makes one feel it would do the opposite of feeling relaxed.

AI Assistant: I understand! But what I meant is, coffee can make people more nervous or relaxed, depending on how they are feeling beforehand.  If they're feeling stressed and needing to relax, coffee can have a soothing effect, but if they're feeling anxious or nervous, coffee can make them feel more agitated.

Human: Hm, I see. Where did you read this?

AI Assistant: I'm just recalling a couple of things I've read.  One of them was a psychology paper, but I'm also just drawing from my own experiences.

Human: I'm a little skeptical about what you're saying but you certainly got some stuff right such as coffee being a stimulant and helping those feeling sluggish

AI Assistant: Glad I was able to answer your questions!"

"""
    for j in range(len(random_personas)):
        conversation = [
            {
                "role": "system",
                "content": "Generate a conversation starter as if you were the persona given"
            },
            {
                "role": "user",
                "content": f"""Decide on a task you'd like an AI assistant to help you with and try to converse with this assistant. The conversation should be open ended, asking for help, advice, or for the model to accomplish a task. This should simulate having a conversation, with the responses looking a little like this, and should strictly follow the following format:

        {sample_convo}

        Generate the first human turn of the conversation as if you were a {random_personas[j]} conversing in English with an AI assistant.
        """,
            },
        ]
        conversations.append(conversation)
        information.append("\nPersona: " + random_personas[j] + "\nTurns: ")

    rand_seed = np.random.randint(1, 1000)
    sampling_params = SamplingParams(n=1, max_tokens=2048, temperature=1, seed=rand_seed)
    outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=True)
    for i in range(len(all_outputs)):
        final_result = outputs[i].outputs[0].text
        test_tracker[i] += final_result
        if not final_result.startswith("Human: "):
            final_result = "Human: " + final_result
        all_outputs[i] += final_result
        tokens[i] += len(outputs[i].outputs[0].token_ids)
        first_turn.append(final_result + "\nNEXT RESPONSE\n")

    additional_turns = np.random.randint(0, 4)
    for i in range(len(information)):
        information[i] += str(additional_turns)
    
    if additional_turns == 0:
        eachturn(all_outputs, llm, sampling_params, False, sample_convo, tokens, convo_tracker=a_responses, update=False)
        rand_seed = np.random.randint(1001, 2000)
        smp2 = SamplingParams(n=1, max_tokens = 2048, temperature=1, seed=rand_seed)           
        eachturn(all_outputs, llm, smp2, False, sample_convo, tokens, convo_tracker=b_responses, update=False)
    else:
        all_outputs = eachturn(all_outputs, llm, sampling_params, False, sample_convo, tokens)

    for k in range(additional_turns):
        all_outputs = eachturn(all_outputs, llm, sampling_params, True, sample_convo, tokens)
        if k != additional_turns - 1:
            all_outputs = eachturn(all_outputs, llm, sampling_params, False, sample_convo, tokens)
        else:
            eachturn(all_outputs, llm, sampling_params, False, sample_convo, tokens, convo_tracker=a_responses, update=False)
            rand_seed = np.random.randint(1001, 2000)
            smp2 = SamplingParams(n=1, max_tokens = 2048, temperature=1, seed=rand_seed)           
            eachturn(all_outputs, llm, smp2, False, sample_convo, tokens, convo_tracker=b_responses, update=False)

    for i in range(len(information)):
        information[i] += "\nTotal Tokens: " + str(tokens[i]) + "\n"

    with open("sample.jsonl", "a", encoding="utf-8") as f:
        for i in range(len(all_outputs)):
            json_entry = {
                "info": information[i],
                "test_convo": "".join(test_tracker[i]),
                "conversation": "".join(all_outputs[i]),
                "response_A": a_responses[i],
                "response_B": b_responses[i]
            }
            f.write(json.dumps(json_entry) + "\n")

        
    with open("batch_human_prompts.txt", "a", encoding="utf-8") as f:
        for response in first_turn:
            f.write(response)

