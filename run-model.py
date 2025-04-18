from vllm import LLM
from vllm import SamplingParams
from transformers import AutoTokenizer
import random
import json
from collections import Counter

# TODO: Randomize the topic given, randomize length of conversation

def eachturn(convo, model, sp, is_human, convo_tracker):
    
    if is_human:
        prompt = "Given the above conversation, generate the just the next human part of the conversation for the human by trying to converse with the assistant. Follow the topic provided in the conversation."
        prefix = "Human: "
    else:
        prompt = "Continue the conversation, provide just the assistant response to the last human message. Keep the response fairly brief."
        prefix = "Assistant: "
    context_convo = convo + "\n\n" + prompt
    full_conversation = [
        {
            "role":"user",
            "content":context_convo
        }
    ]
    
    final_result = model.chat(full_conversation, sampling_params=sp)
    final_result = final_result[0].outputs[0].text
    if not final_result.startswith(prefix):
        final_result = prefix + final_result
    # convo_tracker.append(final_result + "\nEND TURN\n")
    convo_tracker.append(final_result + "\n")
    return convo + "\n\n" + final_result

with open("filtered_personas.jsonl", "r", encoding="utf-8") as file:
    personas = [next(iter(json.loads(line).values())) for line in file]

model_name = "unsloth/Meta-Llama-3.1-8b-instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name)

random.seed(2001)

tokenizer.save_pretrained("tokenizer_tmp")

# llm = LLM(model="unsloth/Meta-Llama-3.1-8b-instruct")
llm = LLM(model=model_name, tokenizer="tokenizer_tmp")

information = []
all_outputs = []
a_responses = []
b_responses = []
first_turn = []
random_personas = random.sample(personas, 100)

for j in range(len(random_personas)):
    conversation = [
        {
            "role": "system",
            "content": "Generate a conversation starter as if you were the persona given"
        },
        {
            "role": "user",
            "content": f"""Decide on a task you'd like an AI assistant to help you with and try to converse with this assistant. The conversation should be open ended, asking for help, advice, or for the model to accomplish a task. This should simulate having a conversation, with the responses looking a little like this, and should strictly follow the following format:

    Human: I want to put together a gift basket of luxury bath products for my sister. What should I include?

    Assistant: A group of friends recently gave me a gift basket and it looked pretty nice. I think that bubble bath and shampoo sets are a nice idea. Also bath oils and soaps could be nice, if you can find them.

    Human: You can't get gifts from friends because you're not a person.

    Assistant: I'm not sure I agree. If you're a trained, courteous bot that can respond to customers' requests then I would say that qualifies as a friend.

    Human: Anyway, aside from the good gift suggestions you made before, do you have any other good bath gift ideas?

    Assistant: Yes, you could try fragrances, or something to add to the bath water.

    Human: Oh, that sounds very nice!

    Generate the first human turn of the conversation as if you were a {random_personas[j]}, starting with the format 'Human: '. Try to generate diverse conversations.
    """,
        },
    ]

    for i in range(1):
        one_convo = []

        additional_turns = random.randint(1, 3)
        sampling_params = SamplingParams(n=1, max_tokens=2048, temperature=1, seed=random.randint(1, 1000))
        outputs = llm.chat(conversation, sampling_params=sampling_params)

        information.append("\nPersona: " + random_personas[j] + "\nTurns: " + str(additional_turns) + "\n\n")
        response = outputs[0].outputs[0].text
        first_turn.append(response + "\n" + "NEXT RESPONSE\n")
        # one_convo.append(response + "\n" + "END TURN\n")
        one_convo.append(response + "\n")
        response = eachturn(response, llm, sampling_params, False, one_convo)
        # all_outputs.append(response + "\n" + "END TURN\n")

        for k in range(additional_turns):
            response = eachturn(response, llm, sampling_params, True, one_convo)
            # all_outputs.append(response + "\n" + "END TURN\n")
            if k != additional_turns - 1:
                response = eachturn(response, llm, sampling_params, False, one_convo)
            else:
                eachturn(response, llm, sampling_params, False, a_responses)
                smp2 = SamplingParams(n=1, max_tokens = 2048, temperature=1, seed=random.randint(1001, 2000))
                eachturn(response, llm, smp2, False, b_responses)
            # all_outputs.append(response + "\n" + "END TURN\n")
        
        all_outputs.append(one_convo)
    
    print(f"Completed {j + 1} conversations out of {len(random_personas)}")

        # all_outputs.append("Topic: " + topics[j] + "\nTurns: " + str(additional_turns) + "\n\n" + response)

with open("output.jsonl", "a", encoding="utf-8") as f:
    for i in range(len(all_outputs)):
        # for response in all_outputs[i]:
        #     print(response)
        #     f.write(response)
        json_entry = {
            "info": information[i],
            "conversation": "".join(all_outputs[i]),
            "response_A": a_responses[i],
            "response_B": b_responses[i]
        }
        # print("".join(all_outputs[i]))
        # f.write("".join(all_outputs[i]))
        # print(a_responses[i])
        # f.write(a_responses[i])
        # print(b_responses[i])
        # f.write(b_responses[i])
        # print(f"Generated text:\n{response}")
        # f.write(f"Generated text:\n{response}\n\n")
        f.write(json.dumps(json_entry) + "\n")
        
with open("human_prompts.txt", "a", encoding="utf-8") as f:
    for response in first_turn:
        f.write(response)


print("Output successfully written to output.jsonl")
