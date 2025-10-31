import json
from collections import Counter
from nltk.util import ngrams
import nltk
import re

nltk.download('punkt')

def clean_tags_for_ngrams(text):
    return re.sub(r"(Human:|AI Assistant:)", "", text)

def get_5grams(text):
    cleaned = clean_tags_for_ngrams(text)
    tokens = nltk.word_tokenize(cleaned.lower())
    return list(ngrams(tokens, 5))

def convo_to_text(convo_str, response_str):
    return convo_str + " " + response_str

def is_exempt_5gram(gram):
    # Join back for easier checking
    joined = " ".join(gram)
    if "human" in joined or "assistant" in joined:
        return True
    if joined == "i 'd be happy to help":
        return True
    return False

MAX_REPEATS_IN_CONVO = 2  # allow up to 2 repeats, filter if >2

def repeated_5grams(text):
    grams = get_5grams(text)
    counts = Counter(grams)
    return [gram for gram, c in counts.items() if c > MAX_REPEATS_IN_CONVO]

all_5grams = []
convos_data = []

# Step 1: Read all conversations and extract all 5-grams
with open("sample5.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        convo = entry.get("conversation", "")
        response = entry.get("response_A", "")
        convo_text = convo_to_text(convo, response)
        convos_data.append((convo_text, entry))
        all_5grams.extend(get_5grams(convo_text))

# Step 2: Count frequencies of all 5-grams
freq_5grams = Counter(all_5grams)

THRESHOLD = 200  # Adjust this threshold as needed

common_5grams = set(gram for gram, count in freq_5grams.items() if count > THRESHOLD and is_exempt_5gram(gram))

def contains_common_5gram(text):
    grams = get_5grams(text)
    if any(gram in common_5grams for gram in grams):
        return True, None
    repeats =  repeated_5grams(text)
    if repeats:
        return True, repeats
    
    return False, None

# Step 3: Filter out conversations that contain any common 5-gram
filtered_entries = []
for convo_text, entry in convos_data:
    flagged, repeats = contains_common_5gram(convo_text)
    if not flagged:
        filtered_entries.append(entry)
    # else:
    #     if repeats:
    #         print("Filtered due to repeated 5-gram(s):",
    #               [ " ".join(r) for r in repeats ])

# Step 4: Save filtered conversations to a new JSONL file
with open("filtered_sample5.jsonl", "w", encoding="utf-8") as f_out:
    for entry in filtered_entries:
        json.dump(entry, f_out)
        f_out.write("\n")

print(f"Original number of conversations: {len(convos_data)}")
print(f"Number of conversations after filtering: {len(filtered_entries)}")
print(common_5grams)