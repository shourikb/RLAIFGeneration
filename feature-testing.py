import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
import matplotlib.pyplot as plt

# Download required NLTK package (run once)
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def get_ngrams(text, n=8):
    """Extracts n-grams from tokenized text using NLTK."""
    tokens = word_tokenize(text.lower())  # Tokenize & lowercase for consistency
    tokens = remove_stopwords(tokens)
    return set(ngrams(tokens, n))

def compute_match_percentage(test_text, train_text):
    """Computes the percentage of test tokens that are part of an 8-gram match with train text."""
    test_ngrams = get_ngrams(test_text, n=8)
    train_ngrams = get_ngrams(train_text, n=8)

    common_ngrams = test_ngrams.intersection(train_ngrams)
    
    if not test_ngrams:
        return 0.0  # Avoid division by zero

    return (len(common_ngrams) / len(test_ngrams)) * 100  # Percentage of test n-grams that match

def remove_stopwords(tokens):
    """Removes stopwords from the given text."""
    filtered_words = [word for word in tokens if word.lower() not in stop_words]  # Remove stopwords
    return ' '.join(filtered_words)

def distinct_n(sentences, n):
    all_ngrams = []
    for s in sentences:
        tokens = word_tokenize(s.lower())
        all_ngrams.extend(list(ngrams(tokens, n)))
    total = len(all_ngrams)
    unique = len(set(all_ngrams))
    return unique / total if total > 0 else 0

# Example texts
# train_text = "I'm organizing a historical reenactment for this weekend and I need help narrowing down a list of potential locations for the event - I've got a few places in mind, like the former location of the old Tampa Bay Hotel in Ybor City, but are there any other sites in the Tampa Bay area that might be suitable for historical reenactments?"
# test_text = "I'm planning a historical reenactment this weekend and need help selecting a location. I have a few options in mind, such as the former site of the old Tampa Bay Hotel in Ybor City, but I'm looking for other suitable historical sites in the Tampa Bay area. Any suggestions?"

# # Compute match percentage
# match_percentage = compute_match_percentage(test_text, train_text)
# print(f"Match Percentage: {match_percentage:.2f}%")

# # Check if match is significant (above 50%)
# if match_percentage > 50:
#     print("Significant overlap detected!")
# else:
#     print("No significant overlap.")

with open("batch_human_prompts.txt", "r") as file:
    content = file.read()

responses = content.split("NEXT RESPONSE")

# Optionally, strip any leading/trailing whitespace from each response
responses = [response.strip() for response in responses if response.strip()]

print("Starting match testing process")
max_match_pct = 0
for i in range(len(responses)):
    for j in range(i + 1, len(responses)):
        match_percentage = compute_match_percentage(responses[i], responses[j])
        # print(f"Match Percentage: {match_percentage:.2f}%" + " between conversations " + str(i + 1) + " and " + str(j + 1))
        if match_percentage > 15:
             print("Significant overlap between " + responses[i] + " and " + responses[j] + " with match percentage " + str(match_percentage))
        max_match_pct = max(max_match_pct, match_percentage)

# step = 1000  # you can go smaller for smoother curve, e.g., 500
# sample_sizes = list(range(step, len(responses) + 1, step))
# distinct_2_scores = []

# # --- Compute distinct-2 for increasing sample sizes ---
# for size in sample_sizes:
#     sample = responses[:size]
#     score = distinct_n(sample, 2)
#     distinct_2_scores.append(score)
#     print(score)

# # --- Plot ---
# plt.figure(figsize=(10, 6))
# plt.plot(sample_sizes, distinct_2_scores, label='Distinct-2', color='teal', marker='o')
# plt.xlabel('Number of Responses Sampled')
# plt.ylabel('Distinct-2 Score')
# plt.title('Distinct-2 Diversity Over Sample Size')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("distinct2_plot.png")
# plt.show()

# print("N-gram stuff is " + str(unique / total))

print("Greatest match percentage is " + str(max_match_pct))