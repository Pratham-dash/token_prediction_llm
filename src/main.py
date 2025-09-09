import collections
import time
import random
import matplotlib.pyplot as plt
import google.generativeai as genai
from google.api_core import exceptions
import os # Import the os module to access environment variables

# --- Configure Gemini API ---
# IMPORTANT: DO NOT hard-code your API key here.
# Store it in an environment variable and access it like this:
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY environment variable not set. Please set it securely.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Statement to complete ---
statement = "the primary colours are red blue green yellow"

# --- Function to call Gemini API with retries ---
def generate_candidates(statement, candidate_count=8, max_retries=5):
    """
    Generates content from the Gemini API with a retry mechanism for rate limits.

    Args:
        statement (str): The prompt for the Gemini model.
        candidate_count (int): The number of response candidates to generate.
        max_retries (int): The maximum number of retries for API requests.

    Returns:
        A response object from the Gemini API.
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                statement,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1,
                    candidate_count=candidate_count,
                    temperature=2
                )
            )
            return response
        except exceptions.TooManyRequests:
            print(f"⚠️ Rate limit hit (attempt {attempt+1}), retrying...")
            time.sleep(2 ** attempt + random.uniform(0, 1))  # Exponential backoff with jitter
        except exceptions.ServiceUnavailable:
            print(f"⚠️ Service unavailable (attempt {attempt+1}), retrying...")
            time.sleep(5)
    raise RuntimeError("❌ Max retries exceeded.")

# --- Generate completions ---
result = generate_candidates(statement, candidate_count=8)

# --- Extract unique next-word candidates and count frequency ---
next_words = [
    candidate.content.parts[0].text.strip()
    for candidate in result.candidates
    if candidate.content.parts and candidate.content.parts[0].text
]

# --- Count unique words and their frequencies ---
word_counts = collections.Counter(next_words)
total_candidates = len(next_words)

# --- Prepare data for plotting ---
unique_words = list(word_counts.keys())
probs = [(count / total_candidates) * 100 for count in word_counts.values()]

# --- Combine words and probabilities and sort by probability (most probable first) ---
combined = list(zip(unique_words, probs))
combined.sort(key=lambda x: x[1], reverse=True)

# --- Extract sorted words and probabilities ---
sorted_words = [w for w, p in combined]
sorted_probs = [p for w, p in combined]

# --- Print results ---
print("Most probable next-word candidates (approximate probability):")
for w, p in zip(sorted_words, sorted_probs):
    print(f"{w} : {p:.2f}%")

# --- Plot bar chart ---
plt.figure(figsize=(7, 5))
plt.bar(sorted_words, sorted_probs, color="skyblue")
plt.xlabel("Next Word")
plt.ylabel("Approximate Probability (%)")
plt.title(f"Most Probable Next Word Candidates for: \"{statement}\"")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
