import os
import re
import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# === CONFIG ===
SECTIONS_DIR = "data/03_sections"
PROCESSED_DIR = "data/02_processed"
OUTPUT_CSV = "data/04_labeled/labeled_sentences.csv"
os.makedirs("data/04_labeled", exist_ok=True)

# === LOAD AND CLEAN SENTENCES ===
def clean_sentence(sentence):
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()

def load_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [clean_sentence(s) for s in sent_tokenize(text) if len(s.strip()) >= 10]

# === COLLECT POSITIVES ===
def collect_positives():
    data = []
    for fname in os.listdir(SECTIONS_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(SECTIONS_DIR, fname)
            for sent in load_sentences(path):
                data.append({"text": sent, "label": 1, "source_file": fname})
    return data

# === COLLECT NEGATIVES ===
def collect_negatives(positive_map, limit_per_file=None):
    data = []
    for fname in os.listdir(PROCESSED_DIR):
        if fname.endswith(".txt") and fname in positive_map:
            path = os.path.join(PROCESSED_DIR, fname)
            full_sentences = load_sentences(path)
            positives_set = set(positive_map[fname])
            negatives = [s for s in full_sentences if s not in positives_set]
            sampled = negatives[:limit_per_file] if limit_per_file else negatives
            for sent in sampled:
                data.append({"text": sent, "label": 0, "source_file": fname})
    return data

# === MAIN ===
def build_labeled_dataset():
    print("Building labeled dataset...")
    positives = collect_positives()
    positive_map = {}
    for item in positives:
        positive_map.setdefault(item["source_file"], []).append(item["text"])

    negatives = collect_negatives(positive_map)
    negatives = negatives[:len(positives)]  # balance

    df = pd.DataFrame(positives + negatives).drop_duplicates(subset=["text"])
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(df)} labeled examples to {OUTPUT_CSV}")

if __name__ == "__main__":
    build_labeled_dataset()
