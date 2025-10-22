import os
import pandas as pd
import torch
import joblib
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === CONFIG ===
INPUT_DIR = "data/02_processed"
OUTPUT_DIR = "data/05_predictions"
CLASSICAL_MODEL_PATH = "models/classical/logreg_model.pkl"
VECTORIZER_PATH = "models/classical/bow_vectorizer.pkl"
BERT_MODEL_DIR = "models/bert"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === COMMON UTILITIES ===
def clean_sentence(text):
    return text.strip().replace("\n", " ")

def get_sentences_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return [clean_sentence(s) for s in sent_tokenize(raw_text) if len(s.strip()) > 10]

# === PREDICT WITH CLASSICAL MODEL ===
def predict_with_classical():
    model = joblib.load(CLASSICAL_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    for fname in tqdm(os.listdir(INPUT_DIR), desc="Classical model predictions"):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(INPUT_DIR, fname)
        sentences = get_sentences_from_file(path)
        if not sentences:
            continue
        X_vec = vectorizer.transform(sentences)
        preds = model.predict(X_vec)
        probs = model.predict_proba(X_vec)[:, 1]

        df = pd.DataFrame({
            "text": sentences,
            "predicted_label": preds,
            "prob_labeling": probs
        })
        df.to_csv(os.path.join(OUTPUT_DIR, fname.replace(".txt", "_classical.csv")), index=False)

# === PREDICT WITH BERT ===
def predict_with_bert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR).to(device)
    model.eval()

    for fname in tqdm(os.listdir(INPUT_DIR), desc="BERT model predictions"):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(INPUT_DIR, fname)
        sentences = get_sentences_from_file(path)
        if not sentences:
            continue

        results = []
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                prob_labeling = probs[0][1].item()

            results.append({
                "text": sentence,
                "predicted_label": pred,
                "prob_labeling": prob_labeling
            })

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(OUTPUT_DIR, fname.replace(".txt", "_bert.csv")), index=False)

# === MAIN ===
if __name__ == "__main__":
    predict_with_classical()
    predict_with_bert()
    print("\nPredictions completed and saved to:", OUTPUT_DIR)
