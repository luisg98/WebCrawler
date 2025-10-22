import os
import torch
import pandas as pd
import re
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === CONFIG ===
RAW_PDF_DIR = "data/01_raw"
OUTPUT_DIR = "data/07_structured"
MODEL_DIR = "models/bert"
MIN_PROB = 0.5
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# === REQUIREMENT DEFINITIONS ===
REQUIREMENTS = [
    ("Show Ingredients List", ['ingredient', 'composition', 'list of'], "Ingredients must be listed."),
    ("Show Net Quantity", ['net weight', 'net quantity', 'volume', 'quantity'], "Net content must be shown."),
    ("Show Expiration Date", ['expiration', 'expiry', 'best before', 'sell-by'], "Expiration date must be present."),
    ("Show Manufacturer Info", ['manufacturer', 'packager', 'seller', 'name and address'], "Must show responsible party."),
    ("Use French Language", ['french', 'language', 'translation'], "Must include French text."),
    ("Show Storage Instructions", ['storage', 'keep refrigerated', 'temperature'], "Storage conditions must be provided."),
    ("Show Usage Instructions", ['instructions', 'how to use', 'directions'], "Usage instructions must be included."),
    ("Show Brand Name", ['brand', 'product name', 'sales name'], "Must display brand or product name.")
]

# === UTILITIES ===
def extract_text_from_pdf(pdf_path):
    from pdfminer.high_level import extract_text
    return extract_text(pdf_path)

def get_sentences(text):
    return [s.strip().replace("\n", " ") for s in sent_tokenize(text) if len(s.strip()) > 10]

def predict_sentences(sentences):
    results = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            prob = probs[0][1].item()

        results.append({"text": sentence, "predicted_label": pred, "prob_labeling": prob})
    return pd.DataFrame(results)

def extract_country_and_year(filename):
    match = re.search(r"_(.+)_([A-Z]{2}\d{4}-\d{4}|\d{4})", filename)
    if match:
        country = match.group(1).split("_")[-1].replace("'", "")
        year = re.search(r"(\d{4})", match.group(2))
        return country, year.group(1) if year else None
    return "UNKNOWN", None

def consolidate_requirements(df):
    results = []
    for name, keywords, desc in REQUIREMENTS:
        matches = df[(df["prob_labeling"] >= MIN_PROB) & (df["text"].str.lower().str.contains('|'.join(keywords)))]
        if not matches.empty:
            confidence = matches["prob_labeling"].max()
            results.append({
                "requirement_name": name,
                "description": desc,
                "confidence": round(confidence, 3)
            })
    return pd.DataFrame(results)

# === MAIN ===
if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(RAW_PDF_DIR) if f.lower().endswith(".pdf")]

    for fname in tqdm(pdf_files, desc="Processing PDFs"):
        path = os.path.join(RAW_PDF_DIR, fname)
        try:
            raw_text = extract_text_from_pdf(path)
            sentences = get_sentences(raw_text)
            df_preds = predict_sentences(sentences)
            df_filtered = df_preds[df_preds["prob_labeling"] >= MIN_PROB]
            df_reqs = consolidate_requirements(df_filtered)

            country, year = extract_country_and_year(fname)
            base_name = fname.replace(".pdf", "")
            out_dir = os.path.join(OUTPUT_DIR, base_name)
            os.makedirs(out_dir, exist_ok=True)

            df_reqs.to_csv(os.path.join(out_dir, "requirements.csv"), index=False)
            pd.DataFrame([{"country": country, "year": year, "total_requirements": len(df_reqs)}]).to_csv(
                os.path.join(out_dir, "summary.csv"), index=False
            )
        except Exception as e:
            print(f"Error with {fname}: {e}")

    print(f"\nStructured outputs saved to {OUTPUT_DIR}")
