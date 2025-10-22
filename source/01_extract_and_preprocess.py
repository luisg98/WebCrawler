import os
import re
import logging
import pdfplumber
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# === CONFIG ===
RAW_DIR = "data/01_raw"
PROCESSED_DIR = "data/02_processed"
SECTIONS_DIR = "data/03_sections"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SECTIONS_DIR, exist_ok=True)

logging.getLogger("pdfminer").setLevel(logging.ERROR)

# === STOPWORDS & LEMMATIZER ===
STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# === STEP 1: Extract text from PDF ===
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

def extract_all_pdfs(input_dir):
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    for filename in tqdm(pdf_files, desc="Extracting text from PDFs"):
        pdf_path = os.path.join(input_dir, filename)
        text = extract_text_from_pdf(pdf_path)
        output_path = os.path.join(PROCESSED_DIR, filename.replace(".pdf", ".txt"))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

# === STEP 2: Preprocess text ===
def clean_and_lemmatize(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

def preprocess_all():
    txt_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".txt")]
    for filename in tqdm(txt_files, desc="Preprocessing text"):
        file_path = os.path.join(PROCESSED_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        sentences = sent_tokenize(raw_text)
        cleaned = [clean_and_lemmatize(s) for s in sentences if s.strip()]
        with open(file_path, "w", encoding="utf-8") as f:
            for line in cleaned:
                f.write(line + "\n")

# === STEP 3: Extract labeling sections ===
LABELING_PATTERNS = [
    r"section\s+ii\b.*labeling", r"\blabeling requirements\b", r"\blabelling requirements\b",
    r"\blabeling (of|for)\b", r"\blabel(ling)? laws\b", r"\blabel(ling)? regulations\b",
    r"\brequirements for (food )?labeling\b", r"\blabeling.*alcohol",
    r"\betiquetado\b", r"\brotulagem\b", r"\betiquetage\b", r"\betiquette\b"
]

END_PATTERNS = [
    r"section\s+(iii|3)\b", r"\bpackaging and container regulations\b",
    r"\bfood additive regulations\b", r"\bpesticide(s)? and contaminants\b",
    r"\bimport procedures\b", r"\btrade facilitation\b", r"\bappendix\b",
    r"\bother requirements\b", r"\bgeographical indicators\b", r"\btrademarks\b",
    r"\bintellectual property\b", r"\bregistration\b"
]

start_regex = re.compile("|".join(LABELING_PATTERNS), re.IGNORECASE)
end_regex = re.compile("|".join(END_PATTERNS), re.IGNORECASE)

def extract_labeling_sections(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections, current = [], []
    capturing = False
    for line in lines:
        if start_regex.search(line):
            if not capturing:
                capturing = True
                current = [line]
            else:
                current.append(line)
            continue
        if capturing:
            if end_regex.search(line):
                sections.append("\n".join(current))
                capturing = False
                current = []
            else:
                current.append(line)
    if capturing and current:
        sections.append("\n".join(current))
    return sections

def extract_sections_all():
    txt_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".txt")]
    for filename in tqdm(txt_files, desc="Extracting labeling sections"):
        file_path = os.path.join(PROCESSED_DIR, filename)
        out_path = os.path.join(SECTIONS_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        sections = extract_labeling_sections(text)
        if sections:
            with open(out_path, "w", encoding="utf-8") as out:
                for i, sec in enumerate(sections):
                    out.write(f"[SECTION {i+1}]\n{sec.strip()}\n\n")

# === MAIN ===
if __name__ == "__main__":
    extract_all_pdfs(RAW_DIR)
    preprocess_all()
    extract_sections_all()
    print("\nProcessing complete! Sections extracted to:", SECTIONS_DIR)
