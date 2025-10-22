import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import evaluate
import torch
import joblib

# === CONFIG ===
DATA_PATH = "data/04_labeled/labeled_sentences.csv"
CLASSICAL_MODEL_DIR = "models/classical"
BERT_MODEL_DIR = "models/bert"
os.makedirs(CLASSICAL_MODEL_DIR, exist_ok=True)
os.makedirs(BERT_MODEL_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH).dropna()
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === TRAIN CLASSICAL MODELS ===
def train_classical(name, vectorizer, model, model_file, vec_file):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    print(f"\n{name} CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, digits=3))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Not Labeling", "Labeling"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()

    joblib.dump(model, os.path.join(CLASSICAL_MODEL_DIR, model_file))
    joblib.dump(vectorizer, os.path.join(CLASSICAL_MODEL_DIR, vec_file))

print("\n--- CLASSICAL MODELS ---")
train_classical("BoW + Logistic Regression", CountVectorizer(), LogisticRegression(max_iter=1000), "logreg_model.pkl", "bow_vectorizer.pkl")
train_classical("TF-IDF + SVM", TfidfVectorizer(ngram_range=(1,2)), SVC(), "svm_model.pkl", "tfidf_vectorizer.pkl")
train_classical("TF-IDF + Naive Bayes", TfidfVectorizer(), MultinomialNB(), "nb_model.pkl", "tfidf_vectorizer_nb.pkl")

# === TRAIN BERT ===
print("\n--- BERT FINE-TUNING ---")
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, stratify=df["label"], random_state=42
)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize, batched=True)
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

training_args = TrainingArguments(
    output_dir=BERT_MODEL_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir=f"{BERT_MODEL_DIR}/logs",
    seed=42,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="f1",
    greater_is_better=True
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "f1": f1.compute(predictions=preds, references=p.label_ids, average="binary")["f1"]
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
model.save_pretrained(BERT_MODEL_DIR)
tokenizer.save_pretrained(BERT_MODEL_DIR)
print(f"BERT model saved to {BERT_MODEL_DIR}")
