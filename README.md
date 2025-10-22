# Labeling Requirements Extraction Pipeline

This repository contains a complete machine learning pipeline for extracting and structuring food labeling requirements from regulatory PDF documents. The pipeline uses both traditional ML (TF-IDF + Logistic Regression) and modern deep learning (BERT) approaches.

## Pipeline Overview

The pipeline consists of 11 sequential scripts that process PDF documents through text extraction, preprocessing, machine learning model training, and database storage. Each script depends on the output of previous steps.

## Directory Structure

```
data/
├── 1_raw/                     # Input PDF files
├── 2_extracted/               # Extracted text files
├── 3_processed/               # Preprocessed and cleaned text
├── 4_sections/                # Labeling-specific sections
├── 5_dataset/                 # Labeled training dataset
├── 6_predictions/             # Traditional ML predictions
├── 7_predictions_bert/        # BERT model predictions
├── 8_pdf_model_extraction/    # Direct PDF processing with BERT
└── 9_structured/              # Final structured requirements

models/
├── traditional/               # TF-IDF + Logistic Regression models
└── bert_finetuned/           # Fine-tuned BERT model
```

## Execution Flow

### Phase 1: Data Preparation and Preprocessing

#### 1. Text Extraction (`01_extract_text.py`)

**Run when:** PDF files are added to `data/1_raw/` or output doesn't exist

```bash
python 01_extract_text.py
```

- **Input:** PDF files in `data/1_raw/`
- **Output:** Plain text files in `data/2_extracted/`
- **Purpose:** Extracts raw text content from PDF documents using pdfplumber

#### 2. Text Preprocessing (`02_preprocess_text.py`)

**Run when:** Step 1 completes or extracted text files are updated

```bash
python 02_preprocess_text.py
```

- **Input:** Text files from `data/2_extracted/`
- **Output:** Cleaned text files in `data/3_processed/`
- **Purpose:** Normalizes text, removes punctuation, tokenizes into sentences

#### 3. Labeling Section Extraction (`03_extract_labeling_sections.py`)

**Run when:** Step 2 completes or processed text files are updated

```bash
python 03_extract_labeling_sections.py
```

- **Input:** Processed text from `data/3_processed/`
- **Output:** Labeling sections in `data/4_sections/`
- **Purpose:** Identifies and extracts labeling-related sections using regex patterns

#### 4. Dataset Preparation (`04_prepare_labeled_dataset.py`)

**Run when:** Step 3 completes or section files are updated

```bash
python 04_prepare_labeled_dataset.py
```

- **Input:** Section files from `data/4_sections/` and full text from `data/3_processed/`
- **Output:** Labeled dataset at `data/5_dataset/labeled_sentences.csv`
- **Purpose:** Creates balanced training dataset with positive (labeling) and negative examples

### Phase 2: Traditional Machine Learning Pipeline

#### 5. Baseline Model Training (`11_vectorize_and_train_baseline.py`)

**Run when:** Step 4 completes or labeled dataset is updated

```bash
python 11_vectorize_and_train_baseline.py
```

- **Input:** Labeled dataset from `data/5_dataset/labeled_sentences.csv`
- **Output:** Trained models in `models/traditional/`
- **Purpose:** Trains TF-IDF vectorizer and Logistic Regression classifier

#### 6. Baseline Predictions (`12_predict_baseline.py`)

**Run when:** Step 5 completes or baseline model is retrained

```bash
python 12_predict_baseline.py
```

- **Input:** Processed text from `data/3_processed/` and models from `models/traditional/`
- **Output:** Predictions in `data/6_predictions/`
- **Purpose:** Applies baseline model to classify sentences

### Phase 3: BERT Deep Learning Pipeline

#### 7. BERT Fine-tuning (`21_finetune_bert.py`)

**Run when:** Step 4 completes, labeled dataset is updated, or model retraining is needed

```bash
python 21_finetune_bert.py
```

- **Input:** Labeled dataset from `data/5_dataset/labeled_sentences.csv`
- **Output:** Fine-tuned BERT model in `models/bert_finetuned/`
- **Purpose:** Fine-tunes DistilBERT for labeling requirement classification

#### 8. BERT Predictions (`22_predict_bert.py`)

**Run when:** Step 7 completes or BERT model is retrained

```bash
python 22_predict_bert.py
```

- **Input:** Processed text from `data/3_processed/` and BERT model from `models/bert_finetuned/`
- **Output:** BERT predictions in `data/7_predictions_bert/`
- **Purpose:** Applies fine-tuned BERT to classify sentences

### Phase 4: Direct PDF Processing and Structuring

#### 9. Direct PDF Processing (`31_inspect_pdf_content.py`)

**Run when:** Step 7 completes, new PDFs are added, or BERT model is updated

```bash
python 31_inspect_pdf_content.py
```

- **Input:** Raw PDFs from `data/1_raw/` and BERT model from `models/bert_finetuned/`
- **Output:** Direct PDF predictions in `data/8_pdf_model_extraction/`
- **Purpose:** Processes PDFs directly with BERT without intermediate preprocessing

#### 10. Requirement Structuring (`32_extract_structured_requirements.py`)

**Run when:** Step 9 completes or PDF extraction results are updated

```bash
python 32_extract_structured_requirements.py
```

- **Input:** BERT extractions from `data/8_pdf_model_extraction/`
- **Output:** Structured requirements in `data/9_structured/`
- **Purpose:** Consolidates predictions into structured labeling requirements

#### 11. Database Upload (`41_send_to_db.py`)

**Run when:** Step 10 completes or structured requirements are updated

```bash
python 41_send_to_db.py
```

- **Input:** Structured requirements from `data/9_structured/`
- **Output:** Data inserted into SQL Server database
- **Purpose:** Uploads structured requirements to relational database

## Execution Decision Logic

### When to Run Each Script

1. **Always run from the beginning** if starting fresh or if raw PDFs have changed
2. **Skip early steps** if their outputs exist and inputs haven't changed
3. **Run from specific step** if that step's inputs have been modified

### Automated Execution Script

You can create a simple automation script to check dependencies:

```python
import os
from pathlib import Path

def should_run_step(input_paths, output_path):
    """Check if step should run based on file timestamps"""
    if not os.path.exists(output_path):
        return True

    output_time = os.path.getmtime(output_path)
    for input_path in input_paths:
        if os.path.exists(input_path) and os.path.getmtime(input_path) > output_time:
            return True
    return False

# Example usage
steps = [
    ("01_extract_text.py", ["data/1_raw"], "data/2_extracted"),
    ("02_preprocess_text.py", ["data/2_extracted"], "data/3_processed"),
    # ... add other steps
]

for script, inputs, output in steps:
    if should_run_step(inputs, output):
        print(f"Running {script}...")
        # Execute script
```

## Requirements

### Python Dependencies

```bash
pip install pandas numpy torch transformers datasets evaluate sklearn nltk tqdm pdfplumber pdfminer.six matplotlib joblib sqlalchemy pyodbc
```

### System Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster BERT training/inference)
- SQL Server with ODBC Driver 17 (for database upload)

### NLTK Data

```python
import nltk
nltk.download('punkt')
```

## Configuration

### Key Parameters to Adjust

- **Minimum probability threshold**: `MIN_PROB = 0.60` in `32_extract_structured_requirements.py`
- **BERT training epochs**: `NUM_EPOCHS = 4` in `21_finetune_bert.py`
- **Batch size**: `BATCH_SIZE = 8` in `21_finetune_bert.py`
- **Database connection**: Update server details in `41_send_to_db.py`

## Monitoring Progress

Each script provides progress information:

- Progress bars for file processing
- Counts of processed items
- Success/failure messages
- Model performance metrics (for training scripts)

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Install all required packages
2. **GPU memory issues**: Reduce batch size in BERT scripts
3. **File encoding**: Ensure UTF-8 encoding for all text files
4. **Database connection**: Verify SQL Server configuration and ODBC driver

### Performance Optimization

1. **Use GPU**: Ensure CUDA is available for BERT processing
2. **Parallel processing**: Consider multiprocessing for large file sets
3. **Memory management**: Process files in batches if memory is limited

## Output Formats

### Final Structured Requirements

Each document produces:

- `requirements.csv`: Structured labeling requirements with confidence scores
- `summary.csv`: Document metadata and processing statistics

### Database Tables

- `LabelingDocuments`: Document metadata
- `LabelingRequirements`: Individual requirement details

This pipeline provides a complete solution for extracting, structuring, and storing food labeling requirements from regulatory documents.
