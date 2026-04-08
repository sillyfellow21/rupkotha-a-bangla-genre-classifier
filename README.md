# Rupkotha Genre Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-0B3D91?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Transformers-Hugging%20Face-FCC624)](https://huggingface.co/docs/transformers/index)
[![UI](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-1f6feb.svg)](LICENSE)

Rupkotha Genre Engine is an end-to-end Bengali NLP project that predicts a book's genre from its summary using a transformer model.
It includes full data cleaning, EDA, model training, evaluation, and a Streamlit web app ready for free cloud deployment.
The codebase is designed for reproducibility, portfolio publishing, and public demo sharing.

## Problem Statement
Bengali publishing platforms, libraries, and readers often face difficulty in fast genre tagging for new books.
This project automates genre prediction from Bengali summaries with a BERT-style classifier.

## Labels (3)
- ইতিবাচক
- নিরপেক্ষ
- নেতিবাচক

## Dataset Source
- Internet source (Hugging Face): https://huggingface.co/datasets/mHossain/bengali_sentiment_v2
- License: MIT
- Project CSV (prepared from source): [data/raw/bengali_books_demo.csv](data/raw/bengali_books_demo.csv)

## Dataset
- Demo CSV: [data/raw/bengali_books_demo.csv](data/raw/bengali_books_demo.csv)

## Project Structure

```text
Rupkotha-Genre-Engine/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
│
├── data/
│   ├── raw/
│   │   └── bengali_books_demo.csv
│   └── processed/
│       └── .gitkeep
│
├── notebooks/
│   └── 01_pipeline_overview.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
│
├── artifacts/
│   └── .gitkeep
├── model/
│   └── .gitkeep
└── reports/
    └── figures/
        └── .gitkeep
```

## Demo Screenshot
Add screenshot after deployment:

```text
![Demo UI](assets/demo-screenshot.png)
```

## Author
- GitHub: [@sillyfellow21](https://github.com/sillyfellow21)
- Profile: [https://github.com/sillyfellow21](https://github.com/sillyfellow21)

## Quickstart (Local)

### 1. Create environment and install dependencies
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Download dataset from internet
```bash
python -m src.download_dataset --output_csv data/raw/bengali_books_demo.csv
```

### 3. Preprocess dataset
```bash
python -m src.preprocessing \
  --input_csv data/raw/bengali_books_demo.csv \
  --output_csv data/processed/books_clean.csv
```

### 4. Run EDA
```bash
python -m src.eda --input_csv data/processed/books_clean.csv
```

### 5. Train model
```bash
python -m src.train \
  --input_csv data/processed/books_clean.csv \
  --model_name csebuetnlp/banglabert \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 2e-5
```

### 6. Evaluate
```bash
python -m src.evaluate \
  --model_dir model/best_model \
  --test_csv data/processed/test_split.csv
```

### 7. Run Streamlit app
```bash
streamlit run app.py
```

## Quick Mode (Low-Resource Device)
Run this single command to preprocess, build a small balanced subset,
train for 1 epoch, and evaluate:

```bash
python -m src.quick_run --rows_per_class 300 --epochs 1 --batch_size 4
```

## Training Details
- Backbone: `csebuetnlp/banglabert`
- Framework: PyTorch + Hugging Face Transformers
- Split: 85/15 stratified train/validation
- Early stopping: validation macro F1 with patience
- Saved artifacts:
  - `model/best_model/`
  - `artifacts/label_map.json`
  - `artifacts/training_history.json`
  - `artifacts/training_curve.png`

## Evaluation Outputs
After running evaluation, the following files are generated:
- `reports/metrics.json`
- `reports/classification_report.txt`
- `reports/figures/confusion_matrix.png`
- `reports/error_analysis.csv`

## Results Summary (Auto-Updated)
This section refreshes after each evaluation run.

<!-- RESULTS_SUMMARY_START -->
| Metric | Value |
|---|---:|
| Accuracy | 0.7842 |
| Precision (Macro) | 0.7654 |
| Recall (Macro) | 0.7634 |
| F1-score (Macro) | 0.7631 |
_Last updated: 2026-04-08 18:42 UTC_
<!-- RESULTS_SUMMARY_END -->

## Project Outcomes
- Built a complete Bengali NLP pipeline for sentiment classification from raw CSV ingestion to deployable web inference.
- Trained a transformer-based classifier (BanglaBERT) for 3 Bengali sentiment labels (positive/neutral/negative) on 14,852 samples with 78.42% accuracy.
- Sourced internet dataset from mHossain/bengali_sentiment_v2 (Hugging Face, MIT license) for reproducible real-world training.
- Implemented Bengali Unicode-aware normalization (emoji cleanup, punctuation noise removal, whitespace normalization, and text standardization).
- Produced publication-friendly analysis artifacts: class distribution plot, text-length distribution plot, top-word summary, confusion matrix, and error analysis samples.
- Generated reproducible model artifacts (`model/best_model/`, label map, training history, and training curves) for transparent experimentation.
- Created an independent inference module that returns predicted sentiment, confidence score, and per-class probabilities.
- Delivered a modern Streamlit demo app for real-time Bengali text sentiment classification with example inputs and confidence visualization.
- Prepared the repository for free public deployment (Streamlit Cloud or Hugging Face Spaces) and open-source sharing.

## Example Inference (Script)
```bash
python -m src.inference --text "রাতের ট্রেনে ঘটে যাওয়া হত্যাকাণ্ডের রহস্য উন্মোচনে এক তদন্তকারী নামেন।"
```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE).
