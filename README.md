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

## Genre Labels (7)
- উপন্যাস
- কবিতা
- বিজ্ঞান কল্পকাহিনি
- রহস্য-রোমাঞ্চ
- ইতিহাস
- জীবনী
- শিশু-কিশোর

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

### 2. Preprocess dataset
```bash
python -m src.preprocessing \
  --input_csv data/raw/bengali_books_demo.csv \
  --output_csv data/processed/books_clean.csv
```

### 3. Run EDA
```bash
python -m src.eda --input_csv data/processed/books_clean.csv
```

### 4. Train model
```bash
python -m src.train \
  --input_csv data/processed/books_clean.csv \
  --model_name csebuetnlp/banglabert \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 2e-5
```

### 5. Evaluate
```bash
python -m src.evaluate \
  --model_dir model/best_model \
  --test_csv data/processed/test_split.csv
```

### 6. Run Streamlit app
```bash
streamlit run app.py
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
| Accuracy | 0.1429 |
| Precision (Macro) | 0.0238 |
| Recall (Macro) | 0.1429 |
| F1-score (Macro) | 0.0408 |
_Last updated: 2026-04-07 18:49 UTC_
<!-- RESULTS_SUMMARY_END -->

## Project Outcomes
- Built a complete Bengali NLP pipeline for genre classification, from raw CSV ingestion to deployable web inference.
- Trained a transformer-based classifier (BanglaBERT) for 7 Bengali book genres using an 85/15 stratified split.
- Implemented Bengali Unicode-aware normalization (emoji cleanup, punctuation noise removal, whitespace normalization, and text standardization).
- Produced publication-friendly analysis artifacts: class distribution plot, text-length distribution plot, top-word summary, confusion matrix, and error analysis samples.
- Generated reproducible model artifacts (`model/best_model/`, label map, training history, and training curves) for transparent experimentation.
- Created an independent inference module that returns predicted genre, confidence score, and per-class probabilities.
- Delivered a modern Streamlit demo app for real-time Bengali summary classification with example inputs and confidence visualization.
- Prepared the repository for free public deployment (Streamlit Cloud or Hugging Face Spaces) and open-source sharing.

## Example Inference (Script)
```bash
python -m src.inference --text "রাতের ট্রেনে ঘটে যাওয়া হত্যাকাণ্ডের রহস্য উন্মোচনে এক তদন্তকারী নামেন।"
```

## Deployment Guide

### Option A: Streamlit Cloud (Recommended)
1. Push this repository to GitHub.
2. Sign in to https://streamlit.io/cloud with GitHub.
3. Click **New app**.
4. Select repo, branch, and `app.py` as entry point.
5. Add Python version if needed (3.10+ recommended).
6. Deploy.
7. Copy the public app URL and add it to this README.

### Option B: Hugging Face Spaces
1. Create a new Space at https://huggingface.co/spaces.
2. Choose **Streamlit** SDK and set visibility to public.
3. Upload project files (`app.py`, `src/`, `requirements.txt`, artifacts/model files).
4. Commit files; Space will auto-build.
5. Share generated public URL.

## Reproducibility Notes
- Set random seed through `--seed` in `src.train`.
- Keep Unicode CSV encoding as UTF-8.
- For production-level accuracy, replace the demo CSV with a larger open Bengali summary dataset.

## Publish Checklist
- [ ] Add real model artifacts to `model/best_model/` (or provide download instructions).
- [ ] Add a live app URL under Demo Screenshot after deployment.
- [ ] Add 1-2 repository topics on GitHub (for example: `nlp`, `bengali`, `transformers`, `streamlit`).
- [ ] Pin this repository on your GitHub profile.
- [ ] Add a short project post in your GitHub profile README linking this repo.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE).
