import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

WHITESPACE_PATTERN = re.compile(r"\s+")
NOISY_PUNCT_PATTERN = re.compile(
    r"[!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]+"
)
ALLOWED_CHARS_PATTERN = re.compile(r"[^\u0980-\u09FF0-9a-zA-Z\s।]")


def normalize_bengali_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    text = EMOJI_PATTERN.sub(" ", text)
    text = NOISY_PUNCT_PATTERN.sub(" ", text)
    text = ALLOWED_CHARS_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def preprocess_dataset(input_csv: Path, output_csv: Path, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Input CSV must contain columns '{text_col}' and '{label_col}'. Found: {list(df.columns)}"
        )

    print("=== Raw Dataset Audit ===")
    print(f"Shape: {df.shape}")
    print("Null values per column:")
    print(df.isnull().sum())
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("Class distribution:")
    print(df[label_col].value_counts(dropna=False))

    cleaned = df[[text_col, label_col]].copy()
    cleaned = cleaned.dropna(subset=[text_col, label_col])
    cleaned[text_col] = cleaned[text_col].astype(str).map(normalize_bengali_text)
    cleaned[label_col] = cleaned[label_col].astype(str).str.strip()
    cleaned = cleaned[cleaned[text_col].str.len() > 0]
    cleaned = cleaned.drop_duplicates(subset=[text_col, label_col])

    print("\n=== Cleaned Dataset Audit ===")
    print(f"Shape: {cleaned.shape}")
    print("Null values per column:")
    print(cleaned.isnull().sum())
    print(f"Duplicate rows: {cleaned.duplicated().sum()}")
    print("Class distribution:")
    print(cleaned[label_col].value_counts())

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\nSaved cleaned dataset to: {output_csv}")
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Bengali book summary dataset.")
    parser.add_argument("--input_csv", type=Path, default=Path("data/raw/bengali_books_demo.csv"))
    parser.add_argument("--output_csv", type=Path, default=Path("data/processed/books_clean.csv"))
    parser.add_argument("--text_col", type=str, default="summary")
    parser.add_argument("--label_col", type=str, default="genre")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_dataset(args.input_csv, args.output_csv, args.text_col, args.label_col)


if __name__ == "__main__":
    main()
