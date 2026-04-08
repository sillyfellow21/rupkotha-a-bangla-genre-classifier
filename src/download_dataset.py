import argparse
from pathlib import Path

import pandas as pd


DATASET_BASE = (
    "https://huggingface.co/datasets/mHossain/"
    "bengali_sentiment_v2/resolve/main"
)


def load_split(split_name: str) -> pd.DataFrame:
    url = f"{DATASET_BASE}/{split_name}.csv"
    frame = pd.read_csv(url)
    return frame


def download_and_prepare(output_csv: Path) -> pd.DataFrame:
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    merged = pd.concat([train_df, val_df, test_df], ignore_index=True)

    label_map = {
        0: "নেতিবাচক",
        1: "নিরপেক্ষ",
        2: "ইতিবাচক",
    }

    prepared = pd.DataFrame(
        {
            "summary": merged["text"].astype(str).str.strip(),
            "genre": merged["label"].map(label_map),
        }
    )
    prepared = prepared.dropna(subset=["summary", "genre"])
    prepared = prepared[prepared["summary"].str.len() > 0]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved internet dataset to: {output_csv}")
    print(f"Shape: {prepared.shape}")
    print("Class distribution:")
    print(prepared["genre"].value_counts())
    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Bengali dataset from internet and prepare CSV."
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("data/raw/bengali_books_demo.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_and_prepare(args.output_csv)


if __name__ == "__main__":
    main()
