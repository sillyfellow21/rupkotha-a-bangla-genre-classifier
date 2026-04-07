import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STOPWORDS = {
    "এবং", "এই", "একটি", "তার", "তিনি", "থেকে", "যে", "করে", "করা", "হয়", "হয়ে", "জন্য", "সঙ্গে",
    "তাকে", "তাদের", "বই", "গল্প", "উপন্যাস", "না", "বা", "ও", "কে", "কি", "কিন্তু", "পর", "আর",
}


def run_eda(input_csv: Path, text_col: str, label_col: str, output_dir: Path) -> None:
    df = pd.read_csv(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    genre_fig = output_dir / "genre_distribution.png"
    length_fig = output_dir / "text_length_distribution.png"

    plt.figure(figsize=(9, 5))
    df[label_col].value_counts().sort_values(ascending=True).plot(kind="barh", color="#2E8B57")
    plt.title("Genre Distribution")
    plt.xlabel("Number of Samples")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig(genre_fig, dpi=150)
    plt.close()

    lengths = df[text_col].astype(str).map(lambda x: len(x.split()))
    plt.figure(figsize=(9, 5))
    plt.hist(lengths, bins=20, color="#1f77b4", edgecolor="white")
    plt.title("Summary Length Distribution (Words)")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(length_fig, dpi=150)
    plt.close()

    words = " ".join(df[text_col].astype(str).tolist()).split()
    filtered_words = [w for w in words if len(w) > 1 and w not in STOPWORDS]
    most_common = Counter(filtered_words).most_common(25)

    summary_file = output_dir / "eda_summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("EDA Findings\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Unique genres: {df[label_col].nunique()}\n")
        f.write(f"Average summary length: {lengths.mean():.2f} words\n")
        f.write(f"Median summary length: {lengths.median():.2f} words\n\n")
        f.write("Top frequent words:\n")
        for word, count in most_common:
            f.write(f"- {word}: {count}\n")

    print(f"Saved: {genre_fig}")
    print(f"Saved: {length_fig}")
    print(f"Saved: {summary_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA for Bengali genre classification.")
    parser.add_argument("--input_csv", type=Path, default=Path("data/processed/books_clean.csv"))
    parser.add_argument("--text_col", type=str, default="summary")
    parser.add_argument("--label_col", type=str, default="genre")
    parser.add_argument("--output_dir", type=Path, default=Path("reports/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_eda(args.input_csv, args.text_col, args.label_col, args.output_dir)


if __name__ == "__main__":
    main()
