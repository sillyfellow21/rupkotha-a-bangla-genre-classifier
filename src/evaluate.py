import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocessing import normalize_bengali_text
from src.utils import compute_metrics, get_device, load_label_map


def update_readme_results(readme_path: Path, metrics: dict) -> None:
    start_marker = "<!-- RESULTS_SUMMARY_START -->"
    end_marker = "<!-- RESULTS_SUMMARY_END -->"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    summary_block = "\n".join(
        [
            start_marker,
            "| Metric | Value |",
            "|---|---:|",
            f"| Accuracy | {metrics['accuracy']:.4f} |",
            f"| Precision (Macro) | {metrics['precision_macro']:.4f} |",
            f"| Recall (Macro) | {metrics['recall_macro']:.4f} |",
            f"| F1-score (Macro) | {metrics['f1_macro']:.4f} |",
            f"_Last updated: {timestamp}_",
            end_marker,
        ]
    )

    if not readme_path.exists():
        return

    text = readme_path.read_text(encoding="utf-8")
    pattern = rf"{re.escape(start_marker)}[\s\S]*?{re.escape(end_marker)}"

    if start_marker in text and end_marker in text:
        updated = re.sub(pattern, summary_block, text, count=1)
    else:
        updated = text + (
            "\n\n## Results Summary (Auto-Updated)\n"
            "This section refreshes after each evaluation run.\n\n"
            f"{summary_block}\n"
        )

    readme_path.write_text(updated, encoding="utf-8")


class EvalDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def evaluate(
    model_dir: Path,
    label_map_path: Path,
    test_csv: Path,
    output_dir: Path,
    text_col: str,
    label_col: str,
    max_length: int,
    batch_size: int,
    readme_path: Path,
    update_readme: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_label_map(label_map_path)
    label_to_id = mapping["label_to_id"]
    id_to_label = {int(k): v for k, v in mapping["id_to_label"].items()}

    df = pd.read_csv(test_csv)
    df[text_col] = df[text_col].astype(str).map(normalize_bengali_text)
    df[label_col] = df[label_col].astype(str).str.strip()
    df["label_id"] = df[label_col].map(label_to_id)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = get_device()
    model.to(device)
    model.eval()

    dataset = EvalDataset(
        df[text_col].tolist(),
        df["label_id"].tolist(),
        tokenizer,
        max_length,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    true_labels, pred_labels = [], []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=1).cpu()

            true_labels.extend(labels.tolist())
            pred_labels.extend(preds.tolist())

    metrics = compute_metrics(true_labels, pred_labels)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    target_names = [id_to_label[i] for i in sorted(id_to_label)]
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    misclassified = df.copy()
    misclassified["predicted_id"] = pred_labels
    misclassified["predicted_label"] = misclassified["predicted_id"].map(id_to_label)
    error_df = misclassified[misclassified[label_col] != misclassified["predicted_label"]].head(10)
    error_path = output_dir / "error_analysis.csv"
    error_df[[text_col, label_col, "predicted_label"]].to_csv(error_path, index=False, encoding="utf-8")

    if update_readme:
        update_readme_results(readme_path, metrics)

    print("Evaluation completed.")
    print("Metrics:", metrics)
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved error analysis to: {error_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Bengali genre classifier.")
    parser.add_argument("--model_dir", type=Path, default=Path("model/best_model"))
    parser.add_argument("--label_map_path", type=Path, default=Path("artifacts/label_map.json"))
    parser.add_argument("--test_csv", type=Path, default=Path("data/processed/test_split.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("reports"))
    parser.add_argument("--text_col", type=str, default="summary")
    parser.add_argument("--label_col", type=str, default="genre")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--readme_path", type=Path, default=Path("README.md"))
    parser.add_argument(
        "--skip_readme_update",
        action="store_true",
        help="Disable auto-updating the README results summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        model_dir=args.model_dir,
        label_map_path=args.label_map_path,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        text_col=args.text_col,
        label_col=args.label_col,
        max_length=args.max_length,
        batch_size=args.batch_size,
        readme_path=args.readme_path,
        update_readme=not args.skip_readme_update,
    )


if __name__ == "__main__":
    main()
