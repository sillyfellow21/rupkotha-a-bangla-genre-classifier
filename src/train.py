import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import (
    ARTIFACTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_PATIENCE,
    DEFAULT_RANDOM_SEED,
    MODEL_DIR,
)
from src.preprocessing import normalize_bengali_text
from src.utils import compute_metrics, ensure_dirs, get_device, plot_training_history, save_label_map, set_seed


@dataclass
class TrainingConfig:
    input_csv: Path
    text_col: str
    label_col: str
    model_name: str
    output_dir: Path
    label_map_path: Path
    train_split_path: Path
    test_split_path: Path
    epochs: int
    batch_size: int
    learning_rate: float
    max_length: int
    patience: int
    random_seed: int


class SummaryDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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


def build_label_map(labels: List[str]) -> Dict[str, int]:
    unique_labels = sorted(set(labels))
    return {label: idx for idx, label in enumerate(unique_labels)}


def evaluate_epoch(model, dataloader, device) -> Tuple[float, Dict[str, float], List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    true_labels, pred_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            true_labels.extend(batch["labels"].cpu().tolist())
            pred_labels.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = compute_metrics(true_labels, pred_labels)
    return avg_loss, metrics, true_labels, pred_labels


def train(config: TrainingConfig) -> None:
    ensure_dirs([config.output_dir, config.label_map_path.parent, config.train_split_path.parent, ARTIFACTS_DIR])
    set_seed(config.random_seed)
    device = get_device()

    df = pd.read_csv(config.input_csv)
    df[config.text_col] = df[config.text_col].astype(str).map(normalize_bengali_text)
    df[config.label_col] = df[config.label_col].astype(str).str.strip()

    label_to_id = build_label_map(df[config.label_col].tolist())
    save_label_map(label_to_id, config.label_map_path)

    df["label_id"] = df[config.label_col].map(label_to_id)

    num_classes = df["label_id"].nunique()
    min_val_size = num_classes / max(len(df), 1)
    val_size = max(0.15, min_val_size)

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=config.random_seed,
        stratify=df["label_id"],
    )

    train_df.to_csv(config.train_split_path, index=False, encoding="utf-8")
    val_df.to_csv(config.test_split_path, index=False, encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label_to_id),
    ).to(device)

    train_dataset = SummaryDataset(
        train_df[config.text_col].tolist(),
        train_df["label_id"].tolist(),
        tokenizer,
        config.max_length,
    )
    val_dataset = SummaryDataset(
        val_df[config.text_col].tolist(),
        val_df["label_id"].tolist(),
        tokenizer,
        config.max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(total_steps // 10, 1),
        num_training_steps=max(total_steps, 1),
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1_macro": [],
    }

    best_f1 = -1.0
    bad_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)
        val_loss, val_metrics, _, _ = evaluate_epoch(model, val_loader, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            bad_epochs = 0
            model.save_pretrained(config.output_dir)
            tokenizer.save_pretrained(config.output_dir)
            print(f"Saved new best checkpoint to {config.output_dir}")
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print("Early stopping triggered.")
                break

    history_path = ARTIFACTS_DIR / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_training_history(history, ARTIFACTS_DIR / "training_curve.png")

    print(f"Best validation macro F1: {best_f1:.4f}")
    print(f"Training history saved to: {history_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Bengali book genre classifier.")
    parser.add_argument("--input_csv", type=Path, default=Path("data/processed/books_clean.csv"))
    parser.add_argument("--text_col", type=str, default="summary")
    parser.add_argument("--label_col", type=str, default="genre")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir", type=Path, default=MODEL_DIR / "best_model")
    parser.add_argument("--label_map_path", type=Path, default=ARTIFACTS_DIR / "label_map.json")
    parser.add_argument("--train_split_path", type=Path, default=Path("data/processed/train_split.csv"))
    parser.add_argument("--test_split_path", type=Path, default=Path("data/processed/test_split.csv"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig(
        input_csv=args.input_csv,
        text_col=args.text_col,
        label_col=args.label_col,
        model_name=args.model_name,
        output_dir=args.output_dir,
        label_map_path=args.label_map_path,
        train_split_path=args.train_split_path,
        test_split_path=args.test_split_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        patience=args.patience,
        random_seed=args.seed,
    )
    train(cfg)


if __name__ == "__main__":
    main()
