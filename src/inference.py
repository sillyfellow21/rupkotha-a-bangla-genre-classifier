import argparse
import os
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocessing import normalize_bengali_text
from src.utils import get_device, load_label_map


# Some environments (including restricted Windows/Cloud setups) block hf_xet.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


class GenrePredictor:
    def __init__(
        self,
        model_dir: Path,
        label_map_path: Path,
        max_length: int = 256,
        allow_fallback: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.label_map_path = Path(label_map_path)
        self.max_length = max_length
        self.allow_fallback = allow_fallback

        self.fallback_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

        local_model_exists = self.model_dir.exists()
        local_label_map_exists = self.label_map_path.exists()

        if local_model_exists and local_label_map_exists:
            mapping = load_label_map(self.label_map_path)
            self.id_to_label = {int(k): v for k, v in mapping["id_to_label"].items()}

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        elif self.allow_fallback:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.fallback_model_name,
                use_fast=False,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.fallback_model_name
            )
            self.id_to_label = self._fallback_labels(self.model.config.id2label)
        else:
            raise FileNotFoundError(
                "Model artifacts were not found and fallback model is disabled."
            )

        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _fallback_labels(id2label: Dict[int, str]) -> Dict[int, str]:
        normalized = {int(k): str(v).lower() for k, v in id2label.items()}
        mapped = {}

        for idx, label_text in normalized.items():
            if "star" in label_text:
                digits = [ch for ch in label_text if ch.isdigit()]
                stars = int(digits[0]) if digits else 3
                if stars <= 2:
                    mapped[idx] = "নেতিবাচক"
                elif stars == 3:
                    mapped[idx] = "নিরপেক্ষ"
                else:
                    mapped[idx] = "ইতিবাচক"
            if "neg" in label_text:
                mapped[idx] = "নেতিবাচক"
            elif "neu" in label_text:
                mapped[idx] = "নিরপেক্ষ"
            elif "pos" in label_text:
                mapped[idx] = "ইতিবাচক"

        if len(mapped) == len(normalized):
            return mapped

        # Fallback to a deterministic 3-class mapping when labels are generic.
        ordered = sorted(normalized)
        default_map = ["নেতিবাচক", "নিরপেক্ষ", "ইতিবাচক"]
        return {
            idx: default_map[min(i, len(default_map) - 1)]
            for i, idx in enumerate(ordered)
        }

    def predict(self, summary_text: str) -> Dict[str, object]:
        cleaned = normalize_bengali_text(summary_text)
        encoded = self.tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

        confidence, pred_id = torch.max(probs, dim=0)
        merged_probabilities = {}
        for idx in range(len(self.id_to_label)):
            label = self.id_to_label[idx]
            merged_probabilities[label] = (
                merged_probabilities.get(label, 0.0) + float(probs[idx])
            )

        return {
            "cleaned_text": cleaned,
            "predicted_genre": self.id_to_label[int(pred_id)],
            "confidence": float(confidence),
            "probabilities": merged_probabilities,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a Bengali summary.")
    parser.add_argument("--model_dir", type=Path, default=Path("model/best_model"))
    parser.add_argument("--label_map_path", type=Path, default=Path("artifacts/label_map.json"))
    parser.add_argument("--text", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = GenrePredictor(args.model_dir, args.label_map_path)
    result = predictor.predict(args.text)

    print("Predicted genre:", result["predicted_genre"])
    print("Confidence:", f"{result['confidence']:.4f}")


if __name__ == "__main__":
    main()
