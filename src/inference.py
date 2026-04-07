import argparse
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocessing import normalize_bengali_text
from src.utils import get_device, load_label_map


class GenrePredictor:
    def __init__(self, model_dir: Path, label_map_path: Path, max_length: int = 256):
        self.model_dir = Path(model_dir)
        self.label_map_path = Path(label_map_path)
        self.max_length = max_length

        mapping = load_label_map(self.label_map_path)
        self.id_to_label = {int(k): v for k, v in mapping["id_to_label"].items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)

        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()

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
        return {
            "cleaned_text": cleaned,
            "predicted_genre": self.id_to_label[int(pred_id)],
            "confidence": float(confidence),
            "probabilities": {
                self.id_to_label[i]: float(probs[i]) for i in range(len(self.id_to_label))
            },
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
