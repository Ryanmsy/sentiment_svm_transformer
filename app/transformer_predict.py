import sqlite3
import pandas as pd
import torch
import os
from pathlib import Path
from datasets import Dataset
from typing import List, Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from config import BERT_MODEL_DIR


basepath = Path(__file__).resolve().parent.parent
db_name = basepath / "data" / "amazon_reviews.db"


class TransformerPredictor:
    """
    Inference-only transformer pipeline for scoring customer review data.
    Loads a pretrained/saved DistilBERT model and predicts sentiment.
    """

    default_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self, checkpoint: str = None, db_path=None):
        self.checkpoint = checkpoint or TransformerPredictor.default_checkpoint
        self.db_path = db_path or db_name

        self.tokenizer = None
        self.model = None
        self.raw_datasets = None
        self.cleaned_dataset = None

    # 1. Load saved model (falls back to HuggingFace Hub if local not found)
    def load_saved_model(self, source_dir: str = BERT_MODEL_DIR):
        source = source_dir if os.path.exists(source_dir) else self.checkpoint
        if source == self.checkpoint:
            print(f"Local model not found at '{source_dir}'. Loading from HuggingFace Hub: {self.checkpoint}")
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source)
        self.model.eval()
        print(f"Model loaded from: {source}")

    # 2. Load customer dataset from SQLite
    def load_dataset(self):
        print(f"Connecting to SQLite DB: {self.db_path}")

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database {self.db_path} not found.")

        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM reviews", conn)
        conn.close()

        # Column mapping: test/text → reviewText
        if "test" in df.columns:
            df.rename(columns={"test": "reviewText"}, inplace=True)
        elif "text" in df.columns:
            df.rename(columns={"text": "reviewText"}, inplace=True)

        if "reviewText" not in df.columns:
            raise KeyError(f"No text column found. Columns: {df.columns.tolist()}")

        self.raw_datasets = Dataset.from_pandas(df)
        print(f"Dataset loaded. Rows: {len(self.raw_datasets)}")

    # 3. Clean dataset
    def cleaning(self):
        print("Cleaning dataset...")
        texts = self.raw_datasets["reviewText"]
        good_indices = [
            i for i, t in enumerate(texts)
            if t is not None and isinstance(t, str)
        ]
        self.cleaned_dataset = self.raw_datasets.select(good_indices)
        print(f"Cleaned dataset size: {len(self.cleaned_dataset)}")

    # 4. Predict single text → "Positive" / "Negative"
    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        prediction = torch.argmax(outputs.logits, dim=-1).item()
        return "Positive" if prediction == 1 else "Negative"

    # 5. Predict single text with confidence score
    def predict_with_confidence(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()
        label = "Positive" if pred == 1 else "Negative"
        return label, confidence

    # 6. Predict a list of texts (bulk scoring)
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        results = []
        for text in texts:
            label, confidence = self.predict_with_confidence(text)
            results.append({"text": text, "label": label, "confidence": round(confidence, 4)})
        return results

    # 7. Score every row in the SQLite DB → returns DataFrame with predictions
    def predict_from_db(self) -> pd.DataFrame:
        self.load_dataset()
        self.cleaning()

        texts = self.cleaned_dataset["reviewText"]
        print(f"Scoring {len(texts)} rows...")

        labels, confidences = [], []
        for text in texts:
            label, confidence = self.predict_with_confidence(text)
            labels.append(label)
            confidences.append(round(confidence, 4))

        df = self.cleaned_dataset.to_pandas()
        df["predicted_label"] = labels
        df["confidence"] = confidences

        print("Scoring complete.")
        return df


def main():
    predictor = TransformerPredictor(db_path=db_name)
    predictor.load_saved_model(source_dir=BERT_MODEL_DIR)

    # Bulk score customer dataset
    results_df = predictor.predict_from_db()
    print(f"\nScored {len(results_df)} reviews.")
    print(results_df[["reviewText", "predicted_label", "confidence"]].head(10))

    # Sample single predictions
    print("\nSample predictions:")
    samples = [
        "Amazing product, exactly what I needed!",
        "Broke after one day. Very disappointed.",
        "It's okay, nothing special.",
    ]
    for text in samples:
        label, conf = predictor.predict_with_confidence(text)
        print(f"  [{label} ({conf:.1%})] {text}")


if __name__ == "__main__":
    main()
