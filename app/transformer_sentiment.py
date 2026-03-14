import sqlite3
import pandas as pd
from datasets import Dataset
from typing import Dict, List, Any
import evaluate
import numpy as np
import torch
import os
import openpyxl
import sklearn
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)



basepath = Path(__file__).resolve().parent.parent
source_file = basepath / 'data' / "amazon_test_2500.xlsx"
db_name = basepath / 'data' / "amazon_reviews.db"

# --- HELPER: Ingest Excel into SQLite ---
def ingest_excel_to_sqlite(source_file=source_file, db_name=db_name):
    """
    Reads the Excel file and saves it into a SQLite database table.
    """
    if os.path.exists(db_name):
        print(f"Database '{db_name}' already exists. Skipping ingestion to avoid overwriting.")
        return
    print(basepath)
    print(f"Reading file: {source_file}...")
    try:
        # Try reading as Excel first (since user specified .xlsx)
        df = pd.read_excel(source_file)
    except Exception:
        # Fallback to CSV if the user actually has a CSV
        print("Excel read failed, trying CSV format...")
        df = pd.read_csv(str(source_file).replace(".xlsx", ".csv"))

    # Cleanup: Remove index columns if they exist
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(f"Ingesting {len(df)} rows into SQLite database: {db_name}...")
    
    conn = sqlite3.connect(db_name)
    # Save the dataframe to a table named 'reviews'
    df.to_sql("reviews", conn, if_exists="replace", index=False)
    conn.close()
    
    print("Ingestion complete.\n")


# --- MAIN MODEL CLASS ---
class SentimentModel:
    """
    Full OOP pipeline for training/evaluating a DistilBERT sentiment model
    using a SQLite database source.
    """

    default_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self, checkpoint: str = None, db_path: str = None):
        self.checkpoint = checkpoint or SentimentModel.default_checkpoint
        self.db_path = db_path or "amazon_reviews.db"

        # Placeholders
        self.tokenizer = None
        self.raw_datasets = None
        self.cleaned_dataset = None
        self.tokenized_datasets = None
        self.data_collator = None
        self.dataset_splits = None
        self.model = None
        self.training_args = None
        self.trainer = None
        self.accuracy_metric = evaluate.load("accuracy")

    def save_model(self, output_dir="./bert_model_saved"):
        """Saves the model and tokenizer to a folder."""
        if self.model is not None:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")
        else:
            print("No model to save! Train it first.")

    def load_saved_model(self, source_dir="./bert_model_saved"):
        """Loads from a local folder; falls back to HuggingFace Hub if not found."""
        source = source_dir if os.path.exists(source_dir) else self.checkpoint
        if source == self.checkpoint:
            print(f"Local model not found at '{source_dir}'. Loading from HuggingFace Hub: {self.checkpoint}")
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source)
        print(f"Model loaded from: {source}")

    # 1. Load Dataset (FROM SQLITE)
    def load_dataset(self):
        print(f"Connecting to SQLite DB: {self.db_path}")

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database {self.db_path} not found. Please run ingestion first.")

        try:
            conn = sqlite3.connect(self.db_path)
            # Read the entire table 'reviews'
            query = "SELECT * FROM reviews"
            df = pd.read_sql(query, conn)
            conn.close()
        except Exception as e:
            raise ConnectionError(f"Failed to load data from SQLite: {e}")

        # --- MAPPING COLUMNS ---
        # Map 'test' OR 'text' -> 'reviewText'
        if "test" in df.columns:
            print("Mapping column 'test' -> 'reviewText'...")
            df.rename(columns={"test": "reviewText"}, inplace=True)
        elif "text" in df.columns:
            print("Mapping column 'text' -> 'reviewText'...")
            df.rename(columns={"text": "reviewText"}, inplace=True)

        if "reviewText" not in df.columns:
            raise KeyError(f"Dataset columns {df.columns.tolist()} do not contain 'test', 'text', or 'reviewText'.")

        # Handle Rating -> Sentiment
        if "sentiment" not in df.columns:
            if "rating" in df.columns:
                print("Converting rating -> sentiment labels (0 or 1)...")
                # 1-2 stars = Negative (0), 4-5 stars = Positive (1), 3 is ignored or treated as negative
                df["sentiment"] = df["rating"].apply(
                    lambda x: 0 if x <= 2 else (1 if x >= 4 else 0)
                )
            else:
                raise KeyError("Dataset must contain either 'sentiment' or 'rating'.")

        # Convert to HuggingFace Dataset
        self.raw_datasets = Dataset.from_pandas(df)
        print(f"Dataset loaded successfully. Rows: {len(self.raw_datasets)}")

    # 2. Tokenizer
    def load_tokenizer(self):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

    # 3. Cleaning
    def cleaning(self):
        print("Cleaning dataset...")
        texts = self.raw_datasets["reviewText"]
        good_indices = [
            i for i, t in enumerate(texts)
            if t is not None and isinstance(t, str)
        ]
        self.cleaned_dataset = self.raw_datasets.select(good_indices)
        print(f"Cleaned dataset size: {len(self.cleaned_dataset)}")

    # 4. Tokenization
    def tokenize_function(self, batch: Dict[str, List[Any]]):
        texts = [str(t) if t else "" for t in batch["reviewText"]]
        return self.tokenizer(texts, truncation=True)

    def tokenization(self):
        print("Tokenizing...")
        self.tokenized_datasets = self.cleaned_dataset.map(
            self.tokenize_function,
            batched=True
        )

    # 5. Data Collator
    def load_data_collator(self):
        self.data_collator = DataCollatorWithPadding(self.tokenizer)

    # 6. Model Loading
    def load_model(self):
        print("Loading model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint,
            num_labels=2
        )

    # 7. Split
    def split_dataset(self, test_ratio=0.2):
        print("Splitting train/test datasets...")
        self.dataset_splits = self.tokenized_datasets.train_test_split(
            test_size=test_ratio
        )
        print(self.dataset_splits)

        if "sentiment" in self.dataset_splits['train'].column_names:
            print("renaming to label")
            self.dataset_splits = self.dataset_splits.rename_column('sentiment','label')

            print(f"Columns in dataset: {self.dataset_splits['train'].column_names}")

    # 8. Metrics
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return self.accuracy_metric.compute(predictions=preds, references=labels)

    # 9. Trainer
    def create_trainer(self):
        print("Creating Trainer...")
        self.training_args = TrainingArguments(
            output_dir="model_output",
            num_train_epochs=2,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy = "epoch",
            fp16=torch.cuda.is_available(),
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset_splits["train"],
            eval_dataset=self.dataset_splits["test"],
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

    # 10. Train
    def train(self):
        print("Training...")
        self.trainer.train()

    # 11. Evaluate
    def evaluate_model(self):
        return self.trainer.evaluate()

    # 12. Predict
    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        return "Positive" if prediction == 1 else "Negative"

    def predict_with_confidence(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()
        label = "Positive" if pred == 1 else "Negative"
        return label, confidence


def main():
    # 1. Ingest Data (Excel -> SQLite)
    # Ensure 'amazon_test_2500.xlsx' is in the same folder
    ingest_excel_to_sqlite(source_file=source_file, db_name=db_name)

    # 2. Run Pipeline
    model = SentimentModel(db_path= db_name)

    model.load_dataset()
    model.load_tokenizer()
    model.cleaning()
    model.tokenization()
    model.load_data_collator()
    model.load_model()
    model.split_dataset()
    model.create_trainer()

    model.train()
    results = model.evaluate_model()
    print("Final Evaluation:", results)

    model.save_model("./bert_model_saved") 


    # Sample predictions
    print("\nTest Predictions:")
    print(f"Product is great: {model.predict('I love this product, amazing quality!')}")
    print(f"Product is bad: {model.predict('Terrible experience, I hate it.')}")


if __name__ == "__main__":
    main()