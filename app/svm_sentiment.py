import pandas as pd
import numpy as np
import pickle
from typing import List, Dict, Any
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import os

class SVMSentimentModel:
    """
    OOP Sentiment Model using TF-IDF + Linear SVM.
    """

    def __init__(self, db_filepath: str):
        # FIX 1: Ensure the attribute name matches what we use later (db_filepath)
        self.db_filepath = db_filepath
        self.df = None
        self.vectorizer = None
        self.model = None
        
        # dataset splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def save_model(self, filename="svm_model.pkl"):
        # FIX 2: Fixed typo 'dum' -> 'dump'
        with open(filename, 'wb') as f:
            pickle.dump((self.vectorizer, self.model), f)
            print("Save_model worked")

    def load_model(self, filename="svm_model.pkl"):
        with open(filename, 'rb') as f:
            self.vectorizer, self.model = pickle.load(f)
            print("load_model worked")

    _ALLOWED_TABLES = {"reviews"}

    def load_dataset_from_db(self, table_name: str = "reviews"):
        """
        Connects to SQLite database and executes a SQL query to fetch data.
        """
        if table_name not in self._ALLOWED_TABLES:
            raise ValueError(f"Invalid table_name '{table_name}'. Allowed: {self._ALLOWED_TABLES}")

        if not os.path.exists(self.db_filepath):
             raise FileNotFoundError(f"Database file not found at {self.db_filepath}. Did you run database_sentimentanalysis.py?")

        try:
            print(f"Connecting to Data Warehouse: {self.db_filepath}...")
            conn = sqlite3.connect(self.db_filepath)

            query = f"""
                SELECT text, rating
                FROM {table_name}
                WHERE text IS NOT NULL
            """

            self.df = pd.read_sql(query, conn)
            conn.close()
            
            required_cols = {"rating", "text"}
            if not required_cols.issubset(self.df.columns):
                raise ValueError(f"Database table must contain {required_cols}")

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

        print("Dataset loaded successfully.")
        print(self.df.head())
        return self.df

    def cleaning(self):
        if self.df is None:
            raise ValueError("Dataset must be loaded before cleaning.")

        try:
            before = len(self.df)
            self.df = self.df.dropna(subset=['text'])
            self.df = self.df[self.df['text'].apply(lambda x: isinstance(x, str))]
            
            # create binary labels: negative (0), positive (1)
            self.df['label'] = self.df['rating'].apply(
                lambda x: 0 if x <= 2 else (1 if x >= 4 else None)
            )

            self.df = self.df.dropna(subset=['label'])
            after = len(self.df)
            print(f"Removed {before - after} rows (bad text or neutral rating).")

            self.df['label'] = self.df['label'].astype(int)

        except Exception as e:
            raise RuntimeError(f"Cleaning failed: {e}")

        print("Cleaning complete.")

    def split_data(self, test_size=0.2):
        if self.df is None:
            raise ValueError("Dataset must be cleaned before splitting.")

        try:
            X = self.df['text']
            y = self.df['label']

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        except Exception as e:
            raise RuntimeError(f"Train/test split failed: {e}")

        print("Dataset split completed.")

    def vectorization(self):
        if self.X_train is None:
            raise ValueError("You must split data before vectorizing.")

        try:
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=50000
            )

            print("Fitting TF-IDF vectorizer...")
            self.X_train = self.vectorizer.fit_transform(self.X_train)
            self.X_test = self.vectorizer.transform(self.X_test)

        except Exception as e:
            raise RuntimeError(f"Vectorization failed: {e}")

        print("Vectorization complete.")

    def train(self):
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted before training.")

        try:
            print("Training Linear SVM...")
            self.model = LinearSVC()
            self.model.fit(self.X_train, self.y_train)

        except Exception as e:
            raise RuntimeError(f"Model training failed: {e}")

        print("Training complete.")

    def evaluate(self):
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        try:
            predictions = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
            f1 = f1_score(self.y_test, predictions)

        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        return {"accuracy": accuracy, "f1": f1}

    def predict(self, text: str):
        if self.model is None:
            raise ValueError("Model must be trained before prediction.")

        try:
            vectorized = self.vectorizer.transform([text])
            pred = self.model.predict(vectorized)[0]
            return "Positive" if pred == 1 else "Negative"

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def predict_with_confidence(self, text: str):
        if self.model is None:
            raise ValueError("Model must be trained before prediction.")

        try:
            vectorized = self.vectorizer.transform([text])
            pred = self.model.predict(vectorized)[0]
            score = self.model.decision_function(vectorized)[0]
            confidence = 1 / (1 + np.exp(-abs(score)))
            label = "Positive" if pred == 1 else "Negative"
            return label, confidence

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

# FIX 3: Wrap execution code so it doesn't run when imported by app.py
if __name__ == "__main__":
    from pathlib import Path as _Path
    # 1. Define DB path anchored to this file's directory
    db_path = str(_Path(__file__).resolve().parent / "corporate_data_warehouse.db")

    # 2. Check if DB exists
    if not os.path.exists(db_path):
        print(" WARNING: Database not found. Please run 'database_sentimentanalysis.py' first.")
    else:
        # 3. Instantiate and Run
        # FIX 4: Use the correct parameter name (db_filepath)
        model = SVMSentimentModel(db_filepath=db_path) 
        
        model.load_dataset_from_db() 
        model.cleaning()
        model.split_data()
        model.vectorization()
        model.train()
        model.evaluate()
        model.save_model() # Save the model so app.py can find it

        print("\n--- Inference Tests ---")
        print(f"Input: 'This product was amazing!' -> Prediction: {model.predict('This product was amazing!')}")