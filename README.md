# Sentiment Analysis: SVM vs. Transformer

A  **Streamlit** web application demonstrating binary sentiment classification on Amazon product reviews. This project compares the trade-offs in speed and accuracy between classical Machine Learning (**SVM**) and Deep Learning (**DistilBERT**).

## Tech Stack
* **NLP & Machine Learning:** Scikit-learn, TF-IDF, HuggingFace Transformers, DistilBERT, Fine-tuning
* **Infrastructure & Data:** Python 3.14+, SQLite, ETL, Docker
* **Frontend:** Streamlit

## Models Compared

| Feature | SVM (Baseline) | DistilBERT (Transformer) |
| :--- | :--- | :--- |
| **Tech** | `LinearSVC` + TF-IDF (50k features) | `distilbert-base-uncased-finetuned-sst-2` |
| **Pros** | Lightweight, near-instant inference | High accuracy, captures semantic context/nuance |
| **Cons** | No semantic understanding | Slower inference (~100s of ms), heavier resource cost |
| **Output**| Label + Confidence (Sigmoid) | Label + Confidence (Softmax) |

*Data Source: 2,500 Amazon reviews (1-2 stars = Negative, 4-5 stars = Positive). Ingested via a custom SQLite ETL layer.*
