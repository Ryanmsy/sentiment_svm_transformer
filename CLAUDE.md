# Project Conventions for Claude Code

## Runtime
- Always use `uv` to run Python: `uv run python <file>`
- Always use `uv run streamlit run <file>` for Streamlit

## Project Layout
- Source data lives in `data/` (xlsx + SQLite .db files)
- Trained models: `bert_model_saved/` (transformer) and `svm_model.pkl` (SVM)
- App modules all live in `app/`
- Path config (DB paths, model dirs) is in `app/config.py` — import from there, don't hardcode

## Code Patterns
- `basepath = Path(__file__).resolve().parent.parent` — used in app modules to anchor paths to project root
- SQLite is the data backend; source Excel is ingested once via helper functions
- Both model classes (`SentimentModel`, `SVMSentimentModel`) are OOP pipelines with sequential step methods
- `transformers>=5.3.0`: use `processing_class=` instead of `tokenizer=` in `Trainer.__init__()`
- Binary sentiment labels: rating ≤2 → 0 (Negative), rating ≥4 → 1 (Positive), rating 3 → dropped

## Dependencies
- Managed with `uv` and `pyproject.toml`
- Key packages: transformers, datasets, evaluate, torch, scikit-learn, streamlit, pandas, openpyxl
