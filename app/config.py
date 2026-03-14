import os
from pathlib import Path

_app_dir = Path(__file__).resolve().parent
_root_dir = _app_dir.parent

# DB files live in app/
DB_LOGS     = os.getenv("DB_LOGS",      str(_app_dir / "production_logs.db"))
DB_WAREHOUSE = os.getenv("DB_WAREHOUSE", str(_app_dir / "corporate_data_warehouse.db"))

# Model artifacts live at project root
SVM_MODEL_PATH = os.getenv("SVM_MODEL_PATH", str(_root_dir / "svm_model.pkl"))
BERT_MODEL_DIR  = os.getenv("BERT_MODEL_DIR",  str(_root_dir / "bert_model_saved"))
