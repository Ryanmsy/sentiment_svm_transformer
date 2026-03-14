import streamlit as st
import sqlite3
import os

# --- IMPORT MODELS ---
try:
    from svm_sentiment import SVMSentimentModel
    from transformer_predict import TransformerPredictor as TransformerModel
    from config import DB_LOGS, DB_WAREHOUSE, SVM_MODEL_PATH, BERT_MODEL_DIR
except ImportError as e:
    st.error(f"Missing model files: {e}")
    st.stop()

# --- DATABASE SETUP ---
def init_log_db():
    conn = sqlite3.connect(DB_LOGS)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_version TEXT,
            input_text TEXT,
            prediction_label TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(model_name, text, pred):
    conn = sqlite3.connect(DB_LOGS)
    c = conn.cursor()
    c.execute(
        "INSERT INTO prediction_logs (model_version, input_text, prediction_label) VALUES (?, ?, ?)",
        (model_name, text, pred)
    )
    conn.commit()
    conn.close()

init_log_db()

# --- PAGE SETUP ---
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["SVM (Fast)", "DistilBERT (Accurate)", "Compare Both"]
)

st.sidebar.divider()
st.sidebar.markdown("""
**About this project**

Two sentiment models trained on Amazon product reviews:

- **SVM** — TF-IDF features + LinearSVC. Fast, lightweight, interpretable.
- **DistilBERT** — Fine-tuned transformer. Slower but captures context and nuance.

The Compare view runs both simultaneously so you can see where they agree or disagree.
""")

# --- LOAD MODELS ---
@st.cache_resource
def get_svm_model():
    model = SVMSentimentModel(db_filepath=DB_WAREHOUSE)
    if os.path.exists(SVM_MODEL_PATH):
        model.load_model(SVM_MODEL_PATH)
        return model
    return None

@st.cache_resource
def get_transformer_model():
    try:
        model = TransformerModel()
        model.load_saved_model(BERT_MODEL_DIR)
        return model
    except Exception:
        return None

# --- HELPER: render a single model result ---
def render_result(label, confidence):
    if label == "Positive":
        st.success(f"**{label}**")
    else:
        st.error(f"**{label}**")
    st.progress(confidence, text=f"Confidence: {confidence:.1%}")

# --- MAIN UI ---
st.title("Sentiment Analyzer")
st.write("Enter text below to detect if the sentiment is Positive or Negative.")

user_text = st.text_area(
    "Input Text", height=150,
    placeholder="e.g., I absolutely loved this product!"
)

if st.button("Analyze", type="primary", use_container_width=True):
    if not user_text:
        st.warning("Please type something first.")
    else:
        st.divider()

        if model_choice == "Compare Both":
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("SVM (Fast)")
                with st.spinner("Running SVM..."):
                    svm = get_svm_model()
                if svm:
                    label, conf = svm.predict_with_confidence(user_text)
                    render_result(label, conf)
                    log_prediction("SVM (Fast)", user_text, label)
                else:
                    st.error("SVM model file not found. Run `svm_sentiment.py` to train and save it.")

            with col2:
                st.subheader("DistilBERT (Accurate)")
                with st.spinner("Running DistilBERT..."):
                    transformer = get_transformer_model()
                if transformer:
                    label, conf = transformer.predict_with_confidence(user_text)
                    render_result(label, conf)
                    log_prediction("DistilBERT (Accurate)", user_text, label)
                else:
                    st.error("Transformer model failed to load.")

        elif "SVM" in model_choice:
            with st.spinner("Analyzing..."):
                svm = get_svm_model()
            if svm:
                label, conf = svm.predict_with_confidence(user_text)
                render_result(label, conf)
                log_prediction(model_choice, user_text, label)
            else:
                st.error("SVM model file not found. Run `svm_sentiment.py` to train and save it.")

        elif "DistilBERT" in model_choice:
            with st.spinner("Analyzing..."):
                transformer = get_transformer_model()
            if transformer:
                label, conf = transformer.predict_with_confidence(user_text)
                render_result(label, conf)
                log_prediction(model_choice, user_text, label)
            else:
                st.error("Transformer model failed to load.")
