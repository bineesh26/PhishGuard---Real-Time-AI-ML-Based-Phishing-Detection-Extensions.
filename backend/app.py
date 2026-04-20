from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import time
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import scipy.sparse as sp
from urllib.parse import urlparse
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MAX_LEN = 150

# Load DL Artifacts
tm_cnn_model = None
tokenizer = None
# Load ML Artifacts
rf_model = None
scaler = None
tfidf_vectorizer = None
selected_features = None

try:
    # DL Models
    MODEL_PATH = os.path.join(MODELS_DIR, 'tm_cnn_model.h5')
    if os.path.exists(MODEL_PATH):
        tm_cnn_model = load_model(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    
    TOKENIZER_PATH = os.path.join(MODELS_DIR, 'tokenizer.pickle')
    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)

    # ML Models
    RF_PATH = os.path.join(MODELS_DIR, 'random_forest.pkl')
    if os.path.exists(RF_PATH):
        rf_model = joblib.load(RF_PATH)
        
    SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        
    TFIDF_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    if os.path.exists(TFIDF_PATH):
        tfidf_vectorizer = joblib.load(TFIDF_PATH)
        
    SEL_FEAT_PATH = os.path.join(MODELS_DIR, 'selected_features.pkl')
    if os.path.exists(SEL_FEAT_PATH):
        selected_features = joblib.load(SEL_FEAT_PATH)

except Exception as e:
    print(f"Error loading artifacts: {e}")

def extract_features(url: str) -> np.array:
    length = len(url)
    dots = url.count('.')
    hyphens = url.count('-')
    digit_ratio = sum(c.isdigit() for c in url) / max(1, length)
    special_chars = sum(not c.isalnum() for c in url)
    
    parsed = urlparse(url if '//' in url else 'http://'+url)
    subdomains = parsed.netloc.count('.')
    
    keywords = ['login', 'secure', 'bank', 'verify', 'account', 'update']
    keyword_count = sum(url.lower().count(kw) for kw in keywords)
    https_usage = 1 if url.startswith('https') else 0
    
    return np.array([length, dots, hyphens, digit_ratio, special_chars, subdomains, keyword_count, https_usage])

class URLData(BaseModel):
    url: str

@app.post("/check")
def check_url(data: URLData):
    if any(x is None for x in [tm_cnn_model, tokenizer, rf_model, scaler, tfidf_vectorizer, selected_features]):
        raise HTTPException(status_code=500, detail="Models or Tokenizer not loaded.")

    try:
        start_time_total = time.time()
        
        # --- DL (TM-CNN) Timing ---
        start_time_tmcnn = time.time()
        sequences = tokenizer.texts_to_sequences([data.url])
        padded = pad_sequences(sequences, maxlen=MAX_LEN)
        tmcnn_prediction_val = tm_cnn_model.predict(padded, verbose=0)[0][0]
        end_time_tmcnn = time.time()
        tmcnn_time_ms = (end_time_tmcnn - start_time_tmcnn) * 1000

        # --- ML (Random Forest) Timing ---
        start_time_rf = time.time()
        struct_feats = np.array([extract_features(data.url)])
        struct_feats_scaled = scaler.transform(struct_feats)
        tfidf_feats = tfidf_vectorizer.transform([data.url])
        comb_feats = sp.hstack((struct_feats_scaled, tfidf_feats)).tocsr()
        model_feats = comb_feats[:, selected_features]

        rf_prediction_val = rf_model.predict_proba(model_feats)[0][1]
        end_time_rf = time.time()
        rf_time_ms = (end_time_rf - start_time_rf) * 1000

        # Hybrid Ensemble Weights
        w_rf = 0.5
        w_tmcnn = 0.5
        prediction_val = (w_rf * rf_prediction_val) + (w_tmcnn * tmcnn_prediction_val)
        
        prediction = 1 if prediction_val > 0.5 else 0
        probability = float(prediction_val)
        
        end_time_total = time.time()
        execution_time_ms = (end_time_total - start_time_total) * 1000
        
        print(f"⏱️ TOTAL Latency: {round(execution_time_ms, 2)} ms | TM-CNN: {round(tmcnn_time_ms, 2)} ms | RF: {round(rf_time_ms, 2)} ms")

        return {
            "prediction": "phishing" if prediction == 1 else "legitimate",
            "probability": probability,
            "status": int(prediction),
            "execution_time_ms": round(execution_time_ms, 2),
            "details": {
                "tmcnn_prob": float(tmcnn_prediction_val),
                "tmcnn_latency_ms": round(tmcnn_time_ms, 2),
                "rf_prob": float(rf_prediction_val),
                "rf_latency_ms": round(rf_time_ms, 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
