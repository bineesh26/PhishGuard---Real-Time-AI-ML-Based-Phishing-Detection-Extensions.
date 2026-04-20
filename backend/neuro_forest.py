import os
import math
import numpy as np
import pandas as pd
import joblib
import time
from urllib.parse import urlparse

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class NeuroForestFeatureExtractor:
    """
    Extracts engineered features for the Random Forest branch.
    Optimized for low-latency inference.
    """
    def __init__(self):
        # List of high risk TLDs (simplified for speed)
        self.risk_tlds = {'.xyz', '.top', '.ml', '.tk', '.ga', '.cf', '.gq', '.cc', '.su', '.pw'}

    def _shannon_entropy(self, data: str) -> float:
        if not data:
            return 0.0
        prob = [float(data.count(c)) / len(data) for c in dict.fromkeys(list(data))]
        return -sum([p * math.log(p, 2) for p in prob])

    def extract_features(self, url: str) -> np.ndarray:
        # Prevent division by zero
        length = len(url) if len(url) > 0 else 1
        
        # 1. URL Entropy Score
        entropy = self._shannon_entropy(url)
        
        # 2. Digit Ratio
        digit_count = sum(c.isdigit() for c in url)
        digit_ratio = digit_count / length
        
        # 3. Special Character Ratio
        # Non-alphanumeric and non-standard URL characters
        special_count = sum(not c.isalnum() for c in url)
        special_ratio = special_count / length
        
        # Parse URL
        parsed = urlparse(url if '//' in url else 'http://' + url)
        netloc = parsed.netloc.lower()
        
        # 4. Subdomain Depth
        # Count number of dots in the domain name
        subdomain_depth = netloc.count('.')
        
        # 5. TLD Risk Score
        tld = ''
        if '.' in netloc:
            tld = '.' + netloc.split('.')[-1]
        tld_risk = 1.0 if tld in self.risk_tlds else 0.0

        return np.array([entropy, digit_ratio, special_ratio, subdomain_depth, tld_risk])

class NeuroForestModel:
    """
    CNN + Random Forest Hybrid Model for Phishing Detection.
    """
    def __init__(self):
        self.max_len = 100
        self.tokenizer = Tokenizer(char_level=True, lower=True)
        self.scaler = MinMaxScaler()
        self.feature_extractor = NeuroForestFeatureExtractor()
        
        # Models
        self.cnn_model = None
        self.rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        
    def _build_cnn(self, vocab_size):
        model = Sequential([
            Embedding(input_dim=vocab_size + 1, output_dim=32, input_length=self.max_len),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, dataset_path: str, url_col='url', label_col='label'):
        """
        Trains both CNN and RF from scratch using the provided dataset.
        label: 1 for phishing, 0 for legitimate
        """
        print(f"Loading dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        
        urls = df[url_col].astype(str).tolist()
        labels = df[label_col].values

        print("Extracting engineered features for Random Forest...")
        X_rf = np.array([self.feature_extractor.extract_features(url) for url in urls])
        X_rf_scaled = self.scaler.fit_transform(X_rf)

        print("Tokenizing URLs for CNN...")
        self.tokenizer.fit_on_texts(urls)
        vocab_size = len(self.tokenizer.word_index)
        sequences = self.tokenizer.texts_to_sequences(urls)
        X_cnn = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

        # Stratified Split
        print("Splitting data (80% train, 20% test)...")
        X_cnn_train, X_cnn_test, X_rf_train, X_rf_test, y_train, y_test = train_test_split(
            X_cnn, X_rf_scaled, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Train CNN
        print("Training lightweight CNN...")
        self.cnn_model = self._build_cnn(vocab_size)
        self.cnn_model.fit(
            X_cnn_train, y_train,
            batch_size=32,
            epochs=5,
            validation_data=(X_cnn_test, y_test),
            verbose=1
        )

        # Train Random Forest
        print("Training constrained Random Forest...")
        self.rf_model.fit(X_rf_train, y_train)

        print("Training complete.")
        
    def predict(self, url: str) -> dict:
        """
        Predicts if a URL is phishing or safe.
        Returns standard interface dictionary.
        """
        start_time = time.time()
        
        if self.cnn_model is None or self.rf_model is None:
            raise Exception("Models are not trained or loaded.")

        # CNN Branch Inference
        seq = self.tokenizer.texts_to_sequences([url])
        padded_seq = pad_sequences(seq, maxlen=self.max_len, padding='post', truncating='post')
        cnn_prob = float(self.cnn_model.predict(padded_seq, verbose=0)[0][0])

        # RF Branch Inference
        feats = self.feature_extractor.extract_features(url)
        feats_scaled = self.scaler.transform([feats])
        rf_prob = float(self.rf_model.predict_proba(feats_scaled)[0][1])

        # Meta-learner: Simple Averaging
        final_prob = (cnn_prob + rf_prob) / 2.0
        
        is_phishing = final_prob > 0.5
        
        latency_ms = (time.time() - start_time) * 1000

        return {
            "label": "phishing" if is_phishing else "safe",
            "confidence": final_prob,
            "model": "cnn_rf_hybrid",
            "details": {
                "cnn_prob": cnn_prob,
                "rf_prob": rf_prob,
                "latency_ms": round(latency_ms, 2)
            }
        }

    def save(self, filepath_prefix: str = "neuro_forest"):
        """Saves models, tokenizer, and scaler."""
        os.makedirs(os.path.dirname(filepath_prefix) or '.', exist_ok=True)
        
        if self.cnn_model is not None:
            self.cnn_model.save(f"{filepath_prefix}_cnn.h5")
        
        artifacts = {
            "rf_model": self.rf_model,
            "tokenizer": self.tokenizer,
            "scaler": self.scaler
        }
        joblib.dump(artifacts, f"{filepath_prefix}_artifacts.pkl")
        print(f"Model saved to {filepath_prefix}_cnn.h5 and {filepath_prefix}_artifacts.pkl")

    def load(self, filepath_prefix: str = "neuro_forest"):
        """Loads models, tokenizer, and scaler."""
        cnn_path = f"{filepath_prefix}_cnn.h5"
        artifacts_path = f"{filepath_prefix}_artifacts.pkl"
        
        if os.path.exists(cnn_path) and os.path.exists(artifacts_path):
            self.cnn_model = load_model(cnn_path)
            artifacts = joblib.load(artifacts_path)
            self.rf_model = artifacts["rf_model"]
            self.tokenizer = artifacts["tokenizer"]
            self.scaler = artifacts["scaler"]
            print(f"Model loaded from {filepath_prefix}")
        else:
            raise FileNotFoundError(f"Files not found: {cnn_path} or {artifacts_path}")

def generate_pipeline_diagram():
    """Generates a text/mermaid representation of the pipeline."""
    diagram = """
    ## Neuro-Forest Architecture Pipeline
    ```mermaid
    graph TD;
        A[Input URL] --> B[Feature Extractor]
        A --> C[Character Tokenizer]
        
        B --> D[Compute Core Features]
        D --> E[MinMaxScaler]
        E --> F[Random Forest n_estimators=50]
        
        C --> G[Padding max_len=100]
        G --> H[Embedding Layer]
        H --> I[Conv1D 32 flts, k=3]
        I --> J[Global MaxPooling 1D]
        J --> K[Dense Layer sigmoid]
        
        F -->|rf_prob| L[Meta-learner: Simple Averaging]
        K -->|cnn_prob| L
        
        L --> M[Output Label & Confidence]
    ```
    """
    return diagram

if __name__ == "__main__":
    print(generate_pipeline_diagram())
    # Example usage:
    # model = NeuroForestModel()
    # model.train("balanced_urls.csv", url_col="url", label_col="label")
    # print(model.predict("http://secure-login-update-paypal.com"))
    # model.save("backend/models/neuro_forest")
