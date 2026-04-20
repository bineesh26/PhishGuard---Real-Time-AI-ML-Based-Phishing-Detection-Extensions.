import time
import joblib
import pickle
import numpy as np
import scipy.sparse as sp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from urllib.parse import urlparse

# Constants
MAX_LEN = 150

def extract_features_dummy(url: str) -> np.array:
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

def measure_inference_times():
    print("Loading data for benchmarking...")
    try:
        df = pd.read_csv('backend/balanced_urls.csv')
    except:
        df = pd.read_csv('balanced_urls.csv')
        
    sample_urls = df['url'].head(100).tolist() # Benchmark on 100 sample URLs
    
    # Load Models and preprocessing artifacts
    base_dir = 'backend/models/'
    try:
        tfidf = joblib.load(base_dir + 'tfidf_vectorizer.pkl')
        scaler = joblib.load(base_dir + 'scaler.pkl')
        selected_indices = joblib.load(base_dir + 'selected_features.pkl')
        
        lr = joblib.load(base_dir + 'logistic_regression.pkl')
        nb = joblib.load(base_dir + 'naive_bayes.pkl')
        rf = joblib.load(base_dir + 'random_forest.pkl')
        
        with open(base_dir + 'tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        cnn = load_model(base_dir + 'cnn_model.h5')
        tmcnn = load_model(base_dir + 'tm_cnn_model.h5')
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    results = {}
    
    # 1. Logistic Regression & Naive Bayes & Random forest (ML Models)
    # Preprocessing time
    start = time.perf_counter()
    struct_feats = np.array([extract_features_dummy(u) for u in sample_urls])
    struct_scaled = scaler.transform(struct_feats)
    tfidf_feats = tfidf.transform(sample_urls)
    comb_feats = sp.hstack((struct_scaled, tfidf_feats)).tocsr()
    model_inputs = comb_feats[:, selected_indices]
    end = time.perf_counter()
    prep_time_ml = (end - start) / len(sample_urls) * 1000 # in ms
    
    # LR
    start = time.perf_counter()
    _ = lr.predict(model_inputs)
    end = time.perf_counter()
    results['Logistic Regression'] = (end - start) / len(sample_urls) * 1000
    
    # NB
    start = time.perf_counter()
    _ = nb.predict(model_inputs)
    end = time.perf_counter()
    results['Naive Bayes'] = (end - start) / len(sample_urls) * 1000
    
    # RF
    start = time.perf_counter()
    _ = rf.predict(model_inputs)
    end = time.perf_counter()
    results['Random Forest'] = (end - start) / len(sample_urls) * 1000

    # 2. DL Models (CNN, TMCNN)
    # Preprocessing
    start = time.perf_counter()
    seqs = tokenizer.texts_to_sequences(sample_urls)
    pads = pad_sequences(seqs, maxlen=MAX_LEN)
    end = time.perf_counter()
    prep_time_dl = (end - start) / len(sample_urls) * 1000
    
    # CNN
    start = time.perf_counter()
    _ = cnn.predict(pads, verbose=0)
    end = time.perf_counter()
    results['CNN (1D)'] = (end - start) / len(sample_urls) * 1000
    
    # TM-CNN
    start = time.perf_counter()
    _ = tmcnn.predict(pads, verbose=0)
    end = time.perf_counter()
    results['TM-CNN (Multi-Channel)'] = (end - start) / len(sample_urls) * 1000

    print("="*60)
    print(f"{'Model':<30} | {'Average Inference Time per URL (ms)'}")
    print("="*60)
    for model_name, t in results.items():
        print(f"{model_name:<30} | {t:.4f} ms")
    print("="*60)
    print(f"Typical Preprocessing Time (ML): {prep_time_ml:.4f} ms")
    print(f"Typical Preprocessing Time (DL Text): {prep_time_dl:.4f} ms")

if __name__ == '__main__':
    measure_inference_times()
