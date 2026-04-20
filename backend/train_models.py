import pandas as pd
import numpy as np
import os
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
import pygad
from urllib.parse import urlparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Input, Concatenate, Dropout

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'balanced_urls.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MAX_LEN = 150  # Maximum length of URL for DL models
EMBEDDING_DIM = 50
vocab_size = 10000  # Will be adjusted based on tokenizer

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def load_and_preprocess_data():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found!")
    
    df = pd.read_csv(DATA_PATH)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Data shape: {df.shape}")
    print(df.head())
    
    # Features and Labels
    X = df['url']
    y = df['result']
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

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

def save_confusion_matrix(y_true, y_pred, model_name):
    print(f"Generating Confusion Matrix for {model_name}...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") + "_cm.png"
    plt.savefig(os.path.join(MODELS_DIR, filename))
    plt.close()
    print(f"Saved {filename}")

def train_ml_models(X_train, X_test, y_train, y_test):
    print("\n--- Training ML Models (TF-IDF & PyGAD Selected) ---")
    
    # TF-IDF Vectorizer
    print("Vectorizing data...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Save Vectorizer
    joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    
    print("Extracting structural features...")
    X_train_struct = np.array([extract_features(url) for url in X_train])
    X_test_struct = np.array([extract_features(url) for url in X_test])
    
    scaler = MinMaxScaler()
    X_train_struct_scaled = scaler.fit_transform(X_train_struct)
    X_test_struct_scaled = scaler.transform(X_test_struct)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    
    print("Combining features...")
    X_train_comb = sp.hstack((X_train_struct_scaled, X_train_tfidf)).tocsr()
    X_test_comb = sp.hstack((X_test_struct_scaled, X_test_tfidf)).tocsr()
    
    print("\n--- Running PyGAD for Feature Selection ---")
    X_train_ga, _, y_train_ga, _ = train_test_split(X_train_comb, y_train, train_size=0.1, random_state=42)
    rf_eval = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1)
    
    def fitness_func(ga_instance, solution, solution_idx):
        selected_indices = np.where(solution == 1)[0]
        if len(selected_indices) == 0:
            return 0.0
        
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(rf_eval, X_train_ga[:, selected_indices], y_train_ga, cv=3, scoring='f1', n_jobs=-1)
        
        penalty = 0.0001 * len(selected_indices)
        return scores.mean() - penalty
    
    num_features = X_train_comb.shape[1]
    
    ga_instance = pygad.GA(
        num_generations=5,
        num_parents_mating=2,
        fitness_func=fitness_func,
        sol_per_pop=5,
        num_genes=num_features,
        init_range_low=0,
        init_range_high=2,
        gene_type=int,
        mutation_percent_genes=5,
        random_seed=42,
        suppress_warnings=True
    )
    ga_instance.run()
    
    best_solution, best_fitness, _ = ga_instance.best_solution()
    selected_indices = np.where(best_solution == 1)[0]
    
    if len(selected_indices) == 0:
        selected_indices = np.array([0])
        
    print(f"Selected {len(selected_indices)} out of {num_features} features.")
    joblib.dump(selected_indices, os.path.join(MODELS_DIR, 'selected_features.pkl'))
    
    X_train_model = X_train_comb[:, selected_indices]
    X_test_model = X_test_comb[:, selected_indices]
    
    metrics_list = []

    # 1. Logistic Regression
    print("\n1. Logistic Regression")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_model, y_train)
    y_pred_lr = lr.predict(X_test_model)
    acc = accuracy_score(y_test, y_pred_lr)
    prec = precision_score(y_test, y_pred_lr)
    rec = recall_score(y_test, y_pred_lr)
    f1 = f1_score(y_test, y_pred_lr)
    print(f"Accuracy: {acc:.4f}")
    joblib.dump(lr, os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    metrics_list.append({'Model': 'Logistic Regression', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1})
    save_confusion_matrix(y_test, y_pred_lr, 'Logistic Regression')
    
    # 2. Naive Bayes
    print("\n2. Naive Bayes")
    nb = MultinomialNB()
    nb.fit(X_train_model, y_train)
    y_pred_nb = nb.predict(X_test_model)
    acc = accuracy_score(y_test, y_pred_nb)
    prec = precision_score(y_test, y_pred_nb)
    rec = recall_score(y_test, y_pred_nb)
    f1 = f1_score(y_test, y_pred_nb)
    print(f"Accuracy: {acc:.4f}")
    joblib.dump(nb, os.path.join(MODELS_DIR, 'naive_bayes.pkl'))
    metrics_list.append({'Model': 'Naive Bayes', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1})
    save_confusion_matrix(y_test, y_pred_nb, 'Naive Bayes')
    
    # 3. Random Forest
    print("\n3. Random Forest")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_model, y_train)
    y_pred_rf = rf.predict(X_test_model)
    acc = accuracy_score(y_test, y_pred_rf)
    prec = precision_score(y_test, y_pred_rf)
    rec = recall_score(y_test, y_pred_rf)
    f1 = f1_score(y_test, y_pred_rf)
    print(f"Accuracy: {acc:.4f}")
    joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest.pkl'))
    metrics_list.append({'Model': 'Random Forest', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1})
    save_confusion_matrix(y_test, y_pred_rf, 'Random Forest')

    return metrics_list

def train_dl_models(X_train, X_test, y_train, y_test):
    print("\n--- Training DL Models (Keras) ---")
    
    # Tokenizer
    print("Tokenizing data...")
    tokenizer = Tokenizer(num_words=vocab_size, char_level=True) # Using char-level for URLs can be effective
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)
    
    # Save Tokenizer
    with open(os.path.join(MODELS_DIR, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    metrics_list = []
        
    # 4. CNN (1D)
    print("\n4. CNN (1D)")
    model_cnn = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_cnn.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_data=(X_test_pad, y_test), verbose=1)
    
    y_pred_cnn = model_cnn.predict(X_test_pad)
    y_pred_cnn_binary = (y_pred_cnn > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred_cnn_binary)
    prec = precision_score(y_test, y_pred_cnn_binary)
    rec = recall_score(y_test, y_pred_cnn_binary)
    f1 = f1_score(y_test, y_pred_cnn_binary)

    print(f"CNN Accuracy: {acc:.4f}")
    model_cnn.save(os.path.join(MODELS_DIR, 'cnn_model.h5'))
    metrics_list.append({'Model': 'CNN (1D)', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1})
    save_confusion_matrix(y_test, y_pred_cnn_binary, 'CNN (1D)')
    
    # 5. TM-CNN (Multi-Channel)
    print("\n5. TM-CNN (Multi-Channel CNN)")
    input_layer = Input(shape=(MAX_LEN,))
    embedding = Embedding(vocab_size, EMBEDDING_DIM)(input_layer)
    
    # Parallel convolutions
    conv_blocks = []
    for kernel_size in [3, 4, 5]:
        conv = Conv1D(128, kernel_size, activation='relu')(embedding)
        pool = GlobalMaxPooling1D()(conv)
        conv_blocks.append(pool)
        
    concat = Concatenate()(conv_blocks)
    dense = Dense(10, activation='relu')(concat)
    dropout = Dropout(0.5)(dense)
    output = Dense(1, activation='sigmoid')(dropout)
    
    model_tmcnn = Model(inputs=input_layer, outputs=output)
    model_tmcnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_tmcnn.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_data=(X_test_pad, y_test), verbose=1)
    
    y_pred_tmcnn = model_tmcnn.predict(X_test_pad)
    y_pred_tmcnn_binary = (y_pred_tmcnn > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred_tmcnn_binary)
    prec = precision_score(y_test, y_pred_tmcnn_binary)
    rec = recall_score(y_test, y_pred_tmcnn_binary)
    f1 = f1_score(y_test, y_pred_tmcnn_binary)
    
    print(f"TM-CNN Accuracy: {acc:.4f}")
    model_tmcnn.save(os.path.join(MODELS_DIR, 'tm_cnn_model.h5'))
    metrics_list.append({'Model': 'TM-CNN (Multi-Channel)', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1})
    save_confusion_matrix(y_test, y_pred_tmcnn_binary, 'TM-CNN (Multi-Channel)')

    return metrics_list

def evaluate_hybrid_model(X_test, y_test):
    print("\n--- Evaluating Hybrid Ensemble (RF + TM-CNN) ---")
    
    # Load Artifacts
    tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    selected_indices = joblib.load(os.path.join(MODELS_DIR, 'selected_features.pkl'))
    rf = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    
    # Process for RF
    X_test_tfidf = tfidf.transform(X_test)
    X_test_struct = np.array([extract_features(url) for url in X_test])
    X_test_struct_scaled = scaler.transform(X_test_struct)
    X_test_comb = sp.hstack((X_test_struct_scaled, X_test_tfidf)).tocsr()
    X_test_model = X_test_comb[:, selected_indices]
    
    # Predict probabilities for RF
    rf_probs = rf.predict_proba(X_test_model)[:, 1]
    
    # Process for TM-CNN
    with open(os.path.join(MODELS_DIR, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    from tensorflow.keras.models import load_model
    tm_cnn = load_model(os.path.join(MODELS_DIR, 'tm_cnn_model.h5'))
    
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)
    tmcnn_probs = tm_cnn.predict(X_test_pad).flatten()
    
    # Weighted vote
    w_rf = 0.5
    w_tmcnn = 0.5
    final_probs = (w_rf * rf_probs) + (w_tmcnn * tmcnn_probs)
    y_pred_hybrid = (final_probs > 0.5).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred_hybrid)
    prec = precision_score(y_test, y_pred_hybrid)
    rec = recall_score(y_test, y_pred_hybrid)
    f1 = f1_score(y_test, y_pred_hybrid)
    
    print(f"Hybrid Accuracy: {acc:.4f}")
    save_confusion_matrix(y_test, y_pred_hybrid, 'Hybrid Ensemble')
    
    return [{'Model': 'Hybrid Ensemble', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}]

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    ml_metrics = train_ml_models(X_train, X_test, y_train, y_test)
    dl_metrics = train_dl_models(X_train, X_test, y_train, y_test)
    hybrid_metrics = evaluate_hybrid_model(X_test, y_test)
    
    all_metrics = ml_metrics + dl_metrics + hybrid_metrics
    
    # Print Table
    print("\n" + "="*85)
    print(f"{'Model':<25} | {'Accuracy':<12} | {'Precision':<12} | {'Recall':<12} | {'F1 Score':<12}")
    print("-" * 85)
    for m in all_metrics:
        print(f"{m['Model']:<25} | {m['Accuracy']:<12.4f} | {m['Precision']:<12.4f} | {m['Recall']:<12.4f} | {m['F1 Score']:<12.4f}")
    print("="*85)

    print(f"\nTraining Complete. Models saved in {MODELS_DIR}/")
