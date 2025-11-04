# train.py

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

EMBEDDINGS_FILE = 'data/embeddings.npy'
LABELS_FILE = 'data/labels.csv'
MODEL_OUTPUT_PATH = 'models/model.joblib'
SCALER_OUTPUT_PATH = 'models/scaler.joblib'
METRICS_OUTPUT_PATH = 'reports/metrics.json'

def train_model():
    try:
        X = np.load(EMBEDDINGS_FILE)
        y = pd.read_csv(LABELS_FILE)['label']

        print(f"Loaded {len(X)} embeddings and {len(y)} labels.")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = LogisticRegression(max_iter=200, class_weight='balanced', random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred_train = model.predict(X_train_scaled)
        y_pred_val = model.predict(X_val_scaled)

        y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
        y_prob_val = model.predict_proba(X_val_scaled)[:, 1]

        metrics = {
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "validation_accuracy": accuracy_score(y_val, y_pred_val),
            "train_auc": roc_auc_score(y_train, y_prob_train),
            "validation_auc": roc_auc_score(y_val, y_prob_val)
        }

        print("Model performance:")
        print(json.dumps(metrics, indent=2))

        joblib.dump(model, MODEL_OUTPUT_PATH)
        joblib.dump(scaler, SCALER_OUTPUT_PATH)

        with open(METRICS_OUTPUT_PATH, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"\nTrained model saved to {MODEL_OUTPUT_PATH}")
        print(f"Scaler saved to {SCALER_OUTPUT_PATH}")
        print(f"Metrics saved to {METRICS_OUTPUT_PATH}")

    except FileNotFoundError:
        print("Embeddings or labels file not found. Please run scripts/embeddings.py first.")
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    train_model()
