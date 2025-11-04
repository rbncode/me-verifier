import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

EMBEDDINGS_FILE = 'data/embeddings.npy'
LABELS_FILE = 'data/labels.csv'
MODEL_PATH = 'models/model.joblib'
SCALER_PATH = 'models/scaler.joblib'
CONFUSION_MATRIX_OUTPUT = 'reports/confusion_matrix.png'
PR_CURVE_OUTPUT = 'reports/precision_recall_curve.png'
ROC_CURVE_OUTPUT = 'reports/roc_curve.png'
EVALUATION_REPORT_OUTPUT = 'reports/evaluation_report.json'

def evaluate_model():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        X = np.load(EMBEDDINGS_FILE)
        y = pd.read_csv(LABELS_FILE)['label']

        _, X_val, _, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_val_scaled = scaler.transform(X_val)

        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]

        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Me', 'Me'], yticklabels=['Not Me', 'Me'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(CONFUSION_MATRIX_OUTPUT)
        print(f"Confusion matrix saved to {CONFUSION_MATRIX_OUTPUT}")
        plt.close()

        precision, recall, _ = precision_recall_curve(y_val, y_prob)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', label=f'PR AUC = {pr_auc:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(PR_CURVE_OUTPUT)
        print(f"Precision-Recall curve saved to {PR_CURVE_OUTPUT}")
        plt.close()

        fpr, tpr, _ = roc_curve(y_val, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, marker='.', label=f'ROC AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(ROC_CURVE_OUTPUT)
        print(f"ROC curve saved to {ROC_CURVE_OUTPUT}")
        plt.close()

        report = classification_report(y_val, y_pred, target_names=['Not Me', 'Me'], output_dict=True)
        report['pr_auc'] = pr_auc
        report['roc_auc'] = roc_auc

        with open(EVALUATION_REPORT_OUTPUT, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"Evaluation report saved to {EVALUATION_REPORT_OUTPUT}")


    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure you have run train.py first.")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

if __name__ == "__main__":
    evaluate_model()
