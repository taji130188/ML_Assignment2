"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset, evaluates each with 6 metrics, saves:
  - trained models (model/*.pkl)
  - a fitted StandardScaler (model/scaler.pkl)
  - the test split as CSV (test_data.csv) for use in the Streamlit app
  - a metrics comparison table (model/metrics_comparison.csv)

Dataset: Breast Cancer Wisconsin (Diagnostic)
  - Source: UCI ML Repository / sklearn.datasets (built-in copy of the
    public UCI dataset: https://archive.ics.uci.edu/dataset/17)
  - Instances: 569  (>= 500 required)
  - Features: 30    (>= 12 required)
  - Task: Binary classification (malignant vs benign)
"""

import pandas as pd
import numpy as np
import pickle
import json
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODEL_DIR)

# ---------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df.rename(columns={"target": "diagnosis"}, inplace=True)
# In sklearn's encoding: 0 = malignant, 1 = benign

feature_cols = [c for c in df.columns if c != "diagnosis"]
X = df[feature_cols]
y = df["diagnosis"]

print(f"Dataset shape: {df.shape}")
print(f"Number of features: {len(feature_cols)}")
print(f"Class balance:\n{y.value_counts()}")

# ---------------------------------------------------------------------
# 2. Train/test split (stratified)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------
# 3. Scale features (fit on train only)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ---------------------------------------------------------------------
# 4. Save the test split as test_data.csv (features + true label)
#    This is the file used for the "upload CSV" feature in the Streamlit app.
# ---------------------------------------------------------------------
test_df = X_test.copy()
test_df["diagnosis"] = y_test.values
test_df.to_csv(os.path.join(ROOT_DIR, "test_data.csv"), index=False)
print(f"\nSaved test_data.csv with shape {test_df.shape}")

# ---------------------------------------------------------------------
# 5. Define models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = y_pred

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(f"\n{name}: {metrics}")

    # Save the trained model
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
    with open(os.path.join(MODEL_DIR, fname), "wb") as f:
        pickle.dump(model, f)

# ---------------------------------------------------------------------
# 6. Save comparison table
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(MODEL_DIR, "metrics_comparison.csv"), index=False)

# Save feature column order + target names (needed by the Streamlit app)
meta = {
    "feature_cols": feature_cols,
    "target_names": list(data.target_names),  # ['malignant', 'benign']
}
with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n\n=== FINAL COMPARISON TABLE ===")
print(results_df.to_string(index=False))
print("\nAll models, scaler, metadata, metrics table, and test_data.csv saved.")
