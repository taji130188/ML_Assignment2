"""
app.py — Streamlit app for ML Assignment 2
Breast Cancer Wisconsin (Diagnostic) — Binary Classification Demo

Features:
  a. Dataset upload option (CSV) — upload the provided test_data.csv (or any
     CSV with the same 30 feature columns + a 'diagnosis' column)
  b. Model selection dropdown (5 trained classifiers)
  c. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  d. Confusion matrix + full classification report
  e. ROC curve comparison across all 5 models
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    roc_curve
)

st.set_page_config(page_title="ML Assignment 2 — Classifier Demo", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}

# Fixed colors so each model keeps the same color across all charts
MODEL_COLORS = {
    "Logistic Regression": "#2ca02c",
    "Decision Tree": "#ff7f0e",
    "kNN": "#1f77b4",
    "Naive Bayes": "#9467bd",
    "Random Forest (Ensemble)": "#d62728",
}


@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "meta.json"), "r") as f:
        meta = json.load(f)
    models = {}
    for name, fname in MODEL_FILES.items():
        with open(os.path.join(MODEL_DIR, fname), "rb") as f:
            models[name] = pickle.load(f)
    metrics_table = pd.read_csv(os.path.join(MODEL_DIR, "metrics_comparison.csv"))
    return scaler, meta, models, metrics_table


scaler, meta, models, metrics_table = load_artifacts()
feature_cols = meta["feature_cols"]
target_names = meta["target_names"]  # ['malignant', 'benign'] -> 0, 1

st.title("🩺 Breast Cancer Classification — Model Demo")
st.caption(
    "ML Assignment 2 · Dataset: Breast Cancer Wisconsin (Diagnostic) · "
    "569 instances, 30 features, binary classification"
)

# ---------------------------------------------------------------------
# Sidebar — model selection
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")
selected_model_name = st.sidebar.selectbox(
    "Select a classification model", list(models.keys())
)
selected_model = models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.subheader("📊 All-model comparison")
st.sidebar.dataframe(metrics_table.set_index("ML Model Name"), use_container_width=True)

# ---------------------------------------------------------------------
# Main — CSV upload
# ---------------------------------------------------------------------
st.subheader("1️⃣ Upload Test Data (CSV)")
st.write(
    "Upload the provided `test_data.csv` (30 feature columns + a `diagnosis` "
    "column, where 0 = malignant, 1 = benign). Only test data should be "
    "uploaded, per Streamlit free-tier limits."
)

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded yet — using the bundled sample `test_data.csv` for demo purposes.")
    input_df = pd.read_csv(os.path.join(BASE_DIR, "test_data.csv"))

st.dataframe(input_df.head(10), use_container_width=True)
st.caption(f"Loaded {input_df.shape[0]} rows, {input_df.shape[1]} columns.")

# ---------------------------------------------------------------------
# Validate columns
# ---------------------------------------------------------------------
missing_cols = [c for c in feature_cols if c not in input_df.columns]
has_labels = "diagnosis" in input_df.columns

if missing_cols:
    st.error(
        f"The uploaded CSV is missing {len(missing_cols)} required feature "
        f"column(s), e.g. {missing_cols[:5]}. Please upload a CSV with the "
        f"same schema as test_data.csv."
    )
    st.stop()

X_input = input_df[feature_cols]
X_input_scaled = scaler.transform(X_input)

# ---------------------------------------------------------------------
# 2. Run predictions
# ---------------------------------------------------------------------
st.subheader(f"2️⃣ Predictions — {selected_model_name}")

y_pred = selected_model.predict(X_input_scaled)
if hasattr(selected_model, "predict_proba"):
    y_proba = selected_model.predict_proba(X_input_scaled)[:, 1]
else:
    y_proba = y_pred

pred_labels = [target_names[p] for p in y_pred]
result_df = input_df.copy()
result_df["Predicted"] = pred_labels
if has_labels:
    result_df["Actual"] = [target_names[a] for a in input_df["diagnosis"]]

st.dataframe(
    result_df[["Predicted"] + (["Actual"] if has_labels else [])].head(20),
    use_container_width=True,
)

# ---------------------------------------------------------------------
# 3. Evaluation metrics (only possible if ground-truth labels present)
# ---------------------------------------------------------------------
st.subheader("3️⃣ Evaluation Metrics")

if has_labels:
    y_true = input_df["diagnosis"]
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("AUC", f"{auc:.4f}")
    c3.metric("Precision", f"{prec:.4f}")
    c4.metric("Recall", f"{rec:.4f}")
    c5.metric("F1 Score", f"{f1:.4f}")
    c6.metric("MCC", f"{mcc:.4f}")

    # -------------------------------------------------------------
    # 4. Confusion matrix + classification report
    # -------------------------------------------------------------
    st.subheader("4️⃣ Confusion Matrix & Classification Report")

    col_a, col_b = st.columns(2)

    with col_a:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {selected_model_name}")
        st.pyplot(fig)

    with col_b:
        report = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True
        )
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

    # -------------------------------------------------------------
    # 5. ROC Curve — All Models
    # -------------------------------------------------------------
    st.subheader("5️⃣ ROC Curve — All Models")
    st.write(
        "Receiver Operating Characteristic curves for all 5 trained models, "
        "computed on the currently loaded data, with each model's AUC shown "
        "in the legend."
    )

    fig_roc, ax_roc = plt.subplots(figsize=(7, 5.5))
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random (AUC=0.50)")

    for name, mdl in models.items():
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(X_input_scaled)[:, 1]
        else:
            proba = mdl.predict(X_input_scaled)
        fpr, tpr, _ = roc_curve(y_true, proba)
        model_auc = roc_auc_score(y_true, proba)
        ax_roc.plot(
            fpr, tpr,
            label=f"{name} (AUC={model_auc:.4f})",
            color=MODEL_COLORS.get(name, None),
            linewidth=2,
        )

    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves — Breast Cancer Classification")
    ax_roc.legend(loc="lower right", fontsize=8)
    st.pyplot(fig_roc)

else:
    st.warning(
        "Uploaded CSV has no `diagnosis` column, so ground-truth metrics, the "
        "confusion matrix, and ROC curves can't be computed — predictions only "
        "are shown above."
    )

# ---------------------------------------------------------------------
# 6. Full model comparison table
# ---------------------------------------------------------------------
st.subheader("6️⃣ All Models — Comparison Table (on held-out test split)")
st.dataframe(metrics_table.set_index("ML Model Name"), use_container_width=True)

st.markdown("---")
st.caption("ML Assignment 2 · Streamlit Community Cloud Deployment Demo")
