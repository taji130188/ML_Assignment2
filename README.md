# ML Assignment 2 — Breast Cancer Classification with Streamlit

## a. Problem Statement

Breast cancer diagnosis from digitized images of a fine needle aspirate (FNA)
of a breast mass is a critical, time-sensitive task. Features are computed
from the image describing characteristics of the cell nuclei present (radius,
texture, perimeter, area, smoothness, etc.). The goal of this project is to
build and compare multiple supervised classification models that predict
whether a breast mass is **malignant** or **benign** based on these
numeric features, and to expose the trained models through an interactive
Streamlit web application for evaluation.

This is a **binary classification** problem.

## b. Dataset Description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository
  (https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic),
  accessed here via `sklearn.datasets.load_breast_cancer()`, which bundles
  the same public UCI dataset.
- **Instances:** 569 (≥ 500 required ✅)
- **Features:** 30 numeric features (≥ 12 required ✅) — three summary
  statistics (mean, standard error, "worst"/largest) computed for 10 real-valued
  characteristics of each cell nucleus: radius, texture, perimeter, area,
  smoothness, compactness, concavity, concave points, symmetry, and fractal
  dimension.
- **Target variable:** `diagnosis` — 0 = malignant (212 cases), 1 = benign
  (357 cases)
- **Train/Test split:** 80% train (455 rows) / 20% test (114 rows),
  stratified by class, `random_state=42`. The 20% test split is saved as
  `test_data.csv` and is the file used by the Streamlit app.

## c. GitHub Repository Link

> **`<<(https://github.com/taji130188/ML_Assignment2) >>`**

Repository contains: `app.py`, `requirements.txt`, `README.md`,
`test_data.csv`, and the `model/` folder (training script + saved model
files for all 5 implemented models).

## d. Models Used

All 5 models below were trained on the **same** dataset and the same
80/20 stratified train/test split, after standardizing features with
`StandardScaler` (fit on the training set only).

### Comparison Table (evaluation on the 20% held-out test set, 114 samples)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9932 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

*(Reproducible via `model/train_models.py`; raw numbers are also written to
`model/metrics_comparison.csv`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer on this dataset — the classes are close to linearly separable once features are standardized, so a simple linear decision boundary generalizes very well. Highest accuracy, precision, recall, F1, and MCC of all five models, with almost no misclassifications on the test set. |
| Decision Tree | Weakest performer here. A single unconstrained tree overfits the training data (high variance), which hurts generalization on unseen test rows — visible in the lowest accuracy, AUC, and MCC of the group. |
| kNN | Solid performance, close to Random Forest. Because it relies on distance between standardized feature vectors, scaling was essential; it captures local structure well but is a bit more sensitive to noisy/overlapping feature regions than Logistic Regression here. |
| Naive Bayes | Reasonable accuracy despite its strong (and technically violated) assumption that the 30 features are conditionally independent given the class. Its AUC is high (0.9868) even though its hard-label metrics (accuracy/F1) trail the top models, meaning it ranks/scores cases well but its default 0.5 threshold is slightly less optimal. |
| Random Forest (Ensemble) | Strong performer — bagging many trees reduces the variance/overfitting problem seen in the single Decision Tree and recovers most of the lost performance, essentially tying kNN on accuracy/F1 and posting the second-highest AUC overall. |
| **Overall Winner for your dataset?** | **Logistic Regression** — it achieves the highest Accuracy (0.9825), AUC (0.9954), Precision, Recall, F1 (0.9861), and MCC (0.9623) among all five models, and is also the simplest/most interpretable model here, making it the clear winner for this dataset. |

## Repository Structure

```
project-folder/
│-- app.py                    # Streamlit application
│-- requirements.txt          # Python dependencies
│-- README.md                 # This file
│-- test_data.csv             # Held-out test split (20%, 114 rows) used by the app
│-- model/
│   │-- train_models.py       # Full training + evaluation pipeline
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- scaler.pkl            # Fitted StandardScaler
│   │-- meta.json             # Feature column order + class names
│   └-- metrics_comparison.csv
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # optional — regenerates models/metrics/test_data.csv
streamlit run app.py
```

## Live Streamlit App Link

> **`<< PASTE YOUR LIVE STREAMLIT COMMUNITY CLOUD APP LINK HERE >>`**

## Streamlit App Features

1. **CSV Upload** — upload `test_data.csv` (or any CSV with the same 30
   feature columns, optionally with a `diagnosis` column) via the file
   uploader.
2. **Model Selection Dropdown** — choose between all 5 trained models in
   the sidebar.
3. **Evaluation Metrics Display** — Accuracy, AUC, Precision, Recall, F1,
   and MCC computed live on the uploaded data.
4. **Confusion Matrix & Classification Report** — heatmap confusion matrix
   plus a full per-class classification report table.
5. A live, always-visible **comparison table across all 5 models** in the
   sidebar and at the bottom of the page.

## Academic Integrity Note

Model choices, hyperparameters, and app structure in this repository are
implemented independently. AI tools were used only for learning support
during development, not for direct copy-paste submission, per the
assignment's Anti-Plagiarism & Academic Integrity Guidelines.
