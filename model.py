"""
Heart Disease Risk Predictor — Model Training
Trains a Random Forest classifier with full evaluation metrics.
Run this script to regenerate model.pkl, scaler.pkl, and metadata.pkl.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report
)

# ── 1. Load & validate data ────────────────────────────────────────────────────
df = pd.read_csv("health_data.csv")
assert "target" in df.columns, "Dataset must contain a 'target' column"

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Human-readable labels for display in the app
FEATURE_LABELS = {
    "age":      "Age",
    "sex":      "Sex",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol":     "Cholesterol",
    "fbs":      "Fasting Blood Sugar",
    "restecg":  "Resting ECG",
    "thalach":  "Max Heart Rate",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (Oldpeak)",
    "slope":    "ST Slope",
    "ca":       "Number of Major Vessels",
    "thal":     "Thalassemia",
}

print("── Dataset Overview ──────────────────────────────")
print(f"  Samples : {len(df)}")
print(f"  Features: {len(FEATURE_NAMES)}")
print(f"  Positive (disease): {df['target'].sum()}  |  Negative: {(df['target'] == 0).sum()}")

# ── 2. Prepare features & target ──────────────────────────────────────────────
df = df.fillna(df.median(numeric_only=True))
X = df[FEATURE_NAMES]
y = df["target"]   # 1 = heart disease present, 0 = absent

# ── 3. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# ── 5. Train Random Forest ────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)
f1   = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

cv_auc = cross_val_score(
    model, X_scaled, y,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="roc_auc", n_jobs=-1
)

print("\n── Test-Set Metrics ──────────────────────────────")
print(f"  Accuracy  : {acc:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"\n── 5-Fold CV AUC : {cv_auc.mean():.4f} ± {cv_auc.std():.4f} ──")
print("\n── Confusion Matrix ──────────────────────────────")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
print("\n── Classification Report ─────────────────────────")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# ── 7. Feature importance ─────────────────────────────────────────────────────
importances = model.feature_importances_
importance_dict = dict(zip(FEATURE_NAMES, importances.tolist()))

# ── 8. Build metadata bundle ──────────────────────────────────────────────────
metadata = {
    "feature_names":   FEATURE_NAMES,
    "feature_labels":  FEATURE_LABELS,
    "feature_importance": importance_dict,
    "metrics": {
        "accuracy":  round(acc,  4),
        "roc_auc":   round(auc,  4),
        "f1":        round(f1,   4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_auc_std":  round(float(cv_auc.std()),  4),
    },
    "confusion_matrix": cm.tolist(),
}

# ── 9. Save artifacts ─────────────────────────────────────────────────────────
pickle.dump(model,    open("model.pkl",    "wb"))
pickle.dump(scaler,   open("scaler.pkl",   "wb"))
pickle.dump(metadata, open("metadata.pkl", "wb"))

print("\n✅  Saved: model.pkl  |  scaler.pkl  |  metadata.pkl")
