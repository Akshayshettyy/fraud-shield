"""
Module 3: trainer.py
Trains LGBM, XGBoost, and Logistic Regression; evaluates a weighted ensemble;
saves every artifact needed for reproducible inference.
"""

import os
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


# ── Weights for the final blend ───────────────────────────────────────────────
LGBM_W = 0.45
XGB_W  = 0.45
LR_W   = 0.10


def run_multi_model_pipeline(
    df: pd.DataFrame,
    artifacts_dir: str = "models/",
):
    """
    Trains all models, blends them into an ensemble, evaluates, saves artifacts.

    Returns
    -------
    results        : dict of {model_name: roc_auc}
    ensemble_probs : np.ndarray of blended probabilities on the val set
    y_val          : ground-truth labels for the val set
    lgbm_model     : fitted LGBMClassifier (for feature importance plots)
    """
    print("--- 3. Training Multi-Model Suite ---")
    os.makedirs(artifacts_dir, exist_ok=True)

    # ── Time-series split (no data leakage) ───────────────────────────────────
    df = df.sort_values("TransactionDT")
    drop_cols = ["isFraud", "TransactionID", "TransactionDT"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["isFraud"]

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    feature_names = X_train.columns.tolist()

    # ── Save feature list ─────────────────────────────────────────────────────
    with open(os.path.join(artifacts_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    print(f"Saved {len(feature_names)} feature names.")

    # ── Scaler (used ONLY by Logistic Regression) ─────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))

    # ── Model 1: Logistic Regression ──────────────────────────────────────────
    print("Fitting Logistic Regression...")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_val_scaled)[:, 1]
    joblib.dump(lr, os.path.join(artifacts_dir, "lr_model.pkl"))

    # ── Model 2: LightGBM ─────────────────────────────────────────────────────
    print("Fitting LightGBM...")
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=500, is_unbalance=True, verbose=-1
    )
    lgbm_model.fit(X_train, y_train)
    lgbm_probs = lgbm_model.predict_proba(X_val)[:, 1]
    joblib.dump(lgbm_model, os.path.join(artifacts_dir, "lgbm_model.pkl"))

    # ── Model 3: XGBoost ──────────────────────────────────────────────────────
    print("Fitting XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=500, scale_pos_weight=20, eval_metric="auc", verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
    joblib.dump(xgb_model, os.path.join(artifacts_dir, "xgb_model.pkl"))

    # ── Weighted ensemble ─────────────────────────────────────────────────────
    ensemble_probs = (lgbm_probs * LGBM_W) + (xgb_probs * XGB_W) + (lr_probs * LR_W)

    results = {
        "Logistic Regression": roc_auc_score(y_val, lr_probs),
        "LightGBM":            roc_auc_score(y_val, lgbm_probs),
        "XGBoost":             roc_auc_score(y_val, xgb_probs),
        "Ensemble":            roc_auc_score(y_val, ensemble_probs),
    }

    print("\n=== ROC-AUC Results ===")
    for name, score in results.items():
        print(f"  {name:<25}: {score:.4f}")

    return results, ensemble_probs, y_val, lgbm_model


# ── Diagnostics ───────────────────────────────────────────────────────────────

def plot_conclusion(
    y_val,
    probs: np.ndarray,
    lgbm_model,
    feature_names: list[str],
    threshold: float = 0.2,
):
    """Confusion matrix + LGBM feature importance."""
    print("--- 4. Generating Final Visuals ---")

    preds_binary = (probs > threshold).astype(int)
    cm   = confusion_matrix(y_val, preds_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues")
    plt.title(f"Ensemble Confusion Matrix (Threshold: {threshold})")
    plt.tight_layout()
    plt.show()

    if hasattr(lgbm_model, "feature_importances_"):
        feat_imp = (
            pd.Series(lgbm_model.feature_importances_, index=feature_names)
            .sort_values(ascending=False)
            .head(15)
        )
        plt.figure(figsize=(10, 6))
        feat_imp.plot(kind="barh", color="teal")
        plt.title("Top 15 Fraud Indicators (LGBM Feature Importance)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()