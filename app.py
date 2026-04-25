"""
Seller Fraud Shield API
Run from project root:
    $env:ARTIFACTS_DIR="notebooks/models/"; uvicorn app:app --reload
"""

import json
import os
import sys
import traceback

# ── Make src/ importable ──────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Artifact paths ────────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "notebooks/models/")

# ── Ensemble weights — must match training exactly ────────────────────────────
LGBM_W = 0.45
XGB_W  = 0.45
LR_W   = 0.10

# ── Globals ───────────────────────────────────────────────────────────────────
LGBM_MODEL     = None
XGB_MODEL      = None
LR_MODEL       = None
SCALER         = None
MODEL_FEATURES = []

app = FastAPI(title="Seller Fraud Shield API", version="2.0")


@app.on_event("startup")
def load_artifacts():
    global LGBM_MODEL, XGB_MODEL, LR_MODEL, SCALER, MODEL_FEATURES

    files = ["lgbm_model.pkl", "xgb_model.pkl", "lr_model.pkl",
             "scaler.pkl", "feature_names.json"]
    missing = [f for f in files
               if not os.path.exists(os.path.join(ARTIFACTS_DIR, f))]
    if missing:
        raise RuntimeError(
            f"Missing artifacts in '{ARTIFACTS_DIR}': {missing}\n"
            "Run: cd notebooks && python main.py --data_dir ../data/ --artifacts_dir models/"
        )

    LGBM_MODEL = joblib.load(os.path.join(ARTIFACTS_DIR, "lgbm_model.pkl"))
    XGB_MODEL  = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"))
    LR_MODEL   = joblib.load(os.path.join(ARTIFACTS_DIR, "lr_model.pkl"))
    SCALER     = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

    with open(os.path.join(ARTIFACTS_DIR, "feature_names.json")) as f:
        MODEL_FEATURES = json.load(f)

    print(f"✅ All artifacts loaded from '{ARTIFACTS_DIR}'. "
          f"Expecting {len(MODEL_FEATURES)} features.")


@app.get("/")
def health_check():
    return {
        "status":            "active",
        "features_expected": len(MODEL_FEATURES),
        "ensemble_weights":  {"lgbm": LGBM_W, "xgb": XGB_W, "lr": LR_W},
    }


@app.post("/predict")
async def predict(transaction: dict):
    try:
        # Build DataFrame aligned to training feature order
        df = pd.DataFrame([transaction])
        df = df.reindex(columns=MODEL_FEATURES, fill_value=-999)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(-999)
        df = df.astype(float)

        # LGBM and XGBoost use RAW features (no scaling)
        lgbm_prob = LGBM_MODEL.predict_proba(df)[:, 1][0]
        xgb_prob  = XGB_MODEL.predict_proba(df.values)[:, 1][0]

        # Logistic Regression uses SCALED features
        lr_prob = LR_MODEL.predict_proba(SCALER.transform(df))[:, 1][0]

        # Weighted ensemble
        prob = (lgbm_prob * LGBM_W) + (xgb_prob * XGB_W) + (lr_prob * LR_W)

        if prob > 0.8:
            recommendation = "BLOCK"
        elif prob > 0.2:
            recommendation = "MANUAL_REVIEW"
        else:
            recommendation = "APPROVE"

        return {
            "transaction_id":        transaction.get("TransactionID", "unknown"),
            "fraud_probability":     round(float(prob), 4),
            "model_probabilities":   {
                "lgbm": round(float(lgbm_prob), 4),
                "xgb":  round(float(xgb_prob),  4),
                "lr":   round(float(lr_prob),    4),
            },
            "action_recommendation": recommendation,
            "features_used":         len(MODEL_FEATURES),
        }

    except Exception:
        print(traceback.format_exc())
        raise HTTPException(status_code=400,
                            detail=traceback.format_exc())


@app.post("/predict/batch")
async def predict_batch(transactions: list[dict]):
    try:
        df = pd.DataFrame(transactions)
        df = df.reindex(columns=MODEL_FEATURES, fill_value=-999)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(-999)
        df = df.astype(float)

        lgbm_probs = LGBM_MODEL.predict_proba(df)[:, 1]
        xgb_probs  = XGB_MODEL.predict_proba(df.values)[:, 1]
        lr_probs   = LR_MODEL.predict_proba(SCALER.transform(df))[:, 1]

        ensemble = (lgbm_probs * LGBM_W) + (xgb_probs * XGB_W) + (lr_probs * LR_W)

        return [
            {
                "index":                 i,
                "transaction_id":        transactions[i].get("TransactionID", "unknown"),
                "fraud_probability":     round(float(p), 4),
                "action_recommendation": "BLOCK" if p > 0.8
                                         else "MANUAL_REVIEW" if p > 0.2
                                         else "APPROVE",
            }
            for i, p in enumerate(ensemble)
        ]

    except Exception:
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=traceback.format_exc())