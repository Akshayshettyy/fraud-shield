"""
Module 2: feature_engineer.py
Cleans the dataframe, builds behavioural features, and fits + saves one
LabelEncoder per categorical column so inference can reuse them.
"""

import os
import json
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def engineer_features(
    df: pd.DataFrame,
    encoders_dir: str = "models/",
    fit: bool = True,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df          : raw merged dataframe
    encoders_dir: where to save / load fitted LabelEncoders
    fit         : True during training  → fits and saves encoders
                  False during inference → loads and applies saved encoders
    """
    print("--- 2. Feature Engineering & Cleaning ---")

    # ── Drop columns with > 90 % nulls ───────────────────────────────────────
    null_pct = df.isnull().sum() / len(df)
    df = df.drop(columns=null_pct[null_pct > 0.90].index)

    # ── UID behavioural aggregation ───────────────────────────────────────────
    df["uid"] = (
        df["card1"].astype(str) + "_"
        + df["card2"].astype(str) + "_"
        + df["addr1"].astype(str)
    )
    df["uid_mean_amt"] = df.groupby("uid")["TransactionAmt"].transform("mean")
    df["amt_to_mean"]  = df["TransactionAmt"] / df["uid_mean_amt"].replace(0, 1)

    # ── Fill remaining nulls ──────────────────────────────────────────────────
    df = df.fillna(-999)

    # ── Label Encoding ────────────────────────────────────────────────────────
    os.makedirs(encoders_dir, exist_ok=True)
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != "uid"]

    encoders: dict[str, LabelEncoder] = {}

    if fit:
        # Training path: fit a fresh encoder per column and save it
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        joblib.dump(encoders, os.path.join(encoders_dir, "label_encoders.pkl"))
        print(f"Fitted and saved {len(encoders)} LabelEncoders.")

    else:
        # Inference path: load saved encoders, handle unseen categories gracefully
        encoders = joblib.load(os.path.join(encoders_dir, "label_encoders.pkl"))
        for col in cat_cols:
            if col in encoders:
                le = encoders[col]
                known = set(le.classes_)
                # Map unseen values to a placeholder seen during training
                fallback = le.classes_[0]
                df[col] = df[col].astype(str).map(
                    lambda v, k=known, fb=fallback: v if v in k else fb
                )
                df[col] = le.transform(df[col])

    df.drop(columns=["uid"], inplace=True)
    return df