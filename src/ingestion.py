"""
Module 1: data_loader.py
Loads, merges, and memory-optimises the raw transaction data.
"""

import pandas as pd
import numpy as np
import gc


def load_and_merge(data_dir: str = "data/") -> pd.DataFrame:
    print("--- 1. Loading & Merging Data ---")

    train_trans = pd.read_csv(f"{data_dir}train_transaction.csv")
    train_id    = pd.read_csv(f"{data_dir}train_identity.csv")

    df = pd.merge(train_trans, train_id, on="TransactionID", how="left")

    # Downcast numeric columns to save ~60 % memory
    for col in df.columns:
        if df[col].dtype == "object":
            continue
        col_min, col_max = df[col].min(), df[col].max()
        if str(df[col].dtype).startswith("int"):
            if col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

    del train_trans, train_id
    gc.collect()

    print(f"Dataset loaded. Shape: {df.shape}")
    return df