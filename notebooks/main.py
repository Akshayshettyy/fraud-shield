"""
Module 5: main.py
Orchestrates the full training pipeline end-to-end.

Usage:
    python main.py                      # train on data/
    python main.py --data_dir raw/      # custom data location
    python main.py --artifacts_dir out/ # custom model output location
"""



import argparse
import json

import sys
sys.path.append('../src')

from ingestion import load_and_merge
from features import engineer_features
from model import run_multi_model_pipeline, plot_conclusion

ARTIFACTS_DIR = "models/"


def main(data_dir: str, artifacts_dir: str, plot: bool = True):
    # 1. Load & merge
    df = load_and_merge(data_dir=data_dir)

    # 2. Feature engineering (fit=True → fits + saves LabelEncoders)
    df = engineer_features(df, encoders_dir=artifacts_dir, fit=True)

    # 3. Train all models + save all artifacts
    results, ensemble_probs, y_val, lgbm_model = run_multi_model_pipeline(
        df, artifacts_dir=artifacts_dir
    )

    # 4. Optional diagnostics
    if plot:
        with open(f"{artifacts_dir}feature_names.json") as f:
            feature_names = json.load(f)
        plot_conclusion(y_val, ensemble_probs, lgbm_model, feature_names)

    print("\n Pipeline complete. Artifacts saved to:", artifacts_dir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Shield Training Pipeline")
    parser.add_argument("--data_dir",      default="data/",   help="Directory with raw CSVs")
    parser.add_argument("--artifacts_dir", default="models/", help="Where to save model artifacts")
    parser.add_argument("--no_plot",       action="store_true", help="Skip diagnostic plots")
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        plot=not args.no_plot,
    )