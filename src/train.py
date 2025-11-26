"""
End-to-end training and evaluation.

  1. load the pairs and split them          (sts.data)
  2. train the two Siamese LSTMs            (sts.lstm)
  3. build the six pair features            (sts.features)
       - lstm_prob on the training set is out-of-fold (5 anchor-grouped folds)
       - TF-IDF is fit on training text only
  4. zero-shot baselines: MiniLM, LaBSE, E5-large, simple score average
       (decision thresholds are picked on the training set)
  5. meta-models: Random Forest, XGBoost, LightGBM
  6. write CSVs to --out and, for the clean protocol, model files to models/

Protocols:
  clean   (default) deduplicated data, anchor-grouped split, out-of-fold stacking
  legacy  the original setup: raw rows, random split, in-sample lstm_prob, TF-IDF fit
          on all text. Only used to measure how much the leakage inflated the scores.

Usage (from the repository root):
  python src/train.py
  python src/train.py --protocol legacy --out results/legacy
"""
import argparse
import json
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from sts import data, features, lstm

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def meta_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                     eval_metric="logloss", random_state=SEED, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=SEED, verbose=-1),
    }


def metrics(name, kind, y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    return {
        "Model": name, "Type": kind, "Threshold": round(float(threshold), 4),
        "Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred), "F1": f1_score(y, pred), "AUC": roc_auc_score(y, scores),
    }


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["clean", "legacy"], default="clean")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sample-size", type=int, default=62_000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--skip-extra-baselines", action="store_true", help="skip LaBSE and E5-large")
    args = ap.parse_args()
    legacy = args.protocol == "legacy"
    os.makedirs(args.out, exist_ok=True)
    start = time.time()
    print(f"protocol={args.protocol}  device={DEVICE}  out={args.out}")
