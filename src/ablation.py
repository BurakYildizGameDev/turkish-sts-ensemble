"""
Feature ablation for the XGBoost meta-model.

Each configuration is trained on the training split and scored on the held-out
test split (the same split as train.py). The train-set cross-validation F1 over
anchor-grouped folds is reported as well, to show how stable each number is.

Reads results/features.csv written by train.py. No GPU needed.
Usage:  python src/ablation.py
"""
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, cross_val_score

from sts.features import FEATURES

FEATURE_CSV = "results/features.csv"
OUT_CSV = "results/ablation.csv"
LEXICAL = ["tfidf_sim", "jaccard_sim", "token_overlap", "levenshtein_dist"]


def model():
    return xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                             eval_metric="logloss", random_state=42, verbosity=0)
