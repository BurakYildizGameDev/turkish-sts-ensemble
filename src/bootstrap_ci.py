"""
Bootstrap confidence intervals on the held-out test set.

Test pairs that share an anchor sentence are not independent, so the bootstrap
resamples whole anchor groups (cluster bootstrap), not single pairs.

For every model it reports F1 with a 95% interval, and the paired difference
to zero-shot MiniLM (same resamples for both), which answers the real question:
does the learned ensemble beat the transformer it is built on?

Reads results/features.csv written by train.py. No GPU needed.
Usage:  python src/bootstrap_ci.py
"""
import numpy as np
import pandas as pd

from sts.features import FEATURES, best_threshold
from train import meta_models

FEATURE_CSV = "results/features.csv"
OUT_CSV = "results/bootstrap_ci.csv"
N_BOOT = 1000
SEED = 42


def group_counts(groups, y, pred):
    """Per-group TP / FP / FN counts, so each resample is a weighted sum instead of a refit."""
    df = pd.DataFrame({"g": groups, "tp": (pred == 1) & (y == 1), "fp": (pred == 1) & (y == 0),
                       "fn": (pred == 0) & (y == 1)})
    return df.groupby("g")[["tp", "fp", "fn"]].sum()
