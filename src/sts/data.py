"""
Loading, deduplication and leakage-free splitting of the sentence-pair dataset.

Two split protocols are supported:

  grouped  (default)  exact duplicate pairs are removed, pairs with conflicting
                      labels are dropped, and the split is grouped by the anchor
                      sentence (text_a), so no anchor appears on both sides.
  random              the original protocol: raw rows, stratified random split.
                      Kept only to measure how much the leakage inflated scores.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

DATA_PATH = "data/processed/semantic_dataset_binary.csv"


def normalize(text):
    return " ".join(str(text).lower().split())


def load_pairs(path=DATA_PATH):
    df = pd.read_csv(path, usecols=["text_a", "text_b", "label_value", "source"])
    df = df.dropna(subset=["text_a", "text_b", "label_value"])
    df["text_a"] = df["text_a"].astype(str)
    df["text_b"] = df["text_b"].astype(str)
    df["label"] = df["label_value"].astype(int)
    return df.drop(columns="label_value").reset_index(drop=True)


def pair_key(a, b):
    """Order-independent key of a normalised sentence pair."""
    a, b = normalize(a), normalize(b)
    return a + "\t" + b if a <= b else b + "\t" + a


def deduplicate(df):
    """Drop exact duplicate pairs (in either order) and pairs whose copies disagree on the label."""
    keys = [pair_key(a, b) for a, b in zip(df["text_a"], df["text_b"])]
    df = df.assign(pair_key=keys)
    n_labels = df.groupby("pair_key")["label"].transform("nunique")
    df = df[n_labels == 1]
    return df.drop_duplicates("pair_key").drop(columns="pair_key").reset_index(drop=True)
