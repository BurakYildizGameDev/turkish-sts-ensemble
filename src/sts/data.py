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


def anchor_groups(df):
    """Integer group id per row: pairs sharing the same normalised anchor sentence share a group."""
    return pd.factorize(df["text_a"].map(normalize))[0]


def make_split(df, protocol="grouped", sample_size=62_000, test_size=0.2, seed=42):
    """
    Return (train_df, test_df), each with a `group` column.

    sample_size is taken after deduplication for the grouped protocol and from the
    raw rows for the random protocol, so both runs train on the same number of pairs.
    """
    if protocol == "grouped":
        df = deduplicate(df)
    elif protocol != "random":
        raise ValueError(f"unknown split protocol: {protocol}")

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    df = df.assign(group=anchor_groups(df))

    if protocol == "grouped":
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_idx, te_idx = next(splitter.split(df, groups=df["group"]))
    else:
        tr_idx, te_idx = train_test_split(
            np.arange(len(df)), test_size=test_size, random_state=seed, stratify=df["label"]
        )
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)


def group_folds(train_df, n_folds=5):
    """Fold indices over the training set that never split an anchor group."""
    return list(GroupKFold(n_splits=n_folds).split(train_df, groups=train_df["group"]))
