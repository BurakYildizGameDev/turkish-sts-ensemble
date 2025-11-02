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
