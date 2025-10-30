"""
Build the combined Turkish sentence-pair dataset from Hugging Face sources.

Outputs (in data/processed/):
  semantic_dataset.csv         STS-B rows keep their 0-5 score (label_type="score")
  semantic_dataset_binary.csv  every row binarised (STS-B score >= 3.0 -> 1)

Sources:
  dogukanvzr/ml-paraphrase-tr              60,000 labelled paraphrase pairs
  mertcobanov/all-nli-triplets-turkish     277,386 triplets -> 2 pairs each
                                           (anchor, positive)=1 / (anchor, negative)=0
  figenfikri/stsb_tr                       5,749 train pairs with 0-5 scores

Usage:  python scripts/build_dataset.py
"""
import os
import uuid

import pandas as pd
from datasets import load_dataset

OUT_DIR = "data/processed"
STS_THRESHOLD = 3.0


def pair(a, b, label_type, label_value, source, score=None):
    return {"id": str(uuid.uuid4()), "text_a": a, "text_b": b,
            "label_type": label_type, "label_value": label_value,
            "source": source, "original_score": score}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    print("ml-paraphrase-tr ...")
    ml = load_dataset("dogukanvzr/ml-paraphrase-tr", split="train")
    for r in ml:
        rows.append(pair(r["sentence1"], r["sentence2"], "binary", int(r["label"]), "ml_paraphrase"))

    print("all-nli-triplets-turkish ...")
    nli = load_dataset("mertcobanov/all-nli-triplets-turkish", split="train")
    for r in nli:
        a = r["anchor_translated"]
        rows.append(pair(a, r["positive_translated"], "binary", 1, "nli_triplet"))
        rows.append(pair(a, r["negative_translated"], "binary", 0, "nli_triplet"))

    print("stsb_tr ...")
    sts = load_dataset("figenfikri/stsb_tr", split="train")
    for r in sts:
        rows.append(pair(r["sentence1"], r["sentence2"], "score", float(r["score"]), "stsb_tr"))
