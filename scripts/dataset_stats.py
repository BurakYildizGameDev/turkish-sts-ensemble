"""
Per-source statistics of the merged dataset (the numbers in the README "Dataset" section).

Reads data/processed/semantic_dataset_binary.csv (built by scripts/build_dataset.py).
Usage:  python scripts/dataset_stats.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sts import data  # noqa: E402

OUT_CSV = "results/dataset_stats.csv"


def words(s):
    return s.str.split().str.len()


def summarize(name, raw, unique):
    return {
        "Source": name,
        "Raw_pairs": len(raw),
        "Unique_pairs": len(unique),
        "Duplicate_share": 1 - len(unique) / len(raw),
        "Positive_rate": unique["label"].mean(),
        "Unique_anchors": unique["text_a"].map(data.normalize).nunique(),
        "Mean_words_a": words(unique["text_a"]).mean(),
        "Mean_words_b": words(unique["text_b"]).mean(),
        "Mean_chars": (unique["text_a"].str.len() + unique["text_b"].str.len()).mean() / 2,
    }


def main():
    raw = data.load_pairs()
    unique = data.deduplicate(raw)
    rows = [summarize(src, raw[raw.source == src], unique[unique.source == src])
            for src in raw["source"].value_counts().index]
    rows.append(summarize("all", raw, unique))
    out = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False, float_format="%.3f"))
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
