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


def main():
    df = pd.read_csv(FEATURE_CSV)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    y_tr, y_te = tr["label"].values, te["label"].values

    preds = {}
    for name, col in [("MiniLM-L12 (zero-shot)", "minilm_sim"), ("Bi-LSTM + attention", "lstm_prob")]:
        preds[name] = (te[col].values >= best_threshold(y_tr, tr[col].values)).astype(int)
    avg = lambda d: d[["minilm_sim", "tfidf_sim", "jaccard_sim"]].mean(axis=1).values
    preds["Score average"] = (avg(te) >= best_threshold(y_tr, avg(tr))).astype(int)
    for name, model in meta_models().items():
        model.fit(tr[FEATURES].values, y_tr)
        preds[name] = model.predict(te[FEATURES].values)

    counts = {m: group_counts(te["group"].values, y_te, p) for m, p in preds.items()}
    n_groups = len(next(iter(counts.values())))
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=N_BOOT)  # (N_BOOT, groups)

    def f1_samples(c):
        tp, fp, fn = (weights @ c[k].values for k in ("tp", "fp", "fn"))
        return 2 * tp / (2 * tp + fp + fn)

    samples = {m: f1_samples(c) for m, c in counts.items()}
    ref = samples["MiniLM-L12 (zero-shot)"]
    rows = []
    for m, s in samples.items():
        c = counts[m].sum()
        diff = s - ref
        rows.append({
            "Model": m,
            "F1": 2 * c.tp / (2 * c.tp + c.fp + c.fn),
            "F1_CI_low": np.percentile(s, 2.5), "F1_CI_high": np.percentile(s, 97.5),
            "Diff_vs_MiniLM": diff.mean(),
            "Diff_CI_low": np.percentile(diff, 2.5), "Diff_CI_high": np.percentile(diff, 97.5),
            "P(diff<=0)": float(np.mean(diff <= 0)) if m != "MiniLM-L12 (zero-shot)" else np.nan,
        })

    out = pd.DataFrame(rows).sort_values("F1", ascending=False)
    out.to_csv(OUT_CSV, index=False)
    print(f"{len(te)} test pairs in {n_groups} anchor groups, {N_BOOT} cluster-bootstrap resamples\n")
    print(out.to_string(index=False, float_format="%.4f"))
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
