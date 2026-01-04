"""
Test-set scores broken down by data source.

The three sources differ a lot (translated NLI captions, ML textbook sentences,
translated STS-B), so a single F1 hides where the model is strong or weak.
MiniLM uses the threshold tuned on the whole training set; the meta-model is
the LightGBM configuration from train.py.

Reads results/features.csv written by train.py. No GPU needed.
Usage:  python src/per_source.py
"""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sts.features import FEATURES, best_threshold
from train import meta_models

FEATURE_CSV = "results/features.csv"
OUT_CSV = "results/per_source.csv"


def main():
    df = pd.read_csv(FEATURE_CSV)
    tr, te = df[df.split == "train"], df[df.split == "test"].copy()

    t = best_threshold(tr["label"].values, tr["minilm_sim"].values)
    te["minilm_pred"] = (te["minilm_sim"] >= t).astype(int)
    meta = meta_models()["LightGBM"].fit(tr[FEATURES].values, tr["label"].values)
    te["ens_prob"] = meta.predict_proba(te[FEATURES].values)[:, 1]
    te["ens_pred"] = (te["ens_prob"] >= 0.5).astype(int)

    rows = []
    for src, g in list(te.groupby("source")) + [("all", te)]:
        y = g["label"].values
        rows.append({
            "Source": src, "Test_pairs": len(g), "Positive_rate": y.mean(),
            "MiniLM_F1": f1_score(y, g["minilm_pred"]), "MiniLM_Acc": accuracy_score(y, g["minilm_pred"]),
            "Ensemble_F1": f1_score(y, g["ens_pred"]), "Ensemble_Acc": accuracy_score(y, g["ens_pred"]),
            "Ensemble_AUC": roc_auc_score(y, g["ens_prob"]),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False, float_format="%.4f"))
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
