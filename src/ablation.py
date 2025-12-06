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


def main():
    df = pd.read_csv(FEATURE_CSV)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    folds = list(GroupKFold(n_splits=5).split(tr, groups=tr["group"]))

    configs = [("All features", FEATURES)]
    configs += [(f"without {f}", [x for x in FEATURES if x != f]) for f in FEATURES]
    configs += [("minilm_sim only", ["minilm_sim"]),
                ("minilm_sim + lstm_prob", ["minilm_sim", "lstm_prob"]),
                ("lexical only", LEXICAL)]

    rows = []
    for name, cols in configs:
        cv = cross_val_score(model(), tr[cols].values, tr["label"].values, cv=folds, scoring="f1")
        m = model().fit(tr[cols].values, tr["label"].values)
        f1 = f1_score(te["label"].values, m.predict(te[cols].values))
        rows.append({"Configuration": name, "Features": ", ".join(cols), "N": len(cols),
                     "Test_F1": f1, "CV_F1_mean": cv.mean(), "CV_F1_std": cv.std()})
        print(f"  {name:<26} test F1 = {f1:.4f}   train-CV F1 = {cv.mean():.4f} ± {cv.std():.4f}")

    out = pd.DataFrame(rows)
    out["Delta_vs_all"] = out["Test_F1"] - out.loc[0, "Test_F1"]
    out.to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
