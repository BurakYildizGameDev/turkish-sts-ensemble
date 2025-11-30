"""
End-to-end training and evaluation.

  1. load the pairs and split them          (sts.data)
  2. train the two Siamese LSTMs            (sts.lstm)
  3. build the six pair features            (sts.features)
       - lstm_prob on the training set is out-of-fold (5 anchor-grouped folds)
       - TF-IDF is fit on training text only
  4. zero-shot baselines: MiniLM, LaBSE, E5-large, simple score average
       (decision thresholds are picked on the training set)
  5. meta-models: Random Forest, XGBoost, LightGBM
  6. write CSVs to --out and, for the clean protocol, model files to models/

Protocols:
  clean   (default) deduplicated data, anchor-grouped split, out-of-fold stacking
  legacy  the original setup: raw rows, random split, in-sample lstm_prob, TF-IDF fit
          on all text. Only used to measure how much the leakage inflated the scores.

Usage (from the repository root):
  python src/train.py
  python src/train.py --protocol legacy --out results/legacy
"""
import argparse
import json
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from sts import data, features, lstm

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def meta_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                     eval_metric="logloss", random_state=SEED, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=SEED, verbose=-1),
    }


def metrics(name, kind, y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    return {
        "Model": name, "Type": kind, "Threshold": round(float(threshold), 4),
        "Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred), "F1": f1_score(y, pred), "AUC": roc_auc_score(y, scores),
    }


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["clean", "legacy"], default="clean")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sample-size", type=int, default=62_000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--skip-extra-baselines", action="store_true", help="skip LaBSE and E5-large")
    args = ap.parse_args()
    legacy = args.protocol == "legacy"
    os.makedirs(args.out, exist_ok=True)
    start = time.time()
    print(f"protocol={args.protocol}  device={DEVICE}  out={args.out}")

    # ------------------------------------------------------------------ 1. data
    section("1/5  Data")
    pairs = data.load_pairs()
    train, test = data.make_split(pairs, "random" if legacy else "grouped",
                                  sample_size=args.sample_size, seed=SEED)
    leak = data.leakage_report(train, test)
    leak.update(protocol=args.protocol, train_pairs=len(train),
                train_positive_rate=float(train["label"].mean()), test_positive_rate=float(test["label"].mean()))
    print(json.dumps(leak, indent=2))
    with open(f"{args.out}/split_info.json", "w") as f:
        json.dump(leak, f, indent=2)
    y_tr, y_te = train["label"].values, test["label"].values

    # ------------------------------------------------------------------ 2. LSTMs
    section("2/5  Siamese LSTMs")
    rows, histories, ckpts = [], {}, {}
    for kind, name, epochs in [("baseline", "LSTM baseline", 5), ("advanced", "Bi-LSTM + attention", 7)]:
        print(f"  {name}")
        ckpt, hist = lstm.train_lstm(train["text_a"], train["text_b"], y_tr, kind=kind, epochs=epochs,
                                     groups=None if legacy else train["group"].values,
                                     seed=SEED, device=DEVICE)
        prob = lstm.predict_texts(ckpt, test["text_a"], test["text_b"], DEVICE)
        rows.append(metrics(name, "Deep learning", y_te, prob, 0.5))
        print(f"    test F1 = {rows[-1]['F1']:.4f}")
        histories[kind], ckpts[kind] = hist, ckpt
    with open(f"{args.out}/lstm_history.json", "w") as f:
        json.dump(histories, f, indent=2)

    # ------------------------------------------------------------------ 3. features
    section("3/5  Features")
    lstm_te = lstm.predict_texts(ckpts["advanced"], test["text_a"], test["text_b"], DEVICE)
    if legacy:
        lstm_tr = lstm.predict_texts(ckpts["advanced"], train["text_a"], train["text_b"], DEVICE)
    else:
        lstm_tr = np.zeros(len(train))
        for k, (fit_idx, oof_idx) in enumerate(data.group_folds(train, args.folds), 1):
            print(f"  out-of-fold lstm_prob, fold {k}/{args.folds}")
            part = train.iloc[fit_idx]
            ckpt, _ = lstm.train_lstm(part["text_a"], part["text_b"], part["label"].values, kind="advanced",
                                      groups=part["group"].values, seed=SEED + k, device=DEVICE, verbose=False)
            lstm_tr[oof_idx] = lstm.predict_texts(ckpt, train["text_a"].iloc[oof_idx],
                                                  train["text_b"].iloc[oof_idx], DEVICE)

    print("  MiniLM similarities")
    minilm = features.load_minilm(DEVICE)
    sim_tr = features.embedding_similarity(minilm, train["text_a"].tolist(), train["text_b"].tolist())
    sim_te = features.embedding_similarity(minilm, test["text_a"].tolist(), test["text_b"].tolist())
    del minilm

    print("  TF-IDF and lexical features")
    tfidf_text = train["text_a"].tolist() + train["text_b"].tolist()
    if legacy:
        tfidf_text += test["text_a"].tolist() + test["text_b"].tolist()
    tfidf = features.fit_tfidf(tfidf_text)

    def frame(df, sim, prob):
        cols = {"minilm_sim": sim, "lstm_prob": prob,
                "tfidf_sim": features.tfidf_similarity(tfidf, df["text_a"], df["text_b"])}
        cols.update(features.lexical_features(df["text_a"], df["text_b"]))
        return pd.DataFrame(cols)[features.FEATURES]

    X_tr, X_te = frame(train, sim_tr, lstm_tr), frame(test, sim_te, lstm_te)
    feat = pd.concat([
        X_tr.assign(label=y_tr, group=train["group"].values, source=train["source"].values, split="train"),
        X_te.assign(label=y_te, group=test["group"].values, source=test["source"].values, split="test"),
    ], ignore_index=True)
    feat.to_csv(f"{args.out}/features.csv", index=False)

    # ------------------------------------------------------------------ 4. zero-shot baselines
    section("4/5  Zero-shot baselines")
    t = features.best_threshold(y_tr, X_tr["minilm_sim"].values)
    rows.append(metrics("MiniLM-L12 (zero-shot)", "Pre-trained", y_te, X_te["minilm_sim"].values, t))
    avg = lambda X: X[["minilm_sim", "tfidf_sim", "jaccard_sim"]].mean(axis=1).values
    t = features.best_threshold(y_tr, avg(X_tr))
    rows.append(metrics("Score average (MiniLM, TF-IDF, Jaccard)", "Simple ensemble", y_te, avg(X_te), t))

    if not args.skip_extra_baselines:
        from sentence_transformers import SentenceTransformer
        calib = train.sample(n=min(5000, len(train)), random_state=SEED)
        for name, model_id, prefix in [("LaBSE (zero-shot)", "sentence-transformers/LaBSE", ""),
                                       ("E5-large (zero-shot)", "intfloat/multilingual-e5-large", "query: ")]:
            print(f"  {name}")
            model = SentenceTransformer(model_id, device=DEVICE)
            enc = lambda df: features.embedding_similarity(model, df["text_a"].tolist(), df["text_b"].tolist(),
                                                           batch_size=128, prefix_a=prefix, prefix_b=prefix)
            t = features.best_threshold(calib["label"].values, enc(calib))
            rows.append(metrics(name, "Pre-trained", y_te, enc(test), t))
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------ 5. meta-models
    section("5/5  Meta-models")
    importances, fitted, cv_f1 = {}, {}, {}
    folds = data.group_folds(train, args.folds)
    for name, model in meta_models().items():
        cv_f1[name] = float(cross_val_score(model, X_tr.values, y_tr, cv=folds, scoring="f1").mean())
        model.fit(X_tr.values, y_tr)
        prob = model.predict_proba(X_te.values)[:, 1]
        rows.append(metrics(name, "Ensemble", y_te, prob, 0.5))
        importances[name] = model.feature_importances_ / model.feature_importances_.sum()
        fitted[name] = model
        print(f"  {name:<14} train-CV F1 = {cv_f1[name]:.4f}   test F1 = {rows[-1]['F1']:.4f}")

    table = pd.DataFrame(rows).sort_values("F1", ascending=False)
    table.to_csv(f"{args.out}/model_comparison.csv", index=False)
    pd.DataFrame(importances, index=features.FEATURES).to_csv(f"{args.out}/feature_importance.csv")

    if not legacy:
        # The demo model is chosen by cross-validation on the training set, not by test score.
        best = max(cv_f1, key=cv_f1.get)
        os.makedirs("models", exist_ok=True)
        torch.save(ckpts["baseline"], "models/lstm_baseline.pt")
        torch.save(ckpts["advanced"], "models/lstm_advanced.pt")
        joblib.dump({"name": best, "model": fitted[best], "tfidf": tfidf, "features": features.FEATURES,
                     "minilm": features.MINILM_NAME, "lstm": "models/lstm_advanced.pt"},
                    "models/ensemble.joblib", compress=3)
        print(f"\n  saved models/ensemble.joblib ({best}), models/lstm_*.pt")

    section("Results (test set)")
    print(table[["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]].to_string(index=False, float_format="%.4f"))
    print(f"\ndone in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
