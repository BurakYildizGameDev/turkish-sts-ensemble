"""
Build everything the in-browser demo (web/) needs, so the page has no runtime
dependency on any CDN or on other people's model repositories.

Writes:
  web/model/config.json      feature order, LSTM shape + vocabulary, TF-IDF vocabulary + idf,
                             MiniLM threshold tuned on the training set
  web/model/lstm.bin         Bi-LSTM weights, little-endian float32 (offsets in config.json)
  web/model/lightgbm.json    the LightGBM trees (booster.dump_model)
  web/data/results.json      result tables shown on the page (from results/*.csv)
  web/models/<MiniLM id>/    8-bit ONNX MiniLM + tokenizer, copied from the Hugging Face Hub
  web/vendor/                transformers.js and its ONNX Runtime WebAssembly files

Downloads happen here, at build time; the published page only loads its own files.
Usage:  python scripts/export_web.py
"""
import json
import os
import urllib.request

import joblib
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

WEB = "web"
MINILM_ONNX = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
MINILM_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model_quantized.onnx"]
TRANSFORMERS_JS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/"
# name on the CDN -> name in web/vendor/ (.mjs is saved as .js: some static hosts serve .mjs as text/plain,
# which browsers refuse to run as a module)
VENDOR_FILES = {"transformers.min.js": "transformers.min.js",
                "ort-wasm-simd-threaded.jsep.mjs": "ort-wasm-simd-threaded.jsep.js",
                "ort-wasm-simd-threaded.jsep.wasm": "ort-wasm-simd-threaded.jsep.wasm"}


def export_model(bundle):
    out = f"{WEB}/model"
    os.makedirs(out, exist_ok=True)
    if bundle["name"] != "LightGBM":
        raise SystemExit(f"the web demo only supports a LightGBM meta-model, got {bundle['name']}")
    ckpt = torch.load(bundle["lstm"], map_location="cpu", weights_only=False)
    if ckpt["config"]["kind"] != "advanced":
        raise SystemExit("the web demo expects the advanced (Bi-LSTM + attention) model")

    tensors, offset = {}, 0
    with open(f"{out}/lstm.bin", "wb") as f:
        for name, t in ckpt["state_dict"].items():
            arr = t.detach().cpu().numpy().astype("<f4")
            tensors[name] = {"offset": offset, "shape": list(arr.shape)}
            f.write(arr.tobytes())
            offset += arr.size

    comparison = pd.read_csv("results/model_comparison.csv").set_index("Model")
    tfidf = bundle["tfidf"]
    config = {
        "features": bundle["features"],
        "minilm": MINILM_ONNX,
        "minilm_threshold": float(comparison.loc["MiniLM-L12 (zero-shot)", "Threshold"]),
        "lstm": {"vocab": ckpt["vocab"], "max_len": ckpt["max_len"], "tensors": tensors},
        "tfidf": {"vocabulary": {w: int(i) for w, i in tfidf.vocabulary_.items()},
                  "idf": np.round(tfidf.idf_, 7).tolist()},
    }
    with open(f"{out}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, separators=(",", ":"))
    with open(f"{out}/lightgbm.json", "w") as f:
        json.dump(bundle["model"].booster_.dump_model(), f, separators=(",", ":"))


def export_results():
    os.makedirs(f"{WEB}/data", exist_ok=True)
    table = lambda path, cols: pd.read_csv(path)[cols].round(6).to_dict(orient="records")
    split = json.load(open("results/split_info.json"))
    legacy = json.load(open("results/legacy/split_info.json"))
    results = {
        "models": table("results/model_comparison.csv", ["Model", "Type", "Accuracy", "Precision", "Recall", "F1", "AUC"]),
        "ablation": table("results/ablation.csv", ["Configuration", "Test_F1", "Delta_vs_all"]),
        "bootstrap": table("results/bootstrap_ci.csv", ["Model", "F1", "Diff_vs_MiniLM", "Diff_CI_low", "Diff_CI_high"]),
        "per_source": table("results/per_source.csv", ["Source", "Test_pairs", "MiniLM_F1", "Ensemble_F1"]),
        "datasets": table("results/dataset_stats.csv", ["Source", "Raw_pairs", "Unique_pairs", "Duplicate_share",
                                                        "Positive_rate", "Mean_words_a", "Mean_words_b"]),
        "legacy_models": table("results/legacy/model_comparison.csv", ["Model", "F1"]),
        "split": {k: split[k] for k in ["train_pairs", "test_pairs", "pair_seen_in_train", "anchor_seen_in_train"]},
        "legacy_split": {k: legacy[k] for k in ["pair_seen_in_train", "anchor_seen_in_train"]},
    }
    with open(f"{WEB}/data/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


def export_minilm():
    target = f"{WEB}/models/{MINILM_ONNX}"
    for name in MINILM_FILES:
        path = hf_hub_download(MINILM_ONNX, name)
        dst = f"{target}/{name}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(path, "rb") as src, open(dst, "wb") as out:
            out.write(src.read())


def export_vendor():
    os.makedirs(f"{WEB}/vendor", exist_ok=True)
    for name, local in VENDOR_FILES.items():
        dst = f"{WEB}/vendor/{local}"
        if not os.path.exists(dst):
            urllib.request.urlretrieve(TRANSFORMERS_JS + name, dst)


def main():
    export_model(joblib.load("models/ensemble.joblib"))
    export_results()
    export_minilm()
    export_vendor()
    for root, _, files in os.walk(WEB):
        for name in files:
            path = os.path.join(root, name)
            if os.path.getsize(path) > 100_000:
                print(f"  {path.replace(os.sep, '/')}  {os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
