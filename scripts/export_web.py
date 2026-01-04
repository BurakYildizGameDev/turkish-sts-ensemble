"""
Export the trained ensemble for the in-browser demo (web/).

Writes web/model/:
  config.json       feature order, LSTM shape, vocabulary, TF-IDF vocabulary + idf
  lstm.bin          Bi-LSTM weights as little-endian float32, offsets listed in config.json
  lightgbm.json     the LightGBM trees (booster.dump_model)

MiniLM is not exported: the page loads the ONNX version from the Hugging Face Hub
with transformers.js.

Usage:  python scripts/export_web.py
"""
import json
import os

import joblib
import numpy as np
import torch

OUT = "web/model"


def main():
    os.makedirs(OUT, exist_ok=True)
    bundle = joblib.load("models/ensemble.joblib")
    if bundle["name"] != "LightGBM":
        raise SystemExit(f"the web demo only supports a LightGBM meta-model, got {bundle['name']}")

    ckpt = torch.load(bundle["lstm"], map_location="cpu", weights_only=False)
    if ckpt["config"]["kind"] != "advanced":
        raise SystemExit("the web demo expects the advanced (Bi-LSTM + attention) model")

    tensors, offset, blobs = {}, 0, []
    for name, t in ckpt["state_dict"].items():
        arr = t.detach().cpu().numpy().astype("<f4")
        tensors[name] = {"offset": offset, "shape": list(arr.shape)}
        blobs.append(arr.tobytes())
        offset += arr.size
    with open(f"{OUT}/lstm.bin", "wb") as f:
        for b in blobs:
            f.write(b)

    tfidf = bundle["tfidf"]
    config = {
        "features": bundle["features"],
        "minilm": "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
        "lstm": {"vocab": ckpt["vocab"], "max_len": ckpt["max_len"], "tensors": tensors},
        "tfidf": {
            "vocabulary": {w: int(i) for w, i in tfidf.vocabulary_.items()},
            "idf": np.round(tfidf.idf_, 7).tolist(),
        },
    }
    with open(f"{OUT}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, separators=(",", ":"))

    booster = bundle["model"].booster_
    with open(f"{OUT}/lightgbm.json", "w") as f:
        json.dump(booster.dump_model(), f, separators=(",", ":"))

    for name in os.listdir(OUT):
        print(f"  {OUT}/{name}  {os.path.getsize(f'{OUT}/{name}') / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
