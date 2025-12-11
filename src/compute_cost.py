"""
Load time, latency and GPU memory of each model.

Latency is the time to score one sentence pair, averaged over N_PAIRS pairs in
batches of BATCH (after a warm-up batch). For the ensemble it covers the whole
pipeline: MiniLM + Bi-LSTM + TF-IDF + lexical features + meta-model.

Needs models/ensemble.joblib and models/lstm_advanced.pt (train.py or the release).
Usage:  python src/compute_cost.py
"""
import gc
import os
import time

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from sts import features, lstm
from sts.ensemble import Ensemble

OUT_CSV = "results/compute_cost.csv"
N_PAIRS, BATCH = 1000, 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sample_pairs():
    path = "data/processed/semantic_dataset_binary.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, usecols=["text_a", "text_b"]).dropna().sample(n=N_PAIRS, random_state=42)
        return df["text_a"].tolist(), df["text_b"].tolist()
    return ([f"Bu {i}. örnek cümledir." for i in range(N_PAIRS)],
            [f"Bu da {i}. karşılaştırma cümlesi." for i in range(N_PAIRS)])


def measure(name, load_fn, score_fn, texts_a, texts_b):
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    obj = load_fn()
    load_s = time.perf_counter() - t0

    score_fn(obj, texts_a[:BATCH], texts_b[:BATCH])  # warm-up
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(0, len(texts_a), BATCH):
        score_fn(obj, texts_a[s:s + BATCH], texts_b[s:s + BATCH])
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / len(texts_a) * 1000
    vram = (torch.cuda.max_memory_allocated() - base) / 1024 ** 3 if DEVICE == "cuda" else float("nan")
    print(f"  {name:<22} load {load_s:6.2f} s   {ms:6.3f} ms/pair   peak VRAM {vram:.2f} GB")
    del obj
    return {"Model": name, "Load_s": load_s, "ms_per_pair": ms, "Peak_VRAM_GB": vram}


def encoder(model_id, prefix=""):
    return (lambda: SentenceTransformer(model_id, device=DEVICE),
            lambda m, a, b: features.embedding_similarity(m, a, b, batch_size=BATCH,
                                                          prefix_a=prefix, prefix_b=prefix))


def main():
    a, b = sample_pairs()
    gpu = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    print(f"device: {gpu}, {N_PAIRS} pairs, batch {BATCH}\n")
    rows = [
        measure("MiniLM-L12", *encoder(features.MINILM_NAME), a, b),
        measure("LaBSE", *encoder("sentence-transformers/LaBSE"), a, b),
        measure("E5-large", *encoder("intfloat/multilingual-e5-large", "query: "), a, b),
        measure("Bi-LSTM + attention", lambda: lstm.load("models/lstm_advanced.pt", DEVICE),
                lambda m, x, y: lstm.predict(m[0], m[1].encode(x, m[2]), m[1].encode(y, m[2]), DEVICE), a, b),
        measure("Full ensemble", lambda: Ensemble(device=DEVICE),
                lambda m, x, y: m.predict_proba(x, y), a, b),
    ]
    out = pd.DataFrame(rows).assign(Device=gpu)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
