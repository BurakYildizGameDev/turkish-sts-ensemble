# Turkish Paraphrase Detection — Hybrid Ensemble

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Task](https://img.shields.io/badge/task-paraphrase%20detection-orange)
![Language](https://img.shields.io/badge/language-Turkish-red)
[![tests](https://github.com/BurakYildizGameDev/turkish-sts-ensemble/actions/workflows/tests.yml/badge.svg)](https://github.com/BurakYildizGameDev/turkish-sts-ensemble/actions/workflows/tests.yml)

Decides whether two **Turkish** sentences mean the same thing. A multilingual transformer (MiniLM), a Siamese Bi-LSTM and four lexical similarity features are stacked into a tree ensemble (Random Forest / XGBoost / LightGBM).

On a deduplicated 62K-pair sample with an **anchor-grouped** train/test split, the ensemble reaches **F1 = 0.880** (LightGBM). Zero-shot MiniLM alone reaches **0.832**. The +4.7 point gain is significant: the 95% cluster-bootstrap interval is [+4.2, +5.3].

<p align="center">
  <img src="results/figures/model_comparison.png" width="780" alt="F1 comparison of all models">
</p>

---

## How it works

```
 sentence A ─┐
             ├─► MiniLM-L12 (multilingual)  ──► cosine similarity ─────┐
 sentence B ─┤                                                         │
             ├─► Siamese Bi-LSTM            ──► P(paraphrase) ─────────┤
             │                                                         ├─► RF / XGBoost / LightGBM ─► 0 / 1
             └─► lexical features           ──► TF-IDF cosine          │
                                                Jaccard                │
                                                token overlap          │
                                                Levenshtein distance ──┘
```

| Feature | What it captures |
|---|---|
| `minilm_sim` | Cosine similarity of [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) embeddings (zero-shot) |
| `lstm_prob` | Output of a Siamese Bi-LSTM with attention, trained from scratch (PyTorch) |
| `tfidf_sim` | Cosine similarity of TF-IDF vectors (5,000 features, fit on training text only) |
| `jaccard_sim` | Word-set intersection over union |
| `token_overlap` | Number of shared lower-cased tokens |
| `levenshtein_dist` | Character-level edit distance |
