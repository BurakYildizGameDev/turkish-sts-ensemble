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

## Dataset

Three public Hugging Face datasets are merged into **620,089 sentence pairs**. Almost half of them are exact duplicates, mostly because every NLI anchor appears in several triplets. After removing duplicate pairs (in either order) and 62 pairs whose copies carry conflicting labels, **326,279 unique pairs** remain (54.9% positive).

| Source | Unique pairs | Label construction |
|---|---:|---|
| [mertcobanov/all-nli-triplets-turkish](https://huggingface.co/datasets/mertcobanov/all-nli-triplets-turkish) | 260,840 | Each triplet becomes (anchor, positive) = 1 and (anchor, negative) = 0 |
| [dogukanvzr/ml-paraphrase-tr](https://huggingface.co/datasets/dogukanvzr/ml-paraphrase-tr) | 59,772 | Binary labels as provided |
| [figenfikri/stsb_tr](https://huggingface.co/datasets/figenfikri/stsb_tr) | 5,667 | STS score ≥ 3.0 → 1, otherwise 0 |

`scripts/build_dataset.py` downloads the sources and rebuilds the merged CSVs.

<details>
<summary>Data exploration plots</summary>

| | |
|---|---|
| ![class balance](results/figures/data/Class_balance.png) | ![sentence lengths](results/figures/data/sentence_lengths.png) |
| ![length difference](results/figures/data/length_diff_boxplot.png) | ![word cloud](results/figures/data/wordcloud.png) |

</details>

## Evaluation protocol

The first version of this project evaluated on a random split of the raw rows. About 23% of the test pairs were also in the training set, and 63% of the test anchors had been seen in training. The stacking features were computed in-sample as well. The current protocol fixes all of this:

| | Original | Current |
|---|---|---|
| Data | raw rows, duplicates included | duplicates and label conflicts removed |
| Split | random 80/20 | 80/20 **grouped by anchor sentence** |
| Test pairs also in training | 23.2% | **0%** |
| Test anchors seen in training (in any role) | 63.1% | 6.1% |
| `lstm_prob` for training rows | predicted by a model trained on those rows | **out-of-fold** (5 anchor-grouped folds) |
| TF-IDF vocabulary | fit on train + test | fit on train only |
| Zero-shot thresholds | tuned on the evaluation data | tuned on the training set |
| Meta-model for the demo | — | chosen by cross-validation on the training set |

Sample size is 62,000 pairs in both cases: 49.6K for training and 12.4K for testing.
