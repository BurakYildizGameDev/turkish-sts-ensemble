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

## Results

### Model comparison (held-out test set, 12,383 pairs)

| Model | Type | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|
| **LightGBM** | Ensemble | **0.867** | 0.864 | 0.897 | **0.880** | **0.940** |
| XGBoost | Ensemble | 0.866 | 0.864 | 0.895 | 0.879 | 0.940 |
| Random Forest | Ensemble | 0.859 | 0.856 | 0.893 | 0.874 | 0.932 |
| MiniLM-L12 (zero-shot) | Pre-trained | 0.806 | 0.787 | 0.883 | 0.832 | 0.891 |
| E5-large (zero-shot) | Pre-trained | 0.793 | 0.782 | 0.862 | 0.820 | 0.875 |
| Bi-LSTM + attention | Deep learning | 0.794 | 0.794 | 0.842 | 0.817 | 0.879 |
| Score average (MiniLM, TF-IDF, Jaccard) | Simple ensemble | 0.767 | 0.734 | 0.897 | 0.808 | 0.841 |
| LaBSE (zero-shot) | Pre-trained | 0.665 | 0.639 | 0.888 | 0.743 | 0.763 |
| LSTM baseline | Deep learning | 0.698 | 0.701 | 0.778 | 0.737 | 0.752 |

The larger multilingual encoders do not beat MiniLM zero-shot, which is the only one of the three trained for paraphrase identification. A hand-weighted score average is worse than MiniLM alone. The gain comes from the learned meta-model.

### Did the leakage inflate the original numbers?

No. The same code run with the original protocol (`--protocol legacy`) gives Random Forest F1 = 0.872, which reproduces the 0.871 reported by the first version. The fixed protocol gives 0.874–0.880. The two runs use different test sets, and the deduplicated data has a slightly higher positive rate (54.6% vs 52.2%), which makes F1 a little easier. The comparison therefore shows that the leak did not inflate the scores. It does not show that the fixed protocol is harder.

<p align="center"><img src="results/figures/leakage.png" width="640" alt="Original vs fixed protocol"></p>

### Ablation

Each feature is removed in turn. The XGBoost meta-model is retrained on the training split and scored on the test split.

<p align="center"><img src="results/figures/ablation.png" width="640" alt="Ablation study"></p>

- Removing `minilm_sim` costs **−5.3 F1**, and removing `lstm_prob` costs **−3.3**. Both carry signal the other lacks.
- The four lexical features add about 1.2 points together (MiniLM + LSTM alone: 86.7), and 0.1–0.6 points each.
- Lexical features alone reach only 72.3.

Train-set cross-validation F1 for every configuration is in [`results/ablation.csv`](results/ablation.csv).

### Is the gain over MiniLM real?

Test pairs that share an anchor are not independent, so the bootstrap resamples whole anchor groups (1,000 resamples, 9,545 groups). Each model is compared with MiniLM on the same resamples.

<p align="center"><img src="results/figures/bootstrap_ci.png" width="640" alt="Paired bootstrap vs MiniLM"></p>

| Model | F1 | 95% CI | Δ vs MiniLM | 95% CI of Δ |
|---|---:|---|---:|---|
| LightGBM | 0.880 | [0.872, 0.886] | +4.7 | [+4.2, +5.3] |
| XGBoost | 0.879 | [0.872, 0.886] | +4.7 | [+4.1, +5.2] |
| Random Forest | 0.873 | [0.866, 0.880] | +4.1 | [+3.5, +4.7] |
| MiniLM-L12 (zero-shot) | 0.832 | [0.825, 0.840] | — | — |
| Bi-LSTM + attention | 0.819 | [0.812, 0.826] | −1.3 | [−2.1, −0.6] |
| Score average | 0.808 | [0.800, 0.815] | −2.5 | [−3.0, −2.0] |

<sub>In this table the Bi-LSTM threshold is tuned on the training set, so its F1 differs slightly from the 0.5-threshold value in the model comparison.</sub>

### Inference cost (RTX 5070 Ti Laptop, 1,000 pairs, batch 32)

| Model | Load time | Latency (ms / pair) | Peak VRAM |
|---|---:|---:|---:|
| MiniLM-L12 | 4.0 s | 0.82 | 0.48 GB |
| LaBSE | 4.5 s | 1.80 | 1.82 GB |
| E5-large | 4.4 s | 6.24 | 2.18 GB |
| Bi-LSTM + attention | 0.03 s | 0.17 | 0.03 GB |
| **Full ensemble** (all features + LightGBM) | 3.7 s | **1.49** | 0.49 GB |

The full ensemble is still about 4× faster than E5-large while scoring 6 F1 points higher.

## Limitations

- **Residual sentence overlap.** No test pair and no test anchor group appears in training. However, 6% of test anchors occur elsewhere in training as the *second* sentence of a pair, and 20% of test pairs share at least one sentence with training. A split over connected sentence components would remove this, but one component covers 42% of the data, so such a split is not practical here.
- **Domain skew.** About 80% of the pairs come from machine-translated NLI data (SNLI/MultiNLI captions). The model is biased toward short, descriptive sentences and has seen little formal or domain-specific text.
- **Word order.** Every feature except the Bi-LSTM is order-insensitive. Pairs such as *"Adam köpeği parkta gezdiriyor"* / *"Köpek parkta adamı kovalıyor"* get a high paraphrase probability (0.90).
- **Single seed and sample.** All numbers come from one 62K sample and one split (seed 42). The bootstrap intervals cover test-set sampling, not training variance.

## Project structure

```
├── app/app.py                  Streamlit demo (full ensemble)
├── scripts/
│   ├── build_dataset.py        download + merge + binarise the datasets
│   └── make_readme_figures.py  redraw the README figures from result CSVs
├── src/
│   ├── sts/                    shared package
│   │   ├── data.py             loading, deduplication, grouped split, leakage report
│   │   ├── features.py         the six pair features
│   │   ├── lstm.py             Siamese LSTM / Bi-LSTM + attention (PyTorch)
│   │   └── ensemble.py         inference with the trained ensemble
│   ├── train.py                full pipeline: split → LSTMs → features → baselines → meta-models
│   ├── bootstrap_ci.py         cluster bootstrap and paired comparison vs MiniLM
│   ├── ablation.py             feature ablation
│   ├── compute_cost.py         load time / latency / VRAM benchmark
│   └── analyze_data.py         EDA plots
├── tests/                      pytest suite (runs in CI)
├── results/                    CSVs, figures, results/legacy/ = original protocol
├── models/                     trained weights (see Releases)
└── data/                       created by build_dataset.py (git-ignored)
```

## Quick start

```bash
git clone https://github.com/BurakYildizGameDev/turkish-sts-ensemble.git
cd turkish-sts-ensemble
pip install -r requirements.txt      # use the CUDA build of torch for GPU training

# 1. build the dataset (~620K pairs, downloads from Hugging Face)
python scripts/build_dataset.py

# 2. train and evaluate everything (~6 min on an RTX 5070 Ti)
python src/train.py
python src/train.py --protocol legacy --out results/legacy    # optional: original protocol

# 3. analyses (CPU only, read results/features.csv)
python src/bootstrap_ci.py
python src/ablation.py
python src/compute_cost.py
python scripts/make_readme_figures.py

# tests
pytest

# demo
streamlit run app/app.py
```

All commands are run from the repository root. To use the demo without training, download `ensemble.joblib`, `lstm_advanced.pt` and `lstm_baseline.pt` from the [Releases](../../releases) page into `models/`.

---
