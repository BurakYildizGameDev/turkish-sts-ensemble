"""
Draw the README figures from the result CSVs. No training or GPU needed.

Inputs:  results/model_comparison.csv, results/legacy/model_comparison.csv,
         results/ablation.csv, results/bootstrap_ci.csv
Outputs: results/figures/{model_comparison,leakage,ablation,bootstrap_ci}.png

Usage:  python scripts/make_readme_figures.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = "results/figures"
COLORS = {"Ensemble": "#2e86de", "Pre-trained": "#10ac84", "Simple ensemble": "#8395a7",
          "Deep learning": "#ee5253"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{name}", dpi=200)
    plt.close(fig)
    print(f"  {OUT_DIR}/{name}")


def model_comparison():
    df = pd.read_csv("results/model_comparison.csv").sort_values("F1")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(df["Model"], df["F1"] * 100, color=[COLORS[t] for t in df["Type"]])
    for b, v in zip(bars, df["F1"] * 100):
        ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=9)
    lo = max(0, df["F1"].min() * 100 - 10)
    ax.set_xlim(lo, 100)
    ax.set_xlabel("F1 on the held-out test set (%)")
    ax.set_title("Model comparison — deduplicated data, anchor-grouped split")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for t, c in COLORS.items() if t in set(df["Type"])]
    ax.legend(handles, [t for t in COLORS if t in set(df["Type"])], loc="lower right", frameon=False)
    save(fig, "model_comparison.png")


def leakage():
    clean = pd.read_csv("results/model_comparison.csv").set_index("Model")["F1"]
    legacy = pd.read_csv("results/legacy/model_comparison.csv").set_index("Model")["F1"]
    models = [m for m in clean.sort_values(ascending=False).index if m in legacy.index]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = range(len(models))
    ax.barh([i + 0.2 for i in y], [legacy[m] * 100 for m in models], height=0.4, color="#c8d6e5",
            label="Original protocol (duplicates, random split)")
    ax.barh([i - 0.2 for i in y], [clean[m] * 100 for m in models], height=0.4, color="#2e86de",
            label="Fixed protocol (deduplicated, grouped split)")
    ax.set_yticks(list(y), models)
    ax.invert_yaxis()
    ax.set_xlim(max(0, min(clean.min(), legacy.min()) * 100 - 10), 100)
    ax.set_xlabel("Test F1 (%)")
    ax.set_title("Original vs. fixed evaluation protocol (different test sets)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    save(fig, "leakage.png")


def ablation():
    df = pd.read_csv("results/ablation.csv")
    df = df[df["Configuration"] != "All features"].sort_values("Delta_vs_all")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#ee5253" if d < -0.01 else "#8395a7" for d in df["Delta_vs_all"]]
    bars = ax.barh(df["Configuration"], df["Delta_vs_all"] * 100, color=colors)
    for b, d in zip(bars, df["Delta_vs_all"] * 100):
        ax.text(d - 0.2 if d < 0 else d + 0.2, b.get_y() + b.get_height() / 2, f"{d:+.1f}",
                va="center", ha="right" if d < 0 else "left", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    full = pd.read_csv("results/ablation.csv").iloc[0]["Test_F1"] * 100
    ax.set_xlabel("Change in test F1 vs. all six features (points)")
    ax.set_title(f"Feature ablation — XGBoost meta-model (all features: F1 = {full:.1f})")
    lo = df["Delta_vs_all"].min() * 100
    ax.set_xlim(lo * 1.25 if lo < 0 else -1, max(1.5, df["Delta_vs_all"].max() * 100 + 1))
    save(fig, "ablation.png")
