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
