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
