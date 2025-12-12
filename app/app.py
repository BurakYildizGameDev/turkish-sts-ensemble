"""
Streamlit demo of the trained ensemble.

Needs models/ensemble.joblib and models/lstm_advanced.pt: run `python src/train.py`,
or download them from the GitHub release into models/.

Usage (from the repository root):  streamlit run app/app.py
"""
import os
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sts.ensemble import Ensemble  # noqa: E402

MODEL_PATH = "models/ensemble.joblib"
RELEASES = "https://github.com/BurakYildizGameDev/turkish-sts-ensemble/releases"

FEATURE_INFO = {
    "minilm_sim": "MiniLM kosinüs benzerliği",
    "lstm_prob": "Bi-LSTM paraphrase olasılığı",
    "tfidf_sim": "TF-IDF kosinüs benzerliği",
    "jaccard_sim": "Jaccard (kelime kümesi) benzerliği",
    "token_overlap": "Ortak kelime sayısı",
    "levenshtein_dist": "Karakter düzenleme mesafesi",
}
EXAMPLES = {
    "Paraphrase": ("Derin öğrenme, yapay sinir ağlarını kullanan bir makine öğrenmesi yöntemidir.",
                   "Yapay sinir ağlarına dayanan makine öğrenmesi yöntemine derin öğrenme denir."),
    "Farklı anlam": ("Hava bugün çok sıcak.", "Kediler çok tatlı hayvanlardır."),
    "Ortak kelimeler, farklı anlam": ("Adam köpeği parkta gezdiriyor.", "Köpek parkta adamı kovalıyor."),
}

st.set_page_config(page_title="Türkçe Paraphrase Tespiti", page_icon="🔍", layout="wide")
