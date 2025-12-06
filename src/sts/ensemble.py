"""Inference with the trained ensemble (used by the demo and the latency benchmark)."""
import joblib
import pandas as pd
import torch

from sts import features, lstm


class Ensemble:
    def __init__(self, path="models/ensemble.joblib", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        bundle = joblib.load(path)
        self.name = bundle["name"]
        self.meta = bundle["model"]
        self.tfidf = bundle["tfidf"]
        self.columns = bundle["features"]
        self.minilm = features.load_minilm(self.device)
        self.lstm, self.vocab, self.max_len = lstm.load(bundle["lstm"], self.device)
