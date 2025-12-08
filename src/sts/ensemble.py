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

    def features(self, texts_a, texts_b):
        texts_a, texts_b = list(texts_a), list(texts_b)
        cols = {
            "minilm_sim": features.embedding_similarity(self.minilm, texts_a, texts_b),
            "lstm_prob": lstm.predict(self.lstm, self.vocab.encode(texts_a, self.max_len),
                                      self.vocab.encode(texts_b, self.max_len), self.device),
            "tfidf_sim": features.tfidf_similarity(self.tfidf, texts_a, texts_b),
        }
        cols.update(features.lexical_features(texts_a, texts_b))
        return pd.DataFrame(cols)[self.columns]

    def predict_proba(self, texts_a, texts_b):
        X = self.features(texts_a, texts_b)
        return self.meta.predict_proba(X.values)[:, 1], X
