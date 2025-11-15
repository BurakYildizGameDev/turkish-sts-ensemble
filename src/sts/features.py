"""
The six pair features fed to the meta-model.

  minilm_sim        cosine similarity of multilingual MiniLM embeddings (zero-shot)
  lstm_prob         P(paraphrase) from the Siamese Bi-LSTM (see sts.lstm)
  tfidf_sim         cosine similarity of TF-IDF vectors (vectoriser fit on training text only)
  jaccard_sim       word-set intersection over union
  token_overlap     number of shared lower-cased tokens
  levenshtein_dist  character edit distance
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURES = ["minilm_sim", "lstm_prob", "tfidf_sim", "jaccard_sim", "token_overlap", "levenshtein_dist"]
MINILM_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

try:
    from Levenshtein import distance as _levenshtein
except ImportError:  # pure-Python fallback
    def _levenshtein(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def _tokens(text):
    return set(str(text).lower().split())


def jaccard(a, b):
    sa, sb = _tokens(a), _tokens(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def token_overlap(a, b):
    return len(_tokens(a) & _tokens(b))


def levenshtein(a, b):
    return _levenshtein(str(a), str(b))


def lexical_features(texts_a, texts_b):
    return {
        "jaccard_sim": np.array([jaccard(a, b) for a, b in zip(texts_a, texts_b)]),
        "token_overlap": np.array([token_overlap(a, b) for a, b in zip(texts_a, texts_b)]),
        "levenshtein_dist": np.array([levenshtein(a, b) for a, b in zip(texts_a, texts_b)]),
    }


def fit_tfidf(texts, max_features=5000):
    return TfidfVectorizer(max_features=max_features).fit(texts)


def tfidf_similarity(tfidf, texts_a, texts_b):
    # rows are L2-normalised, so the row-wise dot product is the cosine similarity
    va, vb = tfidf.transform(texts_a), tfidf.transform(texts_b)
    return np.asarray(va.multiply(vb).sum(axis=1)).ravel()
