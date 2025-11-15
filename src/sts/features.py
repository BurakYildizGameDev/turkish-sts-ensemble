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
