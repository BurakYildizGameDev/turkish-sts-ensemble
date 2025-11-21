"""
Siamese LSTM models in PyTorch.

  baseline  shared embedding + LSTM, last hidden states concatenated
  advanced  shared embedding + Bi-LSTM + additive attention pooling,
            combined as [u, v, |u - v|, u * v]

Both are trained from scratch on the training split only. A checkpoint stores the
weights, the vocabulary and the config, so it can be reloaded without the data.
"""
import copy
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, OOV = 0, 1
_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    return _TOKEN.findall(str(text).lower())


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, texts, max_size=30000):
        counts = Counter(tok for t in texts for tok in tokenize(t))
        return cls(["<pad>", "<oov>"] + [w for w, _ in counts.most_common(max_size - 2)])

    def encode(self, texts, max_len=32):
        out = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            ids = [self.stoi.get(tok, OOV) for tok in tokenize(t)][:max_len]
            out[i, :len(ids)] = ids
        return out

    def __len__(self):
        return len(self.itos)
