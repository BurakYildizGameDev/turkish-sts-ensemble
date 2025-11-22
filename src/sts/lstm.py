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


class SiameseLSTM(nn.Module):
    def __init__(self, vocab_size, kind="advanced", embed_dim=128, hidden=64, dropout=0.3):
        super().__init__()
        self.kind = kind
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        bidir = kind == "advanced"
        self.rnn = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=bidir)
        dim = hidden * (2 if bidir else 1)
        if bidir:
            self.attn = nn.Linear(dim, 1)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * dim, 64), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(64, 1))
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * dim, 1))

    def encode(self, x):
        mask = x != PAD
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.embed(x), lengths, batch_first=True,
                                                   enforce_sorted=False)
        out, (h, _) = self.rnn(packed)
        if self.kind != "advanced":
            return h[-1]
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        scores = self.attn(torch.tanh(out)).squeeze(-1).masked_fill(~mask, -1e9)
        return (out * F.softmax(scores, dim=1).unsqueeze(-1)).sum(1)

    def forward(self, a, b):
        u, v = self.encode(a), self.encode(b)
        z = torch.cat([u, v, (u - v).abs(), u * v], 1) if self.kind == "advanced" else torch.cat([u, v], 1)
        return self.head(z).squeeze(-1)
