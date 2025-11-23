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


def _batches(n, batch_size, shuffle, rng=None):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, batch_size):
        yield idx[s:s + batch_size]


def predict(model, xa, xb, device, batch_size=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for b in _batches(len(xa), batch_size, shuffle=False):
            logits = model(torch.from_numpy(xa[b]).to(device), torch.from_numpy(xb[b]).to(device))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def train_lstm(texts_a, texts_b, y, kind="advanced", val_frac=0.1, groups=None, epochs=7,
               batch_size=512, lr=1e-3, patience=2, max_len=32, vocab_size=30000, seed=42,
               device=None, verbose=True):
    """
    Train a Siamese LSTM with early stopping on a held-out slice of the given data.
    If `groups` is given, the validation slice is grouped the same way as the test split.
    Returns (checkpoint dict, history dict).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    texts_a, texts_b, y = list(texts_a), list(texts_b), np.asarray(y, dtype=np.float32)

    n = len(y)
    if groups is not None:
        uniq = rng.permutation(np.unique(groups))
        val_groups = set(uniq[:max(1, int(len(uniq) * val_frac))])
        is_val = np.array([g in val_groups for g in groups])
    else:
        is_val = np.zeros(n, dtype=bool)
        is_val[rng.permutation(n)[:int(n * val_frac)]] = True
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]

    vocab = Vocab.build([texts_a[i] for i in tr] + [texts_b[i] for i in tr], vocab_size)
    xa, xb = vocab.encode(texts_a, max_len), vocab.encode(texts_b, max_len)

    config = {"kind": kind, "vocab_size": len(vocab)}
    model = SiameseLSTM(**config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
