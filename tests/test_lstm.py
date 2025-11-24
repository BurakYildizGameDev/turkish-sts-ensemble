import numpy as np
import pytest

torch = pytest.importorskip("torch")
from sts import lstm  # noqa: E402


def test_vocab_encode_pads_and_marks_unknown_words():
    vocab = lstm.Vocab.build(["kedi süt içer", "kedi uyur"])
    x = vocab.encode(["kedi uçar"], max_len=4)
    assert x.shape == (1, 4)
    assert x[0, 0] == vocab.stoi["kedi"]
    assert x[0, 1] == lstm.OOV
    assert (x[0, 2:] == lstm.PAD).all()
