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


@pytest.mark.parametrize("kind", ["baseline", "advanced"])
def test_training_round_trip(kind):
    a = ["kedi süt içer", "köpek havlar", "kuş uçar", "balık yüzer"] * 10
    b = ["kedi süt içiyor", "araba hızlı", "kuş uçuyor", "ev büyük"] * 10
    y = np.array([1, 0, 1, 0] * 10)
    ckpt, hist = lstm.train_lstm(a, b, y, kind=kind, epochs=2, batch_size=8, device="cpu", verbose=False)
    assert len(hist["loss"]) >= 1
    prob = lstm.predict_texts(ckpt, a[:4], b[:4], device="cpu")
    assert prob.shape == (4,)
    assert ((prob >= 0) & (prob <= 1)).all()
