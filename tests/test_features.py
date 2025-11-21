import numpy as np
import pytest

from sts import features


def test_jaccard():
    assert features.jaccard("a b c", "A b d") == pytest.approx(2 / 4)
    assert features.jaccard("", "") == 0.0


def test_token_overlap_is_case_insensitive():
    assert features.token_overlap("Kedi uyuyor", "kedi koşuyor") == 1


def test_levenshtein():
    assert features.levenshtein("kitap", "kitap") == 0
    assert features.levenshtein("kitap", "kilit") == 3


def test_tfidf_similarity_matches_cosine():
    tfidf = features.fit_tfidf(["kedi süt içer", "köpek kemik yer", "kedi balık yer"])
    sim = features.tfidf_similarity(tfidf, ["kedi süt içer", "kedi süt içer"],
                                    ["kedi süt içer", "köpek kemik yer"])
    assert sim[0] == pytest.approx(1.0)
    assert sim[1] == pytest.approx(0.0)


def test_best_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = features.best_threshold(y, scores)
    assert 0.3 < t <= 0.7
