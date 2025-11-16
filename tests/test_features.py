import numpy as np
import pytest

from sts import features


def test_jaccard():
    assert features.jaccard("a b c", "A b d") == pytest.approx(2 / 4)
    assert features.jaccard("", "") == 0.0


def test_token_overlap_is_case_insensitive():
    assert features.token_overlap("Kedi uyuyor", "kedi koşuyor") == 1
