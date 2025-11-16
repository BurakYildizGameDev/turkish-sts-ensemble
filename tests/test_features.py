import numpy as np
import pytest

from sts import features


def test_jaccard():
    assert features.jaccard("a b c", "A b d") == pytest.approx(2 / 4)
    assert features.jaccard("", "") == 0.0
