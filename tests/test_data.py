import pandas as pd
import pytest

from sts import data


def toy_pairs():
    rows = []
    for i in range(40):
        anchor = f"anchor sentence {i}"
        rows += [(anchor, f"positive {i}", 1, "nli"), (anchor, f"negative {i}", 0, "nli")]
    rows += [("anchor sentence 0", "positive 0", 1, "nli"),    # exact duplicate
             ("POSITIVE  0", "anchor sentence 0", 1, "nli"),   # same pair, other order / case
             ("conflict a", "conflict b", 1, "ml"),
             ("conflict a", "conflict b", 0, "ml")]            # conflicting labels
    return pd.DataFrame(rows, columns=["text_a", "text_b", "label", "source"])


def test_deduplicate_removes_duplicates_and_conflicts():
    out = data.deduplicate(toy_pairs())
    assert len(out) == 80
    assert not (out["text_a"] == "conflict a").any()


def test_grouped_split_has_no_anchor_overlap():
    train, test = data.make_split(toy_pairs(), "grouped", sample_size=None, test_size=0.25)
    anchors_tr = set(train["text_a"].map(data.normalize))
    anchors_te = set(test["text_a"].map(data.normalize))
    assert anchors_tr.isdisjoint(anchors_te)
    report = data.leakage_report(train, test)
    assert report["pair_seen_in_train"] == 0
    assert report["anchor_seen_in_train"] == 0


def test_random_split_leaks():
    # the legacy protocol keeps duplicates, so the same anchor lands on both sides
    df = pd.concat([toy_pairs()] * 3, ignore_index=True)
    train, test = data.make_split(df, "random", sample_size=None, test_size=0.3)
    assert data.leakage_report(train, test)["pair_seen_in_train"] > 0.5
