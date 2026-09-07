"""Guards on the generated result tables.

The motivating incident: a contrast was reported in prose as "B − D +0.0700"
when the correct statement is "D − B +0.0700". The CSV was right and the prose
was wrong, but a label inversion in the generator would be just as easy to make
and much harder to notice, because +0.07 and −0.07 are both plausible-looking
numbers. These tests make the label a checkable claim rather than a caption.
"""

import csv
import glob
import json
import re

import pytest

from src.data.loader import ROOT
from src.eval.metrics import accuracy

TABLES = ROOT / "results" / "tables"
RAW = ROOT / "results" / "raw"
COMPARISON = TABLES / "table1_comparison.csv"

LABEL_RE = re.compile(r"^(?P<b>\S+) minus (?P<a>\S+)$")


def _load_cell(cell: str):
    """Most recent raw JSONL for a cell, or None if it was never run."""
    hits = sorted(glob.glob(str(RAW / f"attribute_{cell}_dev_*.jsonl")))
    if not hits:
        return None
    with open(hits[-1], encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _rows():
    if not COMPARISON.exists():
        pytest.skip("table1_comparison.csv not generated yet")
    with COMPARISON.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_every_comparison_label_parses():
    for row in _rows():
        assert LABEL_RE.match(row["comparison"]), (
            f"comparison label {row['comparison']!r} is not of the form 'X minus Y'"
        )


def test_point_equals_the_difference_its_label_claims():
    """The core guard: 'X minus Y' must report accuracy(X) - accuracy(Y).

    This catches an inverted label, which a sign check alone cannot: a negative
    value is perfectly legitimate, so only recomputing the stated difference
    distinguishes a real negative from a flipped label.
    """
    checked = 0
    for row in _rows():
        m = LABEL_RE.match(row["comparison"])
        rb, ra = _load_cell(m.group("b")), _load_cell(m.group("a"))
        if rb is None or ra is None:
            continue
        expected = accuracy(rb) - accuracy(ra)
        reported = float(row["point"])
        assert reported == pytest.approx(expected, abs=1e-12), (
            f"{row['comparison']}: reports {reported:+.6f} but "
            f"accuracy({m.group('b')}) - accuracy({m.group('a')}) = {expected:+.6f}. "
            "If these differ by exactly a sign, the label is inverted."
        )
        checked += 1
    assert checked > 0, "no comparison rows could be verified against raw results"


def test_no_inverted_duplicate_rows():
    """A table must not contain both 'X minus Y' and 'Y minus X'."""
    seen = set()
    for row in _rows():
        m = LABEL_RE.match(row["comparison"])
        pair = (m.group("b"), m.group("a"))
        assert pair[::-1] not in seen, (
            f"table contains both {pair[0]} minus {pair[1]} and its inverse; "
            "one of them is almost certainly a mislabel"
        )
        seen.add(pair)


def test_ci_brackets_the_point_estimate():
    for row in _rows():
        lo, pt, hi = float(row["ci_low"]), float(row["point"]), float(row["ci_high"])
        assert lo <= hi, f"{row['comparison']}: ci_low {lo} > ci_high {hi}"
        assert lo <= pt <= hi, (
            f"{row['comparison']}: point {pt} lies outside its CI [{lo}, {hi}]"
        )


def test_excludes_zero_flag_agrees_with_the_ci():
    for row in _rows():
        lo, hi = float(row["ci_low"]), float(row["ci_high"])
        claimed = row["excludes_zero"].strip().lower() == "true"
        actual = lo > 0 or hi < 0
        assert claimed == actual, (
            f"{row['comparison']}: excludes_zero={row['excludes_zero']} but "
            f"CI is [{lo}, {hi}]"
        )


def test_ablation_table_cells_are_known():
    path = TABLES / "table1_ablation.csv"
    if not path.exists():
        pytest.skip("table1_ablation.csv not generated yet")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    known = {"A", "B", "C", "D", "Ap", "Cp", "position-only"}
    for row in rows:
        assert row["cell"] in known, f"unknown cell in ablation table: {row['cell']!r}"


def test_position_only_row_is_labelled_an_artifact():
    """The 1.0000 row must never appear without its warning."""
    path = TABLES / "table1_ablation.csv"
    if not path.exists():
        pytest.skip("table1_ablation.csv not generated yet")
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["cell"] == "position-only":
                assert "ARTIFACT" in row["label"].upper()
                assert "ARTIFACT" in row["notes"].upper()
                return
    pytest.skip("no position-only row present")
