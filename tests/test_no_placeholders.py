"""TRD §14 — test_no_placeholders.py.

PRD §8 rule 1: a metric that was not computed is written as NOT_COMPUTED,
never as 0, 0.0, an empty cell, or null.

Per the M0 decision on §14, this checks that every metric field in
``results/tables/*.csv`` and ``results/raw/*.json`` is either a real number or
the literal string NOT_COMPUTED -- rather than grepping for the substring
"0.0", which would flag legitimately-computed zeros.
"""

import csv
import json

from src.data.loader import ROOT

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
RAW = RESULTS / "raw"

# Column names that hold a metric and must therefore be a number or NOT_COMPUTED.
METRIC_TOKENS = (
    "accuracy", "precision", "recall", "f1", "rate", "ci_low", "ci_high",
    "mean", "point", "tau", "confidence", "cost", "seconds", "ms",
)
FORBIDDEN = ("", "null", "None", "nan", "N/A", "TODO", "PLACEHOLDER", "-")


def _is_metric_column(name: str) -> bool:
    return any(tok in name.lower() for tok in METRIC_TOKENS)


def _check_metric_value(where: str, column: str, value) -> None:
    if value == "NOT_COMPUTED":
        return
    assert value is not None, f"{where}: metric {column!r} is null; use NOT_COMPUTED"
    text = str(value).strip()
    assert text not in FORBIDDEN, (
        f"{where}: metric {column!r} is {value!r}; an uncomputed metric must be "
        "the literal string NOT_COMPUTED"
    )
    try:
        float(text)
    except ValueError:
        raise AssertionError(
            f"{where}: metric {column!r} is {value!r}, which is neither a number "
            "nor NOT_COMPUTED"
        )


def test_no_placeholder_metrics_in_tables():
    for path in sorted(TABLES.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                for column, value in row.items():
                    if column and _is_metric_column(column):
                        _check_metric_value(f"{path.name} row {i}", column, value)


def _walk(obj, path, where):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                _walk(v, f"{path}.{k}", where)
            elif _is_metric_column(k):
                _check_metric_value(where, f"{path}.{k}", v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk(v, f"{path}[{i}]", where)


def test_no_placeholder_metrics_in_raw_json():
    for path in sorted(RAW.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            _walk(json.load(fh), "", path.name)


def test_results_directories_exist():
    assert TABLES.is_dir() and RAW.is_dir()


def test_checker_rejects_a_bare_zero_placeholder():
    """The check itself must fire on the thing it is meant to catch."""
    import pytest

    with pytest.raises(AssertionError):
        _check_metric_value("synthetic", "accuracy", "")
    with pytest.raises(AssertionError):
        _check_metric_value("synthetic", "accuracy", None)
    with pytest.raises(AssertionError):
        _check_metric_value("synthetic", "f1", "TODO")
    # A genuinely computed zero is legal.
    _check_metric_value("synthetic", "accuracy", 0.0)
    _check_metric_value("synthetic", "accuracy", "NOT_COMPUTED")
