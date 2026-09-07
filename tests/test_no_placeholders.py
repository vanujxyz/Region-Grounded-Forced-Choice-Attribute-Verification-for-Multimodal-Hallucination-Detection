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


# Two distinct sentinels, deliberately not collapsed into one:
#   NOT_COMPUTED   -- this metric could have been computed and was not
#                     (empty subgroup, undefined denominator, stage not run)
#   NOT_APPLICABLE -- this quantity does not exist for this configuration.
#                     Cells B and D are forced choice and have no tau at all;
#                     that is the free-parameter asymmetry recorded in D-012,
#                     and flattening it to NOT_COMPUTED would hide it.
# Neither may ever stand in for a number that was actually produced.
LEGAL_SENTINELS = ("NOT_COMPUTED", "NOT_APPLICABLE")


def _check_metric_value(where: str, column: str, value) -> None:
    if value in LEGAL_SENTINELS:
        return
    assert value is not None, f"{where}: metric {column!r} is null; use NOT_COMPUTED"
    text = str(value).strip()
    assert text not in FORBIDDEN, (
        f"{where}: metric {column!r} is {value!r}; an uncomputed metric must be "
        f"one of {LEGAL_SENTINELS}"
    )
    try:
        float(text)
    except ValueError:
        raise AssertionError(
            f"{where}: metric {column!r} is {value!r}, which is neither a number "
            f"nor one of {LEGAL_SENTINELS}"
        )


def test_no_placeholder_metrics_in_tables():
    for path in sorted(TABLES.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                for column, value in row.items():
                    if column and _is_metric_column(column):
                        _check_metric_value(f"{path.name} row {i}", column, value)


def _walk(obj, path, where, skip_prefix=None):
    if skip_prefix and path.startswith(skip_prefix):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                _walk(v, f"{path}.{k}", where, skip_prefix)
            elif _is_metric_column(k):
                if skip_prefix and f"{path}.{k}".startswith(skip_prefix):
                    continue
                _check_metric_value(where, f"{path}.{k}", v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk(v, f"{path}[{i}]", where, skip_prefix)


# A manifest echoes the resolved config verbatim as a record of the run's INPUTS.
# A null there means "this parameter has not been fitted yet" (e.g. relation.tau
# before the M4 relation module exists), which is a different thing from "a metric
# was not computed". The echo is excluded from the metric check; every RESULT
# field in the manifest is still checked, by the test below.
_INPUT_ECHO_PREFIX = ".config."


def test_no_placeholder_metrics_in_raw_json():
    for path in sorted(RAW.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            _walk(json.load(fh), "", path.name, skip_prefix=_INPUT_ECHO_PREFIX)


def test_manifest_result_fields_are_real_or_not_computed():
    """The run's own outputs -- tau, tie rate, fit accuracy -- are still checked."""
    checked = 0
    for path in sorted(RAW.glob("*.manifest.json")):
        with path.open(encoding="utf-8") as fh:
            m = json.load(fh)
        for key in ("tau", "ties"):
            if key not in m or m[key] in LEGAL_SENTINELS:
                continue
            block = m[key]
            assert isinstance(block, dict), f"{path.name}: {key} should be a dict"
            for k, v in block.items():
                if _is_metric_column(k):
                    _check_metric_value(path.name, f"{key}.{k}", v)
                    checked += 1
    assert checked > 0, "no manifest result fields were checked -- has the schema changed?"


def test_config_echo_nulls_are_only_unfitted_thresholds():
    """A null in the config echo must be an unfitted parameter, nothing else."""
    allowed = {"existence.threshold", "relation.tau", "detector.revision",
               "attribute.revision", "tau"}
    for path in sorted(RAW.glob("*.manifest.json")):
        with path.open(encoding="utf-8") as fh:
            cfg = json.load(fh).get("config", {})
        for section, body in cfg.items():
            if not isinstance(body, dict):
                continue
            for k, v in body.items():
                if v is None:
                    assert f"{section}.{k}" in allowed or k in allowed, (
                        f"{path.name}: unexpected null config value {section}.{k}"
                    )


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
    _check_metric_value("synthetic", "tau", "NOT_APPLICABLE")


def test_the_two_sentinels_are_distinct_and_not_numbers():
    """NOT_APPLICABLE must not silently become NOT_COMPUTED or a number."""
    import pytest as _pt

    assert LEGAL_SENTINELS == ("NOT_COMPUTED", "NOT_APPLICABLE")
    for sentinel in LEGAL_SENTINELS:
        with _pt.raises(ValueError):
            float(sentinel)


def test_forced_choice_manifests_declare_tau_not_applicable():
    """Cells B and D have no threshold; their manifests must say so explicitly."""
    seen = 0
    for path in sorted(RAW.glob("*.manifest.json")):
        with path.open(encoding="utf-8") as fh:
            m = json.load(fh)
        if m.get("cell") in ("B", "D"):
            assert m.get("tau") == "NOT_APPLICABLE", (
                f"{path.name}: forced-choice cell {m['cell']} must report "
                f"tau=NOT_APPLICABLE, got {m.get('tau')!r}"
            )
            seen += 1
    if seen == 0:
        _pt = __import__("pytest")
        _pt.skip("no forced-choice manifest present yet")
