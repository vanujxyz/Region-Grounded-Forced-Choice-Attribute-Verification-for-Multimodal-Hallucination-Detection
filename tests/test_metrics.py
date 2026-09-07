"""TRD §14 — test_metrics.py: hand-computed cases."""

import pytest

from src.eval.metrics import (
    NOT_COMPUTED,
    accuracy,
    compute_metrics,
    confusion,
    fallback_rate,
    precision_recall_f1,
)


def rec(i, pred, gold, **kw):
    r = {"id": i, "image": kw.pop("image", "AMBER_1.jpg"), "pred": pred, "gold": gold}
    r.update(kw)
    return r


# --------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------
def test_accuracy_all_correct():
    assert accuracy([rec(1, "yes", "yes"), rec(2, "no", "no")]) == 1.0


def test_accuracy_all_wrong():
    """A genuinely computed 0.0 is a real number, not a placeholder."""
    assert accuracy([rec(1, "no", "yes"), rec(2, "yes", "no")]) == 0.0


def test_accuracy_hand_computed_three_of_four():
    rs = [rec(1, "yes", "yes"), rec(2, "no", "no"), rec(3, "yes", "no"), rec(4, "no", "no")]
    assert accuracy(rs) == 0.75


def test_accuracy_one_of_three():
    rs = [rec(1, "yes", "yes"), rec(2, "yes", "no"), rec(3, "no", "yes")]
    assert accuracy(rs) == pytest.approx(1 / 3)


def test_accuracy_empty_is_not_computed():
    """An undefined metric is NOT_COMPUTED, never 0 (PRD §8 rule 1)."""
    assert accuracy([]) == NOT_COMPUTED


# --------------------------------------------------------------------------
# Confusion / P / R / F1, yes = positive
# --------------------------------------------------------------------------
def test_confusion_hand_computed():
    rs = [
        rec(1, "yes", "yes"),  # tp
        rec(2, "yes", "yes"),  # tp
        rec(3, "yes", "no"),   # fp
        rec(4, "no", "yes"),   # fn
        rec(5, "no", "no"),    # tn
    ]
    assert confusion(rs) == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}


def test_precision_recall_f1_hand_computed():
    rs = [
        rec(1, "yes", "yes"), rec(2, "yes", "yes"),
        rec(3, "yes", "no"), rec(4, "no", "yes"), rec(5, "no", "no"),
    ]
    m = precision_recall_f1(rs)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_precision_undefined_when_no_yes_predicted():
    rs = [rec(1, "no", "yes"), rec(2, "no", "no")]
    m = precision_recall_f1(rs)
    assert m["precision"] == NOT_COMPUTED
    assert m["recall"] == 0.0        # tp+fn = 1, so recall IS defined and is 0
    assert m["f1"] == NOT_COMPUTED


def test_recall_undefined_when_no_gold_yes():
    rs = [rec(1, "yes", "no"), rec(2, "no", "no")]
    m = precision_recall_f1(rs)
    assert m["precision"] == 0.0     # tp+fp = 1, so precision IS defined and is 0
    assert m["recall"] == NOT_COMPUTED
    assert m["f1"] == NOT_COMPUTED


def test_f1_undefined_when_precision_and_recall_both_zero():
    rs = [rec(1, "yes", "no"), rec(2, "no", "yes")]
    m = precision_recall_f1(rs)
    assert m["precision"] == 0.0 and m["recall"] == 0.0
    assert m["f1"] == NOT_COMPUTED


def test_perfect_f1():
    rs = [rec(1, "yes", "yes"), rec(2, "no", "no")]
    m = precision_recall_f1(rs)
    assert (m["precision"], m["recall"], m["f1"]) == (1.0, 1.0, 1.0)


# --------------------------------------------------------------------------
# Fallback
# --------------------------------------------------------------------------
def test_fallback_rate_hand_computed():
    rs = [
        rec(1, "yes", "yes", fell_back=True),
        rec(2, "no", "no", fell_back=False),
        rec(3, "no", "no", fell_back=False),
        rec(4, "no", "no", fell_back=True),
    ]
    assert fallback_rate(rs) == 0.5


def test_fallback_rate_zero_is_real():
    rs = [rec(1, "yes", "yes", fell_back=False)]
    assert fallback_rate(rs) == 0.0


def test_fallback_rate_not_computed_when_field_absent():
    assert fallback_rate([rec(1, "yes", "yes")]) == NOT_COMPUTED


def test_fallback_rate_empty_is_not_computed():
    assert fallback_rate([]) == NOT_COMPUTED


# --------------------------------------------------------------------------
# Validation: failures are loud (PRD §8 rule 2)
# --------------------------------------------------------------------------
def test_invalid_pred_label_raises():
    with pytest.raises(ValueError, match="pred"):
        accuracy([{"id": 1, "pred": "maybe", "gold": "yes"}])


def test_invalid_gold_label_raises():
    with pytest.raises(ValueError, match="gold"):
        accuracy([{"id": 1, "pred": "yes", "gold": "unknown"}])


def test_missing_key_raises():
    with pytest.raises(ValueError, match="missing"):
        accuracy([{"id": 1, "pred": "yes"}])


def test_duplicate_id_raises():
    with pytest.raises(ValueError, match="duplicate"):
        accuracy([rec(1, "yes", "yes"), rec(1, "no", "no")])


# --------------------------------------------------------------------------
# Breakdowns (TRD §11.2)
# --------------------------------------------------------------------------
def test_breakdowns_hand_computed():
    rs = [
        rec(1, "yes", "yes", cell="A", qtype="discriminative-attribute-state", fell_back=False),
        rec(2, "no", "yes", cell="A", qtype="discriminative-attribute-state", fell_back=True),
        rec(3, "yes", "yes", cell="D", qtype="discriminative-attribute-action", fell_back=False),
        rec(4, "yes", "yes", cell="D", qtype="discriminative-attribute-number", fell_back=False),
    ]
    m = compute_metrics(rs)
    assert m["overall"]["accuracy"] == 0.75
    assert m["overall"]["n"] == 4
    assert m["by_cell"]["A"]["accuracy"] == 0.5
    assert m["by_cell"]["D"]["accuracy"] == 1.0
    assert m["by_subtype"]["state"]["accuracy"] == 0.5
    assert m["by_subtype"]["action"]["accuracy"] == 1.0
    assert m["by_subtype"]["number"]["accuracy"] == 1.0
    assert m["by_fallback"]["true"]["accuracy"] == 0.0
    assert m["by_fallback"]["false"]["accuracy"] == 1.0
    assert m["fallback_rate"] == 0.25


def test_compute_metrics_on_empty():
    m = compute_metrics([])
    assert m["overall"]["accuracy"] == NOT_COMPUTED
    assert m["overall"]["n"] == 0
    assert m["fallback_rate"] == NOT_COMPUTED
    assert m["by_cell"] == {} and m["by_subtype"] == {} and m["by_fallback"] == {}
