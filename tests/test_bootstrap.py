"""TRD §11.3 — bootstrap over images, on synthetic predictions."""

import pytest

from src.eval.bootstrap import (
    N_RESAMPLES,
    SEED,
    bootstrap_metric,
    bootstrap_paired_difference,
)
from src.eval.metrics import NOT_COMPUTED


def rec(i, image, pred, gold):
    return {"id": i, "image": image, "pred": pred, "gold": gold}


def _perfect(n_images=20, per_image=4):
    return [
        rec(im * per_image + k, f"AMBER_{im}.jpg", "yes", "yes")
        for im in range(n_images)
        for k in range(per_image)
    ]


def _mixed(n_images=20, per_image=4, correct=2):
    out = []
    for im in range(n_images):
        for k in range(per_image):
            gold = "yes"
            pred = "yes" if k < correct else "no"
            out.append(rec(im * per_image + k, f"AMBER_{im}.jpg", pred, gold))
    return out


def test_seed_and_resamples_match_spec():
    assert SEED == 20260907
    assert N_RESAMPLES == 10_000


def test_bootstrap_of_a_constant_is_degenerate():
    b = bootstrap_metric(_perfect(), n_resamples=200)
    assert b["point"] == 1.0 and b["mean"] == 1.0
    assert b["ci_low"] == 1.0 and b["ci_high"] == 1.0
    assert b["n_images"] == 20


def test_bootstrap_ci_brackets_the_point_estimate():
    b = bootstrap_metric(_mixed(), n_resamples=500)
    assert b["point"] == 0.5
    assert b["ci_low"] <= b["point"] <= b["ci_high"]


def test_bootstrap_is_deterministic():
    a = bootstrap_metric(_mixed(), n_resamples=300)
    b = bootstrap_metric(_mixed(), n_resamples=300)
    assert a == b


def test_bootstrap_resamples_images_not_questions():
    """With one image, every resample is that same image, so the CI collapses."""
    rs = [rec(k, "AMBER_1.jpg", "yes" if k < 2 else "no", "yes") for k in range(4)]
    b = bootstrap_metric(rs, n_resamples=200)
    assert b["n_images"] == 1
    assert b["ci_low"] == b["ci_high"] == 0.5


def test_bootstrap_empty_is_not_computed():
    b = bootstrap_metric([], n_resamples=100)
    assert b["point"] == NOT_COMPUTED and b["ci_low"] == NOT_COMPUTED


def test_paired_difference_of_identical_arms_is_zero():
    rs = _mixed()
    d = bootstrap_paired_difference(rs, rs, n_resamples=300)
    assert d["point"] == 0.0
    assert d["ci_low"] == 0.0 == d["ci_high"]
    assert d["excludes_zero"] is False


def test_paired_difference_detects_a_real_gain():
    a = _mixed(correct=1)   # 25%
    b = _mixed(correct=4)   # 100%
    d = bootstrap_paired_difference(a, b, n_resamples=500)
    assert d["point"] == pytest.approx(0.75)
    assert d["excludes_zero"] is True
    assert d["ci_low"] > 0


def test_paired_difference_requires_matching_images():
    a = _mixed(n_images=5)
    b = _mixed(n_images=6)
    with pytest.raises(ValueError, match="same images"):
        bootstrap_paired_difference(a, b, n_resamples=10)


def test_records_without_image_raise():
    with pytest.raises(ValueError, match="image"):
        bootstrap_metric([{"id": 1, "pred": "yes", "gold": "yes"}], n_resamples=10)
