"""TRD §16 — a threshold must never be fitted on the test split."""

import inspect

import pytest

from src.run import frozen_tau_from_dev, run_attribute_cell


def test_frozen_tau_reads_a_dev_manifest():
    tau, info = frozen_tau_from_dev("A")
    assert isinstance(tau, float)
    assert "dev" in info["frozen_from"]
    assert info["protocol"] == "frozen-from-dev (NOT fitted on test)"


def test_frozen_tau_available_for_every_threshold_cell():
    for cell in ("A", "C", "Ap", "Cp"):
        tau, info = frozen_tau_from_dev(cell)
        assert isinstance(tau, float)
        assert info["frozen_from"].startswith(f"attribute_{cell}_dev_")


def test_frozen_tau_raises_for_a_cell_never_run_on_dev():
    with pytest.raises(RuntimeError, match="never fitted on the test split"):
        frozen_tau_from_dev("NoSuchCell")


def test_test_split_takes_the_frozen_branch_not_the_fitting_branch():
    """Static guard: the fitting path must be unreachable when split == 'test'."""
    src = inspect.getsource(run_attribute_cell)
    assert 'split == "test"' in src, "test split is not special-cased"
    assert "frozen_tau_from_dev(cell, dataset)" in src
    frozen_at = src.index('split == "test"')
    fit_at = src.index("fit_tau(fit_data)")
    assert frozen_at < fit_at, (
        "the frozen-tau branch must be checked BEFORE the fitting branch"
    )


def test_no_tau_means_a_loud_failure():
    src = inspect.getsource(run_attribute_cell)
    assert "has no tau after fitting/freezing" in src
