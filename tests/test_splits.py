"""TRD §14 — test_splits.py.

Asserts: disjoint image sets; deterministic across runs; test split blocked
without the env var.
"""

import json

import pytest

from src.data.loader import N_IMAGES, SPLITS_PATH
from src.data.splits import (
    DEV_FRACTION,
    SEED,
    TEST_ENV_VAR,
    _compute,
    build_splits,
    get_split,
    load_splits,
)


def test_splits_file_exists_and_is_frozen():
    assert SPLITS_PATH.exists(), "splits.json must be frozen during M0"


def test_dev_and_test_are_disjoint():
    s = load_splits()
    assert set(s["dev"]).isdisjoint(set(s["test"]))


def test_splits_cover_every_image_exactly_once():
    s = load_splits()
    allocated = s["dev"] + s["test"]
    assert len(allocated) == N_IMAGES
    assert len(set(allocated)) == N_IMAGES


def test_split_sizes():
    s = load_splits()
    assert len(s["dev"]) == int(DEV_FRACTION * N_IMAGES) == 702
    assert len(s["test"]) == N_IMAGES - 702 == 302


def test_deterministic_across_runs():
    """Recomputing from the seed must reproduce the frozen file exactly."""
    recomputed = _compute()
    frozen = load_splits()
    assert recomputed["dev"] == frozen["dev"]
    assert recomputed["test"] == frozen["test"]


def test_recompute_is_stable():
    assert _compute() == _compute()


def test_seed_recorded_in_file():
    with SPLITS_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["seed"] == SEED == 20260907


def test_rebuild_without_force_raises():
    with pytest.raises(RuntimeError, match="frozen"):
        build_splits(force_resplit=False)


def test_dev_split_readable():
    assert len(get_split("dev")) == 702


def test_test_split_blocked_without_env_var(monkeypatch):
    monkeypatch.delenv(TEST_ENV_VAR, raising=False)
    with pytest.raises(PermissionError):
        get_split("test")


def test_test_split_blocked_when_env_var_is_not_1(monkeypatch):
    monkeypatch.setenv(TEST_ENV_VAR, "0")
    with pytest.raises(PermissionError):
        get_split("test")


def test_test_split_readable_with_env_var(monkeypatch):
    monkeypatch.setenv(TEST_ENV_VAR, "1")
    assert len(get_split("test")) == 302


def test_invalid_split_name_raises():
    with pytest.raises(ValueError):
        get_split("train")


def test_split_is_by_image_not_question():
    """No image may appear in both splits -- PRD §8 rule 4."""
    from src.data.loader import load_questions

    s = load_splits()
    dev, test = set(s["dev"]), set(s["test"])
    for q in load_questions():
        assert (q.image in dev) != (q.image in test)
