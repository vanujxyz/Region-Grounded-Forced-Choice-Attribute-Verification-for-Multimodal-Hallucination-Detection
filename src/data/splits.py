"""TRD §5 — image-level train/test splits.

Splits are by image, never by question: the same image must not appear in both
a tuning and an evaluation split (PRD §8 rule 4).  Written once, then frozen.
"""

from __future__ import annotations

import json
import os
import random

from src.data.loader import N_IMAGES, SPLITS_PATH, all_images

SEED = 20260907
DEV_FRACTION = 0.7
TEST_ENV_VAR = "ALLOW_TEST_SPLIT"


def _compute() -> dict[str, list[str]]:
    """Sort by numeric index, shuffle with the fixed seed, cut at 70%."""
    images = all_images()  # already sorted by numeric index
    if len(images) != N_IMAGES:
        raise AssertionError(f"expected {N_IMAGES} images, got {len(images)}")
    shuffled = list(images)
    random.Random(SEED).shuffle(shuffled)
    n_dev = int(DEV_FRACTION * len(shuffled))
    return {"dev": shuffled[:n_dev], "test": shuffled[n_dev:]}


def build_splits(force_resplit: bool = False) -> dict[str, list[str]]:
    """Create ``splits.json`` once.  Refuse to regenerate without --force-resplit."""
    if SPLITS_PATH.exists() and not force_resplit:
        raise RuntimeError(
            f"{SPLITS_PATH} already exists and splits are frozen. "
            "Pass --force-resplit to overwrite (this invalidates every prior result)."
        )
    splits = _compute()
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SPLITS_PATH.open("w", encoding="utf-8") as fh:
        json.dump({"seed": SEED, "dev_fraction": DEV_FRACTION, **splits}, fh, indent=2)
    return splits


def load_splits() -> dict[str, list[str]]:
    """Load the frozen splits, creating them on first use."""
    if not SPLITS_PATH.exists():
        build_splits()
    with SPLITS_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    dev, test = data["dev"], data["test"]
    if set(dev) & set(test):
        raise AssertionError(
            f"dev and test overlap on {len(set(dev) & set(test))} images"
        )
    if len(dev) + len(test) != N_IMAGES:
        raise AssertionError(
            f"splits cover {len(dev) + len(test)} images, expected {N_IMAGES}"
        )
    return {"dev": dev, "test": test}


def get_split(split: str) -> list[str]:
    """Return the image list for ``split``.

    Reading ``test`` requires ALLOW_TEST_SPLIT=1 in the environment (TRD §5).
    The test split is opened once, at step 14 (PRD §8 rule 6).
    """
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
    if split == "test" and os.environ.get(TEST_ENV_VAR) != "1":
        raise PermissionError(
            "Refusing to read the test split. It may be opened once, at TRD §15 "
            f"step 14. Set {TEST_ENV_VAR}=1 only when that time has come."
        )
    return load_splits()[split]
