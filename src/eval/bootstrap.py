"""TRD §11.3 — bootstrap confidence intervals, resampled over images.

Questions from the same image are correlated, so the resampling unit is the
image, not the question.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from src.eval.metrics import NOT_COMPUTED, Metric, Record, accuracy

N_RESAMPLES = 10_000
SEED = 20260907


def _by_image(records: Iterable[Record]) -> dict[str, list[Record]]:
    out: dict[str, list[Record]] = {}
    for r in records:
        if "image" not in r:
            raise ValueError(f"record id={r.get('id')} has no 'image'; cannot bootstrap over images")
        out.setdefault(r["image"], []).append(r)
    return out


def _resample_indices(n_images: int, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_images, size=(n_resamples, n_images))


def bootstrap_metric(
    records: Iterable[Record],
    statistic: Callable[[list[Record]], Metric] = accuracy,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> dict[str, Any]:
    """Bootstrap ``statistic`` over images; report mean and a 95% percentile CI."""
    grouped = _by_image(records)
    images = sorted(grouped)
    if not images:
        return {"point": NOT_COMPUTED, "mean": NOT_COMPUTED, "ci_low": NOT_COMPUTED,
                "ci_high": NOT_COMPUTED, "n_resamples": n_resamples, "n_images": 0}

    buckets = [grouped[im] for im in images]
    idx = _resample_indices(len(images), n_resamples, seed)

    values: list[float] = []
    for row in idx:
        sample: list[Record] = []
        for j, b in enumerate(row):
            # Re-key duplicated draws so the id-uniqueness check still holds.
            for r in buckets[b]:
                sample.append({**r, "id": (r["id"], j)})
        v = statistic(sample)
        if v != NOT_COMPUTED:
            values.append(float(v))

    if not values:
        return {"point": NOT_COMPUTED, "mean": NOT_COMPUTED, "ci_low": NOT_COMPUTED,
                "ci_high": NOT_COMPUTED, "n_resamples": n_resamples, "n_images": len(images)}

    arr = np.asarray(values)
    return {
        "point": statistic(list(records)),
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "n_resamples": n_resamples,
        "n_images": len(images),
    }


def bootstrap_paired_difference(
    records_a: Iterable[Record],
    records_b: Iterable[Record],
    statistic: Callable[[list[Record]], Metric] = accuracy,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> dict[str, Any]:
    """TRD §11.3: bootstrap the paired difference (b - a), e.g. Cell D minus Cell A.

    The same resampled images are used for both arms, which is what makes it
    paired.  The headline claim is that this CI excludes zero.
    """
    ga, gb = _by_image(records_a), _by_image(records_b)
    images = sorted(set(ga) & set(gb))
    dropped = (set(ga) | set(gb)) - set(images)
    if dropped:
        raise ValueError(
            f"paired bootstrap requires the same images in both arms; "
            f"{len(dropped)} image(s) appear in only one, e.g. {sorted(dropped)[:5]}"
        )
    if not images:
        return {"point": NOT_COMPUTED, "mean": NOT_COMPUTED, "ci_low": NOT_COMPUTED,
                "ci_high": NOT_COMPUTED, "excludes_zero": NOT_COMPUTED,
                "n_resamples": n_resamples, "n_images": 0}

    idx = _resample_indices(len(images), n_resamples, seed)
    diffs: list[float] = []
    for row in idx:
        sa: list[Record] = []
        sb: list[Record] = []
        for j, b in enumerate(row):
            im = images[b]
            for r in ga[im]:
                sa.append({**r, "id": (r["id"], j)})
            for r in gb[im]:
                sb.append({**r, "id": (r["id"], j)})
        va, vb = statistic(sa), statistic(sb)
        if va != NOT_COMPUTED and vb != NOT_COMPUTED:
            diffs.append(float(vb) - float(va))

    if not diffs:
        return {"point": NOT_COMPUTED, "mean": NOT_COMPUTED, "ci_low": NOT_COMPUTED,
                "ci_high": NOT_COMPUTED, "excludes_zero": NOT_COMPUTED,
                "n_resamples": n_resamples, "n_images": len(images)}

    arr = np.asarray(diffs)
    pa, pb = statistic(list(records_a)), statistic(list(records_b))
    point = (float(pb) - float(pa)) if NOT_COMPUTED not in (pa, pb) else NOT_COMPUTED
    lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return {
        "point": point,
        "mean": float(arr.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_resamples": n_resamples,
        "n_images": len(images),
    }
