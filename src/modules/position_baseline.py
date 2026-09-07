"""D-009 — position-only baseline. A benchmark artifact, not a method.

In all 2,774 clean AMBER attribute pairs the true attribute has the LOWER
question id, with zero exceptions, and the two ids are always adjacent (gap
exactly 1).  AMBER generates the correct attribute first.

A detector that ignores the image entirely and answers "yes" to whichever
question of a pair has the lower id therefore scores 100%.  This module
implements exactly that, so the number appears as a labelled row in Table 1
rather than being discovered by a reviewer.

It loads no model and never opens an image.  That is the point.
"""

from __future__ import annotations

from src.data.pairs import AttrPair

CELL = "position-only"
IS_BENCHMARK_ARTIFACT = True

LABEL = (
    "Position-only baseline (BENCHMARK ARTIFACT -- exploits AMBER pair id "
    "ordering; sees no image)"
)


def predict_pair(pair: AttrPair) -> list[dict]:
    """Answer yes to the lower-id question, no to the higher-id one.

    Deliberately derived from ids alone: this is the artifact being measured.
    """
    lower_id = min(pair.positive_id, pair.negative_id)
    out = []
    for qid, attr, gold in (
        (pair.positive_id, pair.positive_attr, "yes"),
        (pair.negative_id, pair.negative_attr, "no"),
    ):
        out.append(
            {
                "id": qid,
                "image": pair.image,
                "obj": pair.obj,
                "attr": attr,
                "cell": CELL,
                "pred": "yes" if qid == lower_id else "no",
                "confidence": 1.0,
                "gold": gold,
                "triple": (pair.obj, pair.positive_attr, pair.negative_attr),
                "fell_back": False,
                "benchmark_artifact": True,
            }
        )
    return out


def predict(pairs: list[AttrPair]) -> list[dict]:
    records: list[dict] = []
    for p in pairs:
        records.extend(predict_pair(p))
    return records


def audit_id_ordering(pairs: list[AttrPair]) -> dict:
    """Quantify the artifact: how often is the true attribute the lower id?"""
    n = len(pairs)
    lower = sum(1 for p in pairs if p.positive_id < p.negative_id)
    gaps = {abs(p.positive_id - p.negative_id) for p in pairs}
    return {
        "n_pairs": n,
        "positive_has_lower_id": lower,
        "fraction": (lower / n) if n else "NOT_COMPUTED",
        "distinct_id_gaps": sorted(gaps),
    }
