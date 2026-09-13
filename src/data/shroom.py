"""SHROOM-Vis: a second attribute benchmark, in AMBER's schema.

2,495 images from the SHROOM vision set, annotated to AMBER's two
discriminative-attribute templates (see ``data/shroom/ANNOTATION_POLICY.md``),
so that every cell of the 2x2 runs on it with **no change to the method**.

This module mirrors the contract of ``src.data.loader`` and ``src.data.pairs``
rather than extending them. The AMBER loader asserts AMBER's exact record
counts on every call -- those assertions are load-bearing (TRD §2/§3) and are
left untouched; a second dataset gets a second reader.

Two differences from AMBER are deliberate and are **not** defects:

* Image filenames are the original SHROOM names, not ``AMBER_n.jpg``. Nothing
  downstream parses an index out of a filename.
* Within a pair, which question holds the lower id is a seeded coin flip.
  AMBER gives the true attribute the lower id in all 2,774 of its pairs, which
  is the benchmark artifact recorded in ``README.md``. This dataset does not
  reproduce it, so ``--module position-only`` should score about 0.5 here.
"""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from pathlib import Path

from src.config import ROOT
from src.data.loader import Question
from src.data.pairs import (
    ACTION_QTYPE,
    ACTION_RE,
    STATE_QTYPE,
    STATE_RE,
    AttrPair,
)

SHROOM_DIR = ROOT / "data" / "shroom"
IMAGES_DIR = ROOT / "shroom-visions-images" / "shroom-vis-images"
ANNOTATIONS_PATH = SHROOM_DIR / "annotations.json"
QUERY_ALL_PATH = SHROOM_DIR / "query" / "query_all.json"
PAIR_META_PATH = SHROOM_DIR / "pair_meta.json"
SPLITS_PATH = SHROOM_DIR / "splits.json"

# Same seed and same fraction as the AMBER splits, so the two datasets are
# split under one rule (src/data/splits.py).
SEED = 20260907
DEV_FRACTION = 0.7
TEST_ENV_VAR = "ALLOW_TEST_SPLIT"

QTYPES = (STATE_QTYPE, ACTION_QTYPE)
PARSE_RE = {STATE_QTYPE: STATE_RE, ACTION_QTYPE: ACTION_RE}


def _read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"SHROOM file missing: {path}. Build it with "
            "`python scripts/shroom_build.py` after annotating."
        )
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _load_all() -> tuple[Question, ...]:
    annotations = _read_json(ANNOTATIONS_PATH)
    queries = _read_json(QUERY_ALL_PATH)

    if len(annotations) != len(queries):
        raise AssertionError(
            f"{len(annotations)} annotations vs {len(queries)} queries; "
            "the two files must be parallel"
        )
    questions: list[Question] = []
    for q in queries:
        ann = annotations[q["id"] - 1]
        if ann["id"] != q["id"]:
            raise AssertionError(
                f"id/index misalignment: annotations[{q['id'] - 1}]['id'] == "
                f"{ann['id']}, expected {q['id']}"
            )
        if ann["type"] not in QTYPES:
            raise AssertionError(
                f"question {q['id']} has type {ann['type']!r}; SHROOM-Vis "
                f"annotates only {QTYPES}"
            )
        if ann["truth"] not in ("yes", "no"):
            raise AssertionError(
                f"question {q['id']} has truth {ann['truth']!r}, expected yes|no"
            )
        questions.append(
            Question(
                id=q["id"],
                image=q["image"],
                query=q["query"],
                qtype=ann["type"],
                truth=ann["truth"],
            )
        )
    return tuple(questions)


def load_questions(qtype: str | None = None) -> list[Question]:
    questions = _load_all()
    if qtype is None:
        return list(questions)
    if qtype not in QTYPES:
        raise ValueError(f"unknown qtype {qtype!r}; expected one of {QTYPES}")
    return [q for q in questions if q.qtype == qtype]


def all_images() -> list[str]:
    """Every image carrying at least one question, in sorted filename order."""
    return sorted({q.image for q in _load_all()})


def load_attr_pairs() -> list[AttrPair]:
    """Rebuild the contrastive pairs from the questions alone.

    ``pair_meta.json`` records the same structure, but it is provenance, not the
    evaluation path: reconstructing the pairs by parsing the queries is exactly
    what ``src.data.pairs`` does for AMBER, and it means a mismatch between the
    two files surfaces as a failure instead of being papered over.
    """
    from collections import defaultdict

    groups: dict[tuple[str, str], list[tuple[str, str, int]]] = defaultdict(list)
    for q in _load_all():
        m = PARSE_RE[q.qtype].match(q.query)
        if m is None:
            raise AssertionError(
                f"question {q.id} does not parse with the {q.qtype} regex: {q.query!r}"
            )
        groups[(q.image, m.group("obj"))].append((m.group("attr"), q.truth, q.id))

    pairs: list[AttrPair] = []
    discarded = 0
    for (image, obj), members in groups.items():
        truths = sorted(t for _, t, _ in members)
        if len(members) == 2 and truths == ["no", "yes"]:
            pos = next(m for m in members if m[1] == "yes")
            neg = next(m for m in members if m[1] == "no")
            pairs.append(
                AttrPair(
                    image=image,
                    obj=obj,
                    positive_attr=pos[0],
                    negative_attr=neg[0],
                    positive_id=pos[2],
                    negative_id=neg[2],
                )
            )
        else:
            discarded += 1
    if discarded:
        raise AssertionError(
            f"{discarded} (image, object) groups are not clean yes/no pairs. "
            "scripts/shroom_build.py is supposed to make this impossible; "
            "rebuild rather than tolerating it."
        )
    pairs.sort(key=lambda p: p.positive_id)
    return pairs


def load_pair_meta() -> dict[str, dict]:
    """Per-pair provenance, keyed by the positive question id as a string."""
    return _read_json(PAIR_META_PATH)


def _compute_splits() -> dict[str, list[str]]:
    images = all_images()
    shuffled = list(images)
    random.Random(SEED).shuffle(shuffled)
    n_dev = int(DEV_FRACTION * len(shuffled))
    return {"dev": shuffled[:n_dev], "test": shuffled[n_dev:]}


def load_splits() -> dict[str, list[str]]:
    """Load the frozen SHROOM splits, creating them on first use."""
    if not SPLITS_PATH.exists():
        splits = _compute_splits()
        SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SPLITS_PATH.open("w", encoding="utf-8") as fh:
            json.dump(
                {"seed": SEED, "dev_fraction": DEV_FRACTION, **splits}, fh, indent=2
            )
    with SPLITS_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    dev, test = data["dev"], data["test"]
    if set(dev) & set(test):
        raise AssertionError(f"dev and test overlap on {len(set(dev) & set(test))} images")
    if len(dev) + len(test) != len(all_images()):
        raise AssertionError(
            f"splits cover {len(dev) + len(test)} images, expected {len(all_images())}"
        )
    return {"dev": dev, "test": test}


def get_split(split: str) -> list[str]:
    """Image list for ``split``; ``test`` needs ALLOW_TEST_SPLIT=1, as for AMBER."""
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
    if split == "test" and os.environ.get(TEST_ENV_VAR) != "1":
        raise PermissionError(
            "Refusing to read the SHROOM test split. Set ALLOW_TEST_SPLIT=1 only "
            "when development on this dataset is complete."
        )
    return load_splits()[split]
