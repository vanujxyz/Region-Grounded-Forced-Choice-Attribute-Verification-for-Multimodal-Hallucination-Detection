"""TRD §3 — load and join AMBER's annotations and queries.

Records are matched by index: the annotation for query id *n* is
``annotations[n-1]``.  Nothing here reads pixels, so the loader works before
the images have been downloaded.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

# --------------------------------------------------------------------------
# Paths.  The repository root is the parent of ``src/``.
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
AMBER_DIR = ROOT / "data" / "amber"
IMAGES_DIR = ROOT / "data" / "images"
SPLITS_PATH = ROOT / "data" / "splits" / "splits.json"

ANNOTATIONS_PATH = AMBER_DIR / "annotations.json"
QUERY_ALL_PATH = AMBER_DIR / "query" / "query_all.json"
RELATION_PATH = AMBER_DIR / "relation.json"

# TRD §2 startup assertions.
N_RECORDS = 15220
N_RELATION_KEYS = 340
N_IMAGES = 1004
IMAGE_RE = re.compile(r"^AMBER_(\d+)\.jpg$")

# TRD §3 expected counts, asserted on every load.
EXPECTED_COUNTS: dict[str, int] = {
    "generative": 1004,
    "discriminative-hallucination": 4924,
    "discriminative-attribute-state": 4764,
    "discriminative-attribute-number": 2072,
    "discriminative-attribute-action": 792,
    "discriminative-relation": 975,
    "relation": 689,
}


class Question(BaseModel):
    """A single AMBER question, joined from the query and annotation files."""

    id: int
    image: str  # "AMBER_1.jpg"
    query: str  # "Is the sky sunny in this image?"
    qtype: str  # annotations[id-1]["type"]
    truth: str | list  # "yes"/"no" for discriminative; list for generative


def _read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required AMBER file missing: {path}. Run the TRD §2 acquisition step."
        )
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def assert_data_integrity() -> None:
    """TRD §2 startup assertions, excluding the image check.

    ``assert_images`` is separate because the images arrive by manual download
    and no part of the data pipeline needs them.
    """
    annotations = _read_json(ANNOTATIONS_PATH)
    queries = _read_json(QUERY_ALL_PATH)
    relation = _read_json(RELATION_PATH)

    if not isinstance(annotations, list) or len(annotations) != N_RECORDS:
        raise AssertionError(
            f"annotations.json must be a list of length {N_RECORDS}, "
            f"got {type(annotations).__name__} of length {len(annotations)}"
        )
    if not isinstance(queries, list) or len(queries) != N_RECORDS:
        raise AssertionError(
            f"query_all.json must be a list of length {N_RECORDS}, "
            f"got {type(queries).__name__} of length {len(queries)}"
        )
    if len(relation) != N_RELATION_KEYS:
        raise AssertionError(
            f"relation.json must have exactly {N_RELATION_KEYS} keys, got {len(relation)}"
        )
    for q in queries:
        got = annotations[q["id"] - 1]["id"]
        if got != q["id"]:
            raise AssertionError(
                f"id/index misalignment: annotations[{q['id'] - 1}]['id'] == {got}, "
                f"expected {q['id']}"
            )


def assert_images() -> None:
    """TRD §2 image assertion.  Called at model time, not by the data pipeline.

    The images are a manual Google Drive download and are not in the AMBER repo.
    """
    if not IMAGES_DIR.is_dir():
        raise AssertionError(
            f"{IMAGES_DIR} does not exist. Download the AMBER images from the "
            "Google Drive link in data/amber_repo/README.md and unzip them there."
        )
    files = [p.name for p in IMAGES_DIR.iterdir() if p.is_file()]
    matching = [f for f in files if IMAGE_RE.match(f)]
    if len(files) != N_IMAGES or len(matching) != N_IMAGES:
        raise AssertionError(
            f"{IMAGES_DIR} must contain exactly {N_IMAGES} files matching "
            rf"AMBER_(\d+).jpg; found {len(files)} files, {len(matching)} matching."
        )
    indices = sorted(int(IMAGE_RE.match(f).group(1)) for f in matching)
    if indices != list(range(1, N_IMAGES + 1)):
        raise AssertionError(
            f"image indices must be exactly 1..{N_IMAGES}; got "
            f"min={indices[0]}, max={indices[-1]}, n_unique={len(set(indices))}"
        )


@lru_cache(maxsize=1)
def _load_all() -> tuple[Question, ...]:
    assert_data_integrity()
    annotations = _read_json(ANNOTATIONS_PATH)
    queries = _read_json(QUERY_ALL_PATH)

    questions: list[Question] = []
    for q in queries:
        ann = annotations[q["id"] - 1]
        questions.append(
            Question(
                id=q["id"],
                image=q["image"],
                query=q["query"],
                qtype=ann["type"],
                truth=ann["truth"],
            )
        )

    counts: dict[str, int] = {}
    for question in questions:
        counts[question.qtype] = counts.get(question.qtype, 0) + 1
    if counts != EXPECTED_COUNTS:
        raise AssertionError(
            f"qtype counts do not match TRD §3.\n  expected: {EXPECTED_COUNTS}\n  observed: {counts}"
        )
    return tuple(questions)


def load_questions(qtype: str | None = None) -> list[Question]:
    """Return all questions, or only those of ``qtype``."""
    questions = _load_all()
    if qtype is None:
        return list(questions)
    if qtype not in EXPECTED_COUNTS:
        raise ValueError(
            f"unknown qtype {qtype!r}; expected one of {sorted(EXPECTED_COUNTS)}"
        )
    return [q for q in questions if q.qtype == qtype]


def all_images() -> list[str]:
    """The 1,004 distinct image filenames, sorted by their numeric index.

    Derived from the query file so that splits can be frozen before the manual
    image download.  ``assert_images`` checks the directory agrees.
    """
    names = {q.image for q in _load_all()}
    indices = sorted(int(IMAGE_RE.match(n).group(1)) for n in names)
    if indices != list(range(1, N_IMAGES + 1)):
        raise AssertionError(
            f"image indices from the query file must be exactly 1..{N_IMAGES}; "
            f"got {len(indices)} indices, min={indices[0]}, max={indices[-1]}"
        )
    return [f"AMBER_{i}.jpg" for i in indices]
