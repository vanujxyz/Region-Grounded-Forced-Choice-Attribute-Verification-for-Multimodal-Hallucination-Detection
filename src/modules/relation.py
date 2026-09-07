"""TRD §10 — relation checking.

Deliberately off-the-shelf and **not** claimed as a contribution (PRD §7): a
SigLIP score of the full query text against the full image, thresholded, with tau
fitted on dev.

**Corrected regex (D-001).** TRD §10 specifies:

    ^Is there direct contact between the (?P<a>.+?) and (?P<b>.+?) in this image\\?$

which parses **0 of 1664** relation questions. The data has no " in this image"
suffix and no article before the second object:

    "Is there direct contact between the person and grass?"

The corrected pattern below parses 1664/1664. This was found at M0, recorded as
D-001, confirmed independently by the project owner, and applied here at M4 as
agreed.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from src.data.loader import load_questions

# D-001: the corrected pattern. Parses 1664/1664.
REL_RE = re.compile(r"^Is there direct contact between the (?P<a>.+?) and (?P<b>.+?)\?$")

# The pattern as written in TRD §10, kept only so the defect stays testable.
TRD_REL_RE = re.compile(
    r"^Is there direct contact between the (?P<a>.+?) and (?P<b>.+?) in this image\?$"
)

RELATION_QTYPES = ("discriminative-relation", "relation")
EXPECTED_COUNTS = {"discriminative-relation": 975, "relation": 689}
EXPECTED_TOTAL = 1664


class RelationQuestion(BaseModel):
    id: int
    image: str
    query: str
    qtype: str
    a: str  # first object, e.g. "person"
    b: str  # second object, e.g. "grass"
    truth: str


def parse_relation_questions() -> list[RelationQuestion]:
    """Parse every relation question; assert a 100% parse rate."""
    out: list[RelationQuestion] = []
    failures: list[str] = []
    for qtype in RELATION_QTYPES:
        questions = load_questions(qtype)
        if len(questions) != EXPECTED_COUNTS[qtype]:
            raise AssertionError(
                f"{qtype}: expected {EXPECTED_COUNTS[qtype]} questions, got {len(questions)}"
            )
        for q in questions:
            m = REL_RE.match(q.query)
            if m is None:
                failures.append(q.query)
                continue
            out.append(
                RelationQuestion(
                    id=q.id, image=q.image, query=q.query, qtype=q.qtype,
                    a=m.group("a"), b=m.group("b"), truth=q.truth,
                )
            )
    if failures:
        raise AssertionError(
            f"REL_RE failed to parse {len(failures)}/{EXPECTED_TOTAL} relation "
            f"questions. Examples: {failures[:5]}"
        )
    if len(out) != EXPECTED_TOTAL:
        raise AssertionError(f"expected {EXPECTED_TOTAL} relation questions, got {len(out)}")
    out.sort(key=lambda r: r.id)
    return out


def predict(
    scorer, questions: list[RelationQuestion], images_dir, tau: float
) -> list[dict[str, Any]]:
    """Score the full query text against the full image and threshold it.

    Off-the-shelf by design (TRD §10). No cropping, no forced choice: this module
    exists so the demo is a complete system, not as a contribution.
    """
    from PIL import Image

    records = []
    for q in questions:
        img = Image.open(images_dir / q.image).convert("RGB")
        score = scorer.score(img, [q.query])[0]
        records.append({
            "id": q.id,
            "image": q.image,
            "qtype": q.qtype,
            "obj": q.a,
            "attr": q.b,
            "cell": "relation",
            "pred": "yes" if score >= tau else "no",
            "confidence": 1.0 / (1.0 + pow(2.718281828459045, -scorer.confidence_k * (score - tau))),
            "gold": q.truth,
            "fell_back": False,
            "detector_score": None,
            "raw_scores": [score],
            "runtime_ms": None,
        })
    return records


def score_all(scorer, questions: list[RelationQuestion], images_dir) -> list[tuple[float, str]]:
    """Raw scores paired with gold, for fitting tau on dev."""
    from PIL import Image

    out = []
    for q in questions:
        img = Image.open(images_dir / q.image).convert("RGB")
        out.append((scorer.score(img, [q.query])[0], q.truth))
    return out
