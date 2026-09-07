"""TRD §8 — existence checking.

For the 4,924 `discriminative-hallucination` questions ("Is there a cloud in this
image?"). Answer yes iff the detector returns at least one detection above
``existence.threshold``, which is fitted on dev.

Off-the-shelf by design (PRD §7) and not claimed as a contribution.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from src.data.loader import load_questions

EXIST_RE = re.compile(r"^Is there an? (?P<obj>.+?) in this image\?$")

QTYPE = "discriminative-hallucination"
EXPECTED_COUNT = 4924


class ExistenceQuestion(BaseModel):
    id: int
    image: str
    query: str
    obj: str
    truth: str


def parse_existence_questions() -> list[ExistenceQuestion]:
    """Parse every existence question; assert the parse rate and log failures."""
    questions = load_questions(QTYPE)
    if len(questions) != EXPECTED_COUNT:
        raise AssertionError(f"expected {EXPECTED_COUNT} questions, got {len(questions)}")

    out, failures = [], []
    for q in questions:
        m = EXIST_RE.match(q.query)
        if m is None:
            failures.append(q.query)
            continue
        out.append(
            ExistenceQuestion(
                id=q.id, image=q.image, query=q.query, obj=m.group("obj"), truth=q.truth
            )
        )
    if failures:
        raise AssertionError(
            f"EXIST_RE failed to parse {len(failures)}/{len(questions)} existence "
            f"questions. Examples: {failures[:5]}"
        )
    out.sort(key=lambda q: q.id)
    return out


def fit_threshold(scored: list[tuple[float, str]]) -> tuple[float, dict[str, Any]]:
    """Sweep the existence threshold on dev, maximising accuracy.

    ``scored`` is (best detection score, gold) per question, where a question
    with no detection at all scores ``-inf`` and can never be answered yes.
    """
    if not scored:
        raise ValueError("cannot fit a threshold on an empty set")
    finite = sorted({s for s, _ in scored if s > float("-inf")})
    if not finite:
        raise ValueError("no detections at all; cannot fit an existence threshold")

    candidates = [finite[0] - 1.0] + [
        (finite[i] + finite[i + 1]) / 2.0 for i in range(len(finite) - 1)
    ] + [finite[-1] + 1.0]

    best_t, best_acc = candidates[0], -1.0
    for t in candidates:
        correct = sum(1 for s, g in scored if (("yes" if s >= t else "no") == g))
        acc = correct / len(scored)
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, {
        "threshold": best_t,
        "selection": "accuracy_maximising",
        "fit_accuracy": best_acc,
        "fit_n": len(scored),
        "protocol": "fit-on-eval",
    }


def best_score(dets) -> float:
    """Highest detection score, or -inf when nothing was detected."""
    return max((d[4] for d in dets), default=float("-inf"))


def predict(questions, detections_for, threshold: float, confidence_k: float = 10.0):
    """Answer yes iff at least one detection clears ``threshold``."""
    records = []
    for q in questions:
        dets = detections_for(q.image, q.obj)
        above = [d for d in dets if d[4] >= threshold]
        s = best_score(dets)
        finite = s if s > float("-inf") else 0.0
        records.append({
            "id": q.id,
            "image": q.image,
            "qtype": QTYPE,
            "obj": q.obj,
            "attr": None,
            "cell": "existence",
            "pred": "yes" if above else "no",
            "confidence": 1.0 / (1.0 + pow(2.718281828459045, -confidence_k * (finite - threshold))),
            "gold": q.truth,
            "n_detections": len(above),
            "fell_back": not dets,
            "detector_score": (s if s > float("-inf") else None),
            "raw_scores": [d[4] for d in dets],
            "runtime_ms": None,
        })
    return records
