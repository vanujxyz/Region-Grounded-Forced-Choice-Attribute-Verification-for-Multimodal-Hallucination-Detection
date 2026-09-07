"""TRD §9 — counting.

For `discriminative-attribute-number`. Parsed with NUMBER_RE (TRD §4). Number
questions do not pair reliably (1,506 singletons vs 283 pairs, measured), so they
are routed here rather than to the attribute module's forced choice.

Closing D1 and D3: these are the 2,072 questions the M3 gate deferred (D-029).

    dets = detect(image, obj)          # after NMS at IoU 0.5
    answer = "yes" if len(dets) == claimed_number else "no"
"""

from __future__ import annotations

from typing import Any

# TRD §9: the complete measured set of number words. Explicit dict; anything
# outside it raises rather than being guessed.
NUMBER_WORDS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}

EXPECTED_COUNT = 2072


def word_to_int(word: str) -> int:
    """Map a number word to an integer. Raises on anything outside TRD §9's set."""
    key = word.strip().lower()
    if key not in NUMBER_WORDS:
        raise ValueError(
            f"unknown number word {word!r}; TRD §9 fixes the set to "
            f"{sorted(NUMBER_WORDS)}. A new word means the data changed -- "
            "do not guess it."
        )
    return NUMBER_WORDS[key]


def load_number_questions():
    """The 2,072 number questions, parsed and count-resolved."""
    from src.data.pairs import load_number_questions as _load

    qs = _load()
    if len(qs) != EXPECTED_COUNT:
        raise AssertionError(f"expected {EXPECTED_COUNT} number questions, got {len(qs)}")
    for q in qs:
        word_to_int(q.num_word)  # raises early on an unknown word
    return qs


class CountConfusion:
    """TRD §9: log the confusion between predicted and claimed counts.

    Useful analysis material -- it separates "the detector miscounted" from
    "the claim was false", which raw accuracy cannot.
    """

    def __init__(self) -> None:
        self.cells: dict[tuple[int, int], int] = {}
        self.abs_err_sum = 0
        self.n = 0

    def record(self, claimed: int, predicted: int) -> None:
        self.cells[(claimed, predicted)] = self.cells.get((claimed, predicted), 0) + 1
        self.abs_err_sum += abs(predicted - claimed)
        self.n += 1

    def as_dict(self) -> dict[str, Any]:
        if not self.n:
            return {"n": 0, "mean_abs_error": "NOT_COMPUTED", "exact_match_rate": "NOT_COMPUTED",
                    "predicted_zero_rate": "NOT_COMPUTED", "confusion": {}}
        exact = sum(v for (c, p), v in self.cells.items() if c == p)
        zeros = sum(v for (_, p), v in self.cells.items() if p == 0)
        return {
            "n": self.n,
            "mean_abs_error": self.abs_err_sum / self.n,
            "exact_match_rate": exact / self.n,
            "predicted_zero_rate": zeros / self.n,
            "confusion": {f"claimed{c}_pred{p}": v for (c, p), v in sorted(self.cells.items())},
        }


def predict(
    questions, detections_for, confusion: CountConfusion | None = None
) -> list[dict[str, Any]]:
    """Answer each number question by comparing detection count to claim.

    ``detections_for(image, obj)`` returns the post-NMS detection list. Injected
    so this module is testable without loading OWLv2.
    """
    confusion = confusion if confusion is not None else CountConfusion()
    records = []
    for q in questions:
        claimed = word_to_int(q.num_word)
        dets = detections_for(q.image, q.obj)
        predicted = len(dets)
        confusion.record(claimed, predicted)
        records.append({
            "id": q.id,
            "image": q.image,
            "qtype": "discriminative-attribute-number",
            "obj": q.obj,
            "attr": q.num_word,
            "cell": "counting",
            "pred": "yes" if predicted == claimed else "no",
            # No calibrated confidence exists here; the decision is an integer
            # comparison. Reported as the normalised closeness of the counts.
            "confidence": 1.0 / (1.0 + abs(predicted - claimed)),
            "gold": q.truth,
            "claimed_count": claimed,
            "predicted_count": predicted,
            "fell_back": predicted == 0,
            "detector_score": (max(d[4] for d in dets) if dets else None),
            "raw_scores": [d[4] for d in dets],
            "runtime_ms": None,
        })
    return records
