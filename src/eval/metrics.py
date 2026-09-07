"""TRD §11 — evaluation metrics.

Written and unit-tested before any model is connected (PRD §8 rule 3): a scorer
written after the fact gets shaped to flatter the system.

A metric that cannot be computed is reported as the string ``NOT_COMPUTED``,
never as 0 or a placeholder (PRD §8 rule 1).
"""

from __future__ import annotations

from typing import Any, Iterable

NOT_COMPUTED = "NOT_COMPUTED"

YES = "yes"
NO = "no"
VALID_LABELS = (YES, NO)

# qtype -> sub-type, for the TRD §11.2 breakdown.
SUBTYPE_OF_QTYPE = {
    "discriminative-attribute-state": "state",
    "discriminative-attribute-action": "action",
    "discriminative-attribute-number": "number",
}

Record = dict[str, Any]
Metric = float | str


def _validate(records: Iterable[Record]) -> list[Record]:
    """Reject malformed records loudly rather than scoring them (PRD §8 rule 2)."""
    out = list(records)
    seen: set[int] = set()
    for i, r in enumerate(out):
        for key in ("id", "pred", "gold"):
            if key not in r:
                raise ValueError(f"record {i} is missing required key {key!r}: {r}")
        if r["pred"] not in VALID_LABELS:
            raise ValueError(
                f"record id={r['id']} has pred={r['pred']!r}; expected one of {VALID_LABELS}"
            )
        if r["gold"] not in VALID_LABELS:
            raise ValueError(
                f"record id={r['id']} has gold={r['gold']!r}; expected one of {VALID_LABELS}"
            )
        if r["id"] in seen:
            raise ValueError(
                f"duplicate question id {r['id']}: accuracy is defined over "
                "individual question ids, so each must appear exactly once"
            )
        seen.add(r["id"])
    return out


def accuracy(records: Iterable[Record]) -> Metric:
    """TRD §11.1 primary metric: accuracy over individual question ids."""
    rs = _validate(records)
    if not rs:
        return NOT_COMPUTED
    correct = sum(1 for r in rs if r["pred"] == r["gold"])
    return correct / len(rs)


def confusion(records: Iterable[Record]) -> dict[str, int]:
    """Counts with ``yes`` as the positive class."""
    rs = _validate(records)
    tp = sum(1 for r in rs if r["pred"] == YES and r["gold"] == YES)
    fp = sum(1 for r in rs if r["pred"] == YES and r["gold"] == NO)
    fn = sum(1 for r in rs if r["pred"] == NO and r["gold"] == YES)
    tn = sum(1 for r in rs if r["pred"] == NO and r["gold"] == NO)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(records: Iterable[Record]) -> dict[str, Metric]:
    """TRD §11.1: precision, recall and F1 treating ``yes`` as positive.

    Each is NOT_COMPUTED when its denominator is zero -- an undefined metric is
    not a zero one.
    """
    c = confusion(records)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]

    precision: Metric = tp / (tp + fp) if (tp + fp) > 0 else NOT_COMPUTED
    recall: Metric = tp / (tp + fn) if (tp + fn) > 0 else NOT_COMPUTED

    if precision == NOT_COMPUTED or recall == NOT_COMPUTED:
        f1: Metric = NOT_COMPUTED
    elif (precision + recall) == 0:
        f1 = NOT_COMPUTED
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def fallback_rate(records: Iterable[Record]) -> Metric:
    """TRD §11.2: the share of records whose detection fell back to the full image.

    This number goes in the paper; it is never hidden.
    """
    rs = _validate(records)
    if not rs:
        return NOT_COMPUTED
    if any("fell_back" not in r for r in rs):
        return NOT_COMPUTED
    return sum(1 for r in rs if r["fell_back"]) / len(rs)


def _group_block(records: list[Record]) -> dict[str, Metric | int]:
    block: dict[str, Metric | int] = {"n": len(records), "accuracy": accuracy(records)}
    block.update(precision_recall_f1(records))
    return block


def _group_by(records: list[Record], key) -> dict[Any, list[Record]]:
    out: dict[Any, list[Record]] = {}
    for r in records:
        out.setdefault(key(r), []).append(r)
    return out


def compute_metrics(records: Iterable[Record]) -> dict[str, Any]:
    """The full TRD §11.1--11.2 report: overall plus every required breakdown."""
    rs = _validate(records)

    report: dict[str, Any] = {"overall": _group_block(rs)}
    report["fallback_rate"] = fallback_rate(rs)

    # by cell (A/B/C/D)
    report["by_cell"] = {
        cell: _group_block(group)
        for cell, group in sorted(
            _group_by([r for r in rs if r.get("cell") is not None], lambda r: r["cell"]).items()
        )
    }

    # by sub-type: state / action / number
    subtyped = [r for r in rs if SUBTYPE_OF_QTYPE.get(r.get("qtype")) is not None]
    report["by_subtype"] = {
        sub: _group_block(group)
        for sub, group in sorted(
            _group_by(subtyped, lambda r: SUBTYPE_OF_QTYPE[r["qtype"]]).items()
        )
    }

    # by whether detection fell back to the full image
    with_fb = [r for r in rs if "fell_back" in r]
    report["by_fallback"] = {
        str(bool(flag)).lower(): _group_block(group)
        for flag, group in sorted(
            _group_by(with_fb, lambda r: bool(r["fell_back"])).items()
        )
    }

    return report
