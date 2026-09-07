"""TRD §11.4 / §14 — an independent second implementation of the metrics.

Written directly from the TRD §11.1--11.2 specification text, deliberately not
importing anything from ``src.eval.metrics``.  Both implementations are run on
20,000 randomly generated prediction sets and asserted to agree exactly.

Independence caveat (recorded in docs/DECISIONS.md): both implementations were
written by the same author in the same session.  The cross-check catches
transcription slips and edge-case divergence -- it cannot catch a shared
misreading of the specification.
"""

import random

from src.eval.metrics import compute_metrics

SEED = 20260907
N_CASES = 20_000

_NC = "NOT_COMPUTED"
_CELLS = ["A", "B", "C", "D"]
_QTYPES = [
    "discriminative-attribute-state",
    "discriminative-attribute-action",
    "discriminative-attribute-number",
    "discriminative-hallucination",  # has no sub-type; must be excluded from by_subtype
    "discriminative-relation",
]
_SUBTYPE = {
    "discriminative-attribute-state": "state",
    "discriminative-attribute-action": "action",
    "discriminative-attribute-number": "number",
}


# --------------------------------------------------------------------------
# Reference implementation, from the specification.
# --------------------------------------------------------------------------
def ref_accuracy(records):
    """"Accuracy over individual question ids." """
    n = len(records)
    if n == 0:
        return _NC
    hits = 0
    for r in records:
        if r["pred"] == r["gold"]:
            hits += 1
    return hits / n


def ref_prf(records):
    """"precision, recall and F1 treating yes as the positive class." """
    tp = fp = fn = 0
    for r in records:
        p, g = r["pred"], r["gold"]
        if p == "yes" and g == "yes":
            tp += 1
        elif p == "yes" and g == "no":
            fp += 1
        elif p == "no" and g == "yes":
            fn += 1

    prec = _NC if (tp + fp) == 0 else tp / (tp + fp)
    rec = _NC if (tp + fn) == 0 else tp / (tp + fn)
    if prec == _NC or rec == _NC or (prec + rec) == 0:
        f1 = _NC
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1}


def ref_block(records):
    block = {"n": len(records), "accuracy": ref_accuracy(records)}
    block.update(ref_prf(records))
    return block


def ref_fallback_rate(records):
    if len(records) == 0:
        return _NC
    for r in records:
        if "fell_back" not in r:
            return _NC
    hits = 0
    for r in records:
        if r["fell_back"]:
            hits += 1
    return hits / len(records)


def ref_compute_metrics(records):
    report = {"overall": ref_block(records), "fallback_rate": ref_fallback_rate(records)}

    by_cell = {}
    for r in records:
        if r.get("cell") is not None:
            by_cell.setdefault(r["cell"], []).append(r)
    report["by_cell"] = {k: ref_block(v) for k, v in sorted(by_cell.items())}

    by_sub = {}
    for r in records:
        sub = _SUBTYPE.get(r.get("qtype"))
        if sub is not None:
            by_sub.setdefault(sub, []).append(r)
    report["by_subtype"] = {k: ref_block(v) for k, v in sorted(by_sub.items())}

    by_fb = {}
    for r in records:
        if "fell_back" in r:
            by_fb.setdefault(bool(r["fell_back"]), []).append(r)
    report["by_fallback"] = {
        str(k).lower(): ref_block(v) for k, v in sorted(by_fb.items())
    }
    return report


# --------------------------------------------------------------------------
# Random case generation
# --------------------------------------------------------------------------
def _random_case(rng):
    """One prediction set.  Sizes include 0 and 1; label distributions include
    the degenerate all-yes / all-no cases that make metrics undefined."""
    n = rng.choice([0, 1, 1, 2, 3, rng.randint(4, 24)])
    # Skewed label priors so undefined-metric branches are exercised often.
    p_pred_yes = rng.choice([0.0, 1.0, 0.5, rng.random()])
    p_gold_yes = rng.choice([0.0, 1.0, 0.5, rng.random()])
    include_fb = rng.random() < 0.75
    # Sometimes only *some* records carry fell_back, so the field is partial.
    partial_fb = include_fb and rng.random() < 0.2

    records = []
    for i in range(n):
        r = {
            "id": i,
            "image": f"AMBER_{rng.randint(1, 5)}.jpg",
            "pred": "yes" if rng.random() < p_pred_yes else "no",
            "gold": "yes" if rng.random() < p_gold_yes else "no",
        }
        if rng.random() < 0.85:
            r["cell"] = rng.choice(_CELLS)
        if rng.random() < 0.85:
            r["qtype"] = rng.choice(_QTYPES)
        if include_fb and (not partial_fb or rng.random() < 0.5):
            r["fell_back"] = rng.random() < 0.3
        records.append(r)
    return records


def test_dual_implementations_agree_on_20000_random_cases():
    rng = random.Random(SEED)
    checked = 0
    for case_i in range(N_CASES):
        records = _random_case(rng)
        got = compute_metrics(records)
        want = ref_compute_metrics(records)
        assert got == want, (
            f"implementations diverged on case {case_i} "
            f"(n={len(records)}):\n  metrics.py: {got}\n  reference : {want}\n"
            f"  records: {records}"
        )
        checked += 1
    assert checked == N_CASES


def test_generator_actually_exercises_the_edge_cases():
    """Guard against a generator that never produces undefined metrics."""
    rng = random.Random(SEED)
    seen_empty = seen_nc_prec = seen_nc_rec = seen_nc_f1 = seen_nc_fb = 0
    seen_acc_zero = seen_acc_one = 0
    for _ in range(N_CASES):
        records = _random_case(rng)
        m = ref_compute_metrics(records)
        o = m["overall"]
        if o["n"] == 0:
            seen_empty += 1
        if o["precision"] == _NC:
            seen_nc_prec += 1
        if o["recall"] == _NC:
            seen_nc_rec += 1
        if o["f1"] == _NC:
            seen_nc_f1 += 1
        if m["fallback_rate"] == _NC:
            seen_nc_fb += 1
        if o["accuracy"] == 0.0:
            seen_acc_zero += 1
        if o["accuracy"] == 1.0:
            seen_acc_one += 1
    for name, count in [
        ("empty sets", seen_empty), ("undefined precision", seen_nc_prec),
        ("undefined recall", seen_nc_rec), ("undefined f1", seen_nc_f1),
        ("undefined fallback", seen_nc_fb), ("accuracy 0.0", seen_acc_zero),
        ("accuracy 1.0", seen_acc_one),
    ]:
        assert count > 0, f"generator never produced: {name}"
