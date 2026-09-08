"""Build table3_pipeline.csv (D4) and table4_cost.csv (D5).

**Table 3 does not report a single pooled accuracy**, because three of the six
question types have a constant gold label (D-042) and pooling them would produce
a number whose value is dominated by how many degenerate questions were included.
Each module is reported with the metric appropriate to it, the trivial baseline
that its label distribution admits, and the gold balance that makes that baseline
possible.

    python scripts/build_tables_34.py
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import NOT_COMPUTED, accuracy, compute_metrics

RAW = ROOT / "results" / "raw"
TABLES = ROOT / "results" / "tables"


# The full development attribute run. Pinned explicitly: selecting "the newest
# matching file" silently picked up a 200-record cache-verification run instead
# of the 3,858-record full-dev run.
FULL_DEV_N = 3858


def newest(pattern: str, expect_n: int | None = None):
    hits = sorted(glob.glob(str(RAW / pattern)), reverse=True)
    for path in hits:
        with open(path, encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh]
        if recs and (expect_n is None or len(recs) == expect_n):
            return recs, Path(path).name
    return None, None


def manifest_for(name: str):
    hits = sorted(glob.glob(str(RAW / f"{name}_dev_*.manifest.json")), reverse=True)
    if not hits:
        return {}
    with open(hits[0], encoding="utf-8") as fh:
        return json.load(fh)


def trivial_baseline(records):
    """Best constant-answer accuracy, and whether the labels are degenerate."""
    golds = [r["gold"] for r in records]
    if not golds:
        return NOT_COMPUTED, NOT_COMPUTED, NOT_COMPUTED
    yes = sum(1 for g in golds if g == "yes")
    n = len(golds)
    always_yes, always_no = yes / n, 1 - yes / n
    return max(always_yes, always_no), always_yes, always_no


def mean_runtime_ms(records):
    vals = [r["runtime_ms"] for r in records if r.get("runtime_ms") is not None]
    return (sum(vals) / len(vals)) if vals else NOT_COMPUTED


def build_table3():
    rows = []

    def add(module, metric_name, metric_value, records, note, extra=""):
        triv, ay, _an = trivial_baseline(records)
        m = compute_metrics(records)["overall"] if records else {}
        rows.append({
            "module": module,
            "n": m.get("n", NOT_COMPUTED),
            "primary_metric": metric_name,
            "primary_value": metric_value,
            "accuracy": m.get("accuracy", NOT_COMPUTED),
            "precision": m.get("precision", NOT_COMPUTED),
            "recall": m.get("recall", NOT_COMPUTED),
            "f1": m.get("f1", NOT_COMPUTED),
            "gold_yes_rate": ay,
            "best_trivial_baseline": triv,
            "labels_degenerate": (ay in (0.0, 1.0)),
            "mean_runtime_ms": mean_runtime_ms(records),
            "notes": note + (f" {extra}" if extra else ""),
        })

    # --- attribute: Cell D, the proposed method, split by sub-type ---
    D, dfile = newest("attribute_D_dev_*.jsonl", FULL_DEV_N)
    if D is None:
        raise SystemExit(
            f"no full-dev Cell D run with {FULL_DEV_N} records found; refusing to "
            "build Table 3 from a partial run"
        )
    if D:
        for sub, qt in (("state", "discriminative-attribute-state"),
                        ("action", "discriminative-attribute-action")):
            rs = [r for r in D if r.get("qtype") == qt]
            add(f"attribute-{sub} (Cell D)", "accuracy", accuracy(rs), rs,
                "proposed method; balanced labels, accuracy is meaningful.")
        add("attribute-ALL (Cell D)", "accuracy", accuracy(D), D,
            "proposed method, state+action pooled.", f"source={dfile}")

    # --- counting ---
    C, cfile = newest("counting_dev_*.jsonl")
    if C:
        man = manifest_for("counting")
        conf = man.get("count_confusion", {})
        add("counting", "accuracy", accuracy(C), C,
            "balanced labels (1036/1036); accuracy is meaningful.",
            f"exact_count_match={conf.get('exact_match_rate', NOT_COMPUTED)} "
            f"mean_abs_count_error={conf.get('mean_abs_error', NOT_COMPUTED)} source={cfile}")

    # --- existence: FPR, not accuracy (D-044) ---
    E, efile = newest("existence_dev_*.jsonl")
    if E:
        man = manifest_for("existence")
        sweep = man.get("fpr_sweep", {})
        thr = man.get("reported_threshold")
        key = f"{float(thr):.2f}" if thr is not None else None
        fpr = sweep.get(key, {}).get("fpr", NOT_COMPUTED) if key else NOT_COMPUTED
        add("existence", "false_positive_rate", fpr, E,
            "DEGENERATE LABELS: all gold 'no' (D-042). Accuracy is NOT "
            "interpretable; always-'no' scores 1.0000 without an image. "
            "Primary metric is the hallucination (false-positive) rate.",
            f"fpr_sweep={json.dumps(sweep)} source={efile}")

    # --- relation: per type AND pooled (D-045) ---
    R, rfile = newest("relation_dev_*.jsonl")
    if R:
        for qt in sorted({r["qtype"] for r in R}):
            rs = [r for r in R if r["qtype"] == qt]
            add(f"relation [{qt}]", "accuracy", accuracy(rs), rs,
                "DEGENERATE LABELS: this type has a constant gold answer "
                "(D-042/D-045); its trivial baseline is 1.0000.")
        add("relation [POOLED]", "accuracy", accuracy(R), R,
            "pooled over both relation types; the only relation set containing "
            "both classes. NEVER report pooled alone (D-045).", f"source={rfile}")

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "table3_pipeline.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows, out


# Measured module wall-clock, from the M4 run log. Existence is deliberately
# NOT_COMPUTED: the machine slept during that stage, so its elapsed time is not a
# measurement of anything. A contaminated timing is not reported as a clean one.
MODULE_WALLCLOCK = {
    "counting": {"seconds": 978, "questions": 1462, "clean": True},
    "existence": {"seconds": None, "questions": 3441, "clean": False,
                  "why": "host slept mid-stage; elapsed time is not a measurement"},
    "relation": {"seconds": 187, "questions": 1169, "clean": True},
}


def build_table4():
    """D5: parameter count, wall-clock per query, API cost = 0."""
    man = manifest_for("attribute_D") or {}
    models = man.get("models", {})
    gpu = man.get("gpu", NOT_COMPUTED)

    # Parameter counts measured at M1 load time, recorded here rather than
    # re-loading the models just to count them.
    PARAMS_M = {"google/owlv2-base-patch16-ensemble": 154.966792,
                "google/siglip-base-patch16-224": 203.16}

    rows = []
    for role, key in (("detector (localisation)", "detector"),
                      ("scorer (attribute verification)", "attribute")):
        mid = models.get(key, {}).get("id", NOT_COMPUTED)
        rows.append({
            "component": role,
            "model_id": mid,
            "revision": models.get(key, {}).get("revision", NOT_COMPUTED),
            "parameters_millions": PARAMS_M.get(mid, NOT_COMPUTED),
            "api_cost_usd": 0.0,
            "notes": "open weights, zero-shot, run locally",
        })

    total_params = sum(v for v in PARAMS_M.values())
    # Attribute cells record per-question timing directly.
    for name, pattern in (("attribute (Cell D, crop+forced choice)", "attribute_D_dev_*.jsonl"),
                          ("attribute (Cell A, whole image+threshold)", "attribute_A_dev_*.jsonl")):
        recs, src = newest(pattern, FULL_DEV_N)
        rows.append({
            "component": f"wall-clock: {name}",
            "model_id": "-", "revision": "-", "parameters_millions": "-",
            "api_cost_usd": 0.0,
            "notes": (f"mean_ms_per_question={mean_runtime_ms(recs) if recs else NOT_COMPUTED}"
                      f" (scoring only, detection cached); source={src}"),
        })

    # The other modules are timed from stage wall-clock.
    for name, info in MODULE_WALLCLOCK.items():
        if info["clean"]:
            ms = 1000.0 * info["seconds"] / info["questions"]
            note = (f"mean_ms_per_question={ms:.1f} "
                    f"({info['seconds']}s / {info['questions']} questions, "
                    "includes detection)")
        else:
            note = f"mean_ms_per_question={NOT_COMPUTED} -- {info['why']}"
        rows.append({
            "component": f"wall-clock: {name}",
            "model_id": "-", "revision": "-", "parameters_millions": "-",
            "api_cost_usd": 0.0, "notes": note,
        })

    rows.append({
        "component": "TOTAL",
        "model_id": "OWLv2-base + SigLIP-base",
        "revision": "-",
        "parameters_millions": total_params,
        "api_cost_usd": 0.0,
        "notes": f"no LLM, no paid API; peak VRAM 1.654 GiB of 4.00; GPU={gpu}",
    })

    out = TABLES / "table4_cost.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows, out


def main() -> int:
    rows3, out3 = build_table3()
    print("=== TABLE 3 — full pipeline (D4) ===")
    for r in rows3:
        pv = r["primary_value"]
        pv_s = pv if isinstance(pv, str) else f"{pv:.4f}"
        tb = r["best_trivial_baseline"]
        tb_s = tb if isinstance(tb, str) else f"{tb:.4f}"
        flag = "  <-- DEGENERATE LABELS" if r["labels_degenerate"] else ""
        print(f"  {r['module']:34s} n={r['n']:>5} {r['primary_metric']:>21s}="
              f"{pv_s}  trivial={tb_s}{flag}")
    print(f"  written: {out3.relative_to(ROOT)}")

    rows4, out4 = build_table4()
    print()
    print("=== TABLE 4 — cost (D5) ===")
    for r in rows4:
        print(f"  {r['component']:38s} params={r['parameters_millions']:>10} "
              f"api=${r['api_cost_usd']}  {r['notes'][:60]}")
    print(f"  written: {out4.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
