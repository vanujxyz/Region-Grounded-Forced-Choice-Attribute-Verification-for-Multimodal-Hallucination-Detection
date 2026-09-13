"""The SHROOM-Vis report: does the AMBER result replicate on a second dataset?

    python scripts/shroom_report.py dev
    python scripts/shroom_report.py test      (needs ALLOW_TEST_SPLIT=1 to have
                                               been set when the cells were run)

Reuses the same metrics and image-level paired bootstrap as the AMBER report, so
the two datasets' numbers are computed by identical code. Adds two breakdowns
that only make sense here:

* **in / out of AMBER's vocabulary** — SHROOM-Vis deliberately names objects
  outside AMBER's closed 340-object list, so generalisation beyond that
  vocabulary is separable rather than averaged away.
* **the position-only artifact** — 1.0000 on AMBER, ~0.5 here.

Writes results/tables/shroom_<split>.csv and shroom_<split>_comparison.csv.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import shroom
from src.eval.bootstrap import (
    bootstrap_metric,
    bootstrap_paired_difference,
)
from src.eval.metrics import accuracy, compute_metrics, confusion

RAW = ROOT / "results" / "raw"
TABLES = ROOT / "results" / "tables"

CELLS = ["A", "B", "C", "D"]
DIAG = ["Ap", "Cp"]
LABEL = {
    "A": "whole image + threshold (baseline)",
    "B": "whole image + forced choice",
    "C": "cropped region + threshold",
    "D": "cropped region + forced choice (PROPOSED)",
    "Ap": "A-prime, base-rate-matched threshold",
    "Cp": "C-prime, base-rate-matched threshold",
}
CONTRASTS = [
    ("A", "C", "region grounding alone"),
    ("A", "B", "forced choice alone"),
    ("Ap", "B", "forced choice, net of base rate"),
    ("C", "D", "forced choice given cropping"),
    ("Cp", "D", "forced choice given cropping, net"),
    ("B", "D", "cropping given forced choice"),
    ("A", "D", "HEADLINE: both"),
]

# The AMBER test-split numbers this is being compared against (README.md).
AMBER_TEST = {"A": 0.5751, "B": 0.7787, "C": 0.5953, "D": 0.8012}
AMBER_TEST_DELTA = {
    "C minus A": (+0.0201, +0.0012, +0.0391),
    "B minus A": (+0.2036, +0.1781, +0.2288),
    "D minus C": (None, None, None),
    "D minus B": (+0.0225, -0.0012, +0.0462),
    "D minus A": (+0.2260, +0.2001, +0.2514),
}


def load(cell: str, split: str):
    pat = str(RAW / f"attribute_{cell}_shroom-{split}_*.jsonl")
    hits = sorted(glob.glob(pat), reverse=True)
    if not hits:
        return None
    with open(hits[0], encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def manifest(cell: str, split: str) -> dict:
    pat = str(RAW / f"attribute_{cell}_shroom-{split}_*.manifest.json")
    hits = sorted(glob.glob(pat), reverse=True)
    if not hits:
        return {}
    with open(hits[0], encoding="utf-8") as fh:
        return json.load(fh)


def position_only(split: str):
    pat = str(RAW / f"position-only_shroom-{split}_*.jsonl")
    hits = sorted(glob.glob(pat), reverse=True)
    if not hits:
        return None
    with open(hits[0], encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def vocab_flags() -> dict[int, bool]:
    """question id -> whether its object is in AMBER's 340-object vocabulary."""
    out: dict[int, bool] = {}
    for m in shroom.load_pair_meta().values():
        out[m["positive_id"]] = m["in_amber_vocab"]
        out[m["negative_id"]] = m["in_amber_vocab"]
    return out


def main(split: str) -> int:
    R = {c: load(c, split) for c in CELLS}
    missing = [c for c, v in R.items() if v is None]
    if missing:
        raise SystemExit(f"missing SHROOM {split} runs for cells {missing}")
    for c in DIAG:
        got = load(c, split)
        if got is not None:
            R[c] = got

    ids = {frozenset(r["id"] for r in v) for v in R.values()}
    if len(ids) != 1:
        raise SystemExit("cells were scored on different question sets")

    cells = [c for c in CELLS + DIAG if c in R]
    n = len(R["A"])
    m0 = compute_metrics(R["A"])["overall"]

    print("=" * 78)
    print(f"  SHROOM-Vis — {split.upper()} split")
    print("  Same method, same frozen constants, a second dataset.")
    print("=" * 78)
    print(f"  n={n} questions  {m0['n_distinct_triples']} triples  "
          f"{m0['n_distinct_images']} images")
    po = position_only(split)
    if po:
        print(f"  position-only baseline: {accuracy(po):.4f}   "
              "(AMBER: 1.0000 — the id-ordering artifact is absent here)")
    print()
    print("  THE 2x2")
    print("                        | threshold        | forced choice")
    print(f"    whole image         | A  {accuracy(R['A']):.4f}       | B  {accuracy(R['B']):.4f}")
    print(f"    cropped region      | C  {accuracy(R['C']):.4f}       | D  {accuracy(R['D']):.4f}")
    if "Ap" in R:
        print(f"    controls: A' {accuracy(R['Ap']):.4f}   C' {accuracy(R['Cp']):.4f}")
    print()

    rows = []
    for c in cells:
        mm = compute_metrics(R[c])["overall"]
        b = bootstrap_metric(R[c])
        cf = confusion(R[c])
        tau = manifest(c, split).get("tau")
        rows.append({
            "cell": c, "label": LABEL[c], "n": mm["n"],
            "n_distinct_triples": mm["n_distinct_triples"],
            "n_distinct_images": mm["n_distinct_images"],
            "accuracy": mm["accuracy"], "ci_low": b["ci_low"], "ci_high": b["ci_high"],
            "precision": mm["precision"], "recall": mm["recall"], "f1": mm["f1"],
            "fallback_rate": compute_metrics(R[c])["fallback_rate"],
            "predicted_yes": cf["tp"] + cf["fp"], "gold_yes": cf["tp"] + cf["fn"],
            "tau": tau.get("tau") if isinstance(tau, dict) else "NOT_APPLICABLE",
            "tau_protocol": tau.get("protocol") if isinstance(tau, dict) else "NOT_APPLICABLE",
            "amber_test_accuracy": AMBER_TEST.get(c, "NOT_APPLICABLE"),
            "split": split, "dataset": "shroom",
        })

    print("  PER CELL  (AMBER test accuracy in the last column, for reference)")
    for r in rows:
        amb = r["amber_test_accuracy"]
        amb_s = f"{amb:.4f}" if isinstance(amb, float) else "    --"
        print(f"    {r['cell']:3s} acc={r['accuracy']:.4f} "
              f"CI[{r['ci_low']:.4f},{r['ci_high']:.4f}] "
              f"yes={r['predicted_yes']}/{r['gold_yes']}  AMBER {amb_s}")
    print()

    comps = []
    print("  ATTRIBUTION — paired bootstrap over images, 10,000 resamples")
    print("    contrast                                 SHROOM            "
          "         AMBER test")
    for a, b_, lab in CONTRASTS:
        if a not in R or b_ not in R:
            continue
        d = bootstrap_paired_difference(R[a], R[b_])
        flag = "EXCLUDES ZERO" if d["excludes_zero"] else "includes zero"
        key = f"{b_} minus {a}"
        amb = AMBER_TEST_DELTA.get(key, (None, None, None))
        amb_s = (f"{amb[0]:+.4f} [{amb[1]:+.4f},{amb[2]:+.4f}]"
                 if amb[0] is not None else "--")
        print(f"    {b_} - {a:2s} {lab:30s} {d['point']:+.4f} "
              f"[{d['ci_low']:+.4f},{d['ci_high']:+.4f}] {flag:13s} {amb_s}")
        comps.append({
            "comparison": key, "isolates": lab, "point": d["point"], "mean": d["mean"],
            "ci_low": d["ci_low"], "ci_high": d["ci_high"],
            "excludes_zero": d["excludes_zero"], "n_images": d["n_images"],
            "n_resamples": d["n_resamples"], "split": split, "dataset": "shroom",
            "amber_test_point": amb[0] if amb[0] is not None else "NOT_APPLICABLE",
            "amber_test_ci_low": amb[1] if amb[1] is not None else "NOT_APPLICABLE",
            "amber_test_ci_high": amb[2] if amb[2] is not None else "NOT_APPLICABLE",
        })
    print()

    # ---- D-ext: external antonym competitor (D-032) ------------------------
    dext = load("Dext", split)
    if dext:
        covered = {r["id"] for r in dext}
        dcov = [r for r in R["D"] if r["id"] in covered]
        acov = [r for r in R["A"] if r["id"] in covered]
        print("  D-ext — competitor from configs/antonyms.yaml, never from the pair")
        print(f"    coverage {len(covered)}/{n} = {len(covered) / n:.4f}   "
              "(map committed for AMBER, before any SHROOM result)")
        print(f"    A on covered {accuracy(acov):.4f}   "
              f"D on covered {accuracy(dcov):.4f}   "
              f"D-ext {accuracy(dext):.4f}")
        for lab, base in (("D-ext - A (covered)", acov), ("D-ext - D (covered)", dcov)):
            d = bootstrap_paired_difference(base, dext)
            flag = "EXCLUDES ZERO" if d["excludes_zero"] else "includes zero"
            print(f"    {lab:22s} {d['point']:+.4f} "
                  f"[{d['ci_low']:+.4f},{d['ci_high']:+.4f}]  {flag}")
            comps.append({
                "comparison": lab, "isolates": "external contrast, matched subset",
                "point": d["point"], "mean": d["mean"], "ci_low": d["ci_low"],
                "ci_high": d["ci_high"], "excludes_zero": d["excludes_zero"],
                "n_images": d["n_images"], "n_resamples": d["n_resamples"],
                "split": split, "dataset": "shroom",
                "amber_test_point": "NOT_APPLICABLE",
                "amber_test_ci_low": "NOT_APPLICABLE",
                "amber_test_ci_high": "NOT_APPLICABLE",
            })
        print()

    # ---- in / out of AMBER's 340-object vocabulary -------------------------
    flags = vocab_flags()
    print("  BY AMBER VOCABULARY MEMBERSHIP of the object")
    print("    subset          n     A       B       C       D      D-A")
    for name, want in (("in vocab", True), ("out of vocab", False)):
        sub = {c: [r for r in R[c] if flags.get(r["id"]) is want] for c in CELLS}
        if not sub["A"]:
            continue
        d = bootstrap_paired_difference(sub["A"], sub["D"])
        print(f"    {name:14s} {len(sub['A']):5d} "
              + "  ".join(f"{accuracy(sub[c]):.4f}" for c in CELLS)
              + f"  {d['point']:+.4f} [{d['ci_low']:+.4f},{d['ci_high']:+.4f}]")
        comps.append({
            "comparison": f"D minus A ({name})", "isolates": "headline, vocab subset",
            "point": d["point"], "mean": d["mean"], "ci_low": d["ci_low"],
            "ci_high": d["ci_high"], "excludes_zero": d["excludes_zero"],
            "n_images": d["n_images"], "n_resamples": d["n_resamples"],
            "split": split, "dataset": "shroom",
            "amber_test_point": "NOT_APPLICABLE",
            "amber_test_ci_low": "NOT_APPLICABLE",
            "amber_test_ci_high": "NOT_APPLICABLE",
        })
    print()

    # ---- by question kind --------------------------------------------------
    print("  BY QUESTION KIND")
    print("    kind            n     A       B       C       D")
    for kind in ("state", "action"):
        qt = f"discriminative-attribute-{kind}"
        sub = {c: [r for r in R[c] if r.get("qtype") == qt] for c in CELLS}
        if not sub["A"]:
            continue
        print(f"    {kind:14s} {len(sub['A']):5d} "
              + "  ".join(f"{accuracy(sub[c]):.4f}" for c in CELLS))
    print()

    TABLES.mkdir(parents=True, exist_ok=True)
    p1 = TABLES / f"shroom_{split}.csv"
    p2 = TABLES / f"shroom_{split}_comparison.csv"
    with p1.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with p2.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comps[0].keys()))
        w.writeheader()
        w.writerows(comps)
    print(f"  written: {p1.relative_to(ROOT)}, {p2.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "dev"))
