"""M5 — the test-split report. The headline result of the project.

Every threshold cell uses tau FROZEN from the development fit (TRD §16); nothing
is fitted here. The test split was opened exactly once.

    python scripts/test_report.py

Writes results/tables/table5_test.csv and table5_test_comparison.csv.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.bootstrap import (
    bootstrap_metric,
    bootstrap_paired_difference,
)
from src.eval.metrics import (
    accuracy,
    compute_metrics,
    confusion,
)

RAW = ROOT / "results" / "raw"
TABLES = ROOT / "results" / "tables"

CELLS = ["A", "B", "C", "D", "Ap", "Cp"]
LABEL = {
    "A": "Whole image + threshold (baseline)",
    "B": "Whole image + forced choice",
    "C": "Cropped region + threshold",
    "D": "Cropped region + forced choice (PROPOSED)",
    "Ap": "A-prime, base-rate-matched threshold",
    "Cp": "C-prime, base-rate-matched threshold",
    "Dext": "D-ext, external antonym competitor",
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


def load(cell: str):
    hits = sorted(glob.glob(str(RAW / f"attribute_{cell}_test_*.jsonl")), reverse=True)
    if not hits:
        return None, None
    with open(hits[0], encoding="utf-8") as fh:
        return [json.loads(line) for line in fh], Path(hits[0]).name


def manifest(cell: str):
    hits = sorted(glob.glob(str(RAW / f"attribute_{cell}_test_*.manifest.json")), reverse=True)
    if not hits:
        return {}
    with open(hits[0], encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    R = {c: load(c)[0] for c in CELLS}
    missing = [c for c, v in R.items() if v is None]
    if missing:
        raise SystemExit(f"missing test runs for cells {missing}")

    ids = {frozenset(r["id"] for r in v) for v in R.values()}
    if len(ids) != 1:
        raise SystemExit("cells were scored on different test question sets")

    Dext, _ = load("Dext")

    print("=" * 78)
    print("  M5 TEST SPLIT — opened once. Thresholds FROZEN from dev, nothing fitted.")
    print("=" * 78)
    n = len(R["A"])
    m0 = compute_metrics(R["A"])["overall"]
    print(f"  n={n} questions  {m0['n_distinct_triples']} triples  "
          f"{m0['n_distinct_images']} images")
    print()
    print("  THE 2x2")
    print("                        | threshold        | forced choice")
    print(f"    whole image         | A  {accuracy(R['A']):.4f}       | B  {accuracy(R['B']):.4f}")
    print(f"    cropped region      | C  {accuracy(R['C']):.4f}       | D  {accuracy(R['D']):.4f}")
    print(f"    controls: A' {accuracy(R['Ap']):.4f}   C' {accuracy(R['Cp']):.4f}")
    if Dext:
        print(f"    D-ext (external contrast, n={len(Dext)}): {accuracy(Dext):.4f}")
    print()

    rows = []
    for c in CELLS + (["Dext"] if Dext else []):
        recs = Dext if c == "Dext" else R[c]
        mm = compute_metrics(recs)["overall"]
        b = bootstrap_metric(recs)
        cf = confusion(recs)
        man = manifest(c)
        tau = man.get("tau")
        tau_val = tau.get("tau") if isinstance(tau, dict) else NOT_APPLICABLE_STR
        proto = tau.get("protocol") if isinstance(tau, dict) else "NOT_APPLICABLE"
        rows.append({
            "cell": c, "label": LABEL[c], "n": mm["n"],
            "n_distinct_triples": mm["n_distinct_triples"],
            "n_distinct_images": mm["n_distinct_images"],
            "accuracy": mm["accuracy"], "ci_low": b["ci_low"], "ci_high": b["ci_high"],
            "precision": mm["precision"], "recall": mm["recall"], "f1": mm["f1"],
            "fallback_rate": compute_metrics(recs)["fallback_rate"],
            "predicted_yes": cf["tp"] + cf["fp"], "gold_yes": cf["tp"] + cf["fn"],
            "tau": tau_val, "tau_protocol": proto, "split": "test",
        })

    print("  PER CELL")
    for r in rows:
        print(f"    {r['cell']:5s} acc={r['accuracy']:.4f} "
              f"CI[{r['ci_low']:.4f},{r['ci_high']:.4f}] "
              f"yes={r['predicted_yes']}/{r['gold_yes']} tau={r['tau']}")
    print()

    comps = []
    print("  ATTRIBUTION — paired bootstrap over test images, 10,000 resamples")
    for a, b, lab in CONTRASTS:
        d = bootstrap_paired_difference(R[a], R[b])
        flag = "EXCLUDES ZERO" if d["excludes_zero"] else "includes zero"
        print(f"    {b} - {a:2s}  {lab:34s} {d['point']:+.4f} "
              f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  {flag}")
        comps.append({"comparison": f"{b} minus {a}", "isolates": lab, "point": d["point"],
                      "mean": d["mean"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                      "excludes_zero": d["excludes_zero"], "n_images": d["n_images"],
                      "n_resamples": d["n_resamples"], "split": "test",
                      "protocol": "tau frozen from dev; nothing fitted on test"})

    if Dext:
        covered = {r["id"] for r in Dext}
        dcov = [r for r in R["D"] if r["id"] in covered]
        acov = [r for r in R["A"] if r["id"] in covered]
        print()
        print(f"  D-ext MATCHED SUBSET  coverage {len(covered)}/{n} = {len(covered) / n:.4f}")
        print(f"    D     on covered  {accuracy(dcov):.4f}")
        print(f"    D-ext on covered  {accuracy(Dext):.4f}")
        print(f"    A     on covered  {accuracy(acov):.4f}")
        for lab, base in (("D-ext - A (covered)", acov), ("D-ext - D (covered)", dcov)):
            d = bootstrap_paired_difference(base, Dext)
            flag = "EXCLUDES ZERO" if d["excludes_zero"] else "includes zero"
            print(f"    {lab:22s} {d['point']:+.4f} "
                  f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  {flag}")
            comps.append({"comparison": lab, "isolates": "external contrast, matched subset",
                          "point": d["point"], "mean": d["mean"], "ci_low": d["ci_low"],
                          "ci_high": d["ci_high"], "excludes_zero": d["excludes_zero"],
                          "n_images": d["n_images"], "n_resamples": d["n_resamples"],
                          "split": "test", "protocol": "matched covered subset"})

    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / "table5_test.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (TABLES / "table5_test_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comps[0].keys()))
        w.writeheader()
        w.writerows(comps)
    print()
    print("  written: results/tables/table5_test.csv, table5_test_comparison.csv")
    return 0


NOT_APPLICABLE_STR = "NOT_APPLICABLE"

if __name__ == "__main__":
    raise SystemExit(main())
