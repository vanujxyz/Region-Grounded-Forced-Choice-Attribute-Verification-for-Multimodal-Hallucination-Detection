"""D-033 — matched-subset comparison for Cell D-ext.

D-ext does not cover every question (state 94.79%, action 79.29%), so comparing
it against Cell D on their raw record sets would confound the method with the
subset. This script reports the three numbers D-033 requires, plus the state and
action breakout, and writes results/tables/table5_dext.csv.

    python scripts/dext_report.py                # limit-100 runs
    python scripts/dext_report.py --n 3858       # full dev runs

Written before any D-ext result existed.
"""

from __future__ import annotations

import argparse
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
    NOT_COMPUTED,
    compute_metrics,
    confusion,
)

SUBTYPE = {
    "discriminative-attribute-state": "state",
    "discriminative-attribute-action": "action",
}


def load(cell: str, n: int | None):
    """Raw records for a cell, choosing the run whose size matches ``n``."""
    hits = sorted(glob.glob(str(ROOT / "results" / "raw" / f"attribute_{cell}_dev_*.jsonl")),
                  reverse=True)
    for path in hits:
        with open(path, encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh]
        if n is None or len(recs) == n:
            return recs, Path(path).name
    return None, None


def restrict(records, ids):
    return [r for r in records if r["id"] in ids]


def by_subtype(records, sub):
    return [r for r in records if SUBTYPE.get(r.get("qtype")) == sub]


def line(label, records):
    if not records:
        return {"label": label, "n": 0, "accuracy": NOT_COMPUTED, "precision": NOT_COMPUTED,
                "recall": NOT_COMPUTED, "f1": NOT_COMPUTED, "predicted_yes": NOT_COMPUTED,
                "gold_yes": NOT_COMPUTED, "ci_low": NOT_COMPUTED, "ci_high": NOT_COMPUTED}
    m = compute_metrics(records)["overall"]
    cf = confusion(records)
    b = bootstrap_metric(records)
    return {
        "label": label, "n": m["n"], "accuracy": m["accuracy"],
        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
        "predicted_yes": cf["tp"] + cf["fp"], "gold_yes": cf["tp"] + cf["fn"],
        "ci_low": b["ci_low"], "ci_high": b["ci_high"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="record count identifying the run (e.g. 200 or 3858)")
    args = ap.parse_args()

    D, dfile = load("D", args.n)
    A, afile = load("A", args.n)
    if D is None or A is None:
        raise SystemExit(f"missing Cell D or Cell A results for n={args.n}")
    # D-ext is smaller than the others by construction; find it by its own size.
    Dext, xfile = load("Dext", None)
    if Dext is None:
        raise SystemExit("no Cell D-ext results found; run --cell Dext first")

    covered = {r["id"] for r in Dext}
    all_ids = {r["id"] for r in D}
    if not covered <= all_ids:
        raise SystemExit(
            f"D-ext scored {len(covered - all_ids)} ids absent from Cell D; "
            "the two runs used different pair selections"
        )

    print(f"Cell D    : {dfile}  ({len(D)} records)")
    print(f"Cell A    : {afile}  ({len(A)} records)")
    print(f"Cell D-ext: {xfile}  ({len(Dext)} records)")
    print()
    print("=" * 78)
    print("  D-033 MATCHED-SUBSET COMPARISON")
    print("=" * 78)
    cov = len(covered) / len(all_ids) if all_ids else NOT_COMPUTED
    print(f"  covered subset: {len(covered)}/{len(all_ids)} questions = {cov:.4f}")
    print()

    rows = []
    blocks = [
        ("ALL", D, A, Dext, all_ids),
    ]
    for sub in ("state", "action"):
        blocks.append((sub, by_subtype(D, sub), by_subtype(A, sub),
                       by_subtype(Dext, sub), {r["id"] for r in by_subtype(D, sub)}))

    for name, d_all, a_all, x, ids in blocks:
        sub_covered = {r["id"] for r in x}
        d_cov = restrict(d_all, sub_covered)
        a_cov = restrict(a_all, sub_covered)
        n_ids = len(ids)
        rate = (len(sub_covered) / n_ids) if n_ids else NOT_COMPUTED

        print(f"  --- {name} ---")
        print(f"    coverage: {len(sub_covered)}/{n_ids} = "
              f"{rate if rate == NOT_COMPUTED else f'{rate:.4f}'}")
        for label, recs in (
            ("D    on ALL pairs      ", d_all),
            ("D    on covered subset ", d_cov),
            ("D-ext on covered subset", x),
            ("A    on covered subset ", a_cov),
        ):
            r = line(f"{name}: {label.strip()}", recs)
            rows.append({"subset": name, **r, "coverage_rate": rate})
            acc = r["accuracy"]
            acc_s = acc if acc == NOT_COMPUTED else f"{acc:.4f}"
            print(f"    {label}  n={r['n']:5d}  acc={acc_s}  "
                  f"yes={r['predicted_yes']}/{r['gold_yes']} gold-yes")

        if d_cov and x:
            for lab, base in (("D-ext - D (covered)", d_cov), ("D-ext - A (covered)", a_cov)):
                if not base:
                    continue
                dd = bootstrap_paired_difference(base, x)
                flag = "EXCLUDES ZERO" if dd["excludes_zero"] else "includes zero"
                print(f"      {lab:22s} {dd['point']:+.4f}  "
                      f"[{dd['ci_low']:+.4f}, {dd['ci_high']:+.4f}]  {flag}")
        print()

    out = ROOT / "results" / "tables" / "table5_dext.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  written: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
