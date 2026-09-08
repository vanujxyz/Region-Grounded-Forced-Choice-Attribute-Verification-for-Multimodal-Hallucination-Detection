"""One command that reproduces the M2 result end to end.

    python reproduce.py                 # limit 100, the M2 configuration
    python reproduce.py --limit 300     # a larger slice
    python reproduce.py --tables-only   # rebuild tables from existing raw results

Runs all six cells on the dev split, rebuilds every table, and prints the 2x2,
the attribution contrasts, and what has been ruled out.

It will not touch the test split. It fits no threshold on anything but dev.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.eval.bootstrap import (
    bootstrap_metric,
    bootstrap_paired_difference,
)
from src.eval.metrics import compute_metrics, confusion

NEWLINE = chr(10)

CELLS = ["A", "B", "C", "D", "Ap", "Cp"]
LABEL = {
    "A": "Whole image + threshold (baseline)",
    "B": "Whole image + forced choice",
    "C": "Cropped region + threshold",
    "D": "Cropped region + forced choice (proposed)",
    "Ap": "A' whole image + threshold, base-rate matched (D-020 control)",
    "Cp": "C' cropped region + threshold, base-rate matched (D-020 control)",
}
FREE_PARAMS = {"A": 1, "B": 0, "C": 1, "D": 0, "Ap": 1, "Cp": 1}
NOTE = {
    "A": "fit-on-eval, accuracy-maximising tau",
    "B": "no free parameter",
    "C": "fit-on-eval, accuracy-maximising tau",
    "D": "no free parameter",
    "Ap": "fit-on-eval, base-rate-matched tau (control, not part of the 2x2)",
    "Cp": "fit-on-eval, base-rate-matched tau (control, not part of the 2x2)",
}
CONTRASTS = [
    ("A", "C", "region grounding alone"),
    ("Ap", "Cp", "region grounding alone, base-rate matched"),
    ("A", "B", "forced choice alone (raw)"),
    ("Ap", "B", "forced choice alone, NET of base rate"),
    ("C", "D", "forced choice given cropping (raw)"),
    ("Cp", "D", "forced choice given cropping, NET of base rate"),
    ("B", "D", "cropping given forced choice"),
    ("A", "Ap", "cost of base-rate matching, whole image"),
    ("C", "Cp", "cost of base-rate matching, crop"),
    ("A", "D", "both (headline)"),
]


def _available(split: str) -> str:
    """Every raw run for this split, with its record count."""
    lines = []
    for c in CELLS:
        pat = str(ROOT / "results" / "raw" / f"attribute_{c}_{split}_*.jsonl")
        for path in sorted(glob.glob(pat)):
            with open(path, encoding="utf-8") as fh:
                k = sum(1 for _ in fh)
            lines.append(f"    {c:5s} {k:5d} records  {Path(path).name}")
    return NEWLINE.join(lines) or "    (none)"


def load_cell(cell: str, split: str, n: int | None = None):
    """Newest run for this cell with exactly ``n`` records.

    Taking simply "the newest file" is wrong once runs of different sizes exist:
    it silently pairs a full-dev Cell A with a limit-100 Cell D. The downstream
    id-set check catches that, but the fix belongs here.
    """
    pat = str(ROOT / "results" / "raw" / f"attribute_{cell}_{split}_*.jsonl")
    candidates = []
    for path in sorted(glob.glob(pat), reverse=True):
        with open(path, encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh]
        if not recs:
            continue
        if n is not None:
            if len(recs) == n:
                return recs
        else:
            candidates.append(recs)
    if n is not None or not candidates:
        return None
    # No size requested: take the LARGEST run, i.e. the full split. Taking the
    # newest instead would pair a full-dev cell with a small ad-hoc run.
    return max(candidates, key=len)


def run_cells(split: str, limit: int | None) -> None:
    for cell in CELLS:
        cmd = [sys.executable, "-m", "src.run", "--cell", cell, "--split", split]
        if limit:
            cmd += ["--limit", str(limit)]
        print(f"  running cell {cell} ...", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            print(r.stderr[-3000:])
            raise SystemExit(f"cell {cell} failed with exit code {r.returncode}")
        print(f"    done in {time.perf_counter() - t0:.1f}s", flush=True)


def build_tables(split: str, n: int | None = None) -> dict:
    R = {c: load_cell(c, split, n) for c in CELLS}
    missing = [c for c, v in R.items() if v is None]
    if missing:
        want = f"exactly {n} records" if n else "any size"
        raise SystemExit(
            f"no {split} run with {want} for cells {missing}."
            + NEWLINE + f"Available {split} runs:" + NEWLINE + _available(split)
            + NEWLINE + "Use --limit to pick a size (--limit 0 = full split)."
        )

    id_sets = {frozenset(r["id"] for r in v) for v in R.values()}
    if len(id_sets) != 1:
        raise SystemExit("cells were scored on different question id sets; refusing to compare")

    tables = ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    rows = []
    for c in CELLS:
        m = compute_metrics(R[c])
        o, cf, b = m["overall"], confusion(R[c]), bootstrap_metric(R[c])
        rows.append({
            "cell": c, "label": LABEL[c], "n": o["n"],
            "n_distinct_triples": o["n_distinct_triples"],
            "n_distinct_images": o["n_distinct_images"], "accuracy": o["accuracy"],
            "ci_low": b["ci_low"], "ci_high": b["ci_high"], "precision": o["precision"],
            "recall": o["recall"], "f1": o["f1"], "fallback_rate": m["fallback_rate"],
            "predicted_yes": cf["tp"] + cf["fp"], "gold_yes": cf["tp"] + cf["fn"],
            "tp": cf["tp"], "fp": cf["fp"], "fn": cf["fn"], "tn": cf["tn"],
            "free_params_fitted_on_eval": FREE_PARAMS[c], "notes": NOTE[c],
        })

    from src.modules.position_baseline import LABEL as POS_LABEL
    from src.modules.position_baseline import predict
    from src.run import select_pairs

    P = predict(select_pairs(split, None))
    keep = {r["id"] for r in R["A"]}
    P = [r for r in P if r["id"] in keep]
    for r in P:
        r["qtype"] = "discriminative-attribute-state"
    pm, pcf, pb = compute_metrics(P), confusion(P), bootstrap_metric(P)
    rows.append({
        "cell": "position-only", "label": POS_LABEL, "n": pm["overall"]["n"],
        "n_distinct_triples": pm["overall"]["n_distinct_triples"],
        "n_distinct_images": pm["overall"]["n_distinct_images"],
        "accuracy": pm["overall"]["accuracy"], "ci_low": pb["ci_low"],
        "ci_high": pb["ci_high"], "precision": pm["overall"]["precision"],
        "recall": pm["overall"]["recall"], "f1": pm["overall"]["f1"],
        "fallback_rate": pm["fallback_rate"],
        "predicted_yes": pcf["tp"] + pcf["fp"], "gold_yes": pcf["tp"] + pcf["fn"],
        "tp": pcf["tp"], "fp": pcf["fp"], "fn": pcf["fn"], "tn": pcf["tn"],
        "free_params_fitted_on_eval": 0,
        "notes": "BENCHMARK ARTIFACT: exploits AMBER pair id ordering; sees no image",
    })
    with (tables / "table1_ablation.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    comps = []
    with (tables / "table1_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["comparison", "isolates", "point", "mean", "ci_low", "ci_high",
                    "excludes_zero", "n_images", "n_resamples", "split", "protocol"])
        for a, b, lab in CONTRASTS:
            d = bootstrap_paired_difference(R[a], R[b])
            w.writerow([f"{b} minus {a}", lab, d["point"], d["mean"], d["ci_low"],
                        d["ci_high"], d["excludes_zero"], d["n_images"], d["n_resamples"],
                        split, "fit-on-eval where tau present"])
            comps.append((f"{b} minus {a}", lab, d))

    with (tables / "table2_subtype.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "subtype", "n", "n_distinct_triples", "accuracy",
                    "precision", "recall", "f1"])
        for c in CELLS:
            for k, v in compute_metrics(R[c])["by_subtype"].items():
                w.writerow([c, k, v["n"], v["n_distinct_triples"], v["accuracy"],
                            v["precision"], v["recall"], v["f1"]])

    return {"R": R, "rows": rows, "comps": comps}


def report(res: dict, split: str) -> None:
    rows, comps = res["rows"], res["comps"]
    acc = {r["cell"]: r["accuracy"] for r in rows}
    n = rows[0]["n"]
    tri = rows[0]["n_distinct_triples"]
    img = rows[0]["n_distinct_images"]

    print()
    print("=" * 78)
    print(f"  RESULTS -- split={split}  n={n} questions  "
          f"{tri} distinct triples  {img} images")
    print("=" * 78)
    print()
    print("  THE 2x2")
    print("                        | threshold        | forced choice")
    print(f"    whole image         | A  {acc['A']:.4f}       | B  {acc['B']:.4f}")
    print(f"    cropped region      | C  {acc['C']:.4f}       | D  {acc['D']:.4f}")
    print()
    print("  CONTROLS")
    print(f"    A' base-rate matched  {acc['Ap']:.4f}")
    print(f"    C' base-rate matched  {acc['Cp']:.4f}")
    print(f"    position-only         {acc['position-only']:.4f}   "
          "<-- BENCHMARK ARTIFACT, sees no image")
    print()
    print("  ATTRIBUTION -- paired bootstrap over images, 10,000 resamples, seed 20260907")
    for name, lab, d in comps:
        flag = "EXCLUDES ZERO" if d["excludes_zero"] else "includes zero"
        print(f"    {name:12s} {lab:42s} {d['point']:+.4f}  "
              f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  {flag}")
    inter = (acc["D"] - acc["A"]) - (acc["B"] - acc["A"]) - (acc["C"] - acc["A"])
    print(f"    {'interaction':12s} {'(D-A) - (B-A) - (C-A)':42s} {inter:+.4f}")
    print()
    print("  RULED OUT")
    print("    - Id leakage. AMBER puts the true attribute at the lower question id in")
    print("      2774/2774 pairs, so a position-only detector scores 1.0000 without an")
    print("      image. Under a permuted-id diagnostic it collapses to ~0.50 while cells")
    print("      A and D are bit-identical. No cell reads an id.  [D-009, D-023]")
    print("    - Balanced-pairs confound. Forced choice emits one yes per pair and AMBER's")
    print("      pairs are exactly balanced, so it gets a correct base rate free. Cells")
    print("      A'/C' hand the threshold cells the same prior: the gap WIDENS, it does")
    print("      not close. The prior is not the source of the advantage.  [D-020, D-025]")
    print()
    print("  LIMITS")
    fb = next(r["fallback_rate"] for r in rows if r["cell"] == "D")
    print(f"    - Detector fallback rate {fb}. On those questions no object was found and")
    print("      the crop IS the full image, so Cell D degenerates to Cell B there.")
    print(f"    - {n} questions rest on only {tri} distinct (object, positive, negative)")
    print("      triples; ('sky','sunny','gloomy') alone recurs heavily. Effective sample")
    print("      is far smaller than n.  [D-007]")
    print(f"    - Bootstrap resamples {img} images, not {n} questions.")
    print("    - Every number here is DEV and fit-on-eval where a threshold is involved:")
    print("      cells A/C/A'/C' each fit tau on the data they are scored on. That favours")
    print("      the BASELINE, so the comparison is conservative. The headline result is")
    print("      the M5 test-split run with tau frozen from the dev fit.  [D-012]")
    print("    - Confidences are uncalibrated and are not a claim of this work.  [D-011]")
    print("    - 340-object AMBER vocabulary, English only, zero-shot, no training.")
    print()
    print("  Tables written to results/tables/")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce the attribute-verification result.")
    ap.add_argument("--limit", type=int, default=100,
                    help="pairs to use (default 100, the M2 configuration); 0 = full split")
    ap.add_argument("--split", default="dev", choices=["dev"],
                    help="dev only; the test split is opened once, at M5")
    ap.add_argument("--tables-only", action="store_true",
                    help="rebuild tables from existing raw results without re-running models")
    ap.add_argument("--list-runs", action="store_true",
                    help="list available raw runs with their sizes, then exit")
    args = ap.parse_args()
    limit = None if args.limit == 0 else args.limit
    expected_n = (2 * limit) if limit else None

    if args.list_runs:
        print(f"Available {args.split} runs:")
        print(_available(args.split))
        return 0

    if not args.tables_only:
        print(f"Running all six cells on split={args.split} limit={limit or 'FULL'} ...")
        run_cells(args.split, limit)
    report(build_tables(args.split, expected_n), args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
