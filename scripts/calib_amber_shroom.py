"""D-062/D-063 -- calibration baselines and image-blind diagnostics on AMBER and
SHROOM-Vis, computed from the EXISTING raw logs (no model re-run except the
grey null-image text scores in results/vaw/null_scores_amber_shroom.json).

POST-HOC: these test splits were opened before this script existed.  Nothing
here touches cells A-D; parameters are fitted on each dataset's dev log.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.eval.bootstrap import (
    bootstrap_metric,
    bootstrap_paired_difference,
)

GAMMAS = [round(0.05 * k, 2) for k in range(41)]
MIN_WORD_N = 10
PROMPT = "a photo of a {attr} {obj}"

LOGS = {
    "amber": {
        "dev": {"A": "attribute_A_dev_20260907T162523Z", "C": "attribute_C_dev_20260907T163257Z",
                "B": "attribute_B_dev_20260907T163010Z", "D": "attribute_D_dev_20260907T170000Z"},
        "test": {"A": "attribute_A_test_20260908T031553Z", "C": "attribute_C_test_20260908T031830Z",
                 "B": "attribute_B_test_20260908T031716Z", "D": "attribute_D_test_20260908T032934Z"},
    },
    "shroom": {
        "dev": {c: f"attribute_{c}_shroom-dev_20260913T082730Z" for c in "ABCD"},
        "test": {c: f"attribute_{c}_shroom-test_20260913T103325Z" for c in "ABCD"},
    },
}


def read_json(path, encoding="utf-8"):
    with open(path, encoding=encoding) as fh:
        return json.load(fh)


def write_json(path, obj, **kw):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, **kw)


def read_lines(path, encoding="utf-8"):
    with open(path, encoding=encoding) as fh:
        return fh.readlines()


def load(raw_dir, stem):
    return [json.loads(x) for x in read_lines(Path(raw_dir) / f"{stem}.jsonl")]


def own(r):
    opts = sorted([r["triple"][1], r["triple"][2]])
    return r["raw_scores"][opts.index(r["attr"])]


def fit_tau(sg):
    vals = sorted({s for s, _ in sg})
    cands = [vals[0] - 1] + [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)] + [vals[-1] + 1]
    S = np.array([s for s, _ in sg])
    G = np.array([g == "yes" for _, g in sg])
    best = (cands[0], -1.0)
    for t in cands:
        acc = float(((S >= t) == G).mean())
        if acc > best[1]:
            best = (t, acc)
    return best


def as_rec(r, pred, cell):
    return {"id": r["id"], "image": r["image"], "pred": pred, "gold": r["gold"], "cell": cell}


def run(raw_dir, null_path, out_dir):
    null = read_json(null_path)
    ns = lambda r: null[PROMPT.format(attr=r["attr"], obj=r["triple"][0])]
    out_rows, con_rows, params = [], [], {}
    for ds, sp in LOGS.items():
        L = {(s, c): load(raw_dir, stem) for s in sp for c, stem in sp[s].items()}
        cells = {}
        for c in "ABCD":
            cells[c] = [as_rec(r, r["pred"], c) for r in L[("test", c)]]
        for c, reg in (("A", "A"), ("C", "C")):
            dv, ts = L[("dev", reg)], L[("test", reg)]
            # Cal-null
            best = None
            for g in GAMMAS:
                tau, acc = fit_tau([(own(r) - g * ns(r), r["gold"]) for r in dv])
                if best is None or acc > best[2] + 1e-12:
                    best = (g, tau, acc)
            g, tau, _ = best
            params[f"{ds}_CalNull{c}"] = {"gamma": g, "tau": tau}
            cells[f"CalNull{c}"] = [as_rec(r, "yes" if own(r) - g * ns(r) >= tau else "no", f"CalNull{c}") for r in ts]
            # Cal-word
            gt = fit_tau([(own(r), r["gold"]) for r in dv])[0]
            byw = defaultdict(list)
            for r in dv:
                byw[r["attr"]].append((own(r), r["gold"]))
            wt = {w: fit_tau(v)[0] for w, v in byw.items() if len(v) >= MIN_WORD_N}
            params[f"{ds}_CalWord{c}"] = {"n_words": len(wt), "global_tau": gt}
            cells[f"CalWord{c}"] = [as_rec(r, "yes" if own(r) >= wt.get(r["attr"], gt) else "no", f"CalWord{c}") for r in ts]
        # WordPrior
        byw = defaultdict(list)
        for r in L[("dev", "A")]:
            byw[r["attr"]].append(r["gold"] == "yes")
        mj = {w: bool(np.mean(v) >= 0.5) for w, v in byw.items() if len(v) >= MIN_WORD_N}
        cells["WordPrior"] = [as_rec(r, ("yes" if mj[r["attr"]] else "no") if r["attr"] in mj else "yes", "WordPrior")
                              for r in L[("test", "A")]]
        # FC-null: forced choice on the grey image
        fc = []
        for r in L[("test", "A")]:
            o, p_, n_ = r["triple"]
            opts = sorted([p_, n_])
            vals = [null[PROMPT.format(attr=a, obj=o)] for a in opts]
            win = opts[vals.index(max(vals))]
            fc.append(as_rec(r, "yes" if r["attr"] == win else "no", "FCnull"))
        cells["FCnull"] = fc
        for c, rs in cells.items():
            b = bootstrap_metric(rs)
            yes = np.mean([x["pred"] == "yes" for x in rs])
            fpr = np.mean([x["pred"] == "yes" for x in rs if x["gold"] == "no"])
            out_rows.append({"dataset": ds, "cell": c, "n": len(rs), "accuracy": b["point"],
                             "ci_low": b["ci_low"], "ci_high": b["ci_high"], "yes_ratio": yes, "fpr": fpr})
        for x, y in (("CalNullA", "A"), ("CalNullC", "C"), ("CalWordA", "A"), ("CalWordC", "C"),
                     ("B", "CalNullA"), ("D", "CalNullC"), ("B", "CalWordA"), ("D", "CalWordC"),
                     ("CalWordC", "WordPrior"), ("D", "WordPrior"), ("D", "FCnull")):
            d = bootstrap_paired_difference(cells[y], cells[x])
            con_rows.append({"dataset": ds, "comparison": f"{x} minus {y}",
                             **{k: d[k] for k in ("point", "ci_low", "ci_high", "excludes_zero", "n_images")}})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("calib_cells.csv", out_rows), ("calib_contrasts.csv", con_rows)):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    write_json(out / "calib_params.json", params, indent=1)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
