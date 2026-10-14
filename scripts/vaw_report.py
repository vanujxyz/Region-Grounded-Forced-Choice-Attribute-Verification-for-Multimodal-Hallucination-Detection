"""D-061 -- apply every pre-registered decision rule to the logged VAW logits.

All free parameters (tau, gamma, per-word tau) are fitted on the dev log and
frozen; the test log is only ever evaluated.  Run after vaw_run.py:

    python scripts/vaw_report.py            # uses the newest full dev + test logs
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.eval.bootstrap import (
    bootstrap_metric,
    bootstrap_paired_difference,
)
from src.modules.attribute import fit_tau, fit_tau_base_rate

OUT = ROOT / "results" / "vaw"
GAMMAS = [round(0.05 * k, 2) for k in range(41)]        # 0.00 .. 2.00
MIN_WORD_N = 10
TYPES = ("color", "material", "state")

THRESH_CELLS = {  # cell -> region
    "A": "full", "C": "crop", "Cstar": "gold",
}
FC_CELLS = {"B": "full", "D": "crop", "Dstar": "gold", "FCnull": "null"}
LABEL = {
    "A": "whole image + threshold (baseline)", "B": "whole image + forced choice",
    "C": "OWLv2 crop + threshold", "D": "OWLv2 crop + forced choice (proposed)",
    "Ap": "A-prime, base-rate-matched threshold", "Cp": "C-prime, base-rate-matched threshold",
    "CalNullA": "whole image + VISOR-style null-image calibration",
    "CalNullC": "OWLv2 crop + VISOR-style null-image calibration",
    "CalWordA": "whole image + per-word calibrated threshold",
    "CalWordC": "OWLv2 crop + per-word calibrated threshold",
    "Cstar": "gold box + threshold (oracle grounding)",
    "Dstar": "gold box + forced choice (oracle grounding)",
    "Dext": "OWLv2 crop + external antonym competitor",
    "WordPrior": "image-blind: dev majority label of the attribute word (D-063)",
    "FCnull": "image-blind: forced choice on the grey null image (D-063)",
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


def newest(split):
    files = [f for f in glob.glob(str(OUT / f"vaw_{split}_*.jsonl")) if "_limit" not in f]
    if not files:
        raise SystemExit(f"no full {split} log in {OUT}")
    return max(files)


def load(path):
    return [json.loads(line) for line in read_lines(path)]


def questions(pairs):
    """(pair, attr, gold, qid) for both questions of every pair."""
    for p in pairs:
        yield p, p["positive_attr"], "yes", p["positive_id"]
        yield p, p["negative_attr"], "no", p["negative_id"]


def rec(p, attr, gold, qid, cell, pred):
    return {"id": qid, "image": p["image"], "pred": pred, "gold": gold, "cell": cell,
            "type": p["type"], "attr": attr, "obj": p["obj"], "fell_back": p["fell_back"],
            "triple": (p["obj"], p["positive_attr"], p["negative_attr"])}


# ---------------------------------------------------------------- fitting ----
def fit_params(dev):
    prm = {}
    for cell, reg in THRESH_CELLS.items():
        sg = [(p["scores"][reg][a], g) for p, a, g, _ in questions(dev)]
        prm[cell] = {"tau": fit_tau(sg)[0], "dev_fit_acc": fit_tau(sg)[1]}
    for cell, reg in (("Ap", "full"), ("Cp", "crop")):
        sg = [(p["scores"][reg][a], g) for p, a, g, _ in questions(dev)]
        tau, info = fit_tau_base_rate(sg)
        prm[cell] = {"tau": tau, **info}
    for cell, reg in (("CalNullA", "full"), ("CalNullC", "crop")):
        best = None
        for gm in GAMMAS:
            sg = [(p["scores"][reg][a] - gm * p["scores"]["null"][a], g) for p, a, g, _ in questions(dev)]
            tau, acc = fit_tau(sg)
            if best is None or acc > best["dev_fit_acc"] + 1e-12:   # ties -> smaller gamma
                best = {"gamma": gm, "tau": tau, "dev_fit_acc": acc}
        prm[cell] = best
    for cell, reg in (("CalWordA", "full"), ("CalWordC", "crop")):
        allsg = [(p["scores"][reg][a], g) for p, a, g, _ in questions(dev)]
        glob_tau = fit_tau(allsg)[0]
        byw = defaultdict(list)
        for p, a, g, _ in questions(dev):
            byw[a].append((p["scores"][reg][a], g))
        taus = {w: fit_tau(v)[0] for w, v in byw.items() if len(v) >= MIN_WORD_N}
        prm[cell] = {"global_tau": glob_tau, "word_tau": taus, "min_word_n": MIN_WORD_N,
                     "n_words_fitted": len(taus)}
    byw = defaultdict(list)
    for p, a, g, _ in questions(dev):
        byw[a].append(g == "yes")
    prm["WordPrior"] = {"majority_yes": {w: bool(np.mean(v) >= 0.5) for w, v in byw.items() if len(v) >= MIN_WORD_N},
                        "fallback": "yes", "min_word_n": MIN_WORD_N}
    return prm


# --------------------------------------------------------------- decisions ---
def decide_all(pairs, prm):
    cells = defaultdict(list)
    ties = Counter()
    for p, a, g, qid in questions(pairs):
        S = p["scores"]
        for cell, reg in THRESH_CELLS.items():
            cells[cell].append(rec(p, a, g, qid, cell, "yes" if S[reg][a] >= prm[cell]["tau"] else "no"))
        for cell, reg in (("Ap", "full"), ("Cp", "crop")):
            cells[cell].append(rec(p, a, g, qid, cell, "yes" if S[reg][a] >= prm[cell]["tau"] else "no"))
        for cell, reg in (("CalNullA", "full"), ("CalNullC", "crop")):
            v = S[reg][a] - prm[cell]["gamma"] * S["null"][a]
            cells[cell].append(rec(p, a, g, qid, cell, "yes" if v >= prm[cell]["tau"] else "no"))
        for cell, reg in (("CalWordA", "full"), ("CalWordC", "crop")):
            t = prm[cell]["word_tau"].get(a, prm[cell]["global_tau"])
            cells[cell].append(rec(p, a, g, qid, cell, "yes" if S[reg][a] >= t else "no"))
        mj = prm["WordPrior"]["majority_yes"]
        cells["WordPrior"].append(rec(p, a, g, qid, "WordPrior", ("yes" if mj[a] else "no") if a in mj else "yes"))
        # D-ext: this question alone vs its antonym-map competitor, on the crop
        comp = p["competitor"].get(a)
        if comp:
            opts = sorted([a, comp])
            vals = [S["crop"][o] for o in opts]
            win = opts[vals.index(max(vals))]
            r = rec(p, a, g, qid, "Dext", "yes" if win == a else "no")
            r["competitor"] = comp
            cells["Dext"].append(r)
    # forced choice: one decision per pair, alphabetical options (D-006)
    for p in pairs:
        opts = sorted([p["positive_attr"], p["negative_attr"]])
        for cell, reg in FC_CELLS.items():
            vals = [p["scores"][reg][o] for o in opts]
            top = max(vals)
            ties[cell] += sum(v == top for v in vals) > 1
            win = opts[vals.index(top)]
            for a, g, qid in ((p["positive_attr"], "yes", p["positive_id"]),
                              (p["negative_attr"], "no", p["negative_id"])):
                cells[cell].append(rec(p, a, g, qid, cell, "yes" if a == win else "no"))
    return cells, ties


# ----------------------------------------------------------------- metrics ---
def block(rs):
    n = len(rs)
    tp = sum(r["pred"] == "yes" and r["gold"] == "yes" for r in rs)
    fp = sum(r["pred"] == "yes" and r["gold"] == "no" for r in rs)
    fn = sum(r["pred"] == "no" and r["gold"] == "yes" for r in rs)
    tn = sum(r["pred"] == "no" and r["gold"] == "no" for r in rs)

    def prf(t, f_p, f_n):
        P = t / (t + f_p) if t + f_p else float("nan")
        R = t / (t + f_n) if t + f_n else float("nan")
        F = 2 * P * R / (P + R) if (P + R) and not np.isnan(P + R) else float("nan")
        return P, R, F
    Py, Ry, Fy = prf(tp, fp, fn)            # non-hallucinated (NH) = gold yes
    Ph, Rh, Fh = prf(tn, fn, fp)            # hallucinated (H) = gold no
    return {"n": n, "accuracy": (tp + tn) / n if n else float("nan"),
            "yes_ratio": (tp + fp) / n if n else float("nan"),
            "fpr": fp / (fp + tn) if fp + tn else float("nan"),
            "f1_yes": Fy, "H_precision": Ph, "H_recall": Rh, "H_f1": Fh,
            "NH_precision": Py, "NH_recall": Ry, "NH_f1": Fy,
            "mac_f1": (Fh + Fy) / 2}


def pair_outcomes(rs):
    by = defaultdict(dict)
    for r in rs:
        by[r["triple"] + (r["image"],)][r["gold"]] = r["pred"]
    c = Counter()
    for v in by.values():
        if len(v) < 2:
            continue
        py, pn = v["yes"], v["no"]
        c[{("yes", "no"): "correct", ("yes", "yes"): "both_yes",
           ("no", "no"): "both_no", ("no", "yes"): "reversed"}[(py, pn)]] += 1
    return dict(c)


CONTRASTS = [
    ("B", "A", "decision rule alone (whole image)"),
    ("C", "A", "region grounding alone (threshold)"),
    ("D", "C", "decision rule given crop"),
    ("D", "B", "region grounding given forced choice"),
    ("D", "A", "HEADLINE: both"),
    ("B", "Ap", "forced choice net of base rate"),
    ("CalNullA", "A", "VISOR-style calibration (whole image)"),
    ("CalNullC", "C", "VISOR-style calibration (crop)"),
    ("CalWordC", "C", "per-word calibration (crop)"),
    ("B", "CalNullA", "forced choice vs calibration (whole image)"),
    ("D", "CalNullC", "forced choice vs calibration (crop)"),
    ("D", "CalWordC", "forced choice vs per-word calibration (crop)"),
    ("Cstar", "A", "oracle grounding alone (threshold)"),
    ("Dstar", "D", "oracle box vs detector box (forced choice)"),
    ("Dstar", "Cstar", "decision rule given oracle box"),
    ("CalWordC", "WordPrior", "per-word calibration net of the image-blind word prior"),
    ("D", "FCnull", "forced choice net of the scorer's text-only preference"),
    ("D", "WordPrior", "proposed vs image-blind word prior"),
]


def report(split, cells, ties, prm, tag):
    rows = []
    for cell, cell_label in LABEL.items():
        rs = cells[cell]
        b = block(rs)
        ci = bootstrap_metric(rs)
        rows.append({"cell": cell, "label": cell_label, "split": split, **b,
                     "acc_ci_low": ci["ci_low"], "acc_ci_high": ci["ci_high"],
                     "n_images": ci["n_images"], **{f"pairs_{k}": v for k, v in pair_outcomes(rs).items()}})
        for t in TYPES:
            bt = block([r for r in rs if r["type"] == t])
            rows.append({"cell": cell, "label": cell_label, "split": split, "type": t, **bt,
                         **{f"pairs_{k}": v for k, v in pair_outcomes([r for r in rs if r["type"] == t]).items()}})
    for r in rows:
        r.setdefault("type", "all")
    con = []
    for b_, a_, what in CONTRASTS:
        d = bootstrap_paired_difference(cells[a_], cells[b_])
        con.append({"comparison": f"{b_} minus {a_}", "isolates": what, "split": split, "type": "all",
                    **{k: d[k] for k in ("point", "ci_low", "ci_high", "excludes_zero", "n_images")}})
        for t in TYPES:
            d = bootstrap_paired_difference([r for r in cells[a_] if r["type"] == t],
                                            [r for r in cells[b_] if r["type"] == t])
            con.append({"comparison": f"{b_} minus {a_}", "isolates": what, "split": split, "type": t,
                        **{k: d[k] for k in ("point", "ci_low", "ci_high", "excludes_zero", "n_images")}})
    # D-ext on its covered subset, matched against A on the same questions
    cov = {r["id"] for r in cells["Dext"]}
    a_cov = [r for r in cells["A"] if r["id"] in cov]
    img_cov = {r["image"] for r in cells["Dext"]}
    dx = [r for r in cells["Dext"] if r["image"] in img_cov]
    d = bootstrap_paired_difference(a_cov, dx)
    con.append({"comparison": "Dext minus A (covered)", "isolates": "external contrast, matched subset",
                "split": split, "type": "all", **{k: d[k] for k in ("point", "ci_low", "ci_high", "excludes_zero", "n_images")}})
    acc = {r["cell"]: r["accuracy"] for r in rows if r["type"] == "all"}
    inter = (acc["D"] - acc["C"]) - (acc["B"] - acc["A"])
    extras = {"interaction": inter, "ties": dict(ties),
              "dext_coverage": len(cells["Dext"]) / len(cells["A"]),
              "fallback_rate": float(np.mean([r["fell_back"] for r in cells["D"]])),
              "source_log": tag}

    def dump(name, data):
        keys = list(dict.fromkeys(k for r in data for k in r))
        with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
    dump(f"vaw_{split}_cells.csv", rows)
    dump(f"vaw_{split}_contrasts.csv", con)
    write_json(OUT / f"vaw_{split}_extras.json", extras, indent=1)
    return rows, con, extras


def main():
    dev_log, test_log = newest("dev"), None
    dev = load(dev_log)
    prm = fit_params(dev)
    write_json(OUT / "vaw_fit_params.json", {"fitted_on": dev_log, **prm}, indent=1, default=float)
    cells, ties = decide_all(dev, prm)
    report("dev", cells, ties, prm, Path(dev_log).name)
    try:
        test_log = newest("test")
    except SystemExit:
        print("no test log yet; dev report only")
        return
    test = load(test_log)
    cells, ties = decide_all(test, prm)
    report("test", cells, ties, prm, Path(test_log).name)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
