"""D-064 -- score the generative-VLM probe arms on the VAW test subsample.

Reads results/vaw/_vlm_batch.json (requests + metadata) and results/vaw/vlm_raw.jsonl
(the model's replies) and writes vlm_cells.csv / vlm_contrasts.csv / vlm_extras.json.
Replies are parsed strictly; anything unparseable is counted and excluded.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.bootstrap import bootstrap_metric, bootstrap_paired_difference

OUT = ROOT / "results" / "vaw"
LABEL = {"thr": "VLM-threshold: independent yes/no probing",
         "fc": "VLM-FC: pairwise forced choice",
         "prior": "VLM-prior: independent yes/no on a grey null image"}


def read_json(path, encoding="utf-8"):
    with open(path, encoding=encoding) as fh:
        return json.load(fh)


def write_json(path, obj, **kw):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, **kw)


def read_lines(path, encoding="utf-8"):
    with open(path, encoding=encoding) as fh:
        return fh.readlines()


def parse_yesno(t):
    if not t:
        return None
    w = re.findall(r"[a-z]+", t.strip().lower())
    if not w:
        return None
    return {"yes": "yes", "no": "no"}.get(w[0])


def parse_choice(t, opts):
    if not t:
        return None
    low = t.strip().lower()
    hits = [o for o in opts if re.search(rf"\b{re.escape(o.lower())}\b", low)]
    return hits[0] if len(hits) == 1 else None


def main():
    B = read_json(OUT / "_vlm_batch.json")
    meta = B["meta"]
    reply = {}
    for line in read_lines(OUT / "vlm_raw.jsonl"):
        r = json.loads(line)
        reply[r["i"]] = r
    cells = defaultdict(list)
    bad = Counter()
    models = Counter()
    for i, m in enumerate(meta):
        r = reply.get(i)
        if r is None:
            bad[m["arm"] + ":missing"] += 1
            continue
        models[r.get("model")] += 1
        rec = {"image": m["image"], "type": m["type"], "triple": tuple(m["triple"])}
        if m["arm"] in ("thr", "prior"):
            p = parse_yesno(r["text"])
            if p is None:
                bad[m["arm"] + ":unparsed"] += 1
                continue
            cells[m["arm"]].append({**rec, "id": m["id"], "pred": p, "gold": m["gold"], "attr": m["attr"]})
        else:
            w = parse_choice(r["text"], m["opts"])
            if w is None:
                bad["fc:unparsed"] += 1
                continue
            for attr, gold, qid in ((m["pos"], "yes", m["pos_id"]), (m["neg"], "no", m["neg_id"])):
                cells["fc"].append({**rec, "id": qid, "pred": "yes" if attr == w else "no",
                                    "gold": gold, "attr": attr})
    # Keep only pairs whose BOTH questions survived in EVERY arm, so the design
    # stays paired and all arms score exactly the same questions.
    key = lambda r: (r["image"], r["triple"])
    per_arm = []
    for a in ("thr", "fc", "prior"):
        c = Counter(key(r) for r in cells[a])
        per_arm.append({k for k, v in c.items() if v == 2})
    keep_pairs = set.intersection(*per_arm)
    for a in cells:
        cells[a] = [r for r in cells[a] if key(r) in keep_pairs]
    common = {r["id"] for r in cells["thr"]}

    def block(rs):
        n = len(rs)
        tp = sum(r["pred"] == "yes" and r["gold"] == "yes" for r in rs)
        fp = sum(r["pred"] == "yes" and r["gold"] == "no" for r in rs)
        fn = sum(r["pred"] == "no" and r["gold"] == "yes" for r in rs)
        tn = n - tp - fp - fn

        def prf(t, a, b):
            P = t / (t + a) if t + a else float("nan")
            R = t / (t + b) if t + b else float("nan")
            F = 2 * P * R / (P + R) if (P + R) else float("nan")
            return P, R, F
        Py, Ry, Fy = prf(tp, fp, fn)
        Ph, Rh, Fh = prf(tn, fn, fp)
        return {"n": n, "accuracy": (tp + tn) / n if n else float("nan"),
                "yes_ratio": (tp + fp) / n if n else float("nan"),
                "fpr": fp / (fp + tn) if fp + tn else float("nan"),
                "f1_yes": Fy, "H_precision": Ph, "H_recall": Rh, "H_f1": Fh,
                "NH_precision": Py, "NH_recall": Ry, "NH_f1": Fy, "mac_f1": (Fh + Fy) / 2}

    def pair_out(rs):
        by = defaultdict(dict)
        for r in rs:
            by[(r["image"], r["triple"])][r["gold"]] = r["pred"]
        c = Counter()
        m = {("yes", "no"): "correct", ("yes", "yes"): "both_yes",
             ("no", "no"): "both_no", ("no", "yes"): "reversed"}
        for v in by.values():
            if len(v) == 2:
                c[m[(v["yes"], v["no"])]] += 1
        return dict(c)

    rows = []
    for a in ("thr", "fc", "prior"):
        for t in ("all", "color", "material", "state"):
            rs = cells[a] if t == "all" else [r for r in cells[a] if r["type"] == t]
            b = block(rs)
            ci = bootstrap_metric(rs) if t == "all" else {"ci_low": "", "ci_high": "", "n_images": len({r["image"] for r in rs})}
            rows.append({"cell": a, "label": LABEL[a], "type": t, **b,
                         "acc_ci_low": ci["ci_low"], "acc_ci_high": ci["ci_high"],
                         "n_images": ci["n_images"],
                         **{f"pairs_{k}": v for k, v in pair_out(rs).items()}})
    con = []
    for x, y, what in (("fc", "thr", "decision rule, VLM scorer"),
                       ("thr", "prior", "visual evidence under independent probing"),
                       ("fc", "prior", "decision rule plus evidence")):
        for t in ("all", "color", "material", "state"):
            fa = (lambda rs, t=t: rs if t == "all" else [r for r in rs if r["type"] == t])
            d = bootstrap_paired_difference(fa(cells[y]), fa(cells[x]))
            con.append({"comparison": f"{x} minus {y}", "isolates": what, "type": t,
                        **{k: d[k] for k in ("point", "ci_low", "ci_high", "excludes_zero", "n_images")}})
    for name, data in (("vlm_cells.csv", rows), ("vlm_contrasts.csv", con)):
        keys = list(dict.fromkeys(k for r in data for k in r))
        with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
    write_json(OUT / "vlm_extras.json",
               {"unparsed_or_missing": dict(bad), "models": dict(models),
                "questions_scored": len(common), "pairs_scored": len(keep_pairs)}, indent=1)
    print(json.dumps({"bad": dict(bad), "scored": len(common)}))


if __name__ == "__main__":
    main()
