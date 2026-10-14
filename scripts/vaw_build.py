"""D-061 -- build the VAW contrastive-pair benchmark (see data/vaw/VAW_PROTOCOL.md).

Reads VAW annotation JSON (adobe-research/vaw_dataset) and writes
data/vaw/pairs_{dev,test}.json.  Pure stdlib; reads no pixels.

    python scripts/vaw_build.py --vaw-dir <path to vaw_dataset/data>
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

SEED = 20260907
TYPES = ("color", "material", "state")
QUOTA = {"color": 600, "material": 500, "state": 400}
MIN_BOX = 20.0
SAMPLE_ORDER = ("state", "material", "color")


def read_json(path, encoding="utf-8"):
    with open(path, encoding=encoding) as fh:
        return json.load(fh)


def write_json(path, obj, **kw):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, **kw)


def read_lines(path, encoding="utf-8"):
    with open(path, encoding=encoding) as fh:
        return fh.readlines()


def eligible_pairs(instances, attr_types, rng):
    """One candidate pair per (instance, type). Deterministic given rng state."""
    per_img_obj = Counter((i["image_id"], i["object_name"]) for i in instances)
    out = {t: [] for t in TYPES}
    for ins in sorted(instances, key=lambda r: r["instance_id"]):
        w, h = ins["instance_bbox"][2], ins["instance_bbox"][3]
        if w < MIN_BOX or h < MIN_BOX:
            continue
        if per_img_obj[(ins["image_id"], ins["object_name"])] != 1:
            continue
        for t in TYPES:
            S = attr_types[t]
            P = sorted(a for a in set(ins["positive_attributes"]) if a in S)
            N = sorted(a for a in set(ins["negative_attributes"]) if a in S and a not in P)
            if not P or not N:
                continue
            out[t].append({
                "image_id": str(ins["image_id"]), "instance_id": str(ins["instance_id"]),
                "obj": ins["object_name"], "type": t,
                "positive_attr": rng.choice(P), "negative_attr": rng.choice(N),
                "bbox_xywh": [float(v) for v in ins["instance_bbox"]],
            })
    return out


def sample(cands_by_source, rng, used_images):
    """Fill each type quota, sources in priority order, max one pair per image."""
    chosen = []
    for t in SAMPLE_ORDER:      # scarcest type first, so it is not starved

        need = QUOTA[t]
        for src, cands in cands_by_source:
            pool = [c for c in cands[t] if c["image_id"] not in used_images]
            rng.shuffle(pool)
            for c in pool:
                if need == 0:
                    break
                if c["image_id"] in used_images:
                    continue
                c = dict(c, source=src)
                chosen.append(c)
                used_images.add(c["image_id"])
                need -= 1
        if need:
            raise AssertionError(f"type {t}: quota short by {need}")
    return chosen


def assign_ids(pairs, rng, start):
    qid = start
    for p in pairs:
        ids = [qid, qid + 1]
        if rng.random() < 0.5:          # D-054: no position artifact
            ids.reverse()
        p["positive_id"], p["negative_id"] = ids
        qid += 2
    return qid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vaw-dir", required=True)
    ap.add_argument("--out", default="data/vaw")
    a = ap.parse_args()
    d = Path(a.vaw_dir)
    attr_types = {k: set(v) for k, v in read_json(d / "attribute_types.json").items()}

    def load(f):
        return read_json(d / f)

    rng = random.Random(SEED)

    test_c = eligible_pairs(load("test.json"), attr_types, rng)
    val_c = eligible_pairs(load("val.json"), attr_types, rng)
    tr_c = eligible_pairs(load("train_part1.json"), attr_types, rng)
    tr2_c = eligible_pairs(load("train_part2.json"), attr_types, rng)

    test = sample([("vaw_test", test_c)], rng, set())
    test_imgs = {p["image_id"] for p in test}
    dev = sample([("vaw_val", val_c), ("vaw_train_part1", tr_c), ("vaw_train_part2", tr2_c)], rng, set(test_imgs))
    # sample() seeded used_images with test images, so dev cannot reuse them
    assert not (test_imgs & {p["image_id"] for p in dev})

    nxt = assign_ids(dev, rng, 1)
    assign_ids(test, rng, nxt)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, pairs in (("dev", dev), ("test", test)):
        write_json(out / f"pairs_{name}.json", pairs, indent=0)
        summary[name] = {
            "pairs": len(pairs), "questions": 2 * len(pairs),
            "images": len({p["image_id"] for p in pairs}),
            "by_type": dict(Counter(p["type"] for p in pairs)),
            "by_source": dict(Counter(p["source"] for p in pairs)),
            "distinct_objects": len({p["obj"] for p in pairs}),
            "distinct_triples": len({(p["obj"], p["positive_attr"], p["negative_attr"]) for p in pairs}),
            "positive_lower_id_rate": sum(p["positive_id"] < p["negative_id"] for p in pairs) / len(pairs),
        }
    summary["eligible_candidates"] = {
        "test": {t: len(v) for t, v in test_c.items()},
        "val": {t: len(v) for t, v in val_c.items()},
        "train_part1": {t: len(v) for t, v in tr_c.items()},
        "train_part2": {t: len(v) for t, v in tr2_c.items()},
    }
    write_json(out / "build_summary.json", summary, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
