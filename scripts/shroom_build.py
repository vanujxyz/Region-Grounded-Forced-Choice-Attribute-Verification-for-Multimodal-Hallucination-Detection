"""Turn the raw SHROOM annotation log into AMBER-schema benchmark files.

Input   data/shroom/annotations_raw.jsonl   one JSON object per annotated image
Output  data/shroom/annotations.json        AMBER schema, joined by id-1
        data/shroom/query/query_all.json    AMBER schema
        data/shroom/pair_meta.json          per-pair provenance (vocab, kind)

Every rule in data/shroom/ANNOTATION_POLICY.md that can be checked mechanically
is checked here, and a violation aborts the build rather than being repaired.

The one substantive choice this script makes is id ordering *within* a pair.
AMBER gives the true attribute the lower id in all 2,774 of its pairs, which
makes "answer yes to the lower id" a perfect strategy that never opens an
image. Here a seeded coin flip decides, so position carries no signal.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHROOM = ROOT / "data" / "shroom"
RAW = SHROOM / "annotations_raw.jsonl"
IMAGES = ROOT / "shroom-visions-images" / "shroom-vis-images"

SEED = 20260907
KINDS = {
    "state": "discriminative-attribute-state",
    "action": "discriminative-attribute-action",
}
TEMPLATE = {
    "state": "Is the {obj} {attr} in this image?",
    "action": "Does the {obj} {attr} in this image?",
}

# The two regexes src/data/pairs.py will parse the result with. Building against
# them here means a template typo fails at build time, not at run time.
STATE_RE = re.compile(r"^Is the (?P<obj>.+?) (?P<attr>.+?) in this image\?$")
ACTION_RE = re.compile(r"^Does the (?P<obj>.+?) (?P<attr>.+?) in this image\?$")
PARSE_RE = {"state": STATE_RE, "action": ACTION_RE}

WORD = re.compile(r"^[a-z]([a-z' ]*[a-z])?$")


class BuildError(Exception):
    pass


def _fail(where: str, msg: str) -> None:
    raise BuildError(f"{where}: {msg}")


def load_raw() -> list[dict]:
    if not RAW.exists():
        raise FileNotFoundError(f"no raw annotation log at {RAW}")
    records: list[dict] = []
    with RAW.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                _fail(f"{RAW.name}:{lineno}", f"not valid JSON: {exc}")
            f = rec.get("file")
            if not f:
                _fail(f"{RAW.name}:{lineno}", "record has no 'file'")
            # Later annotations of the same image supersede earlier ones: a
            # re-annotation is a correction, never a duplicate to average.
            records = [r for r in records if r["file"] != f]
            records.append(rec)
    return records


def amber_vocab() -> set[str]:
    with (ROOT / "data" / "amber" / "relation.json").open(encoding="utf-8") as fh:
        return set(json.load(fh))


def validate_pair(where: str, obj: str, pos: str, neg: str, kind: str) -> None:
    if kind not in KINDS:
        _fail(where, f"unknown kind {kind!r}; expected state|action")
    for name, val in (("obj", obj), ("pos", pos), ("neg", neg)):
        if not isinstance(val, str) or not val.strip():
            _fail(where, f"{name} is empty")
        if val != val.strip().lower():
            _fail(where, f"{name}={val!r} must be lowercase and unpadded")
        if not WORD.match(val):
            _fail(where, f"{name}={val!r} must be words only (a-z, space, apostrophe)")
    if pos == neg:
        _fail(where, f"positive and negative attribute are both {pos!r}")
    # Policy: the attribute must not be contained in the object name.
    obj_words = set(obj.split())
    for name, attr in (("positive", pos), ("negative", neg)):
        if obj_words & set(attr.split()):
            _fail(where, f"{name} attribute {attr!r} repeats a word of object {obj!r}")
    # Policy: the question must parse with the regex the pipeline will use.
    for attr in (pos, neg):
        q = TEMPLATE[kind].format(obj=obj, attr=attr)
        m = PARSE_RE[kind].match(q)
        if m is None:
            _fail(where, f"generated query does not parse: {q!r}")
        if (m.group("obj"), m.group("attr")) != (obj, attr):
            _fail(
                where,
                f"query {q!r} parses ambiguously as obj={m.group('obj')!r} "
                f"attr={m.group('attr')!r}, expected obj={obj!r} attr={attr!r}",
            )


def build() -> dict:
    raw = load_raw()
    vocab = amber_vocab()
    on_disk = {p.name for p in IMAGES.iterdir() if p.is_file()}
    rng = random.Random(SEED)

    annotations: list[dict] = []
    queries: list[dict] = []
    meta: dict[str, dict] = {}
    dropped: list[dict] = []
    per_image_pairs: Counter = Counter()
    next_id = 1

    for rec in sorted(raw, key=lambda r: r["file"]):
        f = rec["file"]
        where = f"image {f}"
        if f not in on_disk:
            _fail(where, "named in the annotation log but not present on disk")
        if rec.get("drop"):
            dropped.append({"file": f, "reason": rec["drop"]})
            continue
        pairs = rec.get("pairs") or []
        if not pairs:
            _fail(
                where,
                "has no pairs and no drop reason; every image must be either "
                "annotated or explicitly dropped",
            )
        seen_keys: set[tuple[str, str]] = set()
        for p in pairs:
            obj, pos, neg = p["obj"], p["pos"], p["neg"]
            kind = p.get("kind", "state")
            validate_pair(where, obj, pos, neg, kind)
            # pairs.py groups by (image, obj) and keeps a group only if it holds
            # exactly two members, so an object may carry one pair per image.
            if obj in {o for o, _ in seen_keys}:
                _fail(
                    where,
                    f"object {obj!r} carries more than one pair; pairs.py would "
                    "discard the whole group",
                )
            seen_keys.add((obj, kind))

            # Seeded coin flip: which of the two questions gets the lower id.
            pos_first = rng.random() < 0.5
            order = (
                ((pos, "yes"), (neg, "no")) if pos_first else ((neg, "no"), (pos, "yes"))
            )
            ids: dict[str, int] = {}
            for attr, truth in order:
                qid = next_id
                next_id += 1
                ids[truth] = qid
                queries.append(
                    {
                        "id": qid,
                        "image": f,
                        "query": TEMPLATE[kind].format(obj=obj, attr=attr),
                    }
                )
                annotations.append({"id": qid, "type": KINDS[kind], "truth": truth})
            meta[str(ids["yes"])] = {
                "image": f,
                "obj": obj,
                "kind": kind,
                "positive_attr": pos,
                "negative_attr": neg,
                "positive_id": ids["yes"],
                "negative_id": ids["no"],
                "positive_has_lower_id": ids["yes"] < ids["no"],
                "in_amber_vocab": obj in vocab,
            }
            per_image_pairs[f] += 1

    # AMBER's join contract: annotations[id - 1]["id"] == id.
    for i, ann in enumerate(annotations):
        if ann["id"] != i + 1:
            _fail("annotations", f"index {i} holds id {ann['id']}, expected {i + 1}")
    if len(queries) != len(annotations):
        _fail("build", f"{len(queries)} queries vs {len(annotations)} annotations")

    n_pairs = len(meta)
    lower = sum(1 for m in meta.values() if m["positive_has_lower_id"])
    summary = {
        "images_on_disk": len(on_disk),
        "images_in_log": len(raw),
        "images_annotated": len(per_image_pairs),
        "images_dropped": len(dropped),
        "pairs": n_pairs,
        "questions": len(queries),
        "pairs_per_annotated_image": (
            (n_pairs / len(per_image_pairs)) if per_image_pairs else "NOT_COMPUTED"
        ),
        "kind_counts": dict(Counter(m["kind"] for m in meta.values())),
        "positive_lower_id": lower,
        "positive_lower_id_rate": (lower / n_pairs) if n_pairs else "NOT_COMPUTED",
        "in_amber_vocab_pairs": sum(1 for m in meta.values() if m["in_amber_vocab"]),
        "distinct_objects": len({m["obj"] for m in meta.values()}),
        "distinct_triples": len(
            {(m["obj"], m["positive_attr"], m["negative_attr"]) for m in meta.values()}
        ),
        "dropped": dropped,
        "seed": SEED,
    }

    (SHROOM / "query").mkdir(parents=True, exist_ok=True)
    _write(SHROOM / "annotations.json", annotations)
    _write(SHROOM / "query" / "query_all.json", queries)
    _write(SHROOM / "pair_meta.json", meta)
    _write(SHROOM / "build_summary.json", summary)
    return summary


def _write(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


if __name__ == "__main__":
    try:
        s = build()
    except BuildError as exc:
        print(f"BUILD FAILED -- {exc}")
        raise SystemExit(1) from None
    for k, v in s.items():
        print(f"  {k}: {len(v) if k == 'dropped' else v}")
