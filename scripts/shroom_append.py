"""Append annotation records to the raw SHROOM log, validating as they arrive.

Reads a JSON array of records on stdin and appends one JSON object per line to
``data/shroom/annotations_raw.jsonl``. Validation is the same code the builder
uses, so a malformed record is rejected at the moment it is written rather than
after two thousand more have piled up behind it.

    python scripts/shroom_append.py < batch.json

Records name images by their thumbnail index (``"i": 42``) or by filename
(``"file": "..."``); the index is resolved through data/shroom/thumb_index.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.shroom_build import BuildError, validate_pair

ROOT = Path(__file__).resolve().parents[1]
SHROOM = ROOT / "data" / "shroom"
RAW = SHROOM / "annotations_raw.jsonl"
INDEX = SHROOM / "thumb_index.json"


def main() -> int:
    payload = json.load(sys.stdin)
    if not isinstance(payload, list):
        print("FAILED: stdin must hold a JSON array of records")
        return 1
    index = json.loads(INDEX.read_text(encoding="utf-8"))

    out: list[dict] = []
    for rec in payload:
        if "file" in rec:
            f = rec["file"]
        else:
            key = f"{int(rec['i']):04d}"
            if key not in index:
                print(f"FAILED: no image at thumbnail index {key}")
                return 1
            f = index[key]
        clean = {"file": f}
        if rec.get("drop"):
            clean["drop"] = str(rec["drop"])
            out.append(clean)
            continue
        pairs = rec.get("pairs") or []
        if not pairs:
            print(f"FAILED: {f} has neither pairs nor a drop reason")
            return 1
        try:
            for p in pairs:
                validate_pair(f, p["obj"], p["pos"], p["neg"], p.get("kind", "state"))
        except BuildError as exc:
            print(f"FAILED: {exc}")
            return 1
        except KeyError as exc:
            print(f"FAILED: {f} pair is missing key {exc}")
            return 1
        objs = [p["obj"] for p in pairs]
        if len(objs) != len(set(objs)):
            print(f"FAILED: {f} repeats an object across pairs: {objs}")
            return 1
        clean["pairs"] = [
            {
                "kind": p.get("kind", "state"),
                "obj": p["obj"],
                "pos": p["pos"],
                "neg": p["neg"],
            }
            for p in pairs
        ]
        out.append(clean)

    RAW.parent.mkdir(parents=True, exist_ok=True)
    with RAW.open("a", encoding="utf-8") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    done = len({json.loads(ln)["file"] for ln in RAW.read_text(encoding="utf-8").splitlines() if ln.strip()})
    print(f"appended {len(out)} records; {done}/{len(index)} images now annotated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
