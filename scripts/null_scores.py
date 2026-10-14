"""D-062 -- SigLIP text scores against a uniform grey null image.

Used by the VISOR-style calibration baseline (Cal-null) on AMBER and SHROOM-Vis,
where the existing raw logs hold only real-image scores.

    python scripts/null_scores.py <prompts.json> <out.json>

``prompts.json`` is a list of [attribute, object] pairs; the output maps the
rendered prompt string to its logit on the null image.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.modules.attribute import AttributeScorer

NULL_SIZE = 128
NULL_VALUE = 128
BATCH = int(os.environ.get("NULL_BATCH", "32"))


def main() -> None:
    torch.set_num_threads(int(os.environ.get("NULL_THREADS", "2")))
    cfg = load_config()
    local = os.environ.get("VAW_LOCAL_MODELS")
    if local:
        with open(Path(local) / "sig" / "PROVENANCE.json", encoding="utf-8") as fh:
            prov = json.load(fh)
        if (prov["model_id"], prov["revision"]) != (
            cfg["attribute"]["model_id"], cfg["attribute"]["revision"]
        ):
            raise SystemExit(f"local files are {prov}, config pins "
                             f"{cfg['attribute']['model_id']}@{cfg['attribute']['revision']}")
        cfg["attribute"]["model_id"] = str(Path(local) / "sig")
    cfg["attribute"]["device"] = "cpu"

    with open(sys.argv[1], encoding="utf-8") as fh:
        pairs = json.load(fh)
    null = Image.new("RGB", (NULL_SIZE, NULL_SIZE), (NULL_VALUE,) * 3)
    out: dict[str, float] = {}
    with AttributeScorer(cfg) as sc:
        texts = [sc.prompt(a, o) for a, o in pairs]
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            for text, value in zip(chunk, sc.score(null, chunk)):
                out[text] = value
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(len(out))


if __name__ == "__main__":
    main()
