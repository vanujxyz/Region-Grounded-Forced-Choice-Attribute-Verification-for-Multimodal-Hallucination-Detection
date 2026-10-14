"""D-061 -- run the detector and scorer on the VAW pair set.

Uses the project's own modules unchanged (src.modules.detector.Detector,
src.modules.attribute.AttributeScorer, pad_box, load_antonyms, CropCache) and
configs/main.yaml as-is.  Decisions are NOT taken here: every SigLIP logit the
cells need is logged, and scripts/vaw_report.py applies the decision rules.
Because SigLIP scores each text independently of the others (attribute.py,
AttributeScorer.score docstring), scoring a text once and reusing the logit in
several cells is exactly equivalent to scoring it inside each cell.

    python scripts/vaw_run.py --split dev  --images <dir of VG jpgs>
    python scripts/vaw_run.py --split test --images <dir>   (ALLOW_TEST_SPLIT=1)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from src.config import load_config
from src.modules.attribute import AttributeScorer, load_antonyms
from src.modules.crop_cache import CropCache
from src.modules.detector import Detector, pad_box

NULL_SIZE = 128          # VISOR Sec. III: uniform grey 128x128x3, value 128
NULL_VALUE = 128


def read_json(path, encoding="utf-8"):
    with open(path, encoding=encoding) as fh:
        return json.load(fh)


def write_json(path, obj, **kw):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, **kw)


def read_lines(path, encoding="utf-8"):
    with open(path, encoding=encoding) as fh:
        return fh.readlines()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def log(msg, fh):
    line = f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.split == "test" and os.environ.get("ALLOW_TEST_SPLIT") != "1":
        raise SystemExit("test split is locked; set ALLOW_TEST_SPLIT=1 (D-049 protocol)")
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)

    cfg = load_config()
    # Windows MAX_PATH (D-005): on hosts where the HF cache path is too long the
    # pinned revisions can be pre-fetched into a short local dir.  The files'
    # PROVENANCE.json must name the SAME model id and revision as main.yaml,
    # otherwise we refuse to run -- the config is never silently overridden.
    local = os.environ.get("VAW_LOCAL_MODELS")
    model_source = "hf_hub"
    if local:
        for sect, sub in (("detector", "owl"), ("attribute", "sig")):
            prov = read_json(Path(local) / sub / "PROVENANCE.json")
            if (prov["model_id"], prov["revision"]) != (cfg[sect]["model_id"], cfg[sect]["revision"]):
                raise SystemExit(f"{sub}: local files are {prov}, config pins "
                                 f"{cfg[sect]['model_id']}@{cfg[sect]['revision']}")
            cfg[sect]["model_id_pinned"] = cfg[sect]["model_id"]
            cfg[sect]["model_id"] = str(Path(local) / sub)
        model_source = "local_prefetch_of_pinned_revision"
    pairs = read_json(ROOT / "data" / "vaw" / f"pairs_{a.split}.json")
    if a.limit:
        pairs = pairs[: a.limit]
    img_dir = Path(a.images)
    out_dir = ROOT / "results" / "vaw"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"vaw_{a.split}_{stamp}" + (f"_limit{a.limit}" if a.limit else "")
    progress = open(out_dir / f"{tag}.progress.log", "w", encoding="utf-8")  # noqa: SIM115

    # ---- stage 1: OWLv2, then free it (TRD §0 memory rule) -------------------
    cache = CropCache(cfg, path=ROOT / "results" / "cache" / "vaw")
    t0 = time.time()
    with Detector(cfg) as det:
        for i, p in enumerate(pairs):
            key = f"{p['image_id']}.jpg"
            if cache.get(key, p["obj"]) is not None:
                continue
            img = Image.open(img_dir / key).convert("RGB")
            dets = det.detect(img, p["obj"])
            packed = [(*d.box, d.score) for d in dets]
            if dets:
                box = pad_box(dets[0].box, img.width, img.height, cfg["crop"])
                cache.put(key, p["obj"], box, False, dets[0].score, packed)
            else:
                cache.put(key, p["obj"], (0, 0, img.width, img.height), True, None, [])
            if (i + 1) % 50 == 0:
                cache.save()
                el = time.time() - t0
                log(f"detect {i + 1}/{len(pairs)}  {el / (i + 1):.2f}s/pair", progress)
        cache.save()
    t_detect = time.time() - t0
    log(f"detection done in {t_detect:.0f}s; cache {cache.stats()['entries']} entries", progress)

    # ---- stage 2: SigLIP -----------------------------------------------------
    antonyms = load_antonyms()
    null_img = Image.new("RGB", (NULL_SIZE, NULL_SIZE), (NULL_VALUE,) * 3)
    null_cache: dict[str, float] = {}
    records = []
    t1 = time.time()
    with AttributeScorer(cfg) as sc:
        for i, p in enumerate(pairs):
            key = f"{p['image_id']}.jpg"
            img = Image.open(img_dir / key).convert("RGB")
            hit = cache.get(key, p["obj"])
            det_box = tuple(hit["box"])
            x, y, w, h = p["bbox_xywh"]
            gold_box = pad_box((x, y, x + w, y + h), img.width, img.height, cfg["crop"])
            regions = {
                "full": img,
                "crop": img if hit["fell_back"] else img.crop(det_box),
                "gold": img.crop(gold_box),
            }
            attrs = [p["positive_attr"], p["negative_attr"]]
            comp = {a_: antonyms.get(a_) for a_ in attrs}
            words = sorted(set(attrs) | {c for c in comp.values() if c})
            texts = [sc.prompt(w_, p["obj"]) for w_ in words]
            scores = {}
            t_pair = time.time()
            for rname, rimg in regions.items():
                vals = sc.score(rimg, texts)
                scores[rname] = dict(zip(words, vals))
            todo = [t_ for t_ in texts if t_ not in null_cache]
            if todo:
                for t_, v in zip(todo, sc.score(null_img, todo)):
                    null_cache[t_] = v
            scores["null"] = {w_: null_cache[sc.prompt(w_, p["obj"])] for w_ in words}
            # gold-box IoU with the detector box (diagnostic for C*, D*)
            ix1, iy1 = max(det_box[0], x), max(det_box[1], y)
            ix2, iy2 = min(det_box[2], x + w), min(det_box[3], y + h)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            raw_best = hit["detections"][0][:4] if hit["detections"] else None
            iou_raw = None
            if raw_best:
                bx1, by1, bx2, by2 = raw_best
                jx1, jy1, jx2, jy2 = max(bx1, x), max(by1, y), min(bx2, x + w), min(by2, y + h)
                ji = max(0.0, jx2 - jx1) * max(0.0, jy2 - jy1)
                iou_raw = ji / ((bx2 - bx1) * (by2 - by1) + w * h - ji)
            records.append({
                **p,
                "image": key,
                "image_wh": [img.width, img.height],
                "fell_back": hit["fell_back"],
                "detector_score": hit["detector_score"],
                "det_box": list(det_box),
                "gold_box": list(gold_box),
                "iou_detbox_goldbox": iou_raw,
                "competitor": comp,
                "scores": scores,
                "runtime_ms_scoring": 1000 * (time.time() - t_pair),
                "_inter_unused": inter,
            })
            if (i + 1) % 100 == 0:
                log(f"score {i + 1}/{len(pairs)}  {(time.time() - t1) / (i + 1):.2f}s/pair", progress)
    t_score = time.time() - t1

    with open(out_dir / f"{tag}.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            r.pop("_inter_unused", None)
            fh.write(json.dumps(r) + "\n")
    import transformers
    manifest = {
        "tag": tag, "split": a.split, "n_pairs": len(records), "limit": a.limit,
        "protocol_sha256_16": sha(ROOT / "data" / "vaw" / "VAW_PROTOCOL.md"),
        "pairs_sha256_16": sha(ROOT / "data" / "vaw" / f"pairs_{a.split}.json"),
        "config_sha256_16": sha(ROOT / "configs" / "main.yaml"),
        "antonyms_sha256_16": sha(ROOT / "configs" / "antonyms.yaml"),
        "model_source": model_source,
        "detector": {k: cfg["detector"].get(k) for k in ("model_id_pinned", "model_id", "revision", "threshold", "nms_iou", "prompt_template")},
        "attribute": {k: cfg["attribute"].get(k) for k in ("model_id_pinned", "model_id", "revision", "prompt_template", "score_scale")},
        "crop": cfg["crop"], "seed": cfg["seed"],
        "null_image": {"size": NULL_SIZE, "value": NULL_VALUE},
        "device": "cpu" if not torch.cuda.is_available() else torch.cuda.get_device_name(0),
        "torch_threads": torch.get_num_threads(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "seconds": {"detection": t_detect, "scoring": t_score},
        "fallback_rate": sum(r["fell_back"] for r in records) / max(1, len(records)),
        "crop_cache": cache.stats(),
    }
    write_json(out_dir / f"{tag}.manifest.json", manifest, indent=1)
    log(f"done: {tag}  detection {t_detect:.0f}s  scoring {t_score:.0f}s", progress)


if __name__ == "__main__":
    main()
