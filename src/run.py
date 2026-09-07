"""TRD §12 — CLI.

    python -m src.run --cell A --split dev --limit 100
    python -m src.run --cell all --split dev
    python -m src.run --module position-only --split dev

Memory rule (TRD §0): the detector stage runs to completion and is freed before
the scorer is loaded. Crops are cached to disk so a stage can be re-run without
re-running the previous one.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import load_config, paths
from src.data.loader import load_questions
from src.data.pairs import load_attr_pairs
from src.data.splits import get_split
from src.eval.metrics import compute_metrics


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or "NOT_COMPUTED"
    except Exception:  # noqa: BLE001 - manifest must never crash a finished run
        return "NOT_COMPUTED"


def _package_versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for p in ("torch", "transformers", "numpy", "pillow", "pydantic", "spacy"):
        try:
            out[p] = md.version(p)
        except Exception:  # noqa: BLE001 - absent package is data, not an error
            out[p] = "NOT_INSTALLED"
    return out


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "cpu"
    except Exception:  # noqa: BLE001 - manifest must never crash a finished run
        return "NOT_COMPUTED"


def write_manifest(run_id: str, cfg: dict, args, extra: dict[str, Any]) -> Path:
    """PRD §8 rule 5: every result file records commit, config, revisions, seed, time."""
    raw = paths()["results_raw"]
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": cfg["_config_path"],
        "config": {k: v for k, v in cfg.items() if k != "paths"},
        "paths": {k: str(v) for k, v in cfg["paths"].items()},
        "models": {
            "detector": {
                "id": cfg["detector"]["model_id"],
                "revision": cfg["detector"]["revision"],
            },
            "attribute": {
                "id": cfg["attribute"]["model_id"],
                "revision": cfg["attribute"]["revision"],
            },
        },
        "seed": args.seed if args.seed is not None else cfg["seed"],
        "split": args.split,
        "limit": args.limit,
        "cell": args.cell,
        "module": args.module,
        "packages": _package_versions(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": _gpu_name(),
        **extra,
    }
    path = raw / f"{run_id}.manifest.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return path


def write_jsonl(records: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")
    return path


def select_pairs(split: str, limit: int | None) -> list:
    """Pairs whose image is in ``split``, in deterministic id order."""
    images = set(get_split(split))
    pairs = [p for p in load_attr_pairs() if p.image in images]
    pairs.sort(key=lambda p: p.positive_id)
    return pairs[:limit] if limit else pairs


def _qtype_map() -> dict[int, str]:
    return {q.id: q.qtype for q in load_questions()}


def run_attribute_cell(cell: str, split: str, limit: int | None, cfg: dict) -> dict:
    """Run one attribute cell end to end. Returns records + diagnostics."""
    from PIL import Image

    from src.modules.attribute import (
        CELL_DECISION,
        CELL_REGION,
        AttributeScorer,
        fit_tau,
        predict_pair,
    )

    pairs = select_pairs(split, limit)
    if not pairs:
        raise RuntimeError(f"no pairs selected for split={split} limit={limit}")

    images_dir = paths()["images"]
    needs_crop = CELL_REGION[cell] == "crop"

    # ---- Stage 1: detection (only for crop cells). Freed before stage 2. ----
    crops: dict[str, dict] = {}
    if needs_crop:
        from src.modules.detector import Detector

        with Detector(cfg) as det:
            for p in pairs:
                key = f"{p.image}|{p.obj}"
                if key in crops:
                    continue
                img = Image.open(images_dir / p.image).convert("RGB")
                c = det.crop_region(img, p.obj)
                crops[key] = {
                    "box": c.box,
                    "fell_back": c.fell_back,
                    "detector_score": c.detector_score,
                }

    # ---- Stage 2: scoring ----
    records: list[dict] = []
    tau_info: Any = "NOT_APPLICABLE"
    with AttributeScorer(cfg) as scorer:

        def _image_for(p):
            img = Image.open(images_dir / p.image).convert("RGB")
            if not needs_crop:
                return img, {"fell_back": False, "detector_score": None}
            c = crops[f"{p.image}|{p.obj}"]
            box = c["box"]
            return (
                img if c["fell_back"] else img.crop(tuple(int(v) for v in box)),
                {"fell_back": c["fell_back"], "detector_score": c["detector_score"]},
            )

        tau = None
        if CELL_DECISION[cell] == "threshold":
            # TRD §7: tau fitted on dev by sweeping. See D-012 on the --limit case.
            fit_data: list[tuple[float, str]] = []
            for p in pairs:
                img, _ = _image_for(p)
                texts = [scorer.prompt(a, p.obj) for a in p.options]
                sc = scorer.score(img, texts)
                by = dict(zip(p.options, sc))
                fit_data.append((by[p.positive_attr], "yes"))
                fit_data.append((by[p.negative_attr], "no"))
            tau, fit_acc = fit_tau(fit_data)
            tau_info = {"tau": tau, "fit_accuracy": fit_acc, "fit_n": len(fit_data),
                        "fit_split": split, "fit_limit": limit}

        for p in pairs:
            t0 = time.perf_counter()
            img, crop_info = _image_for(p)
            recs = predict_pair(scorer, p, img, cell, tau=tau, crop_info=crop_info)
            dt = (time.perf_counter() - t0) * 1000.0
            for r in recs:
                r["runtime_ms"] = dt / 2.0
            records.extend(recs)

        ties = scorer.tie_log.as_dict()

    qtypes = _qtype_map()
    for r in records:
        r["qtype"] = qtypes[r["id"]]

    return {"records": records, "tau": tau_info, "ties": ties, "n_pairs": len(pairs)}


def run_position_only(split: str, limit: int | None) -> dict:
    from src.modules.position_baseline import predict

    pairs = select_pairs(split, limit)
    records = predict(pairs)
    qtypes = _qtype_map()
    for r in records:
        r["qtype"] = qtypes[r["id"]]
    return {"records": records, "tau": "NOT_APPLICABLE",
            "ties": "NOT_APPLICABLE", "n_pairs": len(pairs)}


def _report(name: str, result: dict) -> dict:
    m = compute_metrics(result["records"])
    o = m["overall"]
    print(f"\n=== {name} ===")
    print(f"  pairs            : {result['n_pairs']}")
    print(f"  questions (n)    : {o['n']}")
    print(f"  distinct triples : {o['n_distinct_triples']}")
    print(f"  distinct images  : {o['n_distinct_images']}")
    print(f"  ACCURACY         : {o['accuracy']}")
    print(f"  precision        : {o['precision']}")
    print(f"  recall           : {o['recall']}")
    print(f"  f1               : {o['f1']}")
    print(f"  fallback rate    : {m['fallback_rate']}")
    print(f"  tau              : {result['tau']}")
    print(f"  ties             : {result['ties']}")
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.run")
    ap.add_argument("--cell", choices=["A", "B", "C", "D", "all"])
    ap.add_argument(
        "--module",
        choices=["attribute", "existence", "counting", "relation", "position-only", "all"],
        default="attribute",
    )
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force-resplit", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tables", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else paths()["results_raw"]
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.module == "position-only":
        result = run_position_only(args.split, args.limit)
        run_id = f"position-only_{args.split}_{stamp}"
        _report("POSITION-ONLY BASELINE (benchmark artifact)", result)
        write_jsonl(result["records"], out_dir / f"{run_id}.jsonl")
        write_manifest(run_id, cfg, args, {"n_pairs": result["n_pairs"]})
        return 0

    if not args.cell:
        ap.error("--cell is required for the attribute module")

    cells = ["A", "B", "C", "D"] if args.cell == "all" else [args.cell]
    for cell in cells:
        result = run_attribute_cell(cell, args.split, args.limit, cfg)
        run_id = f"attribute_{cell}_{args.split}_{stamp}"
        _report(f"CELL {cell}", result)
        write_jsonl(result["records"], out_dir / f"{run_id}.jsonl")
        write_manifest(
            run_id, cfg, args,
            {"tau": result["tau"], "ties": result["ties"], "n_pairs": result["n_pairs"]},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
