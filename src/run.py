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


class VramTracker:
    """Peak VRAM across the whole pipeline, including two-model staging.

    Reports both torch's allocator peak and the minimum free VRAM observed via
    the driver, which also captures non-torch overhead (context, fragmentation).
    """

    def __init__(self):
        import torch

        self.torch = torch
        self.ok = torch.cuda.is_available()
        self.min_free = None
        self.total = None
        self.stages: dict[str, float] = {}
        if self.ok:
            torch.cuda.reset_peak_memory_stats()
            free, total = torch.cuda.mem_get_info()
            self.total = total
            self.min_free = free

    def sample(self) -> None:
        if not self.ok:
            return
        free, _ = self.torch.cuda.mem_get_info()
        self.min_free = free if self.min_free is None else min(self.min_free, free)

    def mark(self, stage: str) -> None:
        if not self.ok:
            return
        self.sample()
        self.stages[stage] = self.torch.cuda.max_memory_allocated() / 1024**3

    def as_dict(self) -> dict[str, Any]:
        if not self.ok:
            return {"available": False, "peak_allocated_gib": "NOT_COMPUTED",
                    "peak_used_gib": "NOT_COMPUTED", "total_gib": "NOT_COMPUTED",
                    "per_stage_peak_allocated_gib": {}}
        peak_alloc = self.torch.cuda.max_memory_allocated() / 1024**3
        return {
            "available": True,
            "peak_allocated_gib": peak_alloc,
            "peak_used_gib": (self.total - self.min_free) / 1024**3,
            "min_free_gib": self.min_free / 1024**3,
            "total_gib": self.total / 1024**3,
            "per_stage_peak_allocated_gib": dict(self.stages),
        }


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


PERMUTE_SEED = 20260907


def permute_pair_ids(pairs: list, seed: int = PERMUTE_SEED) -> list:
    """D-021 DIAGNOSTIC ONLY. Randomise which id of a pair holds the gold positive.

    In real AMBER the gold positive is ALWAYS the lower id (D-009, 2774/2774),
    so a detector that ignores the image and answers yes to the lower id scores
    1.0000. This builds a permuted variant to falsify the question of whether any
    cell reads ids.

    **This output is never a reported number.** An AMBER id is bound to a
    question's text, so permuting ids fabricates a variant whose ids no longer
    mean what AMBER's mean, forfeiting the comparability TRD §11.1 exists to
    preserve. Records are tagged ``diagnostic_permuted: True``.

    Expected: position-only collapses to ~0.50; cells A-D are unchanged, because
    none of them reads an id.
    """
    import random

    rng = random.Random(seed)
    out = []
    for p in pairs:
        q = p.model_copy(deep=True)
        if rng.random() < 0.5:
            q.positive_id, q.negative_id = p.negative_id, p.positive_id
        out.append(q)
    return out


def select_pairs(split: str, limit: int | None, permute_ids: bool = False) -> list:
    """Pairs whose image is in ``split``, in deterministic id order."""
    images = set(get_split(split))
    pairs = [p for p in load_attr_pairs() if p.image in images]
    pairs.sort(key=lambda p: p.positive_id)
    pairs = pairs[:limit] if limit else pairs
    return permute_pair_ids(pairs) if permute_ids else pairs


def _qtype_map() -> dict[int, str]:
    return {q.id: q.qtype for q in load_questions()}


def run_attribute_cell(
    cell: str, split: str, limit: int | None, cfg: dict, no_cache: bool = False
) -> dict:
    """Run one attribute cell end to end. Returns records + diagnostics."""
    from PIL import Image

    from src.modules.attribute import (
        DIAG_CELLS,
        AttributeScorer,
        CoverageLog,
        decision_of,
        fit_tau,
        fit_tau_base_rate,
        load_antonyms,
        predict_pair,
        predict_pair_external,
        region_of,
    )

    pairs = select_pairs(split, limit)
    if not pairs:
        raise RuntimeError(f"no pairs selected for split={split} limit={limit}")

    images_dir = paths()["images"]
    needs_crop = region_of(cell) == "crop"
    vram = VramTracker()

    # ---- Stage 1: detection (only for crop cells). Freed before stage 2. ----
    crops: dict[str, dict] = {}
    det_scores: list[float] = []
    cache_stats: Any = "NOT_APPLICABLE"
    if needs_crop:
        from src.modules.crop_cache import CropCache

        cache = CropCache(cfg, enabled=not no_cache)
        wanted = []
        for p in pairs:
            key = f"{p.image}|{p.obj}"
            if key in crops:
                continue
            hit = cache.get(p.image, p.obj)
            if hit is not None:
                crops[key] = {
                    "box": tuple(hit["box"]),
                    "fell_back": hit["fell_back"],
                    "detector_score": hit["detector_score"],
                }
            else:
                crops[key] = None  # placeholder; filled by the detector below
                wanted.append(p)

        if wanted:
            from src.modules.detector import Detector

            with Detector(cfg) as det:
                vram.mark("detector_loaded")
                for p in wanted:
                    key = f"{p.image}|{p.obj}"
                    img = Image.open(images_dir / p.image).convert("RGB")
                    c = det.crop_region(img, p.obj)
                    crops[key] = {
                        "box": c.box,
                        "fell_back": c.fell_back,
                        "detector_score": c.detector_score,
                    }
                    cache.put(p.image, p.obj, c.box, c.fell_back, c.detector_score, c.detections)
                    vram.sample()
                vram.mark("detector_stage_end")
            vram.mark("detector_freed")
        cache.save()
        cache_stats = cache.stats()

        missing = [k for k, v in crops.items() if v is None]
        if missing:
            raise RuntimeError(f"{len(missing)} regions were never detected, e.g. {missing[:3]}")
        det_scores = [
            v["detector_score"] for v in crops.values() if v["detector_score"] is not None
        ]

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

        is_external = decision_of(cell) == "forced_choice_external"
        antonyms = load_antonyms() if is_external else {}
        coverage = CoverageLog()

        tau = None
        if decision_of(cell) == "threshold":
            # TRD §7: tau fitted on dev by sweeping. See D-012 on the --limit case.
            fit_data: list[tuple[float, str]] = []
            for p in pairs:
                img, _ = _image_for(p)
                texts = [scorer.prompt(a, p.obj) for a in p.options]
                sc = scorer.score(img, texts)
                by = dict(zip(p.options, sc))
                fit_data.append((by[p.positive_attr], "yes"))
                fit_data.append((by[p.negative_attr], "no"))
            if cell in DIAG_CELLS:
                # D-020: tau matched to the known base rate, not to accuracy.
                tau, tau_info = fit_tau_base_rate(fit_data)
            else:
                tau, fit_acc = fit_tau(fit_data)
                tau_info = {"tau": tau, "selection": "accuracy_maximising",
                            "fit_accuracy": fit_acc, "fit_n": len(fit_data)}
            tau_info["fit_split"] = split
            tau_info["fit_limit"] = limit
            tau_info["protocol"] = "fit-on-eval"

        for p in pairs:
            t0 = time.perf_counter()
            img, crop_info = _image_for(p)
            if is_external:
                recs = predict_pair_external(
                    scorer, p, img, antonyms, coverage, crop_info=crop_info, cell=cell
                )
            else:
                recs = predict_pair(scorer, p, img, cell, tau=tau, crop_info=crop_info)
            dt = (time.perf_counter() - t0) * 1000.0
            for r in recs:
                r["runtime_ms"] = dt / max(1, len(recs))
            records.extend(recs)

        ties = scorer.tie_log.as_dict()
        vram.mark("scorer_stage_end")

    qtypes = _qtype_map()
    for r in records:
        r["qtype"] = qtypes[r["id"]]

    return {
        "records": records,
        "tau": tau_info,
        "ties": ties,
        "n_pairs": len(pairs),
        "vram": vram.as_dict(),
        "detection_scores": summarise_scores(det_scores),
        "n_unique_regions": len(crops) if needs_crop else "NOT_APPLICABLE",
        "coverage": coverage.as_dict() if is_external else "NOT_APPLICABLE",
        "crop_cache": cache_stats,
    }


def summarise_scores(scores: list[float]) -> dict[str, Any]:
    """Distribution of OWLv2 scores for the objects that WERE found."""
    if not scores:
        return {"n": 0, "min": "NOT_COMPUTED", "p25": "NOT_COMPUTED",
                "median": "NOT_COMPUTED", "p75": "NOT_COMPUTED",
                "max": "NOT_COMPUTED", "mean": "NOT_COMPUTED", "histogram": {}}
    import statistics

    xs = sorted(scores)

    def pct(q):
        if len(xs) == 1:
            return xs[0]
        i = q * (len(xs) - 1)
        lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
        return xs[lo] + (i - lo) * (xs[hi] - xs[lo])

    edges = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.01]
    hist: dict[str, int] = {}
    prev = 0.0
    for e in edges:
        hist[f"[{prev:.2f},{e:.2f})"] = sum(1 for x in xs if prev <= x < e)
        prev = e
    return {
        "n": len(xs), "min": xs[0], "p25": pct(0.25), "median": pct(0.50),
        "p75": pct(0.75), "max": xs[-1], "mean": statistics.fmean(xs),
        "histogram": hist,
    }


# D-044: the M4 module pass detects at this threshold so the FPR sweep can
# filter up from one cache. Lower than detector.threshold (0.10) by design.
MODULE_DETECT_THRESHOLD = 0.05
FPR_SWEEP = (0.05, 0.10, 0.20, 0.30)


def _module_cfg(cfg: dict) -> dict:
    """Config copy whose detector threshold is low enough for the FPR sweep."""
    import copy

    out = copy.deepcopy(cfg)
    out["detector"]["threshold"] = MODULE_DETECT_THRESHOLD
    return out


def _detect_all(pairs_of, cfg: dict, no_cache: bool = False):
    """Detect every (image, phrase) once, cached. Returns {key: detections}."""
    from PIL import Image

    from src.modules.crop_cache import CropCache

    mcfg = _module_cfg(cfg)
    cache = CropCache(mcfg, enabled=not no_cache)
    images_dir = paths()["images"]
    vram = VramTracker()

    wanted, out = [], {}
    for image, phrase in pairs_of:
        key = CropCache.key(image, phrase)
        if key in out:
            continue
        hit = cache.get(image, phrase)
        if hit is not None:
            out[key] = hit.get("detections", [])
        else:
            out[key] = None
            wanted.append((image, phrase))

    if wanted:
        from src.modules.detector import Detector

        with Detector(mcfg) as det:
            vram.mark("detector_loaded")
            for image, phrase in wanted:
                img = Image.open(images_dir / image).convert("RGB")
                c = det.crop_region(img, phrase)
                out[CropCache.key(image, phrase)] = [list(d) for d in c.detections]
                cache.put(image, phrase, c.box, c.fell_back, c.detector_score, c.detections)
                vram.sample()
            vram.mark("detector_stage_end")
    cache.save()

    missing = [k for k, v in out.items() if v is None]
    if missing:
        raise RuntimeError(f"{len(missing)} regions never detected, e.g. {missing[:3]}")
    return out, cache.stats(), vram.as_dict()


def run_existence(split: str, limit: int | None, cfg: dict, no_cache: bool = False) -> dict:
    """TRD §8 under the D-044 metric: false-positive rate, not accuracy."""
    from src.modules.existence import parse_existence_questions
    from src.modules.existence import predict as ex_predict

    images = set(get_split(split))
    qs = [q for q in parse_existence_questions() if q.image in images]
    if limit:
        qs = qs[:limit]
    if not qs:
        raise RuntimeError(f"no existence questions for split={split}")

    dets, cache_stats, vram = _detect_all(((q.image, q.obj) for q in qs), cfg, no_cache)
    thr = float(cfg["detector"]["threshold"])

    def at(threshold):
        return lambda i, o: [d for d in dets[f"{i}|{o}"] if d[4] >= threshold]

    records = ex_predict(qs, at(thr), threshold=thr,
                         confidence_k=float(cfg["attribute"]["confidence_k"]))

    n = len(qs)
    sweep = {}
    for t in FPR_SWEEP:
        fp = sum(1 for q in qs if any(d[4] >= t for d in dets[f"{q.image}|{q.obj}"]))
        sweep[f"{t:.2f}"] = {"false_positives": fp, "n": n, "fpr": fp / n,
                             "accuracy_equivalent": 1.0 - fp / n}

    golds = {r["gold"] for r in records}
    return {
        "records": records, "n_questions": n,
        "all_gold_no": golds == {"no"},
        "fpr_sweep": sweep,
        "reported_threshold": thr,
        "trivial_always_no_accuracy": 1.0,
        "trivial_baseline_note": "BENCHMARK ARTIFACT: all 4924 golds are no (D-042)",
        "crop_cache": cache_stats, "vram": vram,
        "tau": "NOT_APPLICABLE", "ties": "NOT_APPLICABLE",
    }


def run_counting(split: str, limit: int | None, cfg: dict, no_cache: bool = False) -> dict:
    """TRD §9. Unaffected by D-042: number gold is balanced 1036/1036."""
    from src.modules.counting import CountConfusion, load_number_questions
    from src.modules.counting import predict as ct_predict

    images = set(get_split(split))
    qs = [q for q in load_number_questions() if q.image in images]
    if limit:
        qs = qs[:limit]
    if not qs:
        raise RuntimeError(f"no number questions for split={split}")

    dets, cache_stats, vram = _detect_all(((q.image, q.obj) for q in qs), cfg, no_cache)
    thr = float(cfg["detector"]["threshold"])
    conf = CountConfusion()
    records = ct_predict(qs, lambda i, o: [d for d in dets[f"{i}|{o}"] if d[4] >= thr], conf)
    return {
        "records": records, "n_questions": len(qs),
        "count_confusion": conf.as_dict(),
        "reported_threshold": thr,
        "crop_cache": cache_stats, "vram": vram,
        "tau": "NOT_APPLICABLE", "ties": "NOT_APPLICABLE",
    }


def run_relation(split: str, limit: int | None, cfg: dict) -> dict:
    """TRD §10 under D-045: tau fitted on the pooled set; reported per type too."""
    from src.modules.attribute import AttributeScorer, fit_tau
    from src.modules.relation import parse_relation_questions, score_all
    from src.modules.relation import predict as rel_predict

    images = set(get_split(split))
    qs = [q for q in parse_relation_questions() if q.image in images]
    if limit:
        qs = qs[:limit]
    if not qs:
        raise RuntimeError(f"no relation questions for split={split}")

    images_dir = paths()["images"]
    vram = VramTracker()
    with AttributeScorer(cfg) as scorer:
        vram.mark("scorer_loaded")
        scored = score_all(scorer, qs, images_dir)
        tau, fit_acc = fit_tau(scored)
        records = rel_predict(scorer, qs, images_dir, tau)
        vram.mark("scorer_stage_end")

    return {
        "records": records, "n_questions": len(qs),
        "tau": {"tau": tau, "selection": "accuracy_maximising", "fit_accuracy": fit_acc,
                "fit_n": len(scored), "fit_split": split, "protocol": "fit-on-eval",
                "fit_scope": "pooled; per-type fitting impossible, each type is constant"},
        "ties": "NOT_APPLICABLE", "vram": vram.as_dict(),
    }


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
    if "vram" in result:
        v = result["vram"]
        print(f"  VRAM peak alloc  : {v['peak_allocated_gib']}")
        print(f"  VRAM peak used   : {v.get('peak_used_gib')} of {v.get('total_gib')} GiB")
        print(f"  VRAM per stage   : {v.get('per_stage_peak_allocated_gib')}")
    if result.get("crop_cache") not in (None, "NOT_APPLICABLE"):
        cc = result["crop_cache"]
        print(f"  crop cache       : {cc['hits']} hits / {cc['misses']} misses "
              f"(rate {cc['hit_rate']}), {cc['entries']} entries")
    if result.get("coverage") not in (None, "NOT_APPLICABLE"):
        c = result["coverage"]
        print(f"  antonym coverage : {c['questions_mapped']}/{c['questions_total']} "
              f"= {c['coverage_rate']}")
        print(f"  unmapped         : {c['questions_unmapped']} questions, "
              f"{c['distinct_unmapped_attrs']} distinct attributes")
        print(f"  top unmapped     : {c['top_unmapped'][:8]}")
    if result.get("detection_scores"):
        d = result["detection_scores"]
        print(f"  detections found : {d['n']} (unique regions: {result.get('n_unique_regions')})")
        print(f"  det score min/med/max : {d['min']} / {d['median']} / {d['max']}")
        print(f"  det score mean/p25/p75: {d['mean']} / {d['p25']} / {d['p75']}")
        print(f"  det score histogram   : {d['histogram']}")
    return m


def _report_module(name: str, result: dict) -> None:
    from src.eval.metrics import accuracy as _acc
    from src.eval.metrics import compute_metrics

    m = compute_metrics(result["records"])
    o = m["overall"]
    print()
    print(f"=== MODULE {name.upper()} ===")
    print(f"  questions        : {o['n']}")
    print(f"  accuracy         : {o['accuracy']}")
    print(f"  precision        : {o['precision']}")
    print(f"  recall           : {o['recall']}")
    print(f"  f1               : {o['f1']}")

    if name == "existence":
        print(f"  *** all gold no  : {result['all_gold_no']} -> accuracy is DEGENERATE")
        print(f"  *** {result['trivial_baseline_note']}")
        print(f"  *** trivial always-no accuracy: {result['trivial_always_no_accuracy']}")
        print("  PRIMARY METRIC - false-positive rate (D-044):")
        for t, v in result["fpr_sweep"].items():
            mark = " <- reported" if abs(float(t) - result["reported_threshold"]) < 1e-9 else ""
            print(f"      thr {t}: FPR {v['fpr']:.4f}  ({v['false_positives']}/{v['n']})"
                  f"  acc-equiv {v['accuracy_equivalent']:.4f}{mark}")

    if name == "counting":
        c = result["count_confusion"]
        print(f"  exact count match: {c['exact_match_rate']:.4f}")
        print(f"  mean abs error   : {c['mean_abs_error']:.4f}")
        print(f"  predicted zero   : {c['predicted_zero_rate']:.4f}")

    if name == "relation":
        print(f"  tau              : {result['tau']}")
        by = {}
        for r in result["records"]:
            by.setdefault(r["qtype"], []).append(r)
        print("  D-045 per type (each degenerate) and pooled:")
        for qt, rs in sorted(by.items()):
            golds = sorted({x["gold"] for x in rs})
            triv = 1.0 if len(golds) == 1 else "n/a"
            print(f"      {qt:26s} n={len(rs):4d} acc={_acc(rs):.4f} "
                  f"gold={golds} trivial-baseline={triv}")
        yes = sum(1 for r in result["records"] if r["gold"] == "yes")
        print(f"      POOLED                     n={o['n']:4d} acc={o['accuracy']:.4f} "
              f"always-yes={yes / o['n']:.4f}")
        print("      (never report pooled alone -- D-045)")

    if result.get("crop_cache") not in (None, "NOT_APPLICABLE"):
        cc = result["crop_cache"]
        print(f"  crop cache       : {cc['hits']} hits / {cc['misses']} misses, "
              f"{cc['entries']} entries")
    v = result.get("vram")
    if isinstance(v, dict) and v.get("available"):
        print(f"  VRAM peak used   : {v['peak_used_gib']:.3f} of {v['total_gib']:.3f} GiB")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.run")
    ap.add_argument(
        "--cell",
        choices=["A", "B", "C", "D", "Ap", "Cp", "Dext", "all", "all-diag"],
    )
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
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass the detection cache and re-run OWLv2 from scratch")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else paths()["results_raw"]
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.module in ("existence", "counting", "relation"):
        if args.module == "existence":
            result = run_existence(args.split, args.limit, cfg, args.no_cache)
        elif args.module == "counting":
            result = run_counting(args.split, args.limit, cfg, args.no_cache)
        else:
            result = run_relation(args.split, args.limit, cfg)
        run_id = f"{args.module}_{args.split}_{stamp}"
        _report_module(args.module, result)
        write_jsonl(result["records"], out_dir / f"{run_id}.jsonl")
        write_manifest(run_id, cfg, args,
                       {k: v for k, v in result.items() if k != "records"})
        return 0

    if args.module == "position-only":
        result = run_position_only(args.split, args.limit)
        run_id = f"position-only_{args.split}_{stamp}"
        _report("POSITION-ONLY BASELINE (benchmark artifact)", result)
        write_jsonl(result["records"], out_dir / f"{run_id}.jsonl")
        write_manifest(run_id, cfg, args, {"n_pairs": result["n_pairs"]})
        return 0

    if not args.cell:
        ap.error("--cell is required for the attribute module")

    if args.cell == "all":
        cells = ["A", "B", "C", "D"]
    elif args.cell == "all-diag":
        cells = ["Ap", "Cp"]
    else:
        cells = [args.cell]
    for cell in cells:
        result = run_attribute_cell(cell, args.split, args.limit, cfg, args.no_cache)
        run_id = f"attribute_{cell}_{args.split}_{stamp}"
        _report(f"CELL {cell}", result)
        write_jsonl(result["records"], out_dir / f"{run_id}.jsonl")
        write_manifest(
            run_id, cfg, args,
            {
                "tau": result["tau"],
                "ties": result["ties"],
                "n_pairs": result["n_pairs"],
                "vram": result.get("vram"),
                "detection_scores": result.get("detection_scores"),
                "n_unique_regions": result.get("n_unique_regions"),
                "coverage": result.get("coverage"),
                "crop_cache": result.get("crop_cache"),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
