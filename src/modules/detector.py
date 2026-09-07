"""TRD §6 — object localisation with OWLv2.

Memory rule (TRD §0): never hold two models on GPU at once. Load, run the whole
stage, free, load the next. ``Detector`` is a context manager that frees on exit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import torch
from PIL import Image

from src.config import apply_hf_cache, load_config


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    score: float


@dataclass(frozen=True)
class Crop:
    image: Image.Image
    box: tuple[float, float, float, float]
    fell_back: bool
    detector_score: float | None
    detections: tuple = ()  # all post-NMS detections: (x1, y1, x2, y2, score)


def _nms(dets: list[Detection], iou_threshold: float) -> list[Detection]:
    """Greedy NMS, highest score first."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: x.score, reverse=True):
        if all(_iou(d.box, k.box) < iou_threshold for k in kept):
            kept.append(d)
    return kept


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Detector:
    """OWLv2 wrapper. Use as a context manager so the GPU is freed on exit."""

    def __init__(self, config: dict[str, Any] | None = None):
        apply_hf_cache()
        self.cfg = config or load_config()
        d = self.cfg["detector"]
        self.model_id = d["model_id"]
        self.revision = d["revision"]
        if not self.revision:
            raise ValueError(
                "detector.revision is not pinned in the config. Pin the real hash; "
                "never run against an unpinned or invented revision (PRD §8 rule 5)."
            )
        self.threshold = float(d["threshold"])
        self.nms_iou = float(d["nms_iou"])
        self.prompt_template = d["prompt_template"]
        self.crop_cfg = self.cfg["crop"]

        want = self.cfg["attribute"].get("device", "cuda")
        self.device = want if (want != "cuda" or torch.cuda.is_available()) else "cpu"

        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.processor = Owlv2Processor.from_pretrained(self.model_id, revision=self.revision)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.model_id, revision=self.revision
        ).to(self.device)
        self.model.eval()

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def detect(self, image: Image.Image, phrase: str) -> list[Detection]:
        """Locate ``phrase`` in ``image``.

        An empty result is a legitimate outcome, not an error (TRD §6).
        """
        prompt = self.prompt_template.format(phrase=phrase)
        inputs = self.processor(text=[[prompt]], images=image, return_tensors="pt").to(
            self.device
        )
        outputs = self.model(**inputs)
        target = torch.tensor([[image.height, image.width]], device=self.device)
        post = self.processor.post_process_grounded_object_detection(
            outputs, threshold=self.threshold, target_sizes=target
        )[0]

        dets = [
            Detection(box=tuple(float(v) for v in box), score=float(score))
            for box, score in zip(post["boxes"], post["scores"])
            if float(score) >= self.threshold
        ]
        dets = _nms(dets, self.nms_iou)
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets

    def crop_region(self, image: Image.Image, phrase: str) -> Crop:
        """Crop to the highest-scoring detection, padded; full image if none."""
        dets = self.detect(image, phrase)
        packed = tuple((*d.box, d.score) for d in dets)
        if not dets:
            # TRD §6 fallback. Reported, never hidden.
            return Crop(image=image, box=(0.0, 0.0, float(image.width), float(image.height)),
                        fell_back=True, detector_score=None, detections=())
        best = dets[0]
        box = pad_box(best.box, image.width, image.height, self.crop_cfg)
        return Crop(image=image.crop(box), box=box, fell_back=False,
                    detector_score=best.score, detections=packed)


def pad_box(box, img_w: int, img_h: int, crop_cfg: dict) -> tuple[int, int, int, int]:
    """Expand by ``padding`` on each side, clip to bounds, enforce a minimum size.

    Pure function so it can be tested without loading a model.
    """
    padding = float(crop_cfg["padding"])
    min_size = int(crop_cfg["min_size"])

    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - padding * w, x2 + padding * w
    y1, y2 = y1 - padding * h, y2 + padding * h

    # Enforce the minimum crop by expanding symmetrically around the centre.
    min_w = min(min_size, img_w)
    min_h = min(min_size, img_h)
    if (x2 - x1) < min_w:
        cx = (x1 + x2) / 2.0
        x1, x2 = cx - min_w / 2.0, cx + min_w / 2.0
    if (y2 - y1) < min_h:
        cy = (y1 + y2) / 2.0
        y1, y2 = cy - min_h / 2.0, cy + min_h / 2.0

    # Shift back inside the image rather than shrinking below the minimum.
    if x1 < 0:
        x2, x1 = x2 - x1, 0.0
    if y1 < 0:
        y2, y1 = y2 - y1, 0.0
    if x2 > img_w:
        x1, x2 = x1 - (x2 - img_w), float(img_w)
    if y2 > img_h:
        y1, y2 = y1 - (y2 - img_h), float(img_h)

    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
    return (round(x1), round(y1), round(x2), round(y2))
