"""TRD §0 — cache detection results to disk so a stage can be re-run without
re-running the previous one.

Deferred from M1 (D-031) and built first at M4. In the M3 run, Cells C, D and C'
each recomputed the identical 1,929 OWLv2 detections — roughly two thirds of that
run's wall clock. M4's existence and counting modules hit OWLv2 far harder.

The cache is keyed by (model, revision, threshold, nms_iou, prompt, image, phrase)
so a change to any detector setting produces a different key rather than a stale
hit. **A cached entry is only ever reused when every one of those matches**; the
cache can make a run faster but can never make it different.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_VERSION = 2


def _settings_fingerprint(cfg: dict[str, Any]) -> str:
    """Everything about the detector that can change a detection."""
    d = cfg["detector"]
    payload = {
        "cache_version": CACHE_VERSION,
        "model_id": d["model_id"],
        "revision": d["revision"],
        "threshold": float(d["threshold"]),
        "nms_iou": float(d["nms_iou"]),
        "prompt_template": d["prompt_template"],
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class CropCache:
    """Disk cache of detector output, one JSON file per settings fingerprint.

    Entries store the **full post-NMS detection list**, not the cropped pixels. One
    cache therefore serves all three consumers: the attribute cells need the best
    box, existence needs whether any detection clears its threshold, and counting
    needs how many do. The crop is recomputed from the box, which is cheap.
    """

    def __init__(self, cfg: dict[str, Any], path: Path | None = None, enabled: bool = True):
        self.cfg = cfg
        self.enabled = enabled
        self.fingerprint = _settings_fingerprint(cfg)
        base = path or (cfg["paths"]["results"] / "cache")
        self.path = Path(base) / f"detections_{self.fingerprint}.json"
        self._data: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        self._dirty = False
        if self.enabled:
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open(encoding="utf-8") as fh:
                blob = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"crop cache at {self.path} is unreadable ({exc}). Delete it to rebuild; "
                "it is never silently ignored, because a half-read cache would "
                "silently change results."
            ) from exc
        if blob.get("fingerprint") != self.fingerprint:
            raise RuntimeError(
                f"crop cache {self.path} has fingerprint {blob.get('fingerprint')!r} "
                f"but this config is {self.fingerprint!r}. Refusing to use it."
            )
        self._data = blob.get("entries", {})

    @staticmethod
    def key(image: str, phrase: str) -> str:
        return f"{image}|{phrase}"

    def get(self, image: str, phrase: str) -> dict | None:
        if not self.enabled:
            return None
        hit = self._data.get(self.key(image, phrase))
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        return hit

    def put(
        self, image: str, phrase: str, box, fell_back: bool, detector_score,
        detections: list | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._data[self.key(image, phrase)] = {
            "box": list(box),
            "fell_back": bool(fell_back),
            "detector_score": detector_score,
            # [[x1, y1, x2, y2, score], ...] after NMS, sorted by score desc.
            "detections": [list(d) for d in (detections or [])],
        }
        self._dirty = True

    def detections_at(self, image: str, phrase: str, threshold: float) -> list | None:
        """Cached detections at or above ``threshold``, or None on a miss.

        The cache was built at ``detector.threshold``; a consumer asking for a
        LOWER threshold cannot be served from it, and gets a loud error rather
        than a silently truncated list.
        """
        base = float(self.cfg["detector"]["threshold"])
        if threshold < base - 1e-12:
            raise ValueError(
                f"requested threshold {threshold} is below the cached detector "
                f"threshold {base}; the cache cannot contain those detections. "
                "Lower detector.threshold and rebuild the cache."
            )
        hit = self.get(image, phrase)
        if hit is None:
            return None
        return [d for d in hit.get("detections", []) if d[4] >= threshold]

    def save(self) -> None:
        if not self.enabled or not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(
                {"fingerprint": self.fingerprint, "cache_version": CACHE_VERSION,
                 "entries": self._data},
                fh,
            )
        tmp.replace(self.path)
        self._dirty = False

    def stats(self) -> dict[str, Any]:
        looked = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "path": str(self.path),
            "fingerprint": self.fingerprint,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / looked) if looked else "NOT_COMPUTED",
            "entries": len(self._data),
        }
