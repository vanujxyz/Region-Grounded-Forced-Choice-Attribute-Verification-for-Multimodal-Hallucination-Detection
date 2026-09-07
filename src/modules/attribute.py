"""TRD §7 — the four attribute-verification cells.

              | independent + threshold | forced choice (argmax)
  whole image | Cell A (baseline)       | Cell B
  cropped     | Cell C                  | Cell D (proposed method)

The prompt template is identical in all four cells and comes from config; it is
never varied per cell (TRD §7, §16).

Option order is alphabetical and content-independent. D-006: because each option
is scored independently, the result is order-invariant by construction; the only
order-dependence is tie-breaking, which is counted rather than resolved silently.
D-009: option order must never derive from a question id.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Self

import torch
from PIL import Image

from src.config import apply_hf_cache, load_config
from src.data.pairs import AttrPair

CELLS = ("A", "B", "C", "D")
CELL_REGION = {"A": "full", "B": "full", "C": "crop", "D": "crop"}
CELL_DECISION = {
    "A": "threshold",
    "B": "forced_choice",
    "C": "threshold",
    "D": "forced_choice",
}


@dataclass
class TieLog:
    """D-006: exact score ties are the only place option order can matter."""

    ties: int = 0
    total: int = 0
    examples: list[str] = field(default_factory=list)

    def record(self, tied: bool, what: str) -> None:
        self.total += 1
        if tied:
            self.ties += 1
            if len(self.examples) < 20:
                self.examples.append(what)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tie_count": self.ties,
            "comparisons": self.total,
            "tie_rate": (self.ties / self.total) if self.total else "NOT_COMPUTED",
            "examples": self.examples,
        }


class AttributeScorer:
    """SigLIP wrapper. Context manager so the GPU is freed on exit (TRD §0)."""

    def __init__(self, config: dict[str, Any] | None = None):
        apply_hf_cache()
        self.cfg = config or load_config()
        a = self.cfg["attribute"]
        self.model_id = a["model_id"]
        self.revision = a["revision"]
        if not self.revision:
            raise ValueError(
                "attribute.revision is not pinned in the config. Pin the real hash; "
                "never run against an unpinned or invented revision (PRD §8 rule 5)."
            )
        self.prompt_template = a["prompt_template"]
        self.confidence_k = float(a["confidence_k"])
        self.score_scale = a.get("score_scale", "logit")
        if self.score_scale not in ("logit", "sigmoid"):
            raise ValueError(
                f"attribute.score_scale must be logit|sigmoid, got {self.score_scale!r}"
            )

        want = a.get("device", "cuda")
        self.device = want if (want != "cuda" or torch.cuda.is_available()) else "cpu"

        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, revision=self.revision
        )
        self.model = AutoModel.from_pretrained(
            self.model_id, revision=self.revision
        ).to(self.device)
        self.model.eval()
        self.tie_log = TieLog()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prompt(self, attr: str, obj: str) -> str:
        return self.prompt_template.format(attr=attr, obj=obj)

    @torch.no_grad()
    def score(self, image: Image.Image, texts: list[str]) -> list[float]:
        """Score each text against the image.

        Each score is independent of the others: SigLIP encodes one text at a
        time against the image, so no option can observe another's presence or
        position. This is why D-006's both-orders averaging is a no-op.
        """
        inputs = self.processor(
            text=texts, images=image, padding="max_length", return_tensors="pt"
        ).to(self.device)
        out = self.model(**inputs)
        logits = out.logits_per_image[0]
        if self.score_scale == "sigmoid":
            logits = torch.sigmoid(logits)
        return [float(v) for v in logits]


def _record(
    qid, pair, attr, cell, pred, confidence, gold, crop_info, raw_scores, runtime_ms
):
    return {
        "id": qid,
        "image": pair.image,
        "obj": pair.obj,
        "attr": attr,
        "cell": cell,
        "pred": pred,
        "confidence": confidence,
        "gold": gold,
        "triple": (pair.obj, pair.positive_attr, pair.negative_attr),
        "fell_back": crop_info["fell_back"],
        "detector_score": crop_info["detector_score"],
        "raw_scores": raw_scores,
        "runtime_ms": runtime_ms,
    }


def decide(
    options: list[str],
    scores: list[float],
    cell: str,
    tau: float | None,
    confidence_k: float,
    tie_log: TieLog | None = None,
) -> tuple[dict[str, str], dict[str, float]]:
    """Turn per-option scores into yes/no answers plus confidences.

    Pure function: no model, no image. Kept separate so the cell logic is
    testable without a GPU.
    """
    by_attr = dict(zip(options, scores))
    answers: dict[str, str] = {}
    confidences: dict[str, float] = {}

    if CELL_DECISION[cell] == "threshold":
        if tau is None:
            raise ValueError(f"cell {cell} needs a fitted tau; none was supplied")
        for a in options:
            s = by_attr[a]
            answers[a] = "yes" if s >= tau else "no"
            confidences[a] = 1.0 / (1.0 + math.exp(-confidence_k * (s - tau)))
    else:
        vals = [by_attr[a] for a in options]
        top = max(vals)
        tied = sum(1 for v in vals if v == top) > 1
        if tie_log is not None:
            tie_log.record(tied, f"{options}")
        winner = options[vals.index(top)]
        mx = max(vals)
        exps = [math.exp(v - mx) for v in vals]
        denom = sum(exps)
        soft = {a: e / denom for a, e in zip(options, exps)}
        for a in options:
            answers[a] = "yes" if a == winner else "no"
            confidences[a] = soft[winner]

    return answers, confidences


def predict_pair(
    scorer: AttributeScorer,
    pair: AttrPair,
    image: Image.Image,
    cell: str,
    tau: float | None = None,
    crop_info: dict | None = None,
    runtime_ms: float | None = None,
) -> list[dict]:
    """Emit exactly two predictions for a pair, one per question id."""
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {CELLS}, got {cell!r}")
    crop_info = crop_info or {"fell_back": False, "detector_score": None}

    options = pair.options  # alphabetical, content-independent, id-independent
    texts = [scorer.prompt(a, pair.obj) for a in options]
    scores = scorer.score(image, texts)
    by_attr = dict(zip(options, scores))

    answers, confidences = decide(
        options, scores, cell, tau, scorer.confidence_k, scorer.tie_log
    )

    out = []
    for qid, attr, gold in (
        (pair.positive_id, pair.positive_attr, "yes"),
        (pair.negative_id, pair.negative_attr, "no"),
    ):
        out.append(
            _record(
                qid,
                pair,
                attr,
                cell,
                answers[attr],
                confidences[attr],
                gold,
                crop_info,
                [by_attr[a] for a in options],
                runtime_ms,
            )
        )
    return out


def fit_tau(scores_and_golds: list[tuple[float, str]]) -> tuple[float, float]:
    """Sweep thresholds; return (tau, dev accuracy) at the best split.

    Fitted on the dev split only (TRD §7, §16). Candidates are midpoints between
    adjacent observed scores, so the sweep is exhaustive over distinct decisions.
    """
    if not scores_and_golds:
        raise ValueError("cannot fit tau on an empty set")
    vals = sorted({s for s, _ in scores_and_golds})
    candidates = (
        [vals[0] - 1.0]
        + [(vals[i] + vals[i + 1]) / 2.0 for i in range(len(vals) - 1)]
        + [vals[-1] + 1.0]
    )

    best_tau, best_acc = candidates[0], -1.0
    for tau in candidates:
        correct = sum(
            1 for s, g in scores_and_golds if (("yes" if s >= tau else "no") == g)
        )
        acc = correct / len(scores_and_golds)
        if acc > best_acc:
            best_tau, best_acc = tau, acc
    return best_tau, best_acc
