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
from functools import lru_cache
from pathlib import Path
from typing import Any, Self

import torch
from PIL import Image

from src.config import apply_hf_cache, load_config
from src.data.pairs import AttrPair

CELLS = ("A", "B", "C", "D")

# D-020 diagnostic cells: identical to A and C, but tau is chosen so that the
# total number of "yes" predictions matches the dataset base rate, instead of
# maximising accuracy. This hands the threshold cells the same structural
# information forced choice gets for free from AMBER's exactly balanced pairs.
# They are NOT part of the 2x2 -- they are a control for D-020.
# D-032: Cell D-ext. Identical to Cell D, but the competing attribute comes from
# an external antonym map (configs/antonyms.yaml), never from the dataset's
# paired negative. Each question is answered ON ITS OWN -- no pair structure is
# used at any point -- so it cannot exploit contrastive structure the baseline
# lacks. Questions whose attribute is unmapped are reported, never guessed.
EXT_CELLS = ("Dext",)
EXT_REGION = {"Dext": "crop"}
EXT_DECISION = {"Dext": "forced_choice_external"}

DIAG_CELLS = ("Ap", "Cp")
DIAG_LABEL = {"Ap": "A-prime", "Cp": "C-prime"}
DIAG_REGION = {"Ap": "full", "Cp": "crop"}
DIAG_DECISION = {"Ap": "threshold", "Cp": "threshold"}
ALL_CELLS = CELLS + DIAG_CELLS + EXT_CELLS
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


def region_of(cell: str) -> str:
    for table in (CELL_REGION, DIAG_REGION, EXT_REGION):
        if cell in table:
            return table[cell]
    raise ValueError(f"unknown cell {cell!r}; expected one of {ALL_CELLS}")


def decision_of(cell: str) -> str:
    for table in (CELL_DECISION, DIAG_DECISION, EXT_DECISION):
        if cell in table:
            return table[cell]
    raise ValueError(f"unknown cell {cell!r}; expected one of {ALL_CELLS}")


@lru_cache(maxsize=4)
def load_antonyms(path: str | None = None) -> dict[str, str]:
    """Load the external antonym map (D-032).

    This map is the ONLY source of a competing attribute for Cell D-ext. It is
    never consulted by cells A-D, and the evaluation path never falls back to
    the dataset's negative attribute.
    """
    import yaml

    from src.config import ROOT

    p = Path(path) if path else (ROOT / "configs" / "antonyms.yaml")
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"antonym map not found: {p}")
    with p.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    table = data["antonyms"]
    if not isinstance(table, dict) or not table:
        raise ValueError(f"{p} has no usable 'antonyms' mapping")
    for k, v in table.items():
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"antonym for {k!r} is not a non-empty string: {v!r}")
        if v.strip().lower() == str(k).strip().lower():
            raise ValueError(f"attribute {k!r} is mapped to itself")
    return {str(k): str(v) for k, v in table.items()}


@dataclass
class CoverageLog:
    """D-032: which questions the antonym map could and could not serve."""

    mapped: int = 0
    unmapped: int = 0
    unmapped_attrs: dict[str, int] = field(default_factory=dict)

    def record(self, attr: str, found: bool) -> None:
        if found:
            self.mapped += 1
        else:
            self.unmapped += 1
            self.unmapped_attrs[attr] = self.unmapped_attrs.get(attr, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        total = self.mapped + self.unmapped
        top = sorted(self.unmapped_attrs.items(), key=lambda kv: -kv[1])[:25]
        return {
            "questions_total": total,
            "questions_mapped": self.mapped,
            "questions_unmapped": self.unmapped,
            "coverage_rate": (self.mapped / total) if total else "NOT_COMPUTED",
            "distinct_unmapped_attrs": len(self.unmapped_attrs),
            "top_unmapped": top,
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

    if decision_of(cell) == "threshold":
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
    if cell not in ALL_CELLS:
        raise ValueError(f"cell must be one of {ALL_CELLS}, got {cell!r}")
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


def fit_tau_base_rate(
    scores_and_golds: list[tuple[float, str]], target_yes: int | None = None
) -> tuple[float, dict[str, Any]]:
    """D-020: pick tau so the predicted-yes COUNT matches the dataset base rate.

    Cells B and D emit exactly one "yes" per pair by construction, and AMBER's
    attribute pairs are exactly balanced, so forced choice is handed a correct
    50/50 prior for free. A threshold cell has no such information: Cell A
    over-predicts yes, Cell C under-predicts, and both are penalised for it.

    A' and C' remove that asymmetry by choosing tau to match the known base rate
    rather than to maximise accuracy. Selection rule: sort scores descending and
    take tau as the ``target_yes``-th largest, so exactly ``target_yes`` scores
    satisfy ``score >= tau`` (ties can push the realised count higher; the
    realised count is reported, never assumed).

    This is still fit on the evaluation data, exactly like every other dev-split
    tau (D-012). It exchanges one kind of oracle knowledge for another: A' is
    told the base rate instead of being tuned for accuracy. That is the point --
    it is the control that isolates how much of B/D's margin is the free prior.
    """
    if not scores_and_golds:
        raise ValueError("cannot fit tau on an empty set")
    if target_yes is None:
        target_yes = sum(1 for _, g in scores_and_golds if g == "yes")
    n = len(scores_and_golds)
    if not 0 < target_yes <= n:
        raise ValueError(f"target_yes must be in (0, {n}], got {target_yes}")

    desc = sorted((s for s, _ in scores_and_golds), reverse=True)
    tau = desc[target_yes - 1]
    realised = sum(1 for s, _ in scores_and_golds if s >= tau)
    correct = sum(1 for s, g in scores_and_golds if (("yes" if s >= tau else "no") == g))
    return tau, {
        "tau": tau,
        "selection": "base_rate_matched",
        "target_yes": target_yes,
        "realised_yes": realised,
        "ties_at_tau": realised - target_yes,
        "fit_accuracy": correct / n,
        "fit_n": n,
    }


def predict_pair_external(
    scorer: AttributeScorer,
    pair: AttrPair,
    image: Image.Image,
    antonyms: dict[str, str],
    coverage: CoverageLog,
    crop_info: dict | None = None,
    cell: str = "Dext",
) -> list[dict]:
    """D-032 Cell D-ext: answer each question ALONE, using an external antonym.

    For the question "Is the sky sunny?" the competitor is ``antonyms['sunny']``
    -- what the map says is the opposite of "sunny" -- **not** the attribute
    AMBER happens to have paired with it. The two questions of a pair are
    answered independently and never see each other, so D-ext can legitimately
    answer yes twice or no twice.

    A question whose attribute is unmapped produces NO record. It is counted in
    ``coverage`` and excluded from the scored set. There is deliberately no
    fallback to the dataset's negative attribute: that would reintroduce the
    exact leak this cell exists to remove.
    """
    crop_info = crop_info or {"fell_back": False, "detector_score": None}
    out: list[dict] = []

    for qid, attr, gold in (
        (pair.positive_id, pair.positive_attr, "yes"),
        (pair.negative_id, pair.negative_attr, "no"),
    ):
        competitor = antonyms.get(attr)
        coverage.record(attr, competitor is not None)
        if competitor is None:
            continue

        # Alphabetical, content-independent order (D-006, D-009).
        options = sorted([attr, competitor])
        texts = [scorer.prompt(a, pair.obj) for a in options]
        scores = scorer.score(image, texts)
        by_attr = dict(zip(options, scores))

        vals = [by_attr[a] for a in options]
        top = max(vals)
        scorer.tie_log.record(sum(1 for v in vals if v == top) > 1, f"{options}")
        winner = options[vals.index(top)]

        mx = max(vals)
        exps = [math.exp(v - mx) for v in vals]
        soft = {a: e / sum(exps) for a, e in zip(options, exps)}

        rec = _record(
            qid, pair, attr, cell,
            "yes" if winner == attr else "no",
            soft[winner], gold, crop_info,
            [by_attr[a] for a in options], None,
        )
        rec["competitor_attr"] = competitor
        rec["competitor_source"] = "external_antonym_map"
        out.append(rec)

    return out
