"""TRD §13 — live demo (deliverable D6).

    python -m src.demo --image data/images/AMBER_1.jpg --claim "the sky is gloomy"

Pipeline: spaCy parse -> extract (obj, attr) -> locate -> crop -> forced choice.

**The antonym map used here is the DEMO map** (`configs/antonyms.yaml`). TRD §13
is explicit that it must never be used in the evaluation path, where options come
from the dataset pairs. It is used here because free text supplies no contrasting
attribute. Cell D-ext (D-032) is a separate, deliberately-evaluated use of the
same file and is not this.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_config, paths


def split_claim(claim: str) -> tuple[str, str, str]:
    """Extract (object, attribute, kind) from free text with a dependency parse.

    Handles the two shapes AMBER-style claims take:
      state  — "the sky is sunny"      -> (sky, sunny, state)
      action — "the man is sitting"    -> (man, sitting, action)
    plus adjectival modification — "a red apple" -> (apple, red, state).
    """
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as exc:  # pragma: no cover - environment problem, not logic
        raise RuntimeError(
            "en_core_web_sm is not installed. Run: python -m spacy download en_core_web_sm"
        ) from exc

    doc = nlp(claim.strip())

    # "the sky is sunny" -> ADJ with dep acomp, subject is the object
    for tok in doc:
        if tok.dep_ == "acomp":
            subj = next((t for t in doc if t.dep_ == "nsubj"), None)
            if subj is not None:
                return subj.text.lower(), tok.text.lower(), "state"

    # "the man is sitting" -> ROOT verb, subject is the object
    for tok in doc:
        if tok.pos_ == "VERB" and tok.dep_ == "ROOT":
            subj = next((t for t in doc if t.dep_ == "nsubj"), None)
            if subj is not None:
                return subj.text.lower(), tok.lemma_.lower(), "action"

    # "a red apple" -> adjectival modifier
    for tok in doc:
        if tok.dep_ == "amod" and tok.head.pos_ in ("NOUN", "PROPN"):
            return tok.head.text.lower(), tok.text.lower(), "state"

    raise ValueError(
        f"could not extract an (object, attribute) pair from {claim!r}. "
        "Supported shapes: 'the X is Y', 'the X is Ying', 'a Y X'."
    )


def contrasting_attribute(attr: str, antonyms: dict[str, str]) -> str:
    """The competing attribute, from the DEMO antonym map only."""
    for key in (attr, attr.rstrip("ing"), attr + "e"):
        if key in antonyms:
            return antonyms[key]
    raise ValueError(
        f"no contrasting attribute for {attr!r} in configs/antonyms.yaml. "
        "Add one, or use a claim whose attribute is mapped. The demo will not "
        "invent a contrast."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.demo", description="Verify a claim about an image.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--claim", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="where to save the crop")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path
    if not image_path.exists():
        raise SystemExit(f"image not found: {image_path}")

    from PIL import Image

    from src.modules.attribute import AttributeScorer, load_antonyms
    from src.modules.detector import Detector

    obj, attr, kind = split_claim(args.claim)
    antonyms = load_antonyms()
    other = contrasting_attribute(attr, antonyms)
    options = sorted([attr, other])  # alphabetical: order carries no signal

    image = Image.open(image_path).convert("RGB")

    # Stage 1: locate. Freed before the scorer loads (TRD §0).
    with Detector(cfg) as det:
        crop = det.crop_region(image, obj)

    out_dir = Path(args.out) if args.out else paths(args.config)["demo_out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_path = out_dir / f"{image_path.stem}_{obj.replace(' ', '_')}.jpg"
    crop.image.save(crop_path)

    # Stage 2: forced choice on the crop.
    with AttributeScorer(cfg) as scorer:
        scores = scorer.score(crop.image, [scorer.prompt(a, obj) for a in options])

    by = dict(zip(options, scores))
    mx = max(scores)
    exps = {a: pow(2.718281828459045, by[a] - mx) for a in options}
    total = sum(exps.values())
    probs = {a: exps[a] / total for a in options}
    winner = max(options, key=lambda a: by[a])
    supported = winner == attr

    score_txt = (
        f"detector score {crop.detector_score:.2f}" if crop.detector_score is not None
        else "NOT DETECTED - fell back to the full image"
    )
    try:
        shown = crop_path.relative_to(Path.cwd())
    except ValueError:
        shown = crop_path

    print(f"Claim:      {args.claim}")
    print(f"Parsed:     object={obj!r}  attribute={attr!r}  ({kind})")
    print(f"Object:     {obj}        ({score_txt})")
    print(f"Crop saved: {shown}")
    print("Options:    " + " | ".join(f"{a} {probs[a]:.2f}" for a in options))
    print(f"Verdict:    {'SUPPORTED' if supported else 'NOT SUPPORTED'}"
          f"  (confidence {probs[winner]:.2f})")
    if crop.fell_back:
        print("Note:       the object was not detected; this verdict used the whole image.")
    print("Note:       confidence is uncalibrated (D-011); the contrast came from the "
          "demo antonym map, never from the evaluation path (TRD §13).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
