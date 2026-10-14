# VAW experiment — pre-registered protocol (D-061)

Frozen on 2026-10-13, **before any model was run on VAW**. If anything below changes after a result exists, the change goes in a new D-entry together with the reason. The file is never edited in place.

## Why VAW

VISOR (Zhang, Zhan & Wang, arXiv:2608.11024) studies attribute hallucination on VAW (Pham et al., CVPR 2021) using yes/no probing. It finds that colour and state errors are *threshold-placement* errors, and it corrects them with calibration (VISOR-Calib). Running the 2×2 on VAW therefore tests our claim in the setting where the calibration remedy was established. VAW is a third dataset, with human labels, that was built independently of AMBER and SHROOM-Vis.

## Data

* Source: `adobe-research/vaw_dataset`, files `data/{val,test,train_part1,train_part2}.json` and `data/attribute_types.json`. Images are Visual Genome v1.4 (`VG_100K`, `VG_100K_2`).
* Attribute types (the same three as VISOR, Sec. IV-A): `color` (61 words), `material` (52), `state` (11), taken verbatim from `attribute_types.json`.
* **Pair construction.** For an instance and a type T, P = positive_attributes ∩ T and N = (negative_attributes ∩ T) \ P. If both are non-empty, one pair is formed: a⁺ = rng.choice(sorted(P)) and a⁻ = rng.choice(sorted(N)), using `random.Random(20260907)`. Questions: "Is the {object} {a⁺}?" (gold yes) and "Is the {object} {a⁻}?" (gold no). Both questions come from VAW's human labels; neither is generated.
* **Eligibility filters**, applied before sampling:
  1. The gold box has width and height ≥ 20 px.
  2. The object name occurs only once among VAW-annotated instances of that image. This is a partial guard against referential ambiguity: VAW does not annotate every instance.
  3. At most one pair per image per split, so the bootstrap clusters are images. Types are sampled in the order state → material → color (scarcest first), so the one-pair-per-image rule does not starve the rare types. (Amended before any model run.)
* **Splits.**
  * *test*: VAW `test.json`, with quotas color 600, material 500, state 400 (1,500 pairs, 3,000 questions).
  * *dev*: VAW `val.json` first. Because val has only 67 eligible state pairs and 342 material pairs, the quotas (600/500/400) are then filled from `train_part1.json` and, if that is still short, from `train_part2.json`. dev and test share no image (asserted). (Amended before any model run: with the eligibility filters, `train_part1` alone left the state quota 47 pairs short.)
* Question ids within a pair are assigned by seeded coin flip, as in SHROOM-Vis (D-054), so the position-only artifact cannot recur.

## Cells (identical code, config, prompt and model revisions to AMBER/SHROOM)

| cell | region | decision | free parameters (fit on dev only) |
|---|---|---|---|
| A | whole image | threshold | τ |
| B | whole image | forced choice | none |
| C | OWLv2 crop | threshold | τ |
| D | OWLv2 crop | forced choice | none |
| A′, C′ | as A, C | base-rate-matched τ | τ |
| Cal-null-A / Cal-null-C | whole / crop | VISOR-style prior correction: yes iff s(R,t) − γ·s(I∅,t) ≥ τ, with I∅ a uniform grey 128×128 image (VISOR, Sec. III) | γ ∈ {0, 0.05, …, 2.0}, τ |
| Cal-word-A / Cal-word-C | whole / crop | per-attribute-word threshold τ_w; τ_w = global τ when the word has < 10 dev questions | τ_w |
| C\*, D\* | VAW gold box (same padding, minimum size) | threshold / forced choice | τ (C\*) |
| D-ext | OWLv2 crop | forced choice against `configs/antonyms.yaml` (unchanged) | none |

Here s is the SigLIP logit for the prompt "a photo of a {attr} {obj}". The 2×2 (A–D) is the primary analysis. Calibration cells, oracle-box cells and D-ext are pre-declared secondary analyses.

## Metrics and statistics

* Primary: accuracy on the test split, with τ/γ frozen from dev. Paired image-level bootstrap, 10,000 resamples, seed 20260907, 95% percentile intervals.
* Primary contrasts: B−A (decision rule), C−A (grounding), D−B, D−A, and the interaction.
* Secondary contrasts: D−Cal-null-C and B−Cal-null-A (forced choice vs calibration), C\*−C and D\*−D (detector error), and D-ext−A on its covered subset.
* Also reported: hallucinated / non-hallucinated precision, recall and F1, plus Mac.F1 (the UNIHD/OmniHallu protocol), and VISOR's Acc / F1 / Yes-ratio / FPR, per type.
* Pair outcomes: correct, both-yes, both-no, reversed.

## What would falsify the paper's claim on VAW

If C−A ≥ B−A on the test split (grounding contributes at least as much as the decision rule), or if Cal-null-C ≥ D (calibration closes the gap), the claim does not transfer to VAW, and that will be reported as the result.
