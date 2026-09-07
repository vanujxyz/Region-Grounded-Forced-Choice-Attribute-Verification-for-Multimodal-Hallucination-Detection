# PRD — Region-Grounded Forced-Choice Attribute Verification for Multimodal Hallucination Detection

**Project owner:** Vanuj Gangrade (23BDS0282), Aryan Bhandari (23BDS0223), Harsh Parmeshwaram (23BDS0007)
**Guide:** Dr. Gautam Majumdar, VIT Vellore
**Document status:** v1.0 — build specification. Read together with `TRD.md`.

---

## 1. What this project is

A **hallucination detector** for vision-language models. It takes a photograph and a claim that some AI made about that photograph, and decides whether the claim is supported by the image.

The system is built from small open-weight vision models. It contains **no large language model** and calls **no paid API**. It runs on a laptop with an RTX 3050 (4–6 GB VRAM) or a free Colab/Kaggle GPU.

## 2. The one contribution

Existing detectors check attribute claims ("the apple is red") by scoring the text against the **whole image** with a contrastive model, then thresholding the score. This is known to be unreliable: contrastive vision-language models bind attributes to objects poorly (Yuksekgonul et al., ICLR 2023; Thrush et al., CVPR 2022). UNIHD (Chen et al., 2024) states the gap explicitly — it reports minimal improvement on attribute-level hallucination and attributes this to the absence of a specialised attribute tool.

**This project builds that tool.**

The method has two changes from the standard approach:

1. **Region grounding** — locate the object named in the claim, crop it out, and evaluate on the crop alone rather than the full image.
2. **Forced choice** — instead of asking "how well does 'sunny sky' match?" and thresholding, present the model with the candidate attribute values as mutually exclusive options and take the argmax.

These are two independent changes, and the evaluation isolates them (see §5).

## 3. Non-goals

State these in the paper's related-work section rather than claiming them:

- Decompose-and-verify pipelines are established (FaithScore, Woodpecker, UNIHD). The pipeline shape is **not** a contribution.
- Region-level, type-specific hallucination scoring exists (ESREAL, ECCV 2024). Region grounding on its own is **not** a contribution.
- Per-type reliability breakdowns exist (DHCP, HALP, CADMP). "Attributes are harder than objects" is **not** a finding.
- Counting via object detection exists (GroundCount, 2026).

The contribution is the specific combination applied to attribute verification, measured against a controlled baseline.

Also out of scope for v1:
- Languages other than English
- OCR / scene-text verification
- Free-form caption span marking (the generative half of AMBER) — deferred to §7
- Any model training or fine-tuning. All models are used zero-shot.

## 4. Deliverables

| # | Deliverable | Definition of done |
|---|---|---|
| D1 | Working detector | `python -m src.run --config configs/main.yaml` produces predictions for all AMBER attribute questions |
| D2 | 2×2 ablation results | `results/tables/table1_ablation.csv` populated with four accuracy figures + 95% CI |
| D3 | Sub-type breakdown | `results/tables/table2_subtype.csv` — accuracy split by state / action / number |
| D4 | Full-pipeline results | `results/tables/table3_pipeline.csv` — attribute module dropped into the existence+count+relation pipeline |
| D5 | Cost comparison | `results/tables/table4_cost.csv` — parameter count, wall-clock per query, API cost = 0 |
| D6 | Live demo | `python -m src.demo --image X.jpg --claim "the sky is sunny"` prints verdict + confidence + crop path |
| D7 | Paper | 6 pages, IEEE two-column |

## 5. The core experiment (this is the paper)

Two changes, tested independently, on AMBER's attribute questions.

|  | **Independent + threshold** | **Forced choice (argmax)** |
|---|---|---|
| **Whole image** | Cell A — the standard baseline | Cell B |
| **Cropped region** | Cell C | Cell D — the proposed method |

- **A vs C** isolates the effect of region grounding.
- **A vs B** isolates the effect of forced choice.
- **A vs D** is the headline number.
- **B, C** prevent a reviewer from asking whether the gain came from only one of the two changes.

**Success criterion:** D beats A by a margin whose 95% bootstrap confidence interval excludes zero.

**If D does not beat A**, the project is not a failure. The paper becomes a measured negative result: "region-grounded forced-choice attribute verification is widely assumed to fix attribute hallucination; we implement it faithfully and show it does not, and we explain why." This is publishable at the target venue. **Do not abandon or silently retune the method to chase a positive result.** Report what the measurement says.

## 6. Data

**AMBER** (Wang et al., 2023). Repository: `https://github.com/junyangwang0410/AMBER`

Verified contents of `data/annotations.json` (15,220 records, flat list, record for id *n* is at index *n−1*):

| Type | Count | Question form | Answer |
|---|---|---|---|
| `generative` | 1,004 | "Describe this image." | object lists (`truth`, `hallu`) |
| `discriminative-hallucination` | 4,924 | "Is there a cloud in this image?" | `yes` / `no` |
| `discriminative-attribute-state` | 4,764 | "Is the sky sunny in this image?" | `yes` / `no` |
| `discriminative-attribute-number` | 2,072 | "Are there three people in this image?" | `yes` / `no` |
| `discriminative-attribute-action` | 792 | "Does the man sit in this image?" | `yes` / `no` |
| `discriminative-relation` | 975 | "Is there direct contact between the person and grass?" | `yes` / `no` |
| `relation` | 689 | same form | `yes` / `no` |

**Critical property, verified.** The `state` and `action` questions occur in **contrastive pairs** over the same (image, object): one true attribute and one false attribute.

```
AMBER_1.jpg, sky      -> ('sunny', yes, id=1005) , ('gloomy', no, id=1006)
AMBER_1.jpg, mountain -> ('short', yes, id=1007) , ('tall',   no, id=1008)
AMBER_1.jpg, grass    -> ('green', yes, id=1011) , ('blue',   no, id=1012)
AMBER_2.jpg, man      -> ('sit',   yes)          , ('stand',  no)
```

Measured: **2,378 of 2,380** state groups are clean yes/no pairs (2 groups have 4 members — exclude them). **396 of 396** action groups are clean pairs. Answers are exactly balanced: 2382/2382 for state, 396/396 for action, 1036/1036 for number.

**This means the forced-choice option set is supplied by the dataset.** No antonym lexicon needs to be written. This was the main implementation risk and it is resolved.

Number questions do **not** pair reliably (1,506 singletons, 283 pairs), so number is handled by the counting module, not by forced choice.

**Vocabulary is closed but non-punitive.** `data/relation.json` holds 340 core object types (418 with synonyms). The official scorer discards nouns outside this vocabulary before scoring rather than penalising them, so correctly identifying an out-of-vocabulary object costs nothing. State this as a scope limitation: results cover a 340-object vocabulary.

**Images** are not in the repo. Download separately from the Google Drive link in the AMBER README (1,004 files named `AMBER_1.jpg` … `AMBER_1004.jpg`).

## 7. Scope beyond the core experiment

Build these so the capstone demo is a complete system, but do not claim novelty for them:

- **Existence checker** — OWLv2 detection, thresholded. Evaluated on the 4,924 `discriminative-hallucination` questions.
- **Counting checker** — OWLv2 detection, NMS de-duplication, integer comparison. Evaluated on the 2,072 `number` questions.
- **Relation checker** — off-the-shelf. Evaluated on the 1,664 relation questions.
- **Claim splitter** — spaCy dependency parse, used by the demo to handle free-text input.

## 8. Rules the build must follow

1. **No number is invented.** A metric not computed is written as `NOT_COMPUTED`, never as `0` or a placeholder.
2. **Failures are loud.** If a model fails to load or a detection returns nothing unexpected, raise. Never silently return a default verdict.
3. **The evaluation harness is written and unit-tested before any model is connected.** A scorer written after the fact gets shaped to flatter the system.
4. **Splits are by image, never by question.** The same image must not appear in both a tuning and an evaluation split.
5. **Every result file records** the git commit hash, the config used, the model revisions, and the run timestamp.
6. **The test split is opened once.** All development uses the dev split.
7. **Every run is seeded and reproducible.** Same config + same commit = identical numbers.

## 9. Milestones

| M | Content | Gate |
|---|---|---|
| M0 | Repo, data downloaded, evaluation harness + unit tests, splits frozen | Harness passes tests with synthetic predictions |
| M1 | Cell A baseline running on 100 dev pairs | One real accuracy number printed |
| M2 | All four ablation cells on 100 dev pairs | Table 1 exists in draft form — **go/no-go decision by the project owner** — **PASSED 2026-09-07** |
| M3 | All four cells on full dev set + bootstrap CIs | D2 complete; **D1 and D3 complete for `state` and `action` only** |
| M4 | Existence, counting, relation modules; full pipeline | D4, D5, D6 complete |
| M5 | Test split opened once; paper written | D7 complete |

**M2 is the decision point.** If the effect is absent at 100 pairs, the negative-result framing of §5 applies and is settled before proceeding to M3.

**Decision authority.** The project owner is the sole decision-maker on gates. An
earlier version of this section routed the M2 gate through a review with the
guide; that no longer applies. See `docs/DECISIONS.md` D-028.

**M3 gate scope - explicit, not ambiguous.** D1 as written names "all AMBER
attribute questions" and D3 names a breakdown by "state / action / number".
Both therefore depend on the 2,072 `discriminative-attribute-number` questions,
which do not pair reliably and are routed by TRD §4 to `counting.py` - an **M4**
module. Accordingly:

- **At M3, D1 and D3 are complete for `state` and `action` only.** They are
  **not** declared complete outright.
- **The `number` breakdown slips to M4**, and D1 and D3 close only when
  `counting.py` exists and the number questions have been scored.
- D2 (the 2x2 ablation table) completes in full at M3.

See `docs/DECISIONS.md` D-029.

**M2 outcome, 2026-09-07: PASSED.** On 100 dev pairs, Cell D 0.8600 vs Cell A
0.6300; paired bootstrap D − A = +0.2300, 95% CI [+0.1515, +0.3021], excluding
zero. The §5 success criterion is met at this sample size, so the negative-result
framing is not triggered. All dev numbers are fit-on-eval where a threshold is
involved (D-012); the headline remains the M5 test-split run with tau frozen from
the dev fit.
