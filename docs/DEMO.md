# Region-Grounded Forced-Choice Attribute Verification — reproduce & read

A hallucination detector for vision-language models: given a photograph and a
claim about it, decide whether the claim is supported. No LLM, no paid API, runs
on a 4 GB laptop GPU.

**Status: M2 complete, M3 in progress.** Everything below is the **development**
split. Nothing here is a final result — see [Limits](#limits).

---

## One command

```bash
python reproduce.py
```

Runs all six cells on 100 dev pairs, rebuilds every table in `results/tables/`,
and prints the 2×2, the attribution contrasts, what has been ruled out, and the
limits. Takes about four minutes on an RTX 3050.

```bash
python reproduce.py --limit 0        # full dev set (~55 min)
python reproduce.py --tables-only    # rebuild tables, no models
```

It cannot touch the test split: `--split` accepts only `dev`, and reading `test`
additionally requires `ALLOW_TEST_SPLIT=1` in the environment.

### Setup

```bash
conda create -n capstone python=3.11 -y
conda activate capstone
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
python -m spacy download en_core_web_sm          # needed for the M4 demo only
```

Then get the data:

```bash
git clone --depth 1 https://github.com/junyangwang0410/AMBER.git data/amber_repo
cp -r data/amber_repo/data/* data/amber/
# Images: download from the Google Drive link in data/amber_repo/README.md.
# Unzip FLAT into data/images/ -- AMBER_1.jpg ... AMBER_1004.jpg, no subfolder.
```

On Windows, set a short HuggingFace cache to avoid MAX_PATH failures, and
**quote it** — unquoted `HF_HOME=C:\hf` collapses to `C:hf`, which resolves
relative to the current directory:

```bash
export HF_HOME='C:\hf'
```

---

## What the system does

For a claim like *"the sky is sunny"* about `AMBER_1.jpg`:

1. **Locate** — OWLv2 detects `"a photo of a sky"`, keeps boxes scoring ≥ 0.10,
   NMS at IoU 0.5, takes the highest-scoring box.
2. **Crop** — pad the box by 10% per side, clip to the image, enforce a minimum
   64×64. If nothing was detected, fall back to the full image and record it.
3. **Choose** — SigLIP scores `"a photo of a sunny sky"` and
   `"a photo of a gloomy sky"` against the crop; the higher score wins.

The contrast set comes from AMBER itself: its state and action questions occur in
contrastive pairs over the same (image, object), one true attribute and one
false. No antonym lexicon is needed on the evaluation path.

Models: `google/owlv2-base-patch16-ensemble` (155M) and
`google/siglip-base-patch16-224` (203M), both zero-shot, both pinned to exact
revision hashes in `configs/main.yaml`. Never loaded simultaneously — one loads,
runs its whole stage, is freed, then the next loads. Peak VRAM 1.65 GiB of 4.00.

---

## The experiment

Two changes to the standard approach, tested independently.

|  | **threshold** | **forced choice (argmax)** |
|---|---|---|
| **whole image** | **A** — the standard baseline | **B** |
| **cropped region** | **C** | **D** — the proposed method |

Plus two controls:

| | |
|---|---|
| **A′, C′** | identical to A and C, but tau chosen so predicted-yes matches the known base rate rather than maximising accuracy |
| **position-only** | answers from the question id alone, never opens the image — a benchmark artifact, reported so no reviewer discovers it first |

### Results — 100 dev pairs, 200 questions, 62 distinct triples, 33 images

|  | threshold | forced choice |
|---|---|---|
| **whole image** | A **0.6300** | B **0.7900** |
| **cropped region** | C **0.6400** | D **0.8600** |

| control | accuracy | |
|---|---|---|
| A′ base-rate matched | 0.6000 | |
| C′ base-rate matched | 0.6100 | |
| position-only | 1.0000 | ⚠ **benchmark artifact, sees no image** |

### Attribution — paired bootstrap over images, 10,000 resamples, seed 20260907

| contrast | isolates | Δ | 95% CI | |
|---|---|---|---|---|
| C − A | region grounding alone | +0.0100 | [−0.0490, +0.0680] | **includes zero** |
| C′ − A′ | region grounding, base-rate matched | +0.0100 | [−0.0348, +0.0510] | **includes zero** |
| B − A | forced choice alone | +0.1600 | [+0.0859, +0.2330] | excludes zero |
| B − A′ | forced choice, net of base rate | +0.1900 | [+0.1094, +0.2588] | excludes zero |
| D − C | forced choice, given cropping | +0.2200 | [+0.1402, +0.2935] | excludes zero |
| D − C′ | forced choice given cropping, net of base rate | +0.2500 | [+0.1700, +0.3182] | excludes zero |
| D − B | cropping, given forced choice | +0.0700 | [+0.0217, +0.1215] | excludes zero |
| **D − A** | **both (headline)** | **+0.2300** | **[+0.1515, +0.3021]** | **excludes zero** |

Interaction (D−A) − (B−A) − (C−A) = **+0.0600**, superadditive.

**Read it this way.** Forced choice carries the effect on its own. **Region
grounding alone does nothing measurable** — +0.01 with a CI straddling zero,
under both threshold rules. Cropping pays only once forced choice is in place
(+0.07). The finding is the interaction, not two independent gains. The raw and
base-rate-matched contrasts are always quoted as a pair — (+0.16, +0.19) and
(+0.22, +0.25) — never one alone.

### By sub-type

| cell | state (n=166) | action (n=34) |
|---|---|---|
| A | 0.6506 | 0.5294 |
| B | 0.7590 | 0.9412 |
| C | 0.6084 | 0.7941 |
| D | 0.8313 | 1.0000 |

Cell D's 1.0000 on action rests on **17 pairs over 14 distinct triples** of easy
contrasts (sit/stand, run/walk, laugh/cry). Treat it as a small-sample number.

---

## What has been ruled out

### Id leakage — falsified

AMBER places the true attribute at the **lower question id in 2774 of 2774
pairs**, with the two ids always adjacent. A detector that ignores the image and
answers yes to the lower id scores **1.0000**.

Tested by permuting which id holds the gold positive (diagnostic only, never a
reported number — an AMBER id is bound to a question's text, so permuting
fabricates a variant and forfeits comparability with published AMBER results):

| | real ids | permuted |
|---|---|---|
| position-only | 1.0000 | **0.4100** |
| Cell A | 0.6300 | **0.6300** (Δ 0.000000) |
| Cell D | 0.8600 | **0.8600** (Δ 0.000000) |

The `(image, obj, attr, pred, gold)` tuples were identical before and after — only
the filing id changed. **No cell reads a question id.** Guarded by static and
behavioural tests, including one asserting pair polarity is selected from the gold
answer and never from an id comparison.

### Balanced-pairs confound — tested, and it does not explain the effect

Forced choice emits exactly one "yes" per pair *by construction*, and AMBER's
pairs are exactly balanced, so it is handed a correct base rate free. Cells A′/C′
give the threshold cells the same prior.

**The gap widens rather than closing:** +0.16 → +0.19 (whole image), +0.22 → +0.25
(cropped). Matching the base rate *costs* the threshold cells accuracy, because
their accuracy-maximising tau was deliberately unbalanced — A predicts yes 136/200,
C predicts 60/200. The prior is not the source of the advantage.

The two constraints differ: A′/C′ impose a **global** base rate, forced choice a
**per-pair** one. A threshold on independent scores has no representation of the
pair and cannot take the pairwise constraint without becoming forced choice.

---

## Limits

Stated plainly. None of these is resolved.

1. **These are development numbers, and they are fit-on-eval.** Cells A, C, A′ and
   C′ each fit tau on the same data they are scored on. That favours the
   *baseline*, so the comparison is conservative — but no dev number is a final
   result. The headline result is the **M5 test-split run with tau frozen from the
   dev fit**. The test split has not been opened.
2. **The effective sample is much smaller than n.** 200 questions rest on 62
   distinct (object, positive, negative) triples; across the full pair set, 2774
   pairs cover only 828 triples over 182 objects, and `('sky','sunny','gloomy')`
   alone appears 235 times. Bootstrap resamples **33 images**, not 200 questions.
3. **Detector fallback rate 0.05.** On 10 of 200 questions OWLv2 found nothing and
   the crop *is* the full image, making Cell D behave as Cell B there. Cell D
   scores 0.8737 on the 190 non-fallback questions and 0.6000 on the 10 fallback
   ones. This bounds what the D-vs-A comparison measures.
4. **Confidences are uncalibrated** and calibration is not a claim of this work.
   On the logit scale used for tau, `sigmoid(10·(s−τ))` saturates to 0 or 1 almost
   everywhere.
5. **Cell D's action score of 1.0000 is 17 pairs.** Not a claim.
6. **Scope:** English only; AMBER's closed 340-object vocabulary; zero-shot, no
   training or fine-tuning; no OCR or scene-text; no free-form caption span
   marking. Number questions do not pair reliably and are routed to a counting
   module, not to forced choice.
7. **Not yet built:** the existence, counting and relation modules; the full
   pipeline (D4); cost measurement (D5); the live demo (D6); the paper (D7). See
   the milestone audit in `docs/DECISIONS.md`.
8. **A spec defect is outstanding.** `REL_RE` in TRD §10 parses 0 of 1664 relation
   questions; the corrected regex is recorded in D-001 and applies at M4.

---

## Layout

```
configs/main.yaml         all paths (relative), thresholds, pinned model revisions
data/amber/               annotations.json, query/, relation.json
data/images/              AMBER_1.jpg … AMBER_1004.jpg  (manual download)
data/splits/splits.json   frozen 702 dev / 302 test, by image, seed 20260907
src/data/                 loader, contrastive pair construction, splits
src/modules/              detector (OWLv2), attribute (the cells), position_baseline
src/eval/                 metrics, bootstrap
results/raw/              per-question JSONL + a manifest per run
results/tables/           table1_ablation, table1_comparison, table2_subtype
docs/DECISIONS.md         every interpretation, deviation and falsified hypothesis
reproduce.py              this document's numbers, end to end
```

Every run writes a manifest recording the git commit, resolved config, both model
revision hashes, seed, split, package versions and GPU. `pytest` runs 178 tests,
none of which require a GPU.

---

## Conventions worth knowing before reading any number

- **`NOT_COMPUTED`** — the metric could have been computed and was not.
- **`NOT_APPLICABLE`** — the quantity does not exist for this configuration
  (forced-choice cells have no tau at all).
- Neither ever stands in for a number that was actually produced. A bare `0` or
  `0.0` in a metric field fails the test suite.
- Every tuning action, spec deviation and falsified hypothesis is logged in
  `docs/DECISIONS.md` with its reason. **The tuning log is empty:** no threshold,
  padding, prompt template or constant has been adjusted to improve a result.
