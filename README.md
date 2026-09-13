# Region-Grounded Forced-Choice Attribute Verification for Multimodal Hallucination Detection

A hallucination detector for vision–language models. Given a photograph and a
claim about it — *"the sky is sunny"* — it decides whether the image supports the
claim.

No large language model. No paid API. Two open-weight vision models totalling
358M parameters, running in **1.65 GiB of VRAM** on a 4 GB laptop GPU.

**Vanuj Gangrade, Aryan Bhandari, Harsh Parmeshwaram** — VIT Vellore
Guide: Dr. Gautam Majumdar

---

## The result

Measured on AMBER, with a held-out test split opened **once** and every threshold
frozen from the development fit.

|  | threshold | forced choice |
|---|---|---|
| **whole image** | A **0.5751** (baseline) | B **0.7787** |
| **cropped region** | C **0.5953** | **D 0.8012** (proposed) |

**D − A = +0.2260**, 95% CI **[+0.2001, +0.2514]** — against +0.2232 on the
development set. The effect replicated out of sample almost exactly.

### But the attribution is not what the title predicts

| contrast | isolates | Δ | 95% CI | |
|---|---|---|---|---|
| B − A | **forced choice alone** | **+0.2036** | [+0.1781, +0.2288] | excludes zero |
| C − A | **region grounding alone** | **+0.0201** | [+0.0012, +0.0391] | excludes zero, *barely* |
| D − B | cropping given forced choice | +0.0225 | [−0.0012, +0.0462] | **includes zero** |

**Forced choice carries essentially the whole effect.** Region grounding clears
zero by roughly one part in a thousand. A positive interaction found on the
development set (+0.0223) **did not replicate** on test (+0.0023).

The honest summary: *changing the decision rule is what matters; localisation
helps far less than is commonly assumed.*

---

## It replicates on a second dataset

The obvious objection to everything above is that the effect might be a property
of AMBER — AMBER supplies the contrastive pair, balances it perfectly, and (see
below) leaks the answer through question ids.

So the method was run, **unchanged**, on a second dataset: **SHROOM-Vis** — 900
images from the SHROOM hallucination-detection image set, annotated in AMBER's
exact schema, of which 898 are usable: **2,523 pairs, 5,046 questions**. Same two
pinned model revisions, same prompt template, same thresholds-frozen-from-dev
protocol, same seed. `configs/main.yaml` was not touched. (The annotations are in
this repo; the images are supplied separately and not redistributed here.)

|  | threshold | forced choice |
|---|---|---|
| **whole image** | A **0.6177** | B **0.8413** |
| **cropped region** | C **0.6541** | **D 0.8532** |

| contrast | SHROOM-Vis test | AMBER test |
|---|---|---|
| **D − A** the headline | **+0.2354** [+0.2103, +0.2608] | +0.2260 [+0.2001, +0.2514] |
| **B − A** forced choice alone | **+0.2235** [+0.1962, +0.2507] | +0.2036 [+0.1781, +0.2288] |
| **C − A** region grounding alone | **+0.0364** [+0.0177, +0.0553] | +0.0201 [+0.0012, +0.0391] |
| **D − B** cropping given forced choice | +0.0119 [−0.0159, +0.0406] | +0.0225 [−0.0012, +0.0462] |

**The headline holds and forced choice is still the mechanism** — +0.2235 of the
+0.2354, i.e. 95% of it. On the *development* split the agreement with AMBER was
uncanny: D − A **+0.2261** vs +0.2260, B − A **+0.2035** vs +0.2036, D − B
**+0.0226** vs +0.0225.

**One claim this revises upward.** `docs/DECISIONS.md` D-050 called region
grounding *"marginal and split-dependent"* because C − A cleared zero by one part
in a thousand on AMBER. On SHROOM-Vis it is **+0.0364 with a lower bound of
+0.0177** — an order of magnitude clear of zero, on both splits. Region grounding
is *reliably* small rather than *barely* non-zero. It is still 6× smaller than
forced choice, so the framing does not change (D-056).

Cropping helps most where you would expect: on objects **outside** AMBER's closed
340-word vocabulary, which SHROOM-Vis deliberately contains (C − A = +0.0418
out-of-vocabulary vs +0.0337 in-vocabulary).

**And the AMBER id artifact is gone.** `--module position-only` — answer yes to
the lower question id, never open the image — scores **1.0000 on AMBER** and
**0.5503** here, because `scripts/shroom_build.py` assigns the ids within each
pair by a seeded coin flip (D-054).

**Every control replicated too.** The base-rate controls (A′/C′) and the
external-contrast control (D-ext) were run after the headline, so nothing above
depends on them:

| control | SHROOM-Vis test | AMBER test |
|---|---|---|
| **B − A′** forced choice, net of the free 50/50 prior | **+0.2249** [+0.1979, +0.2510] | widened the gap |
| **D-ext − A** competitor from a pre-committed antonym map, each question answered alone | **+0.1773** [+0.1489, +0.2054] | +0.1901 |

`configs/antonyms.yaml` was committed for AMBER before any result existed and was
**not modified** for SHROOM-Vis; it covers 84.3% of the test questions as-is
(D-059).

### Two things this does not claim

- **The annotator was a VLM (Claude Opus 5), not a human.** SHROOM-Vis is
  **silver-standard**. Neither model under test was consulted while annotating,
  so the labels are not the system's own output fed back to it — but they are not
  human-verified either. The policy that fixes how negatives are chosen (which is
  what sets difficulty in a forced-choice benchmark) was frozen in
  `data/shroom/ANNOTATION_POLICY.md` before any result existed.
- **SHROOM-Vis is easier than AMBER** — every cell scores 4–6 points higher. So
  the absolute numbers are not comparable across datasets and only the
  *contrasts* are. Every claim above is a difference between cells measured on
  the same questions (D-058).

Reproduce it:

```bat
python scripts\shroom_build.py
python -m src.run --dataset shroom --cell all --split dev
set ALLOW_TEST_SPLIT=1
python -m src.run --dataset shroom --cell all --split test
python scripts\shroom_report.py test
```

## Two claims this project withdrew

Both are reported rather than quietly corrected, and both are in
[`docs/DECISIONS.md`](docs/DECISIONS.md):

1. **D-030 → D-038.** "Region grounding alone contributes nothing measurable,"
   asserted on 100 pairs, falsified by the full development set.
2. **D-041 → D-050.** The superadditive interaction, found on development,
   failed to replicate on test.

Neither withdrawal was accompanied by a change to the method. The tuning log is
empty: no threshold, padding, prompt template or constant was ever adjusted to
improve a reported number.

---

## Three benchmark artifacts found in AMBER

Each lets a strategy that never opens an image score **1.0000**. Reported rather
than exploited.

| artifact | trivial strategy |
|---|---|
| In all 2,774 attribute pairs the true attribute holds the **lower question id** | answer yes to the lower id |
| All 4,924 existence questions ask about **absent** objects (gold is always `no`) | answer no always |
| Relation gold is **constant per type** — 975 all-yes, 689 all-no | answer yes / no respectively |

The three question types the contribution rests on — `state`, `action`, `number`
— are exactly balanced and free of all three. A permutation test confirms no cell
in the system reads question ids.

---

## Controls

| control | rules out |
|---|---|
| **A′, C′** | Forced choice gets a correct 50/50 prior free from AMBER's balanced pairs. Giving the threshold cells that same prior *widened* the gap. |
| **D-ext** | The method might exploit AMBER supplying the contrast. D-ext draws its competing attribute from an antonym list committed **before any result existed**, scoring each question alone. Still **+0.1901** over baseline. |
| **position-only** | The id-ordering artifact above, implemented and labelled explicitly. |

All three were re-run on the second dataset and all three transferred — including
D-ext at **+0.1773** using an antonym map that was not modified for it (D-059).

---

## Quick start

```bat
conda create -n capstone python=3.11 -y
conda activate capstone
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
python -m spacy download en_core_web_sm
```

Get the data (the JSON is included; the images are not):

```bat
git clone --depth 1 https://github.com/junyangwang0410/AMBER.git data/amber_repo
```

Download the 1,004 AMBER images from the Google Drive link in the upstream README
and unzip them **flat** into `data/images/` as `AMBER_1.jpg … AMBER_1004.jpg`.

For the second dataset, place the SHROOM images in
`shroom-visions-images/shroom-vis-images/`. The **annotations are tracked**
(`data/shroom/`); the 2,495 images are not redistributed here. Without them the
SHROOM tests and tables still work — only re-running the models needs the pixels.

Then:

```bat
set HF_HOME=C:\hf
python -m pytest -q                          :: 252 tests, no GPU
python reproduce.py --tables-only --limit 0  :: rebuild every table, no GPU
python scripts\test_report.py                :: the headline test-split result
python reproduce.py                          :: re-run the models, ~4 min
```

Live demo:

```bat
python -m src.demo --image data\images\AMBER_1.jpg --claim "the sky is sunny"
```

Full command reference and a step-by-step walkthrough of how the system works:
**[`docs/GUIDE.md`](docs/GUIDE.md)**.

---

## How it works

1. **Locate** — OWLv2 is prompted with `"a photo of a {object}"`; boxes below
   0.10 are dropped, overlaps removed by NMS at IoU 0.5.
2. **Crop** — the best box is padded 10% per side, clipped, and forced to a
   64×64 minimum. If nothing is detected the crop falls back to the whole image
   and the record is flagged (3.4% of questions).
3. **Choose** — SigLIP scores `"a photo of a {attribute} {object}"` for each
   candidate attribute against the crop. Forced choice takes the argmax;
   the baseline thresholds each score independently.

The two models are **never resident simultaneously** — one loads, runs its whole
stage, is freed, then the next loads. That is what keeps it inside 4 GB.

Confidence intervals bootstrap over **images**, not questions, since questions
from one image are correlated.

---

## Repository

```
configs/         thresholds, pinned model revisions, the external antonym map
data/amber/      AMBER benchmark JSON (Apache-2.0, see data/amber/NOTICE)
data/shroom/     SHROOM-Vis: the second dataset, its frozen annotation policy,
                 the raw annotation log and the built AMBER-schema JSON
src/data/        loader, contrastive pair construction, frozen splits,
                 shroom.py (the second dataset's reader)
src/modules/     detector, the four cells, existence, counting, relation
src/eval/        metrics and image-level bootstrap
scripts/         shroom_build.py, shroom_report.py, shroom_thumbs.py, ...
results/tables/  every reported table
docs/            DECISIONS.md (63 entries), GUIDE.md, paper/main.tex
```

Every run writes a manifest recording the git commit, resolved config, both model
revision hashes, seed, split, package versions and GPU.

---

## Limitations

- Development thresholds are fitted on the evaluation data, which favours the
  **baseline** — the comparison is conservative. The headline uses test-split
  numbers with thresholds frozen.
- 3,858 development questions rest on only 661 distinct attribute triples; the
  bootstrap resamples 696 images.
- Region grounding's effect is small. On AMBER it was *barely* non-zero; the
  second dataset showed it to be *reliably* small (+0.0364, lower bound +0.0177)
  rather than robust-and-large. It is still 6x smaller than forced choice.
- D-ext's antonym coverage is 79% for actions on AMBER, and 84% overall on
  SHROOM-Vis — many verbs (*swim*, *jump*, *surf*) have no English opposite, and
  neither do material words like *metal*, *brick* or *stone*. Forced choice
  requires a competing hypothesis; where none exists, the method does not apply.
- Confidences are uncalibrated; calibration is not a claim of this work.
- English only, zero-shot, no fine-tuning. AMBER's vocabulary is a closed
  340-object list; SHROOM-Vis opens that to 433 objects but is not a systematic
  sample of anything.

On the second dataset specifically:

- **SHROOM-Vis is VLM-annotated (silver-standard), not human-verified.** The
  annotator was Claude Opus 5 — not either model under test, so the labels are
  not the system's own output fed back to it, but that is the whole of the
  independence claim.
- **It is 4–6 points easier than AMBER in every cell**, most likely because its
  policy drops any pair whose negative might arguably hold. Only the *contrasts*
  are comparable across the two datasets, never the absolute accuracies.
- **900 of 2,495 available images are annotated**, chosen as a prefix of sorted
  filenames rather than at random, so the object mix is not a random sample of
  the image set.
- Its 2,523 pairs rest on 1,127 distinct triples and the test bootstrap resamples
  270 images — a smaller image base than AMBER's 302.

---

## Attribution

Benchmark data from [AMBER](https://github.com/junyangwang0410/AMBER)
(Wang et al., 2023), Copyright 2023 Alibaba X-PLUG Team, Apache License 2.0 —
redistributed unmodified under `data/amber/` with its LICENSE and NOTICE.

Models: [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) and
[SigLIP](https://huggingface.co/google/siglip-base-patch16-224), both Google,
both used zero-shot at pinned revisions.
