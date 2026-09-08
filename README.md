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

Then:

```bat
set HF_HOME=C:\hf
python -m pytest -q                          :: 241 tests, no GPU
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
src/data/        loader, contrastive pair construction, frozen splits
src/modules/     detector, the four cells, existence, counting, relation
src/eval/        metrics and image-level bootstrap
results/tables/  every reported table
docs/            DECISIONS.md (54 entries), GUIDE.md, paper/main.tex
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
- Region grounding's effect is marginal and may not be robust.
- D-ext's antonym coverage is 79% for actions — many verbs (*swim*, *jump*,
  *surf*) have no English opposite. Forced choice requires a competing
  hypothesis; where none exists, the method does not apply.
- Confidences are uncalibrated; calibration is not a claim of this work.
- English only, AMBER's closed 340-object vocabulary, zero-shot, no fine-tuning.

---

## Attribution

Benchmark data from [AMBER](https://github.com/junyangwang0410/AMBER)
(Wang et al., 2023), Copyright 2023 Alibaba X-PLUG Team, Apache License 2.0 —
redistributed unmodified under `data/amber/` with its LICENSE and NOTICE.

Models: [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) and
[SigLIP](https://huggingface.co/google/siglip-base-patch16-224), both Google,
both used zero-shot at pinned revisions.
