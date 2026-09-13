# Guide — what was built, how it works, and how to check it yourself

Every command below was run in this environment before being written down.

---

## 0. Setup — Anaconda Prompt (cmd.exe)

```bat
conda activate capstone
cd /d "C:\Users\vgang\Desktop\Region-Grounded Forced-Choice Attribute Verification for Multimodal Hallucination Detection"
set HF_HOME=C:\hf
```

Notes for this shell:

- **`set`, not `export`.** `export` is bash and fails here with
  *"'export' is not recognized as an internal or external command"*.
- **`make` is not installed on this machine.** Use the `python` commands below.
  The `Makefile` targets only work under a POSIX shell that has make.
- In **PowerShell** the env line is instead: `$env:HF_HOME = "C:\hf"`
- If `conda activate` ever fails, use the interpreter directly in place of
  `python`: `"C:\Users\vgang\anaconda3\envs\capstone\python.exe"`

`HF_HOME` points the model cache at a short path. On a long path Windows hits
MAX_PATH and downloads fail.

---

## 1. Fast checks — no GPU, ~20 seconds

```bat
python -m pytest -q
```

241 tests. Nothing here loads a model.

```bat
python -m ruff check src tests scripts reproduce.py
```

**Prove the suite actually blocks bad commits:**

```bat
echo def test_fail(): assert False > tests\test_tmp.py
git add tests\test_tmp.py
git commit -m "should be refused"
del tests\test_tmp.py
git reset
```

The commit is refused with `COMMIT REFUSED: the test suite failed (exit 1)`.

---

## 2. Rebuild every table from stored results — no GPU, ~5 seconds

```bat
python reproduce.py --tables-only --limit 0
```

Full development set (3,858 questions). Regenerates `results\tables\` and prints
the 2x2, every attribution contrast with its confidence interval, what has been
ruled out, and the limits.

```bat
python reproduce.py --tables-only          :: the 100-pair slice instead
python reproduce.py --list-runs            :: what raw runs exist, and their sizes
python scripts\test_report.py              :: THE HEADLINE - test split
python scripts\dext_report.py --n 3858     :: external-contrast control
python scripts\build_tables_34.py          :: pipeline + cost tables
```

`--limit` picks which stored run to build from: `0` = the full split,
`100` = the 100-pair slice. There are runs of both sizes on disk, so this flag
decides which you get.

---

## 3. Re-run the models yourself — needs the GPU

```bat
python reproduce.py                :: six cells on 100 dev pairs, ~4 min
python reproduce.py --limit 0      :: full dev set, ~35 min with warm cache
```

Individual cells:

```bat
python -m src.run --cell A --split dev --limit 100
python -m src.run --cell D --split dev --limit 100
python -m src.run --cell Dext --split dev
python -m src.run --cell all --split dev
python -m src.run --cell all-diag --split dev
```

Supporting modules:

```bat
python -m src.run --module counting  --split dev
python -m src.run --module existence --split dev
python -m src.run --module relation  --split dev
python -m src.run --module position-only --split dev
```

**The live demo:**

```bat
python -m src.demo --image data\images\AMBER_1.jpg --claim "the sky is sunny"
python -m src.demo --image data\images\AMBER_1.jpg --claim "the sky is gloomy"
python -m src.demo --image data\images\AMBER_2.jpg --claim "the man is sitting"
```

**The test split is locked** — it needs an environment variable, and even then no
threshold is fitted on it:

```bat
python -m src.run --cell D --split test
:: -> PermissionError: Refusing to read the test split.

set ALLOW_TEST_SPLIT=1
python -m src.run --cell D --split test
```

---

## 4. What each piece is

```
configs\main.yaml        all paths, thresholds, pinned model revisions
configs\antonyms.yaml    210-entry external antonym map (D-ext and demo only)

src\data\loader.py       joins AMBER annotations to queries by index
src\data\pairs.py        builds the 2,378 state + 396 action contrastive pairs
src\data\splits.py       702 dev / 302 test images, frozen, test gated

src\modules\detector.py       OWLv2: detect, NMS, pad, crop, fallback
src\modules\attribute.py      the four cells, plus A'/C' and D-ext
src\modules\crop_cache.py     disk cache of detections, settings-fingerprinted
src\modules\existence.py      "Is there an X?"        -> false-positive rate
src\modules\counting.py       "Are there three X?"    -> count comparison
src\modules\relation.py       "direct contact X, Y?"  -> threshold
src\modules\position_baseline.py   the id-ordering artifact, reported explicitly

src\eval\metrics.py      accuracy, P/R/F1, breakdowns, NOT_COMPUTED discipline
src\eval\bootstrap.py    CIs resampled over IMAGES, not questions

src\run.py               the CLI
src\demo.py              free-text claim -> verdict
reproduce.py             one command, end to end
docs\DECISIONS.md        54 numbered decisions, including two withdrawn claims
docs\paper\main.tex      the paper, IEEE two-column
```

---

## 5. How the detector works, step by step

Take `AMBER_1.jpg` and the claim *"the sky is sunny"*.

**Step 1 — parse the claim.** (Demo only; the evaluation path takes the object
and attribute from AMBER directly.) A spaCy dependency parse of
`the sky is sunny` marks `sky` as `nsubj` and `sunny` as `acomp`, giving object
`sky`, attribute `sunny`.

**Step 2 — find the competing attribute.** This is the heart of the method.

- *Evaluation path:* AMBER supplies it. Question 1005 asks "Is the sky sunny?"
  (gold yes) and 1006 asks "Is the sky gloomy?" (gold no), over the same image
  and object. The pair yields the option set `{sunny, gloomy}` — no antonym
  dictionary needed.
- *Demo and Cell D-ext:* looked up in `configs\antonyms.yaml`, `sunny -> gloomy`.

The options are then **sorted alphabetically** to `[gloomy, sunny]`, so option
position can never encode which one is correct.

**Step 3 — locate the object.** OWLv2 is prompted with `"a photo of a sky"`. Any
box scoring below `0.10` is dropped, overlapping boxes are removed by NMS at
IoU 0.5, and the survivors are sorted by score. On AMBER_1 this leaves one box at
score **0.42**.

**Step 4 — crop.** The best box is padded 10% per side, clipped to the image, and
forced to at least 64x64 pixels. If **no** box survived step 3, the crop falls
back to the whole image and the record is flagged `fell_back: true`. That happens
on 3.4% of questions and is always reported.

**Step 5 — score both options against the crop.** SigLIP scores two prompts:

```
"a photo of a gloomy sky"   ->  -14.81
"a photo of a sunny sky"    ->   -9.03
```

Each score is computed **independently** — SigLIP never sees the two options
together. This is why option order provably cannot affect the outcome.

**Step 6 — decide.**

- *Forced choice (cells B, D):* take the argmax. `sunny` wins, so the answer is
  **yes** to "Is the sky sunny?" and **no** to "Is the sky gloomy?" Confidence is
  the softmax over the two scores.
- *Threshold (cells A, C):* compare each score independently against a fitted
  tau, answering yes when `score >= tau`. This can answer yes twice or no twice.

**Memory discipline throughout.** OWLv2 and SigLIP are never resident at the same
time. The detector loads, processes every image, is freed, and only then does the
scorer load. Peak VRAM 1.65 GiB of 4.00.

---

## 6. How the experiment works

The whole paper is one 2x2. Same prompt template, same data, same questions; two
things vary.

|  | **threshold** | **forced choice** |
|---|---|---|
| **whole image** | **A** — the standard baseline | **B** |
| **cropped region** | **C** | **D** — proposed |

- **B - A** isolates forced choice (change the rule, keep the image).
- **C - A** isolates region grounding (change the image, keep the rule).
- **D - A** is both together — the headline.
- **D - B** asks whether cropping still helps once forced choice is present.

Confidence intervals bootstrap over **images**, not questions, because several
questions share an image and are correlated. Cell-to-cell comparisons use a
*paired* bootstrap: the same resampled images for both arms.

**Three controls:**

| control | what it rules out |
|---|---|
| **A', C'** | Forced choice gets a correct 50/50 prior free, since AMBER's pairs are balanced. A'/C' hand the threshold cells that same prior. The gap *widened*, so the prior is not the source of the advantage. |
| **D-ext** | The method might just exploit the benchmark supplying the contrast. D-ext uses an antonym list committed *before* any result existed and scores each question alone. Still +0.1901 over baseline. |
| **position-only** | AMBER puts the true attribute at the lower question id in 2774/2774 pairs, so this scores 1.0000 without opening an image. Reported as an artifact, with a permutation test showing no cell reads ids. |

---

## 7. Check the main claims yourself

**The headline replicated out of sample:**

```bat
python scripts\test_report.py
```

Look for `D - A  HEADLINE: both  +0.2260 [+0.2001, +0.2514]`.

**No threshold was fitted on the test split** — every threshold cell's manifest
says `frozen-from-dev (NOT fitted on test)`:

```bat
findstr /C:"frozen-from-dev" results\raw\attribute_*_test_*.manifest.json
```

**The antonym map predates every D-ext result:**

```bat
git log --format="%h %ci %s" -- configs/antonyms.yaml
```

Compare that timestamp against the first `attribute_Dext_dev_*.jsonl`. The map
commit is earlier.

**Nothing was tuned:**

```bat
python -c "print(open('docs/DECISIONS.md',encoding='utf-8').read().split('## Tuning log')[1][:700])"
```

**Every decision, including both withdrawn claims:**

```bat
findstr /R /C:"^### D-" docs\DECISIONS.md
```

54 entries. D-030 to D-038 is the first withdrawal; D-041 to D-050 the second.

---

## 8. The honest summary

- **Headline: +0.2260** on the held-out test split, thresholds frozen from dev.
  It moved 0.003 from the dev estimate, and that stability is the strongest thing
  about it.
- **Forced choice is the mechanism**: +0.2036 of that +0.2260.
- **Region grounding is +0.0201**, with a CI lower bound of +0.0012. It clears
  zero by roughly one part in a thousand — marginal, and split-dependent.
- **The dev interaction did not replicate.** +0.0223 on dev, +0.0023 on test.
- **Region grounding is in the project title but contributes about 2 points of
  22.** That is the thing to be ready to defend.

---

## 9. SHROOM-Vis — running the method on a second dataset

`shroom-visions-images/shroom-vis-images/` holds 2,495 images from the SHROOM
vision set. 900 of them were annotated to AMBER's two attribute templates so
that the four cells can be re-run **with no change to the method** — same
frozen constants, same prompt, same pinned model revisions.

The annotation policy was fixed before any SHROOM number existed:
**[`data/shroom/ANNOTATION_POLICY.md`](../data/shroom/ANNOTATION_POLICY.md)**.
The annotator is Claude (Opus 5) — a vision–language model, but *not* either of
the two models under test. This is a silver-standard benchmark and is reported
as one.

### Rebuild the dataset from the raw annotation log

```bat
python scripts\shroom_build.py
```

Reads `data/shroom/annotations_raw.jsonl` and writes `annotations.json`,
`query/query_all.json` and `pair_meta.json` in AMBER's exact schema. Every rule
in the policy that can be machine-checked is enforced here, and a violation
aborts the build rather than being repaired. Two checks earned their place:

- a **multi-word object** is rejected, because AMBER's regexes are non-greedy on
  both fields and `Is the trash can green in this image?` silently parses as
  object `trash`, attribute `can green`;
- an attribute that **repeats a word of its own object** is rejected.

### The artifact AMBER has and this does not

AMBER gives the true attribute the lower question id in **all 2,774** of its
attribute pairs, so "answer yes to the lower id" scores **1.0000** without ever
opening an image. SHROOM-Vis decides that by a seeded coin flip:

```bat
python -m src.run --dataset shroom --module position-only --split dev
```

Expect **~0.50**. That is the single most important property of this dataset.

### Run the cells

```bat
set HF_HOME=C:\hf
python -m src.run --dataset shroom --cell all --split dev
python -m src.run --dataset shroom --cell all-diag --split dev
python scripts\shroom_report.py dev
```

Then, once and only once, the held-out split — thresholds read from the dev
manifests by `frozen_tau_from_dev`, nothing fitted:

```bat
set ALLOW_TEST_SPLIT=1
python -m src.run --dataset shroom --cell all --split test
python scripts\shroom_report.py test
```

`shroom_report.py` prints the 2x2, the paired-bootstrap attribution with the
AMBER test figures alongside for comparison, and two breakdowns specific to this
dataset: **in / out of AMBER's closed 340-object vocabulary**, and by question
kind.

### What is and is not shared with the AMBER path

`src/data/shroom.py` is a **second reader**, not a generalisation of the first.
The AMBER loader asserts AMBER's exact record counts on every call (TRD §2/§3)
and those assertions are load-bearing, so they were left alone. Everything after
the reader — detector, crop, scorer, the four cells, metrics, the image-level
bootstrap — is the same code running on different data, which is the whole point.

SHROOM detections cache separately, under `results/cache/shroom/`.

### How it compares to AMBER

| | AMBER | SHROOM-Vis |
|---|---|---|
| images | 1,004 | 898 (900 viewed, 2 dropped) |
| attribute pairs | 2,774 | 2,523 |
| questions | 5,548 | 5,046 |
| pairs per image | 2.76 | 2.81 |
| distinct objects | 340 (closed vocabulary) | 433 (open) |
| true attribute has the lower id | **2,774 / 2,774** | 1,303 / 2,523 |

### Timing on the 4 GB laptop GPU

Dev: A ~14 min, B ~8 min, C ~30 min (23 of it OWLv2 over 1,760 regions), D ~3 min
once detections are cached. Test is ~43% of that, and the threshold cells need
only **one** pass there because tau is frozen rather than fitted.

### What it found

- **The headline replicated.** D - A = **+0.2354** [+0.2103, +0.2608] on SHROOM
  test against **+0.2260** on AMBER test. On dev the two agreed to within 0.0002
  (+0.2261 vs +0.2260).
- **Forced choice is still the mechanism** - +0.2235 of the +0.2354, i.e. 95%.
- **Region grounding is stronger here**, which revised D-050 upward:
  C - A = **+0.0364** [+0.0177, +0.0553], an order of magnitude clear of zero,
  where on AMBER it cleared zero by one part in a thousand (D-056). It helps most
  on objects *outside* AMBER's 340-word vocabulary.
- **The interaction is absent again**, now slightly negative (D-057).
- **SHROOM-Vis is easier than AMBER** by 4-6 points in every cell, so only the
  *contrasts* are comparable across the two datasets (D-058).

Full write-up: `docs/DECISIONS.md`, section **M6** (D-052 to D-060).

### The tests

```bat
python -m pytest tests	est_shroom.py -q
```

11 tests, no GPU. They assert the dataset contract and, more importantly, the
artifacts it must *not* have: the id-ordering leak, an attribute repeating a word
of its own object, a multi-word object the non-greedy regexes would mis-parse, an
unbalanced gold distribution, and a filename the pipeline could read an index out
of.
