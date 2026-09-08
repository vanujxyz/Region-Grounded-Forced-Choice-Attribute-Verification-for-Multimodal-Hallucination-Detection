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
