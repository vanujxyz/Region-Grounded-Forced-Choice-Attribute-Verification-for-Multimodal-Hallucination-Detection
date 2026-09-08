# Guide — what was built, how it works, and how to check it yourself

Every command below was run before being written down.

---

## 0. Setup (once per shell)

```bash
conda activate capstone
cd "C:\Users\vgang\Desktop\Region-Grounded Forced-Choice Attribute Verification for Multimodal Hallucination Detection"
export HF_HOME='C:\hf'
```

The quotes around `C:\hf` matter. Unquoted, bash collapses `C:\hf` to `C:hf`,
which Windows reads as *relative to the current directory on drive C:* — that
once dumped 594 MB of model cache into the project folder.

---

## 1. The fast checks (no GPU, seconds)

```bash
make test
```
241 tests. Nothing here loads a model, so it runs in about 20 seconds.

```bash
make lint
make check          # lint + test, exactly what the pre-commit hook runs
```

**Prove the test suite actually blocks bad commits:**
```bash
echo "def test_fail(): assert False" > tests/test_tmp.py
git add tests/test_tmp.py && git commit -m "should be refused"
# -> COMMIT REFUSED: the test suite failed (exit 1)
rm tests/test_tmp.py && git reset
```

---

## 2. Rebuild every table from stored results (no GPU, ~5 seconds)

```bash
make tables
```
Reads `results/raw/*.jsonl` and regenerates `results/tables/`. Prints the 2×2,
all attribution contrasts with confidence intervals, what has been ruled out,
and the limits.

```bash
python scripts/test_report.py         # the headline test-split result
python scripts/dext_report.py --n 3858  # the external-contrast control
python scripts/build_tables_34.py     # pipeline + cost tables
```

---

## 3. Re-run the models yourself (GPU)

```bash
python reproduce.py                    # all six cells on 100 dev pairs, ~4 min
python reproduce.py --limit 0          # full dev set, ~35 min with warm cache
```

Individual cells:
```bash
python -m src.run --cell A --split dev --limit 100   # baseline
python -m src.run --cell D --split dev --limit 100   # proposed method
python -m src.run --cell Dext --split dev            # external-contrast control
python -m src.run --cell all --split dev             # A, B, C, D
python -m src.run --cell all-diag --split dev        # A', C'
```

Supporting modules:
```bash
python -m src.run --module counting  --split dev
python -m src.run --module existence --split dev
python -m src.run --module relation  --split dev
python -m src.run --module position-only --split dev   # the benchmark artifact
```

**The live demo:**
```bash
python -m src.demo --image data/images/AMBER_1.jpg --claim "the sky is sunny"
python -m src.demo --image data/images/AMBER_1.jpg --claim "the sky is gloomy"
python -m src.demo --image data/images/AMBER_2.jpg --claim "the man is sitting"
```

**The test split is locked.** It needs an environment variable, and even then no
threshold is fitted on it:
```bash
python -m src.run --cell D --split test        # PermissionError
ALLOW_TEST_SPLIT=1 python -m src.run --cell D --split test   # works
```

---

## 4. What each piece is

```
configs/main.yaml        all paths + thresholds + pinned model revisions
configs/antonyms.yaml    210-entry external antonym map (D-ext + demo only)

src/data/loader.py       joins AMBER's annotations to its queries by index
src/data/pairs.py        builds the 2,378 state + 396 action contrastive pairs
src/data/splits.py       702 dev / 302 test images, frozen, test gated

src/modules/detector.py       OWLv2 wrapper: detect, NMS, pad, crop, fallback
src/modules/attribute.py      the four cells + A'/C' + D-ext
src/modules/crop_cache.py     disk cache of detections, fingerprinted
src/modules/existence.py      "Is there an X?"  -> false-positive rate
src/modules/counting.py       "Are there three X?" -> count comparison
src/modules/relation.py       "direct contact between X and Y?" -> threshold
src/modules/position_baseline.py   the id-ordering artifact, reported explicitly

src/eval/metrics.py      accuracy, P/R/F1, breakdowns, NOT_COMPUTED discipline
src/eval/bootstrap.py    CIs resampled over IMAGES, not questions

src/run.py               the CLI
src/demo.py              free-text claim -> verdict
reproduce.py             one command, end to end
```

---

## 5. How the detector works, step by step

Take `AMBER_1.jpg` and the claim *"the sky is sunny"*.

**Step 1 — parse the claim.** (Demo only; the evaluation path gets the object and
attribute from AMBER directly.) spaCy dependency parse:
`the sky is sunny` → `sky` is `nsubj`, `sunny` is `acomp` → object `sky`,
attribute `sunny`.

**Step 2 — find the competing attribute.**
- *Evaluation path:* AMBER supplies it. Question 1005 asks "Is the sky sunny?"
  (gold yes) and 1006 asks "Is the sky gloomy?" (gold no). The pair gives
  `{sunny, gloomy}`.
- *Demo / D-ext:* looked up in `configs/antonyms.yaml` → `sunny → gloomy`.

The two options are then **sorted alphabetically** — `[gloomy, sunny]` — so option
position can never encode which one is correct.

**Step 3 — locate the object.** OWLv2 is prompted with `"a photo of a sky"`.
It returns candidate boxes; anything scoring below `0.10` is dropped, overlapping
boxes are removed by NMS at IoU 0.5, and the rest are sorted by score. On
AMBER_1 this yields one box at score **0.42**.

**Step 4 — crop.** The best box is padded by 10% per side, clipped to the image
bounds, and forced to at least 64×64 pixels. If **no** box survived step 3, the
crop falls back to the whole image and the record is flagged `fell_back: true` —
this happens on 3.4% of questions and is reported, never hidden.

**Step 5 — score both options against the crop.** SigLIP scores two prompts:
```
"a photo of a gloomy sky"   ->  -14.81
"a photo of a sunny sky"    ->   -9.03
```
Each score is computed independently — SigLIP never sees the two options
together. (This is why option order provably cannot matter.)

**Step 6 — decide.**
- *Forced choice (cells B, D):* take the argmax. `sunny` wins → answer **yes** to
  "Is the sky sunny?" and **no** to "Is the sky gloomy?" Confidence is the
  softmax over the two scores.
- *Threshold (cells A, C):* compare each score independently against a fitted
  τ. Answer yes if `score ≥ τ`. This can answer yes twice or no twice.

**Memory discipline throughout:** OWLv2 and SigLIP are *never* resident at the
same time. The detector loads, processes every image, is freed, and only then
does the scorer load. Peak VRAM 1.65 GiB of 4.00.

---

## 6. How the experiment works

The whole paper is one 2×2. Same prompt template, same data, same 3,858
questions; only two things vary.

|  | **threshold** | **forced choice** |
|---|---|---|
| **whole image** | **A** — the standard baseline | **B** |
| **cropped region** | **C** | **D** — proposed |

- **B − A** isolates forced choice (change the decision rule, keep the image).
- **C − A** isolates region grounding (change the image, keep the rule).
- **D − A** is both together.
- **D − B** asks whether cropping still helps once forced choice is present.

Confidence intervals come from a bootstrap that resamples **images**, not
questions, because several questions share an image and are correlated.
Comparisons use a *paired* bootstrap — the same resampled images for both arms.

**Three controls:**

| control | what it rules out |
|---|---|
| **A′, C′** | forced choice gets a correct 50/50 prior free, because AMBER's pairs are balanced. A′/C′ give the threshold cells that same prior. The gap *widened*, so the prior is not the source of the advantage. |
| **D-ext** | the method might exploit the benchmark supplying the contrast. D-ext uses an antonym list committed *before* any result existed, and scores each question alone. Still +0.1901 over baseline. |
| **position-only** | AMBER puts the true attribute at the lower question id in 2774/2774 pairs, so this scores 1.0000 without opening an image. Reported as an artifact, with a permutation test showing no cell reads ids. |

---

## 7. Check the main claims yourself

**The headline replicated out of sample:**
```bash
python scripts/test_report.py
```
Look for `D - A  HEADLINE: both  +0.2260 [+0.2001, +0.2514]`.

**No threshold was fitted on the test split:**
```bash
python -c "import json,glob; [print(json.load(open(f))['cell'], (json.load(open(f)).get('tau') or {}).get('protocol','NOT_APPLICABLE') if isinstance(json.load(open(f)).get('tau'),dict) else 'NOT_APPLICABLE') for f in sorted(glob.glob('results/raw/attribute_*_test_*.manifest.json'))]"
```
Every threshold cell says `frozen-from-dev (NOT fitted on test)`.

**The antonym map predates every D-ext result:**
```bash
git log --format="%h %ci %s" -- configs/antonyms.yaml
git log --format="%h %ci %s" --diff-filter=A -- results/raw/attribute_Dext_dev_20260907T175716Z.jsonl
```
The map commit is earlier.

**Nothing was tuned:**
```bash
sed -n '/^## Tuning log/,/^---/p' docs/DECISIONS.md
```

**Every decision, including the two withdrawn claims:**
```bash
grep "^### D-" docs/DECISIONS.md
```
54 entries. D-030 → D-038 is the first withdrawal; D-041 → D-050 the second.

---

## 8. The honest summary

- **Headline: +0.2260** on the held-out test split, thresholds frozen from dev.
  It moved by 0.003 from the dev estimate, which is the strongest thing about it.
- **Forced choice is the mechanism**: +0.2036 of that +0.2260.
- **Region grounding is +0.0201**, with a CI lower bound of +0.0012. It clears
  zero by roughly one part in a thousand. Marginal, and split-dependent.
- **The dev interaction did not replicate.** +0.0223 on dev, +0.0023 on test.
- **Region grounding is in the project title but contributes about 2 points of
  22.** That is the thing to be ready to defend.
