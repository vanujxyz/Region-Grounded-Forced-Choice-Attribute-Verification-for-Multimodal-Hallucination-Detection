# Decisions log

Every interpretation, deviation and tuning action, with its reason. Tuning
entries must record the before number and the after number (TRD §16).

---

## M0 — 2026-09-07

### D-001 — TRD §10 `REL_RE` does not parse any relation question (spec defect)

**Status:** confirmed by the project owner independently. Patch deferred to M4.

The regex as written in TRD §10 requires an ` in this image?` suffix that the
data does not have, and expects an article before the second object:

```
TRD §10: ^Is there direct contact between the (?P<a>.+?) and (?P<b>.+?) in this image\?$
actual  : "Is there direct contact between the person and grass?"
```

Measured parse rate of the TRD regex: **0 / 1664** (0 of 975
`discriminative-relation` + 0 of 689 `relation`).

**Agreed patch, to be applied when M4 is reached:**

```python
REL_RE = re.compile(r"^Is there direct contact between the (?P<a>.+?) and (?P<b>.+?)\?$")
```

Parses 1664/1664. Not yet applied — `src/modules/relation.py` does not exist at M0.

### D-002 — §11.4 independence caveat

Recorded verbatim, as stated before M0 began:

> I'll write the reference implementation from §11.1 alone without importing
> `src.eval.metrics`, but both come from the same author in the same session —
> the cross-check catches transcription and edge-case bugs, not a shared
> misreading of the spec. Worth stating plainly in the paper.

### D-003 — M0 interpretations, all confirmed by the project owner before coding

1. **Images assertion is separate.** TRD §2's image check lives in
   `assert_images()`, called at model time, not by the data pipeline. Nothing in
   M0 reads pixels, so loader/pairs/splits/metrics work before the manual
   Google Drive download. The image check is reported BLOCKED, not passing.
2. **Python 3.11.** TRD §0 requires it; the host had only 3.12.10 and 3.13.9.
   Created conda env `capstone` with Python 3.11.16.
3. **Repository root.** The existing project directory *is* `capstone/` from the
   TRD §1 tree; no nested `capstone/` folder was created.
4. **70/30 rounding.** `int(0.7 * 1004)` → dev 702 / test 302. TRD §5 does not
   specify the rounding direction.
5. **Split image list source.** Derived from the 1,004 distinct `image` fields in
   `query_all.json`, sorted **numerically** by the index in `AMBER_(\d+).jpg`
   (not as strings), with the extracted indices asserted equal to
   `range(1, 1005)`. This lets splits be frozen before the image download;
   `assert_images()` checks the directory agrees once the images arrive.
6. **Cross-check seed.** TRD §11.4 does not specify one; using `20260907`, matching
   §5 and §11.3.
7. **`test_no_placeholders.py` semantics.** TRD §14 describes grepping for `0.0`,
   which would flag legitimately-computed zeros (a real accuracy of 0.0). The
   test instead asserts every metric field in `results/tables/*.csv` and
   `results/raw/*.json` is either a real number or the literal string
   `NOT_COMPUTED` — never null, empty, or a placeholder. It also self-tests that
   the check fires on what it is meant to catch.
8. **Empty subgroup → `NOT_COMPUTED`.** Accuracy, precision, recall, F1 and
   fallback rate are `NOT_COMPUTED` when their denominator is zero, per PRD §8
   rule 1. An undefined metric is not a zero one.
9. **`git init`.** The directory was not a repository; PRD §8 rule 5 requires a
   commit hash in every result file.

### D-004 — Model revision hashes are null in config, not invented

`configs/main.yaml` sets `detector.revision` and `attribute.revision` to `null`
with a comment requiring M1 to pin them after the first real download. A
plausible-looking hash is never written (PRD §8 rule 1).

### D-005 — Windows long-path handling

`git config --global core.longpaths true`. A `git clone` into a nested path
failed with `Filename too long` before this; the project directory name is 96
characters. The HuggingFace cache will need a short path (e.g. `C:\hf`) at M1.

---

## Tuning log

No tuning has been performed. No threshold has been fitted. No model has been run.

---

## M1 — 2026-09-07

> **Numbering note.** The project owner raised these as "D-002" and "D-003", but
> those ids were already taken by the M0 log above. They are recorded here as
> **D-006** and **D-007**. Owner's D-002 = D-006; owner's D-003 = D-007.

### D-005b — AMBER images arrived nested one level deep

The Google Drive archive unzipped to `data/images/image/*.jpg` rather than
`data/images/*.jpg`. The 1,004 files were moved up one level and the empty
`image/` directory removed, so the tree matches TRD §1/§2. No file was renamed
or altered.

Verified after the move: 1004 entries, 1004 matching `AMBER_(\d+)\.jpg`, zero
non-matching entries, indices exactly `range(1, 1005)` with no gaps, and the set
equal to the 1,004 distinct `image` fields in `query_all.json`.
`assert_images()` PASSES.

### D-006 (owner's "D-002") — forced-choice option order

**Measurement, reproduced independently:** alphabetical sorting puts the
positive attribute first in **1156 of 2774 pairs (41.7%)**, not ~50%.

**Owner's requested change:** evaluate each pair in both option orders and
average the two score vectors before the argmax.

**Status: RESOLVED — option (a). Owner accepted the analysis; averaging skipped.**

The requested change is a mathematical no-op under the architecture TRD §7
specifies. Cell B/D score each option independently:

```python
scores = [siglip_score(image, f"a photo of a {a} {obj}") for a in sorted_options]
```

`siglip_score(image, text)` does not depend on which other options exist or where
they sit in the list -- the options are never presented jointly, so no forward
pass can observe a position. The score vector for `[gloomy, sunny]` is
`[s_g, s_s]`; for `[sunny, gloomy]` it is `[s_s, s_g]`; realigned and averaged it
is `[s_g, s_s]`, unchanged. Identical argmax, identical softmax confidence,
bit-identical accuracy -- at double the inference cost, with a test that passes
vacuously and a paper claim that mitigates nothing.

The 41.7% skew is a real descriptive statistic, but it can only bias a method
that sees the options jointly (a single prompt listing both, or joint encoding).

**The one genuine order-dependence is tie-breaking.** On an exact float tie,
`argmax` returns index 0 -- the alphabetically-first option -- and given the
41.7% skew that tie-break slightly favours answering "no" on the positive
question. Exact ties are rare in float32 but must be counted, not hidden.

Implemented now (the part that is real):
  - an order-invariance test, kept as a regression guard so the implementation
    can never silently become order-dependent;
  - explicit tie detection: ties are counted and reported per run rather than
    resolved silently by list position.

**Resolution (owner, option a):** the both-orders averaging is NOT implemented,
because it is a no-op. Tie-counting and the order-invariance regression guard are
kept. The 41.7% figure is recorded as a descriptive limitation only, and is not
presented as a bias that was corrected.

Superseded in importance by **D-009** below, of which this skew is a shadow.

### D-007 (owner's "D-003") — attribute pairs repeat; raw N overstates evidence

**Measurement, reproduced independently:** the 2,774 attribute pairs cover only
**828 distinct (object, positive, negative) triples** across **182 objects**,
over 992 distinct images. `('sky', 'sunny', 'gloomy')` alone appears **235**
times; the top five triples account for 638 pairs (23%).

    235  ('sky', 'sunny', 'gloomy')
    147  ('cloud', 'white', 'black')
    103  ('grass', 'green', 'blue')
     86  ('forest', 'lively', 'withered')
     67  ('ground', 'clean', 'dirty')

Bootstrapping over images (TRD §11.3) already handles the correlation correctly,
so no change to the CI computation. **No code change to the estimator.**

Reporting change implemented: `src/eval/metrics.py` now emits
`n_distinct_triples` and `n_distinct_images` in every metric block, alongside
raw `n`, so N is never reported alone. Both are `NOT_COMPUTED` when the field is
absent. The independent reference implementation in
`tests/test_metrics_reference.py` computes both separately and the 20,000-case
cross-check covers them.

**For the paper's limitations section:** accuracy is computed over 2,774
questions, but those rest on 828 distinct attribute contrasts over a 340-object
vocabulary. Effective sample size is materially smaller than N suggests.

### D-008 — environment

CUDA build installed from the cu124 index per owner instruction; `HF_HOME=C:\hf`
to avoid the Windows MAX_PATH failure recorded in D-005.

### D-009 — AMBER attribute pairs are systematically ordered by construction

**Raised by the project owner; reproduced independently and found to be stronger
than stated.**

In **all 2,774** clean attribute pairs the true attribute has the **lower**
question id. Zero exceptions:

| subset | positive_id < negative_id | share |
|---|---|---|
| state  | 2378 / 2378 | 100.00% |
| action |  396 /  396 | 100.00% |
| **all**| **2774 / 2774** | **100.00%** |

**Stronger than reported:** the id gap within a pair is *always exactly 1* — the
set of distinct gaps is `[1]`. Pairs are strictly adjacent `(n, n+1)` with the
true attribute at `n`. AMBER generates the correct attribute first.

**Measured consequence:** a detector that never opens an image and simply answers
"yes" to whichever question of a pair has the lower id scores **1.0000
(5548/5548)**.

**Three actions taken:**

1. **Reported, not hidden.** `src/modules/position_baseline.py` implements the
   exploit and emits records in the standard format, tagged
   `benchmark_artifact: True` and `cell: "position-only"`. It becomes a labelled
   row in Table 1 — "Position-only baseline (BENCHMARK ARTIFACT — exploits AMBER
   pair id ordering; sees no image)" — so no reviewer discovers it independently.
   It loads no model and opens no image, by construction.

2. **Ordering may never derive from an id.** `tests/test_position_baseline.py`
   asserts option order is invariant to id swapping and to extreme id
   perturbation, that `AttrPair.options` mentions no id field (static check on
   the property source), and — the real risk — that `_build_pairs` selects pair
   polarity from the **gold answer**, never from an id comparison. That last one
   must be a static check: because the artifact is 100%, "positive = truth is
   yes" and "positive = lower id" are behaviourally indistinguishable on real
   AMBER data, so no behavioural test can separate them.

3. **Paper limitation.** AMBER's attribute pairs are systematically ordered by
   construction. Any evaluation that consumes pairs in dataset order, or that
   lets id ordering reach the model, inherits a free 100%. Reported alongside
   D-007 (828 distinct triples) in the limitations section.

**Note for M2 onward:** this makes the Cell A vs Cell D comparison the only
meaningful signal. An implementation bug that leaks id order would produce a
spectacular accuracy that means nothing. If any cell reports near-100%, suspect
leakage before celebrating.

### D-010 — configuration paths

Owner instruction: the image directory, results directory and HF cache are single
config values; no absolute paths anywhere, so the project can move to Kaggle if
4 GB VRAM proves insufficient.

`configs/main.yaml` now carries `paths:` with `amber`, `images`, `splits`,
`results`, `results_raw`, `results_tables`, `demo_out` and `hf_cache`, all
**relative to the repo root**. `src/config.py` resolves them at runtime;
`src/data/loader.py` no longer hardcodes any path.

**One tension, resolved deliberately.** `HF_HOME=C:\hf` is machine-specific and
absolute, and exists to dodge the Windows MAX_PATH failure in D-005 — but a
relative in-repo cache would reintroduce that failure (the repo dir name is 96
chars). So `hf_cache` defaults to the relative `.hf_cache` (portable, correct on
Kaggle/Colab) and the absolute Windows path is supplied through the `HF_HOME`
environment variable, which `src/config.py` honours as an override. No absolute
path is stored in the repository.

### D-011 — the scale of `siglip_score` is unspecified in TRD §7

TRD §7 writes `s = siglip_score(...)` and `"yes" if s >= tau` without fixing a
scale. SigLIP emits a raw logit; on AMBER_1/sky the two options score
`-14.81` and `-9.03`, whose sigmoids are `0.0000` and `0.0001`. Thresholding
sigmoid values that small is numerically degenerate, so **tau is swept on the
logit scale**. `attribute.score_scale: logit` in `configs/main.yaml`; setting it
to `sigmoid` changes what tau means.

**Consequence for the confidence formula.** TRD §7 gives threshold-cell
confidence as `sigmoid(k * (score - tau))` with `k = 10.0`. On the logit scale a
gap of one unit is already large, so `k = 10` saturates confidence to 0.0 or 1.0
almost everywhere -- the first Cell A record has `confidence: 1.0`. The
confidences are therefore near-binary and should not be read as calibrated. `k`
is left at the specified `10.0`; **this is flagged, not changed.**

### D-012 — fitting tau under `--limit`

TRD §7 says tau is fitted on the dev split. With `--limit 100` the run both fits
and evaluates tau on the same 200 questions, which is optimistically biased.

The bias **favours Cell A and Cell C**, the threshold cells -- that is, it
favours the *baseline* in the headline A-vs-D comparison. The comparison is
therefore conservative with respect to the project's own hypothesis, which is the
safe direction for the error to run. Recorded rather than corrected, because
TRD §7 specifies fitting on dev and does not carve out the `--limit` case.

At M3 (full dev run) the fit set is the whole dev split and the same question
arises at a much smaller scale; if a held-out fitting protocol is wanted, it is a
spec change and needs owner sign-off.

### D-013 — transformers 5.16.1

TRD §0 requires `transformers >= 4.45`; pip resolved **5.16.1**, a major version
released after the TRD was written. It satisfies the constraint literally. Both
required APIs were smoke-tested against the pinned revisions before use:
`Owlv2Processor` / `Owlv2ForObjectDetection` with
`post_process_grounded_object_detection`, and `AutoProcessor` / `AutoModel` with
`logits_per_image`. Both work. Two undeclared transitive dependencies of the
SigLIP tokenizer had to be installed: `sentencepiece` and `protobuf`; neither
appears in TRD §0.

Loading SigLIP emits two harmless config warnings from transformers v5
(`bos_token_id` / `eos_token_id` outside the vocabulary). They come from the
upstream checkpoint config, not from this code.

### D-014 — HF_HOME quoting bug, caught and corrected

`HF_HOME=C:\hf` unquoted in bash collapses to `C:hf`, which Windows resolves
*relative to the current directory on drive C:*. The first OWLv2 download
therefore wrote 594 MB into the project directory as `hf/`. The cache was moved
to `C:\hf` and the stray directory removed; `.hf_cache/` is gitignored. Always
quote it: `HF_HOME='C:\hf'`.

### D-015 — manifest config echo vs the no-placeholder rule

`test_no_placeholders.py` fired on the first real manifest: the manifest embeds a
verbatim echo of the resolved config, which contains `relation.tau: null` and
`existence.threshold: null` for modules that do not exist until M4.

These are **unfitted inputs**, not uncomputed metrics -- a different thing from
what PRD §8 rule 1 governs. Rather than blanket-exempting the manifest (which
would have weakened the check), the `.config.` echo subtree is excluded from the
metric scan and two narrower tests were added:

  - `test_manifest_result_fields_are_real_or_not_computed` -- every *result*
    field the run produced (tau, fit accuracy, tie rate) is still checked, and
    the test fails if the schema changes so that nothing gets checked;
  - `test_config_echo_nulls_are_only_unfitted_thresholds` -- a null in the config
    echo must be one of a known allowlist of unfitted parameters; any other null
    fails.

The test catching this is the intended behaviour, not a nuisance.

---

## M2 — 2026-09-07

### D-011 (resolved) — confidence is uncalibrated and is not a claim

`k` stays at the specified **10.0**. Owner ruling: accuracy derives from the tau
comparison, not from the confidence value, so saturation affects **no reported
metric**.

Recorded for the paper: **the confidence output of this system is uncalibrated,
and calibration is not a claim of this work.** On the logit scale used for tau
(D-011 above), `sigmoid(10 * (s - tau))` saturates to 0.0/1.0 almost everywhere;
forced-choice confidences are a 2-way softmax over logits several units apart and
are similarly extreme. Confidences are reported because TRD §7 requires them, not
because they are meaningful as probabilities.

Revisit only if the M4 demo output (D6) proves unusable.

### D-012 (resolved) — fit-on-eval, and the free-parameter asymmetry

The fit-on-eval protocol stays: tau is fitted by sweeping on the same split it is
scored on, per TRD §7. The bias favours the threshold cells, which are the
baseline, so the comparison runs conservative with respect to this project's own
hypothesis.

**1. The asymmetry, stated explicitly for the paper.**

| cell | region | decision | free parameters fitted on the evaluation data |
|---|---|---|---|
| A (baseline) | full image | threshold | **1** (tau) |
| B | full image | forced choice | **0** |
| C | crop | threshold | **1** (its own tau) |
| D (proposed) | crop | forced choice | **0** |

Cells A and C each get a parameter tuned on the very questions they are scored
on. Cells B and D have **no free parameter at all** -- there is nothing in them
to tune. Any advantage this confers goes to the baseline. If D beats A under
these conditions, the margin is a floor, not a ceiling; if D loses, the loss is
not explained by A having been tuned unfairly in D's favour, because it was
tuned in *A's* favour.

**2. Labelling protocol, fixed now, before any dev number exists.**

- Every dev-split number -- M2's 100 pairs and M3's full dev run alike -- is
  labelled **fit-on-eval** wherever a tau is involved (cells A and C). The label
  travels with the number into every table and into the paper.
- The **headline comparison is the M5 test-split run**, with tau **frozen** from
  the dev fit and not refitted. That is the only number where the threshold cells
  have no evaluation-data advantage.
- The test split is opened exactly once, at TRD §15 step 14 (PRD §8 rule 6).
- Dev numbers are development evidence and a go/no-go signal. They are not the
  headline result and must not be presented as one.

Confirmed by the owner and recorded before any dev number existed.

### D-016 — environment pinned exactly

`requirements.txt` pins every version, replacing TRD §0's ranges. A spec saying
`transformers >= 4.45` resolves differently over time and the numbers stop
reproducing (PRD §8 rule 7). torch/torchvision need the cu124 index; spaCy's
`en_core_web_sm` installs separately. Versions recorded are those the reported
numbers were produced with.

### D-017 — M2 result, first and only run

Cell A and Cell D were each executed **once** on the same 100 dev pairs (200
question ids, 62 distinct triples, 33 images). No setting was changed between
them and neither was re-run. The detector threshold (0.10), crop padding (0.10),
prompt template and `k` are exactly as specified.

| cell | accuracy | 95% CI | free params fitted on eval |
|---|---|---|---|
| A (baseline) | **0.63** | [0.5934, 0.6649] | 1 (tau = -13.3014) |
| D (proposed) | **0.86** | [0.7810, 0.9268] | 0 |
| position-only | 1.0000 | [1.0, 1.0] | 0 (artifact) |

Paired bootstrap over images, D - A, 10,000 resamples, seed 20260907:
**+0.2300, 95% CI [+0.1515, +0.3021], excludes zero.**

**No tuning was performed.** The tuning log below remains empty.

**Leakage audit (because Cell D scored 34/34 on action).** Verified on the Cell D
output before reporting:
  - "yes" landed on the lower id in 86% of pairs -- exactly equal to the
    accuracy. This is an identity, not a leak: forced choice emits one yes per
    pair, and the gold positive is always the lower id (D-009), so accuracy and
    yes-on-lower-id are the same quantity. Genuine id leakage would show 100%
    here and 1.00 accuracy, as the position-only baseline does.
  - `pred == argmax(raw_scores)` for all 200 records, 0 mismatches.
  - Action's 34/34 comes from 17 pairs over 14 distinct triples of easy contrasts
    (sit/stand, run/walk, laugh/cry) with median score margin 4.32 logits.
    Small n, wide margins -- not evidence of a defect, but n=17 pairs.

**Fallback rate 0.05.** 10 of 200 questions fell back to the full image, where
Cell D is behaviourally identical to Cell B. Cell D scores 0.8737 on the 190
non-fallback questions and 0.6000 on the 10 fallback ones.

**Peak VRAM 1.654 GiB used of 3.999 GiB** during two-model staging (torch
allocator peak 0.772 GiB). No OOM. Sequential load/free worked as designed.

### D-018 — two sentinels: NOT_COMPUTED vs NOT_APPLICABLE

`test_no_placeholders` fired again on the Cell D manifest, which reports
`tau: NOT_APPLICABLE`. The two sentinels are kept distinct rather than collapsed:

  - **NOT_COMPUTED** -- the metric could have been computed and was not (empty
    subgroup, undefined denominator, stage not run).
  - **NOT_APPLICABLE** -- the quantity does not exist for this configuration.
    Cells B and D are forced choice and have **no tau at all**.

Flattening NOT_APPLICABLE into NOT_COMPUTED would erase exactly the
free-parameter asymmetry recorded in D-012, which is a point the paper makes.
Neither sentinel may ever stand in for a number that was actually produced; a
bare `0`, `0.0`, empty string or null still fails the check. Two tests were added:
one asserting the sentinels stay distinct and non-numeric, one asserting that any
B/D manifest declares `tau: NOT_APPLICABLE` explicitly.

### D-019 — Cells B and C; the attribution is not what the framing assumed

Run once each on the same 100 dev pairs, same 200 ids, no settings changed.

|  | threshold | forced choice |
|---|---|---|
| **whole image** | A **0.6300** | B **0.7900** |
| **cropped region** | C **0.6400** | D **0.8600** |

Paired bootstrap over images (10,000, seed 20260907):

| contrast | isolates | Δ | 95% CI | |
|---|---|---|---|---|
| C − A | **region grounding alone** | **+0.0100** | [−0.0490, +0.0680] | **includes zero** |
| B − A | **forced choice alone** | +0.1600 | [+0.0859, +0.2330] | excludes zero |
| D − A | both (headline) | +0.2300 | [+0.1515, +0.3021] | excludes zero |
| D − C | forced choice, given cropping | +0.2200 | [+0.1402, +0.2935] | excludes zero |
| D − B | cropping, given forced choice | +0.0700 | [+0.0217, +0.1215] | excludes zero |

Interaction (D−A) − (B−A) − (C−A) = **+0.0600**, superadditive.

**The finding: region grounding alone does nothing measurable here.** C − A is
+0.01 with a CI straddling zero. Cropping only pays once forced choice is in
place (D − B = +0.07, excludes zero). Forced choice is the dominant term. PRD §2
presents the contribution as "two independent changes"; at this sample size only
one of them is independently supported, and the other is justified solely by the
interaction. This must not be smoothed over in the paper.

### D-020 — forced choice inherits the dataset's balance prior (raise at the review)

Predicted-yes counts, against a gold of exactly 100 yes / 100 no:

| cell | predicted yes |
|---|---|
| A | 136 |
| B | **100** |
| C | 60 |
| D | **100** |

Forced choice emits exactly one yes per pair **by construction**, so B and D
cannot deviate from a 50/50 yes rate. AMBER's attribute pairs are exactly
balanced (PRD §6: 2382/2382 state, 396/396 action), so that structural constraint
matches the gold distribution perfectly.

The threshold cells have no such information: A over-predicts yes (136/200), C
under-predicts (60/200), and both are penalised for it.

**Part of the B/D advantage is therefore a free prior handed to them by the
dataset's construction, not better attribute binding.** A reviewer will raise
this. Two mitigations to discuss at the M2 review, neither yet implemented:
  - report balanced accuracy or per-class metrics alongside raw accuracy;
  - report a threshold cell with tau fitted to match the base rate, isolating
    calibration from discrimination.

Flagged, not fixed. It changes what the headline number means.

### D-021 — proposal: id-randomised diagnostic set (cost reported, NOT run)

**A correction to the proposal as framed.** "Randomize which id holds the gold
positive" cannot be done to AMBER itself: an id is bound to a question's text, so
id 1005 *is* "Is the sky sunny in this image?" with gold yes. Swapping the labels
fabricates a dataset variant whose ids no longer mean what AMBER's mean -- and
TRD §11.1 chooses per-id accuracy precisely so numbers stay "directly comparable
with published AMBER numbers". Randomising ids in the reported evaluation path
would forfeit that.

**Proposal instead: a diagnostic-only permuted variant**, built behind a flag,
never used for any reported accuracy, used once to falsify the leakage question.

Expected outcomes, stated in advance:
  - **position-only drops to ~0.50** -- the informative result, and the point of
    the exercise;
  - **Cells A and D return bit-identical accuracy.** Neither reads an id: the
    option list is alphabetical (D-009 guards this with static and behavioural
    tests) and the prediction is a function of (image, obj, attr) only. The id
    appears solely as the key a record is filed under. This is the same
    structural argument as D-006.

**Cost, measured, not estimated.** Model load dominates; the scoring loop is
3-4s per 100 pairs.

| run | wall clock at limit 100 | at full dev (1929 pairs, 19.3x) |
|---|---|---|
| Cell B (scorer only) | 23s measured | ~2 min |
| Cell C (detector + scorer) | 85s measured | ~25 min |
| Cell A re-run | ~25s | ~2 min |
| Cell D re-run | ~85s | ~25 min |
| **A + D at limit 100** | **~2 min** | ~27 min |

**Recommendation: do it at limit 100 only, once.** Two minutes converts a static
audit into an empirical falsification, which is cheap insurance on the project's
central claim. Do **not** adopt it as the standing protocol for M3's full dev run:
~27 minutes for an outcome already known to be bit-identical, and reported
accuracy must stay on real AMBER ids.

### D-022 — Cells A' and C': the base-rate confound is REFUTED

D-020 asked whether B/D's margin is really forced choice inheriting AMBER's
exact pair balance for free. Cells A' and C' test it: identical to A and C, but
tau chosen so predicted-yes matches the known base rate (100 of 200) rather than
maximising accuracy.

**Tau selection rule for A'/C'** (`fit_tau_base_rate`): sort the 200 scores
descending, take tau as the `target_yes`-th largest, so exactly `target_yes`
scores satisfy `score >= tau`. Ties can push the realised count above target; the
realised count is reported, never assumed. `target_yes` defaults to the number of
gold "yes" in the fit set. **Fit-on-eval, exactly like every other dev tau
(D-012)** -- A' is *told* the base rate instead of being tuned for accuracy. It
exchanges one oracle for another; that is what makes it the right control.

Realised: A' tau -11.4144, C' tau -7.6909, both realised_yes = 100 exactly,
ties_at_tau = 0.

| cell | accuracy | predicted yes | tp / fp / fn / tn |
|---|---|---|---|
| A | 0.6300 | 136 | 81 / 55 / 19 / 45 |
| A' | **0.6000** | **100** | 60 / 40 / 40 / 60 |
| C | 0.6400 | 60 | 44 / 16 / 56 / 84 |
| C' | **0.6100** | **100** | 61 / 39 / 39 / 61 |
| B | 0.7900 | 100 | 79 / 21 / 21 / 79 |
| D | 0.8600 | 100 | 86 / 14 / 14 / 86 |

| contrast | isolates | Δ | 95% CI | |
|---|---|---|---|---|
| B − A | forced choice alone (raw) | +0.1600 | [+0.0859, +0.2330] | excludes zero |
| **B − A'** | **forced choice NET of base rate** | **+0.1900** | [+0.1094, +0.2588] | excludes zero |
| D − C | forced choice given crop (raw) | +0.2200 | [+0.1402, +0.2935] | excludes zero |
| **D − C'** | **forced choice given crop, NET** | **+0.2500** | [+0.1700, +0.3182] | excludes zero |
| A' − A | cost of base-rate matching | −0.0300 | [−0.0808, +0.0263] | includes zero |
| C' − C | cost of base-rate matching | −0.0300 | [−0.0743, +0.0140] | includes zero |
| C' − A' | region grounding, base-rate matched | +0.0100 | [−0.0348, +0.0510] | includes zero |

**Conclusion: the confound does not explain the effect.** Handing the threshold
cells the correct base rate does not close the gap -- it *widens* it, from +0.16
to +0.19 (whole image) and +0.22 to +0.25 (cropped). Forced choice's advantage is
not the free prior; matching the base rate costs the threshold cells accuracy
because their accuracy-maximising tau was deliberately unbalanced.

**Balanced accuracy was NOT computed as a mitigation.** Owner is correct that it
is a no-op: gold is exactly 100/100, so BA equals accuracy for every cell.
Per-class precision/recall are reported descriptively in
`results/tables/table1_ablation.csv` (tp/fp/fn/tn columns).

Note the D-019 conclusion survives the control: region grounding alone is +0.0100
with a CI straddling zero under *both* tau-selection rules.

### D-023 — permuted-id diagnostic: leakage falsified empirically

Run once at limit 100, seed 20260907, flagged, **diagnostic-only**. Records are
never a reported number; `permute_pair_ids` carries that in its docstring.

| | real ids | permuted ids |
|---|---|---|
| positive holds the lower id | 100/100 (1.0000) | 41/100 (0.4100) |
| **position-only baseline** | **1.0000** | **0.4100** |
| Cell A | 0.6300 | **0.6300** (delta +0.000000) |
| Cell D | 0.8600 | **0.8600** (delta +0.000000) |

`(image, obj, attr, pred, gold)` tuples were **identical** for both A and D
before and after permutation -- only the id each record is filed under changed.

This converts the D-017 static audit into an empirical falsification: no cell
reads a question id. The position-only baseline collapses, as designed.

0.4100 rather than 0.5000 is binomial noise: with 100 pairs at p=0.5 the standard
deviation is 5 pairs, so 41 sits 1.8 SD low. Not a defect; simply the realised
draw at this seed.

**Not adopted for M3**, per the owner's instruction and D-021's costing.

### D-024 — PRD §2 framing revision drafted, NOT applied

`docs/prd_section2_draft.md` holds a proposed replacement for PRD §2. **`PRD.md`
is unmodified.** The draft replaces "two independent changes" with one change
that works alone (forced choice) and a second that pays only in combination
(region grounding), making the +0.06 superadditive interaction the claim.

The draft flags three things for the review: every number is 100 dev pairs and
the magnitudes are not stable; the proposed *mechanism* for why cropping needs
forced choice is a hypothesis I have not measured; and PRD §3's non-goals need a
second look, since disclaiming region grounding as prior art sits oddly beside a
finding that it does nothing on its own.

### D-025 — D-020 was the owner's hypothesis, and it was falsified

Recorded explicitly so the outcome is not lost.

**Origin.** The hypothesis was raised by the project owner: that part of Cells
B/D's margin was forced choice inheriting AMBER's exactly balanced attribute
pairs as a free prior, rather than binding attributes better. It was a reasonable
concern and exactly the kind a reviewer raises.

**Test.** Cells A′ and C′ (D-022): identical to A and C, but tau chosen to match
the known base rate rather than to maximise accuracy — handing the threshold
cells the one piece of structural information forced choice was getting free.

**Outcome: FALSIFIED.** Giving the baseline the correct base rate did not shrink
the gap; it *widened* it, in both region conditions:

| | raw | base-rate matched |
|---|---|---|
| forced choice, whole image | +0.1600 | **+0.1900** |
| forced choice, cropped | +0.2200 | **+0.2500** |

Matching the base rate *costs* the threshold cells accuracy (−0.03 each), because
their accuracy-maximising tau was deliberately unbalanced (A predicted yes
136/200, C 60/200). The prior was not the source of the advantage.

**Reporting rule, owner instruction:** the raw and base-rate-matched contrasts
are always quoted **as a pair** — (+0.16, +0.19) and (+0.22, +0.25) — never one
alone. A′/C′ are a strictly harder baseline than A/C, so B−A′ and D−C′ are upper
bounds on the forced-choice effect, not neutral estimates.

**Why the remaining margin is not an unremoved confound** (added to the §2 draft
at the owner's direction): A′/C′ impose a *global* base rate (100 yes overall);
forced choice imposes a *pairwise* constraint (one yes per pair). The pairwise
constraint is strictly stronger and is **not available to a thresholding rule at
all** — a single threshold on independent scores has no representation of the
pair. Taking the pairwise constraint requires comparing the two options against
each other, which *is* forced choice. **The pairwise constraint is the
contribution**, and A′/C′ establish it does work beyond supplying a correct
global prior, which is the only part a threshold rule could have borrowed.

### D-026 — a mislabelled contrast, in prose only

The owner flagged that `table1_comparison.csv` reported "B − D cropping given
forced choice +0.0700", which should be "D − B".

**The CSV was correct.** Every row reads `D minus B | +0.0700`, and all ten rows
were verified against the raw results. **The inversion was in my report prose**,
where I wrote "B − D" in a summary table. The artifact was right and the
narration was wrong; the correction belongs to the report, not the file.

Label generation happens in exactly one place, `f'{b} minus {a}'` paired with
`bootstrap_paired_difference(R[a], R[b])`, which computes `b - a`. The pairing is
correct and no other row was affected.

**Guard added** (`tests/test_tables.py`, 7 tests). The load-bearing one recomputes
each row: for a label "X minus Y" it asserts the reported `point` equals
`accuracy(X) - accuracy(Y)` exactly. A sign check alone would not do — a negative
value is perfectly legitimate, so only recomputing the stated difference
distinguishes a real negative from a flipped label. Also guarded: labels parse,
no table contains both "X minus Y" and its inverse, CIs bracket their point
estimate, `excludes_zero` agrees with the CI, and the position-only row never
appears without its ARTIFACT warning.

**The guard was verified by injecting the exact inversion**: rewriting the row to
"B minus D" makes the test fail with "reports +0.070000 but accuracy(B) −
accuracy(D) = −0.070000. If these differ by exactly a sign, the label is
inverted." Restoring the file makes all 7 pass.

### D-027 — PRD §3 non-goals: no action, deferred to the review

Owner ruling. PRD §3 disclaims region grounding as prior art (ESREAL), which
reads oddly beside D-019's finding that region grounding alone does nothing
measurable. Noted as a question for the review, not resolved here.

---

## M3 — 2026-09-07

### D-028 — decision authority; M2 gate PASSED; four open questions resolved by the owner

**Authority.** PRD §9 previously routed the M2 gate through a "go/no-go review
with guide". That no longer applies: **the project owner is the sole
decision-maker.** PRD §9 has been updated accordingly -- the first modification
to `PRD.md` since M0, made on explicit owner instruction.

**M2 gate: PASSED**, 2026-09-07, by the owner. Cell D 0.8600 vs Cell A 0.6300 on
100 dev pairs; D − A = +0.2300, 95% CI [+0.1515, +0.3021], excludes zero. PRD §5's
success criterion is met at this sample size, so the negative-result framing is
not triggered.

**The four open questions, resolved by the owner:**

1. **PRD §2 framing** -- keep the current §2 text for now; the replacement in
   `docs/prd_section2_draft.md` stays a draft pending M3. Its central claim is
   built on 100 dev pairs and could invert if the full dev run moves region
   grounding away from zero (D-019).
2. **Pairwise-constraint argument -- ACCEPTED.** A′/C′ impose a global base rate;
   forced choice imposes a per-pair constraint; a threshold rule cannot take the
   latter without becoming forced choice. The remaining margin is therefore the
   contribution, not an unremoved confound (D-025).
3. **Sample size** -- the full dev set is sufficient for now. No additional data
   collection or resampling scheme.
4. **PRD §3 non-goals -- UNCHANGED.** The oddity noted in D-027 stands
   unresolved by choice, not by oversight.

### D-029 — M3 gate scope: state and action only; D1 and D3 do not close at M3

PRD §9 gave M3 the gate *"D1, D2, D3 complete"*, which is not reachable as
written. PRD §4 defines D1 as predictions for *"all AMBER attribute questions"*
and D3 as a breakdown by *"state / action / number"*. Both name the 2,072
`discriminative-attribute-number` questions, which do not pair reliably
(1,506 singletons vs 283 pairs) and which TRD §4 routes to `counting.py` — an
**M4** module that does not exist.

**Owner ruling, recorded in PRD §9:**
- At M3, **D1 and D3 are complete for `state` and `action` only**.
- **A partial D1 is NOT declared complete.**
- The `number` breakdown **slips to M4** with `counting.py`; D1 and D3 close then.
- D2 completes in full at M3.

### D-030 — the original hypothesis is not supported by the data

Recorded plainly, at the owner's instruction, so it cannot be quietly revised
later.

PRD §2 was written on the premise that **region grounding is the primary
mechanism** — that cropping to the named object is what fixes attribute
hallucination, with forced choice as a second, independent improvement. The M2
ablation does not support that premise:

| contrast | isolates | delta | 95% CI | |
|---|---|---|---|---|
| C − A | **region grounding alone** | **+0.0100** | [−0.0490, +0.0680] | **includes zero** |
| C' − A' | region grounding alone, base-rate matched | +0.0100 | [−0.0348, +0.0510] | **includes zero** |
| B − A | forced choice alone | +0.1600 | [+0.0859, +0.2330] | excludes zero |

**Cropping alone contributes nothing measurable, under both threshold rules.
Forced choice carries the effect.** Region grounding pays only in combination
(D − B = +0.0700, and a +0.0600 superadditive interaction).

**The project's original hypothesis is therefore not supported.** The
contribution is now **the ablation itself**: a measurement showing that a widely
assumed mechanism does not work on its own, and that its value is entirely
interactional.

**No method will be adjusted to recover the original story.** No threshold,
padding, prompt template or constant has been changed in response to this
finding, and none will be. The tuning log remains empty. If a later result moves
region grounding away from zero, that will be reported as a change in evidence,
not as a vindication of the framing.

### D-031 — deferred to M4, by owner instruction

- **Crop caching** (TRD §0, specified and unbuilt): NOT added mid-run. Cells C, D
  and C' each repeat the same 1,929 OWLv2 detections in the M3 run, roughly
  two-thirds of its wall clock. Built at the **start of M4**.
- **`REL_RE`** (D-001): the corrected regex is applied at **M4**, when
  `src/modules/relation.py` is created. Not touched before then.

### D-032 — Cell D-ext: an external competing attribute

**The objection this answers.** Cell D takes its competing attribute from AMBER:
for (sky, sunny, gloomy) it scores "sunny" against "gloomy" because the dataset
supplies that contrast. A reviewer can fairly argue the method exploits pair
structure the baseline has no access to, making the comparison unfair. **A'/C' do
not answer this** — they give the threshold cells a *global* 50% prior, which is
strictly weaker than pairwise complementarity (D-025).

**The design.** Cell D-ext is identical to Cell D except that the competitor
comes from `configs/antonyms.yaml`, never from the dataset's paired negative.
**Each question is answered on its own**: "Is the sky sunny?" is scored against
whatever the map says is the opposite of *sunny*, and "Is the sky gloomy?"
against the opposite of *gloomy*. The two questions of a pair never see each
other, so D-ext can legitimately answer yes twice or no twice. **No pair
structure is used at any point.**

**Unmapped attributes produce no record.** They are counted and reported, never
guessed. There is deliberately **no fallback to the dataset's negative** — that
would reintroduce the exact leak the cell exists to remove.

**Pre-registration.** The map was written and committed **before any D-ext result
existed**, at the owner's instruction:

| commit | time | contents |
|---|---|---|
| `ea3cc68` | 2026-09-07 22:33:13 +0530 | the antonym map alone, 210 entries |
| `bcd1480` | 2026-09-07 22:35:11 +0530 | D-ext implementation + tests |
| (first D-ext run) | later | — |

**Construction rule**, stated in the file so it is auditable: each entry is the
opposite of the key *as a word*, from its ordinary English meaning. The dataset's
negative attribute was not consulted for any entry. Colour entries use
conventional visual contrast or complementary hue and are flagged in-file as the
weakest, most arbitrary part of the map. Most actions have no opposite and are
omitted.

**Coverage, computed before running:**

| subset | mapped / total | rate |
|---|---|---|
| state | 4508 / 4756 | 0.9479 |
| action | 628 / 792 | 0.7929 |
| **all** | **5136 / 5548** | **0.9257** |

Largest unmapped: `calm waters` (85), `rolling waves` (58), `swim` (34),
`calm seas` (22), `rippling water` (21), `jump` (21) — mostly scene phrases and
actions without opposites.

**The map is not the dataset pairing rebadged.** On pairs whose positive
attribute is mapped, the map's competitor **differs from AMBER's negative on
44.1%** (1128 of 2555) and agrees on 55.9%. Some agreement is expected and
healthy — sunny/gloomy really are opposites. A test asserts the disagreement
count is non-zero, so the cell can never silently degenerate into Cell D.

**Stakes, per the owner:** this is the strongest single result in the paper if it
holds, and the most important limitation if it does not.
