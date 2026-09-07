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

**Status: RAISED, NOT IMPLEMENTED — awaiting owner decision.**

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

Deferred pending owner decision: the both-orders averaging.

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
