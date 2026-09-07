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
