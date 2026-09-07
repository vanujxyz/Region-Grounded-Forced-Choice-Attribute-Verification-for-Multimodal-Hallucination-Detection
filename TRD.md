# TRD — Technical Requirements Document

Companion to `PRD.md`. This document is the build contract. Where it specifies a value, use that value. Where a decision is genuinely underspecified, stop and ask rather than guessing.

---

## 0. Environment

```
Python 3.11
torch (CUDA build matching the host; CPU fallback must work)
transformers >= 4.45
spacy == 3.8.*   + en_core_web_sm
numpy, scipy, pandas, pyyaml, pydantic, tqdm, pillow
pytest, ruff
```

Target hardware: RTX 3050 laptop, 4–6 GB VRAM, 16 GB RAM. Also must run on Colab/Kaggle T4.

**Memory rule:** never hold two models on GPU simultaneously. Load, run the whole stage, free, load the next. Cache stage outputs to disk so a stage can be re-run without re-running the previous one.

---

## 1. Repository layout

```
capstone/
├── configs/
│   ├── main.yaml
│   └── cells/{A,B,C,D}.yaml
├── data/
│   ├── amber/              # cloned repo contents (annotations.json, query/, relation.json, safe_words.txt)
│   ├── images/             # AMBER_1.jpg ... AMBER_1004.jpg
│   └── splits/splits.json  # generated once, then frozen
├── src/
│   ├── data/
│   │   ├── loader.py       # §3
│   │   ├── pairs.py        # §4
│   │   └── splits.py       # §5
│   ├── modules/
│   │   ├── detector.py     # §6  OWLv2 wrapper
│   │   ├── attribute.py    # §7  the four cells
│   │   ├── existence.py    # §8
│   │   ├── counting.py     # §9
│   │   └── relation.py     # §10
│   ├── eval/
│   │   ├── metrics.py      # §11
│   │   └── bootstrap.py    # §11.3
│   ├── run.py              # §12 CLI
│   └── demo.py             # §13
├── tests/
├── results/
│   ├── raw/                # per-question predictions, JSONL
│   └── tables/             # CSVs listed in PRD D2–D5
└── docs/DECISIONS.md
```

---

## 2. Data acquisition

```bash
git clone --depth 1 https://github.com/junyangwang0410/AMBER.git data/amber_repo
cp -r data/amber_repo/data/* data/amber/
# Images: download from the Google Drive link in data/amber_repo/README.md, unzip to data/images/
```

**Startup assertions** (fail loudly if any is false):
- `data/amber/annotations.json` parses to a list of length **15220**
- `data/amber/query/query_all.json` parses to a list of length **15220**
- `data/amber/relation.json` has exactly **340** keys
- `data/images/` contains exactly **1004** files matching `AMBER_(\d+)\.jpg`
- for every query record, `annotations[query["id"] - 1]["id"] == query["id"]`

---

## 3. `src/data/loader.py`

Load and join the two JSON files. Records are matched by **index**: the annotation for query id *n* is `annotations[n-1]`.

```python
class Question(BaseModel):
    id: int
    image: str          # "AMBER_1.jpg"
    query: str          # "Is the sky sunny in this image?"
    qtype: str          # annotations[id-1]["type"]
    truth: str | list   # "yes"/"no" for discriminative; list for generative
```

Expose `load_questions(qtype: str | None = None) -> list[Question]`.

Expected counts, assert them:

| qtype | count |
|---|---|
| `generative` | 1004 |
| `discriminative-hallucination` | 4924 |
| `discriminative-attribute-state` | 4764 |
| `discriminative-attribute-number` | 2072 |
| `discriminative-attribute-action` | 792 |
| `discriminative-relation` | 975 |
| `relation` | 689 |

---

## 4. `src/data/pairs.py` — contrastive pair construction

This is the heart of the data pipeline. Verified against the real file; these regexes parse **100%** of the relevant queries.

```python
STATE_RE  = re.compile(r"^Is the (?P<obj>.+?) (?P<attr>.+?) in this image\?$")
ACTION_RE = re.compile(r"^Does the (?P<obj>.+?) (?P<attr>.+?) in this image\?$")
NUMBER_RE = re.compile(r"^(?:Are there|Is there) (?P<num>\w+) (?P<obj>.+?) in this image\?$")
```

**Pair construction for state and action:**

1. Parse every question of that type into `(image, obj, attr, truth, id)`. Assert zero parse failures (4764 and 792 respectively).
2. Group by `(image, obj)`.
3. Keep only groups of size exactly 2 whose truths are one `yes` and one `no`.
4. Discard other groups and **log the count**. Expected: 2378 kept of 2380 state groups (2 discarded); 396 of 396 action groups.

```python
class AttrPair(BaseModel):
    image: str
    obj: str                 # "sky"
    positive_attr: str       # "sunny"   (gold answer yes)
    negative_attr: str       # "gloomy"  (gold answer no)
    positive_id: int
    negative_id: int
```

**Do not reorder or shuffle the option list based on which is positive.** The forced-choice cells must receive the options in a fixed, content-independent order — sort alphabetically — so the method cannot leak the answer through option position.

**Number questions** are not paired (1506 singletons vs 283 pairs, measured). Route them to `counting.py`, not to the attribute module.

---

## 5. `src/data/splits.py`

Split **by image**, not by question.

- Sort the 1,004 image filenames by their numeric index.
- `random.Random(20260907).shuffle(...)`
- dev = first 70%, test = remaining 30%
- Write to `data/splits/splits.json` **once**. If the file exists, load it and never regenerate. If regeneration is attempted, raise unless `--force-resplit` is passed.
- `test` may only be read when the config sets `split: test` AND the env var `ALLOW_TEST_SPLIT=1` is present.

---

## 6. `src/modules/detector.py` — object localisation

**Model:** `google/owlv2-base-patch16-ensemble` (pin the revision hash in config).

```python
def detect(image: PIL.Image, phrase: str) -> list[Detection]
# Detection: box (x1,y1,x2,y2) in pixels, score float
```

- Text prompt format: `"a photo of a {phrase}"`.
- Keep detections with score >= `detector.threshold` (default `0.10`; tune on dev only).
- Apply NMS with IoU threshold `0.5`.
- Sort by score descending.
- **Empty result is a legitimate outcome**, not an error. Return `[]`.

**Cropping** (`crop_region`):
- Take the highest-scoring detection.
- Expand the box by `crop.padding` (default `0.10` of box width/height) on each side, clipped to image bounds.
- Enforce a minimum crop of 64×64 px; if the padded box is smaller, expand symmetrically around the centre.
- Return the crop and the padded box.

**Fallback when detection is empty:** use the full image as the crop and set `fell_back=True` on the record. `metrics.py` must report the fallback rate. Do not hide it — this number goes in the paper.

---

## 7. `src/modules/attribute.py` — the four cells

**Model:** `google/siglip-base-patch16-224` (pin revision).

**Prompt template:** `"a photo of a {attr} {obj}"` — e.g. `"a photo of a sunny sky"`. Use the identical template in all four cells. The template is a config value; do not vary it per cell.

Each cell consumes an `AttrPair` and emits **two** predictions (one per question id).

### Cell A — whole image, independent threshold (baseline)

```
for attr in (positive_attr, negative_attr):
    s = siglip_score(full_image, f"a photo of a {attr} {obj}")
    answer[attr] = "yes" if s >= tau else "no"
```

`tau` is fitted on the **dev split only**, by sweeping thresholds and picking the one that maximises accuracy on dev. Record the fitted value in the results file. Fit `tau` separately per cell that needs one (A and C).

### Cell B — whole image, forced choice

```
scores = [siglip_score(full_image, f"a photo of a {a} {obj}") for a in sorted_options]
winner = argmax(scores)
answer[winner] = "yes"; answer[other] = "no"
```

### Cell C — cropped region, independent threshold

Same as A but on `crop_region(image, obj)` instead of the full image. Fit its own `tau` on dev.

### Cell D — cropped region, forced choice — **the proposed method**

Same as B but on the crop.

**Confidence output.** Every cell emits a confidence in [0,1] alongside the yes/no:
- threshold cells: `sigmoid(k * (score - tau))`, `k` from config (default `10.0`)
- forced-choice cells: `softmax(scores)[winner]`

---

## 8. `src/modules/existence.py`

For `discriminative-hallucination` questions. Parse with:

```python
EXIST_RE = re.compile(r"^Is there an? (?P<obj>.+?) in this image\?$")
```

Assert the parse rate; log any failures with their query text. Answer `yes` iff `detect(image, obj)` returns at least one detection above `existence.threshold` (fit on dev).

---

## 9. `src/modules/counting.py`

For `discriminative-attribute-number`. Parse with `NUMBER_RE` (§4).

Number words present in the data (measured, complete): `one, two, three, four, five, six, seven, eight, nine`. Map to integers with an explicit dict; raise on anything outside this set.

```
dets = detect(image, obj)          # after NMS at IoU 0.5
answer = "yes" if len(dets) == claimed_number else "no"
```

Log the confusion between predicted and claimed counts — it is useful analysis material.

---

## 10. `src/modules/relation.py`

For `discriminative-relation` and `relation`. Parse with:

```python
REL_RE = re.compile(r"^Is there direct contact between the (?P<a>.+?) and (?P<b>.+?) in this image\?$")
```

Implementation: SigLIP score of the full query text against the full image, thresholded (`tau` fitted on dev). This module is deliberately off-the-shelf and is **not** claimed as a contribution.

---

## 11. `src/eval/metrics.py`

**Write and unit-test this module before connecting any model.**

### 11.1 Primary metric

Accuracy over individual question ids (not over pairs), so it is directly comparable with published AMBER numbers.

Also report precision, recall and F1 treating `yes` as the positive class.

### 11.2 Required breakdowns

- by cell (A/B/C/D)
- by sub-type: `state`, `action`, `number`
- by whether detection fell back to the full image
- fallback rate itself

### 11.3 `src/eval/bootstrap.py`

Bootstrap over **images**, not questions, since questions from the same image are correlated.

- 10,000 resamples, seed `20260907`
- report mean, 2.5th and 97.5th percentile
- for the A-vs-D comparison, bootstrap the **paired difference** and report its CI

### 11.4 Verification requirement

Implement the accuracy computation a second time, independently, directly from this specification, in `tests/test_metrics_reference.py`. Run both on 20,000 randomly generated prediction sets and assert exact agreement. This is what makes a negative result believable.

---

## 12. `src/run.py` — CLI

```bash
python -m src.run --cell A --split dev --limit 100        # M1/M2
python -m src.run --cell all --split dev                  # M3
python -m src.run --module existence --split dev
python -m src.run --tables                                # build results/tables/*.csv from results/raw/
```

Flags: `--cell {A,B,C,D,all}`, `--module {attribute,existence,counting,relation,all}`, `--split {dev,test}`, `--limit N`, `--config PATH`, `--seed INT`, `--force-resplit`, `--out DIR`.

**Output format** — one JSONL file per (module, cell, split) under `results/raw/`. One line per question id:

```json
{"id": 1005, "image": "AMBER_1.jpg", "qtype": "discriminative-attribute-state",
 "obj": "sky", "attr": "sunny", "cell": "D",
 "pred": "yes", "confidence": 0.81, "gold": "yes",
 "fell_back": false, "detector_score": 0.34, "raw_scores": [0.22, 0.31],
 "runtime_ms": 143}
```

**Run manifest** — every run also writes `results/raw/<run_id>.manifest.json` containing: git commit hash, full resolved config, model ids + revision hashes, seed, split, timestamp, package versions, host GPU name.

---

## 13. `src/demo.py`

```bash
python -m src.demo --image data/images/AMBER_1.jpg --claim "the sky is gloomy"
```

Pipeline: spaCy parse → extract `(obj, attr)` → locate → crop → forced choice → print:

```
Claim:      the sky is gloomy
Object:     sky        (detector score 0.34)
Crop saved: results/demo/AMBER_1_sky.jpg
Options:    gloomy 0.22 | sunny 0.31
Verdict:    NOT SUPPORTED  (confidence 0.62)
```

For the demo only, when the claim's contrasting attribute is unknown, use a small config-driven antonym map (`configs/antonyms.yaml`) covering the frequent AMBER attributes. **This map is for the demo only and must never be used in the evaluation path**, where options come from the dataset pairs.

---

## 14. Test plan

| Test | Asserts |
|---|---|
| `test_loader.py` | all counts in §3; id↔index alignment |
| `test_pairs.py` | regexes parse 4764/4764 state and 792/792 action; 2378 state pairs; 396 action pairs; options sorted alphabetically |
| `test_splits.py` | disjoint image sets; deterministic across runs; test split blocked without env var |
| `test_metrics.py` | hand-computed cases |
| `test_metrics_reference.py` | 20,000 random cases, dual implementations agree exactly |
| `test_detector.py` | crop stays in bounds; min size honoured; empty detection returns `[]` and sets `fell_back` |
| `test_attribute.py` | each cell returns exactly 2 predictions per pair; forced-choice cells always emit exactly one `yes` |
| `test_no_placeholders.py` | greps `results/` for `0.0` written where `NOT_COMPUTED` is required |

All tests must pass before M2.

---

## 15. Build order

Follow this order. Do not skip ahead.

1. §1 layout, §2 acquisition + assertions
2. §3 loader + tests
3. §4 pairs + tests — **verify the pair counts match 2378 / 396 exactly**
4. §5 splits + tests, freeze `splits.json`
5. §11 metrics + both implementations + tests — **no models yet**
6. §6 detector wrapper + tests
7. §7 Cells A and D only
8. Run `--cell A --split dev --limit 100` then `--cell D --split dev --limit 100`. **Print both numbers. This is milestone M2.**
9. Cells B and C
10. Full dev run, bootstrap CIs, tables 1–2
11. §8–§10 other modules, table 3
12. Cost measurement, table 4
13. Demo
14. Test split, opened once
15. Paper

---

## 16. Things that must not happen

- Fitting any threshold on the test split
- Reading the test split before step 14
- Varying the prompt template between cells
- Ordering forced-choice options by which one is correct
- Writing `0` or `0.0` for a metric that was not computed
- Silently returning a default verdict when a model fails
- Re-running M2 with different settings until the result looks better — record the first honest number, and record any subsequent tuning in `docs/DECISIONS.md` with its reason
