# SHROOM-Vis attribute annotation — policy

Frozen **before any SHROOM result existed**. Every rule below was fixed in
advance so that no annotation choice could be made in response to a score.
Written in the same spirit as `configs/antonyms.yaml` (D-032).

## What is being built

An AMBER-style contrastive attribute benchmark over the 2,495 images in
`shroom-visions-images/shroom-vis-images/`, so that the four cells (A, B, C, D)
can be run on a second dataset without any change to the method.

Output is emitted in AMBER's exact schema — `annotations.json` +
`query/query_all.json`, joined by `annotations[id - 1]` — so the existing
loader, pair builder, detector, scorer, metrics and bootstrap all apply
unmodified.

## Who annotated it, and what that costs

The annotator is **Claude (Opus 5), a vision–language model** — not a human,
and not either of the two models under test.

This is a **silver-standard** benchmark and is reported as such. The
independence argument is narrow but real: OWLv2 and SigLIP are the systems
being measured, and neither was consulted at any point during annotation, so
the gold labels are not the system's own output fed back to it. No claim is
made that these labels match what a human panel would produce. AMBER's own
labels are themselves partly model-derived, but AMBER had human verification
and this does not.

The concrete risk is **difficulty calibration**, not correctness: in a
forced-choice benchmark the negative attribute sets the difficulty. An annotator
who writes easy negatives inflates every cell; one who writes adversarial
negatives depresses every cell. The rules in §"Choosing the negative" exist to
stop that choice drifting, and they were fixed before any number was seen.

## Question forms

Exactly AMBER's two discriminative-attribute templates, character for
character, so `src/data/pairs.py` parses them with its existing regexes:

| kind | template | annotation type |
|---|---|---|
| state | `Is the {obj} {attr} in this image?` | `discriminative-attribute-state` |
| action | `Does the {obj} {attr} in this image?` | `discriminative-attribute-action` |

Number questions (`discriminative-attribute-number`) are **not** annotated:
counting is a separate module and is not part of the 2x2 contribution.
Existence and relation questions are not annotated either.

## What gets annotated per image

- **1 to 4 pairs per image.** AMBER averages 2.76 attribute pairs per image and
  this targets a similar density, but an image that supports only one confident
  pair contributes one. Padding a thin image to hit a quota is how weak pairs
  enter a benchmark, and a weak pair is one whose negative is arguably true --
  exactly what rule 4 below forbids. Quantity yields to confidence.
  *(Amended from "2 to 4" after the first 10 images were viewed and before any
  SHROOM result existed or any model had been run on this data.)*
- Every pair is one **object** plus one **true** and one **false** attribute of
  the same kind, over the same object, in the same image.
- Prefer objects that are **large and unambiguous** in the frame. The object
  name is what OWLv2 will be asked to localise; an object that a reader cannot
  point to is not annotatable.
- An action pair is written **only** when the object is animate or a vehicle and
  is visibly doing something. Most images get state pairs only. This mirrors
  AMBER, where actions are 792 of 5,556 attribute questions (14%).

## Object naming

- Lowercase, singular, bare noun: `dog`, `sky`, `plate` — never `the dog`,
  `dogs`, `a brown dog`.
- **One word.** AMBER's parsing regex is non-greedy on both fields, so
  `Is the trash can green in this image?` parses as object `trash`, attribute
  `can green` — silently wrong. A multi-word object cannot be expressed in this
  question form at all, so `bin` is used rather than `trash can`. The builder
  round-trips every generated question through the regex and aborts if it does
  not parse back to the object and attribute that produced it, so this cannot
  slip through. *(Attributes may still be multi-word — `rolling waves`,
  `lie prone` — because they occupy the final field.)*
- **The attribute must not appear in the object name.** `Is the red car red?`
  is unanswerable-by-construction and is forbidden.
- AMBER's 340-object vocabulary (`data/amber/relation.json`) is preferred where
  a term fits. Where it does not — SHROOM contains penguins, flamingos, moose,
  cannons, shovels — the natural word is used instead. Each pair records
  `in_amber_vocab`, so in-vocabulary and out-of-vocabulary performance can be
  reported separately. This is a deliberate test of generalisation beyond
  AMBER's closed vocabulary and must not be silently mixed into one number.

## Choosing the attributes

- The **positive** attribute must be clearly true of that object in that image
  at thumbnail resolution. If it needs a second look, it is not used.
- Attributes are drawn from AMBER's observed vocabulary wherever one fits
  (`white/black`, `sunny/gloomy`, `clean/dirty`, `open/closed`, `wet/dry`,
  `tall/short`, `lively/withered`, `stand/sit`, ...). New attributes are allowed
  where SHROOM's content demands them.
- Single words, or the short phrases AMBER itself uses (`lie prone`,
  `ride a bike`, `rolling waves`). Lowercase.

## Choosing the negative — the rule that sets difficulty

In priority order:

1. **Same dimension, mutually exclusive.** The negative must be an attribute
   that could have been true of that object but is not: colour against colour,
   posture against posture, texture against texture. `white` vs `black`, not
   `white` vs `running`.
2. **Plausible for the object class.** The negative must be something that
   object *can* be. `dry` is a legitimate negative for a towel; it is not for
   the sea.
3. **Not decidable from the object name alone.** If a language model could
   answer the pair without the image, it is not a vision benchmark. `Is the
   penguin feathered or scaled?` is forbidden.
4. **Unambiguously false.** Where a negative might arguably hold, the pair is
   dropped rather than guessed.

Rule 3 is the one that matters most, and it is the hardest to satisfy for the
1,561 SHROOM files whose *filename* names the object and its subtype
(`alligator_Albino_alligator_...`). Filenames were used only to help the
annotator identify an object; **no filename text enters any query, attribute or
label**, and no attribute was written that is true merely by definition of the
object's name.

## Question-id assignment — the artifact AMBER has and this does not

`README.md` records a benchmark artifact in AMBER: in **all 2,774** attribute
pairs the true attribute holds the **lower** question id, so "answer yes to the
lower id" scores 1.0000 without opening an image.

**SHROOM-Vis deliberately does not reproduce this.** Within each pair, which of
the two questions receives the lower id is decided by a seeded coin flip
(`seed = 20260907`, the project seed). The expected position-only accuracy is
therefore 0.5, and `--module position-only` is run on SHROOM to confirm it.

## Splits

By image, never by question — the same rule as `src/data/splits.py`: seed
20260907, 70% dev / 30% test, frozen on first write.

## What is dropped rather than guessed

- Images with no confidently nameable object.
- Images too dark, blurred, or abstract to attribute at thumbnail resolution.
- Pairs where the negative might arguably also be true.

Dropped images are recorded with an explicit reason and counted in the coverage
report. They are never silently skipped.
