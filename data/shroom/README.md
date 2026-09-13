# SHROOM-Vis — an AMBER-style attribute benchmark over the SHROOM vision images

A second dataset, built so the four cells of this project can be re-run **with
no change to the method**: same prompt template, same frozen thresholds
protocol, same pinned model revisions, same metrics, same image-level bootstrap.

| | AMBER (attribute) | SHROOM-Vis |
|---|---|---|
| images | 1,004 | 898 annotated (of 2,495 available) |
| attribute pairs | 2,774 | 2,523 |
| questions | 5,548 | 5,046 |
| pairs per image | 2.76 | 2.81 |
| distinct objects | — | 433 |
| distinct (object, true, false) triples | 417 | **1,127** |
| distinct attributes | — | 157 |
| gold balance | exactly 50/50 | exactly 50/50 |
| **"answer yes to the lower id" scores** | **1.0000** | **0.5020** dev / **0.5503** test |
| objects outside AMBER's 340-word vocabulary | 0 by construction | 831 pairs (33%) |

The last two rows are the reasons this dataset is worth having.

## Files

| file | what it is |
|---|---|
| `ANNOTATION_POLICY.md` | the rules, frozen before any SHROOM result existed |
| `annotations_raw.jsonl` | the raw annotation log, one JSON object per image |
| `annotations.json` | AMBER schema, joined by `annotations[id - 1]` |
| `query/query_all.json` | AMBER schema: `id`, `image`, `query` |
| `pair_meta.json` | per-pair provenance: object, attributes, ids, vocab flag |
| `build_summary.json` | counts and the id-ordering rate, written by the builder |
| `splits.json` | frozen 70/30 image-level split, seed 20260907 |
| `thumb_index.json` | annotation-time thumbnail index -> filename |

Rebuild everything downstream of the raw log with `python scripts/shroom_build.py`.

## Provenance and honesty

**The annotator is Claude (Opus 5)** — a vision–language model, and *not*
either of the two models under test (OWLv2, SigLIP). Neither was consulted
during annotation, so the labels are not the system's own output fed back to
it. But this is a **silver-standard** benchmark: there was no human
verification pass, and it should not be described as though there were. See
`ANNOTATION_POLICY.md` for the specific risk (difficulty calibration through
the choice of the false attribute) and the pre-registered rules that constrain
it.

Two images were dropped rather than guessed — one an abstract colour gradient,
one too dark and grainy to name an object in. Both reasons are recorded in
`build_summary.json`.

## Attribution

The **images** come from the SHROOM vision set and are not redistributed by this
project; they are read in place from `shroom-visions-images/`. The
**annotations** in this directory were produced for this project.

The question templates, the schema, and the two discriminative-attribute types
are AMBER's: [AMBER](https://github.com/junyangwang0410/AMBER) (Wang et al.,
2023), Copyright 2023 Alibaba X-PLUG Team, Apache License 2.0 — see
`../amber/NOTICE`.
