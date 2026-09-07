# DRAFT — proposed replacement for PRD §2 "The one contribution"

**Status: DRAFT. `PRD.md` has NOT been modified.** This is for the M2 review.

**Why a rewrite is needed.** PRD §2 currently presents the method as "two
independent changes", and PRD §5 says the 2×2 exists so "a reviewer cannot ask
whether the gain came from only one of the two changes". On 100 dev pairs the
answer to that question is: **it largely did.** Region grounding alone is
+0.0100 with a 95% CI of [−0.0490, +0.0680] that straddles zero, and it stays
there under the base-rate-matched control (+0.0100, [−0.0348, +0.0510]). The
current framing asserts something the measurement does not support.

---

## Proposed replacement text

> ### 2. The one contribution
>
> Existing detectors check attribute claims ("the apple is red") by scoring the
> text against the **whole image** with a contrastive model, then thresholding
> the score. This is known to be unreliable: contrastive vision-language models
> bind attributes to objects poorly (Yuksekgonul et al., ICLR 2023; Thrush et
> al., CVPR 2022). UNIHD (Chen et al., 2024) states the gap explicitly — it
> reports minimal improvement on attribute-level hallucination and attributes
> this to the absence of a specialised attribute tool.
>
> **This project builds that tool**, from two changes to the standard approach:
>
> 1. **Forced choice** — instead of asking "how well does 'sunny sky' match?"
>    and thresholding, present the candidate attribute values as mutually
>    exclusive options and take the argmax.
> 2. **Region grounding** — locate the object named in the claim, crop it out,
>    and evaluate on the crop rather than the full image.
>
> **These two changes are not equal partners, and the evaluation says so.** On
> the development set the two contribute asymmetrically, and the contribution of
> this work is the *interaction* between them rather than a sum of two
> independent gains:
>
> - **Forced choice carries the effect on its own.** Applied to the whole image
>   it is worth +0.16 over the thresholding baseline (95% CI [+0.086, +0.233]).
> - **Region grounding on its own does nothing measurable.** Cropping while
>   keeping the threshold decision rule is worth +0.01, with a confidence
>   interval spanning zero ([−0.049, +0.068]).
> - **Region grounding pays only once forced choice is in place.** Adding the
>   crop to forced choice is worth a further +0.07 ([+0.022, +0.121]), and the
>   combination is superadditive by +0.06 relative to the sum of the two
>   individual effects.
>
> We therefore do not claim two independent improvements. We claim that
> **cropping to the named object is only useful when the decision rule is a
> contrast between competing attributes.** The measured superadditivity is the
> evidence for that claim, and isolating it is what the 2×2 in §5 is for.
>
> *Hypothesis (not measured).* We offer, as an untested explanation only, that a
> crop removes the surrounding context a thresholded absolute score relies on for
> calibration — so cropping alone trades one failure mode for another — whereas a
> forced choice between two attributes is scale-free and keeps only the
> comparison that matters. This is consistent with Cell C's recall collapsing to
> 0.44 and with base-rate matching failing to rescue it, but **no experiment in
> this work isolates the mechanism, and we do not claim it as a finding.**
>
> Two controls support the claim rather than decorate it.
>
> **First, the base rate.** Forced choice emits exactly one "yes" per contrastive
> pair, and AMBER's attribute pairs are exactly balanced, so it is handed a
> correct base rate for free. We therefore report threshold cells A′ and C′ whose
> threshold is chosen to match that base rate instead of to maximise accuracy.
> Giving the baseline the prior does not close the gap — it widens it. Quoting
> both the raw and the base-rate-matched contrast, forced choice is worth
> (+0.16, +0.19) on the whole image and (+0.22, +0.25) on the crop.
>
> **The two constraints are not the same constraint, and this is the point.**
> A′ and C′ impose a *global* base rate: 100 "yes" answers across the evaluation
> set as a whole. Forced choice imposes a *pairwise* constraint: exactly one
> "yes" within each contrastive pair. The second is strictly stronger, and it is
> not available to a thresholding rule at all — a single threshold applied to
> independent scores cannot enforce a per-pair condition, because it has no
> representation of the pair. To give a threshold cell the pairwise constraint,
> one would have to compare the two options against each other and take the
> better, which *is* forced choice. The remaining margin after A′/C′ is therefore
> not an unremoved confound awaiting a better control. **The pairwise constraint
> is the contribution**, and A′/C′ establish that it is doing work beyond
> supplying a correct global prior, which is the only part of it a threshold rule
> could have borrowed.
>
> **Second, the position artifact.** AMBER orders every attribute pair so the
> true attribute holds the lower question id, which makes a position-only
> detector that never opens an image score 100%. We report that baseline
> explicitly as a benchmark artifact and confirm by permutation that no cell in
> our system reads it.

---

## Numbers this draft cites (all from 100 dev pairs, fit-on-eval, TRD §15 step 9)

| claim in the draft | value | CI | source |
|---|---|---|---|
| forced choice alone | +0.1600 | [+0.0859, +0.2330] | B − A |
| region grounding alone | +0.0100 | [−0.0490, +0.0680] | C − A |
| region grounding, base-rate matched | +0.0100 | [−0.0348, +0.0510] | C′ − A′ |
| cropping given forced choice | +0.0700 | [+0.0217, +0.1215] | D − B |
| superadditive interaction | +0.0600 | — | (D−A)−(B−A)−(C−A) |
| forced choice net of base rate, whole image | +0.1900 | [+0.1094, +0.2588] | B − A′ |
| forced choice net of base rate, cropped | +0.2500 | [+0.1700, +0.3182] | D − C′ |
| headline | +0.2300 | [+0.1515, +0.3021] | D − A |

**Reporting rule (owner instruction):** the raw and base-rate-matched contrasts
are always quoted as a pair — **(+0.16, +0.19)** and **(+0.22, +0.25)** — never
one alone. A′/C′ are a strictly harder baseline than A/C, so B−A′ and D−C′ are
upper bounds on the forced-choice effect, not neutral estimates.

## Caveats the review must weigh before this text is adopted

1. **Every number above is 100 dev pairs — 33 bootstrap units, 62 distinct
   triples.** The signs are probably stable; the magnitudes are not. If M3's full
   dev run moves region grounding away from zero, this paragraph needs rewriting
   again, in the other direction.
2. **The mechanism sentence is a labelled hypothesis, by decision.** It is set
   off in its own paragraph, opens with "*Hypothesis (not measured)*", and ends
   with an explicit disclaimer. Owner ruling: keep it as a hypothesis, do **not**
   design an experiment for it — out of TRD scope and it does not earn its cost.
   If it is ever promoted to a finding, it needs an experiment first.
3. **PRD §3 (non-goals) also needs a look.** It currently disclaims region
   grounding as a contribution on the grounds that ESREAL did it. That disclaimer
   now sits oddly beside a finding that region grounding alone does nothing —
   which is arguably a more interesting negative result than the disclaimer
   allows for.
4. This draft keeps the ordering of §2's two changes swapped relative to the
   current PRD (forced choice first), because that is the order of their
   measured importance.
