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
> contrast between competing attributes**, and we give the mechanism: a crop
> removes the surrounding context that a thresholded absolute score depends on
> for calibration, so cropping alone trades one failure mode for another, while
> a forced choice between two attributes is scale-free and keeps only the
> comparison that matters. The measured superadditivity is the evidence for this
> account, and isolating it is what the 2×2 in §5 is for.
>
> Two controls support the claim rather than decorate it. First, forced choice
> emits exactly one "yes" per contrastive pair, and AMBER's attribute pairs are
> exactly balanced, so forced choice is handed a correct base rate for free. We
> therefore report threshold cells A′ and C′ whose threshold is chosen to match
> that base rate instead of to maximise accuracy. Giving the baseline the prior
> does not close the gap — it widens it (forced choice net of base-rate
> knowledge: +0.19 whole-image, +0.25 cropped). The advantage is not the prior.
> Second, AMBER orders every attribute pair so the true attribute holds the
> lower question id, which makes a position-only detector that never opens an
> image score 100%. We report that baseline explicitly as a benchmark artifact
> and confirm by permutation that no cell in our system reads it.

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

## Caveats the review must weigh before this text is adopted

1. **Every number above is 100 dev pairs — 33 bootstrap units, 62 distinct
   triples.** The signs are probably stable; the magnitudes are not. If M3's full
   dev run moves region grounding away from zero, this paragraph needs rewriting
   again, in the other direction.
2. **The mechanism sentence** ("a crop removes the surrounding context that a
   thresholded absolute score depends on for calibration") is an *explanation I
   have proposed, not a result I have measured.* It is consistent with Cell C's
   collapsed recall (0.44) and with base-rate matching not rescuing it, but no
   experiment here isolates it. Either mark it as a hypothesis in the paper or
   design a test for it.
3. **PRD §3 (non-goals) also needs a look.** It currently disclaims region
   grounding as a contribution on the grounds that ESREAL did it. That disclaimer
   now sits oddly beside a finding that region grounding alone does nothing —
   which is arguably a more interesting negative result than the disclaimer
   allows for.
4. This draft keeps the ordering of §2's two changes swapped relative to the
   current PRD (forced choice first), because that is the order of their
   measured importance.
