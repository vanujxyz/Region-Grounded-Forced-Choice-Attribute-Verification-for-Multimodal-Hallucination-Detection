# DRAFT — proposed replacement for PRD §2 "The one contribution"

**Status: DRAFT, rewritten for the full dev set. `PRD.md` has NOT been modified.**

**Revision history.** The first version of this file was drafted against the
100-pair slice and asserted that region grounding alone "does nothing
measurable". The full dev set falsified that: C − A = +0.0150 with a 95% CI of
[+0.0023, +0.0278], which excludes zero (D-038). That version carried a STOP
banner and is preserved in git history at commit `b800ba3`, so the correction is
traceable. This version states the corrected claim.

**Why a rewrite of PRD §2 is needed.** PRD §2 presents the method as "two
independent changes", and PRD §5 says the 2×2 exists so a reviewer cannot ask
whether the gain came from only one of them. On the full dev set the answer is
that it largely did: forced choice is worth **+0.1858** and region grounding
**+0.0150** — a real effect, but roughly **twelve times smaller**. The current
framing asserts a parity the measurement does not support.

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
> **The two are not equal partners, and the ablation says so.** Measured
> independently on 3,858 development questions, with 95% confidence intervals
> from a bootstrap over 696 images:
>
> - **Forced choice is the dominant mechanism.** Applied to the whole image it is
>   worth **+0.1858** over the thresholding baseline ([+0.1679, +0.2036]).
> - **Region grounding is real but roughly twelve times smaller**: **+0.0150**
>   ([+0.0023, +0.0278]). The interval excludes zero, so the effect is not null —
>   but it is an order of magnitude below the question format.
> - **They interact positively.** Adding the crop to forced choice is worth a
>   further +0.0373 ([+0.0217, +0.0529]), and the combination is superadditive by
>   **+0.0223** over the sum of the two individual effects.
> - **Together: +0.2232** ([+0.2057, +0.2405]).
>
> We therefore do not claim two independent improvements, and we do not claim
> that region grounding is the mechanism. The claim is narrower and better
> supported: **the decision rule is what matters, and cropping to the named
> object is a smaller, complementary gain that pays most when combined with it.**
> Isolating that is what the 2×2 in §5 is for.
>
> *Hypothesis (not measured).* We offer, as an untested explanation only, that a
> crop removes the surrounding context a thresholded absolute score relies on for
> calibration, so cropping alone trades one failure mode for another, whereas a
> forced choice between two attributes is scale-free and keeps only the
> comparison that matters. This is consistent with Cell C's behaviour, but **no
> experiment in this work isolates the mechanism, and we do not claim it as a
> finding.**
>
> Three controls support the claim rather than decorate it.
>
> **First, the base rate.** Forced choice emits exactly one "yes" per contrastive
> pair, and AMBER's attribute pairs are exactly balanced, so it is handed a
> correct base rate for free. Threshold cells A′ and C′ choose their threshold to
> match that base rate instead of to maximise accuracy. Giving the baseline the
> prior does not close the gap — it widens it: forced choice is worth
> **(+0.1858, +0.1918)** on the whole image and **(+0.2081, +0.2157)** on the
> crop, quoting the raw and base-rate-matched contrasts as a pair.
>
> **The two constraints are not the same constraint.** A′/C′ impose a *global*
> base rate; forced choice imposes a *pairwise* one — exactly one "yes" within
> each pair. The second is strictly stronger and is not available to a
> thresholding rule at all, since a single threshold on independent scores has no
> representation of the pair. Taking the pairwise constraint requires comparing
> the two options against each other, which *is* forced choice. The remaining
> margin is therefore the contribution, not an unremoved confound.
>
> **Second, an external contrast.** A reviewer may object that the method
> exploits pair structure the baseline cannot access, since AMBER supplies the
> competing attribute. Cell D-ext removes that entirely: the competitor comes
> from a fixed antonym list written before any result was computed, and **each
> question is answered on its own**, so no pair structure is used and the method
> is free to answer "yes" to both halves of a pair. On the matched covered subset
> D-ext still beats the baseline by **+0.1827** ([+0.1649, +0.2002]), at a cost of
> −0.0482 against Cell D. Restricted further to the questions where the external
> word *disagreed* with AMBER's, it scores 0.7474 against the baseline's 0.6344.
> **The gain does not depend on the dataset's own contrast.**
>
> **Third, a benchmark artifact.** AMBER orders every attribute pair so the true
> attribute holds the lower question id, which makes a position-only detector
> that never opens an image score 100%. We report that baseline explicitly and
> confirm by permutation that no cell in our system reads it.

---

## Numbers this draft cites — full dev, 3,858 questions, 696 images, fit-on-eval

| claim | value | 95% CI | source |
|---|---|---|---|
| forced choice alone | +0.1858 | [+0.1679, +0.2036] | B − A |
| forced choice, base-rate matched | +0.1918 | [+0.1744, +0.2093] | B − A′ |
| region grounding alone | **+0.0150** | **[+0.0023, +0.0278]** | C − A |
| region grounding, base-rate matched | +0.0135 | [+0.0021, +0.0248] | C′ − A′ |
| forced choice given cropping | +0.2081 | [+0.1908, +0.2249] | D − C |
| forced choice given cropping, matched | +0.2157 | [+0.1985, +0.2323] | D − C′ |
| cropping given forced choice | +0.0373 | [+0.0217, +0.0529] | D − B |
| superadditive interaction | +0.0223 | — | (D−A)−(B−A)−(C−A) |
| D-ext vs baseline, matched subset | +0.1827 | [+0.1649, +0.2002] | D-ext − A |
| D-ext cost vs Cell D | −0.0482 | [−0.0589, −0.0377] | D-ext − D |
| **headline** | **+0.2232** | **[+0.2057, +0.2405]** | D − A |

**Reporting rules (D-025, D-041).** Raw and base-rate-matched contrasts are
always quoted as a pair. Cropping is described as **neither null nor the
mechanism**.

## Caveats the paper must carry

1. **All of the above is the development split, and fit-on-eval** wherever a
   threshold is involved (A, C, A′, C′ each fit tau on the data they are scored
   on). That favours the *baseline*, so the comparison is conservative. The
   headline result is the M5 test-split run with tau frozen from the dev fit.
2. **The correction is part of the record.** An earlier version of this claim,
   drafted on 100 pairs, asserted that region grounding does nothing measurable.
   The full dev set falsified it and the claim was withdrawn in writing (D-030 →
   D-038). Per D-041 the correction is reported in the paper, not just the
   corrected number.
3. **D-ext coverage is 0.9194 overall but 0.7885 for action** (D-040). The
   unmapped attributes are systematically those with no opposite (`swim`, `jump`,
   `surf`) plus scene phrases (`calm waters`). The matched-subset design controls
   the comparison but cannot speak to performance where no antonym exists. This
   is a scope condition of the method, not a defect.
4. **The antonym map was written by the same author as the method.** Bounded by
   three things — committed before any result, disagrees with AMBER's negative on
   44.1% of pairs, construction rule stated in-file — but it is not blind.
5. **Effective sample is smaller than n.** 3,858 questions rest on 661 distinct
   (object, positive, negative) triples; the bootstrap resamples 696 images.
6. **PRD §3 (non-goals)** disclaims region grounding as prior art (ESREAL), which
   now reads oddly beside a finding that it contributes +0.0150 rather than the
   bulk of the effect. Left unchanged by owner decision (D-027).
