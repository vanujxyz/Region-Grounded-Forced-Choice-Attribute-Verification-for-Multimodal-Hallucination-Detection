# DECISIONS addendum (October 2026 revision, after faculty review)

These entries continue `docs/DECISIONS.md` (D-001 … D-060).

## D-061 — third dataset: VAW, following VISOR's construction

The faculty review asked for a comparison with VISOR (arXiv:2608.11024) on the dataset VISOR uses, VAW. The protocol is in `data/vaw/VAW_PROTOCOL.md` and was frozen before any model was run on VAW. Two clauses were amended before any model run, and both amendments are recorded inline: train_part2 was added as a dev source, and the scarcest type is sampled first. `configs/main.yaml` is unchanged.

Execution note: VAW was run on the same NVIDIA RTX 3050 Laptop GPU. The environment was a fresh one (torch 2.14.1, transformers 5.16.1, which matches the pinned transformers version), the model revisions were the same pinned ones, and the weights were fetched at those revisions into a short local path (Windows MAX_PATH, see D-005). Every run manifest records the device, versions and model source. The grey null-image text scores used for the post-hoc AMBER/SHROOM calibration baselines (D-062) were computed on CPU.

## D-062 — calibration baselines on all three datasets (post-hoc for AMBER/SHROOM)

The faculty's novelty statement contrasts *calibrating* the threshold (VISOR) with *removing* it (forced choice). To test that directly, two calibration baselines are applied to the same scorer:

* **Cal-word**: a per-attribute-word threshold fit on dev, falling back to the global τ for words with fewer than 10 dev questions.
* **Cal-null**: a VISOR-style null-image prior correction, s − γ·s(I∅), with γ and τ fit on dev.

On VAW both were pre-declared (D-061). On AMBER and SHROOM-Vis they are **post-hoc**: they were added after those datasets' test splits had already been opened. They are baselines, not changes to the proposed method, and no parameter of cells A–D is touched. They are labelled post-hoc wherever they are reported.

## D-063 — two image-blind diagnostics, declared before the VAW test logits were read

The first Cal-word numbers on AMBER (test 0.7746 for whole image, 0.7787 for crop) were close to forced choice. That raised the question of whether a per-word threshold reads the image or simply learns which attribute words tend to be the true ones. Two image-blind cells answer this, and they are applied to all three datasets:

* **WordPrior**: answer each question with the majority dev gold label of its attribute word (at least 10 dev questions; otherwise "yes"). It never opens an image.
* **FC-null**: forced choice on the grey null image I∅. It measures how much of forced choice could come from the scorer's text-only preference between the two words.

These were declared after the AMBER/SHROOM Cal-word numbers were seen, but before any VAW test result was read. At the time of writing, the VAW run was still in its detection stage.

## D-064 — generalisation check with a generative VLM scorer (pre-registered)

Frozen before any VLM result existed. The contrastive scorer is held fixed throughout the main study by design, so the obvious question is whether the decision-rule effect is a property of SigLIP. This check answers it with a scorer of a different class: a multimodal LLM answering in natural language.

**Scorer.** A Haiku-class multimodal LLM (`claude-haiku-4-5`), accessed through the host API at temperature 0. It is not either of the models under test and was not used to annotate VAW (VAW labels are human). It *is* from the same family as the SHROOM-Vis annotator, so this check is reported on VAW only, where the labels are human.

**Sample.** The VAW **test** split, stratified: the first 150 pairs of each attribute type (color, material, state) in the frozen pair order, i.e. 450 pairs and 900 questions over 450 images. A subsample is used because each question costs an API call; 450 image clusters give a bootstrap interval of roughly ±0.04, which is far narrower than the effect under test (about +0.18 with SigLIP).

**Arms (whole image only; no region arm is run).**
* **VLM-threshold** — independent yes/no probing, the shape VISOR uses: *"Is the {object} {attribute} in this image? Answer only yes or no."* Each question is answered alone. The model's own absolute decision boundary plays the role of τ.
* **VLM-FC** — pairwise forced choice on the same evidence: *"Which word better describes the {object} in this image: {a} or {b}? Answer with exactly one of those two words."* The two candidates are given in alphabetical order, as in cells B and D. The winner is answered \yes{} and the loser \no.
* **VLM-prior** — the same independent questions asked over a uniform grey 128×128 image, the discrete analogue of VISOR's δ_prior. It measures how often the scorer answers yes with no visual evidence.

**Analysis.** Accuracy, Yes-ratio, FPR and the pair-outcome classification (correct / both-yes / both-no / reversed), with the same image-level paired bootstrap (10,000 resamples, seed 20260907). The primary contrast is VLM-FC − VLM-threshold.

**Scope limits, stated in advance.** Token log-probabilities are not available through this API, so δ_vis cannot be computed and **no calibration arm is possible for this scorer**: VISOR-Calib cannot be reproduced here, and the comparison of calibration with forced choice remains a SigLIP-only result (D-062). Unparseable replies are counted and excluded rather than guessed.

**Falsification.** If VLM-FC − VLM-threshold is zero or negative on this sample, the decision-rule effect does not generalise beyond a contrastive scorer, and that is what will be reported.

### D-064a — what was actually executed (amended after the run stopped, before any reply was read)

The run stopped part-way through on an API budget limit of the execution environment, not on anything in the data. Two routes to finish it were unavailable in the same session: the sandbox had no outbound network, so an open-weight VLM could not be downloaded and run locally with logits, and parallel execution was disabled, so the remaining calls could not be dispatched elsewhere.

The requests were issued in the frozen order of the pair list, which is grouped by type (colour, then material, then state), so the completed calls are a prefix: **colour only**. After keeping only pairs whose two questions survived in all three arms, the executed sample is **88 colour pairs, 176 questions, 88 images** out of the 450 pairs planned. 68 replies were unparseable and were excluded, 40 of them in the grey-null-image arm, where the model usually declines to answer rather than guess.

The reduction is reported in the paper as a scope limitation, and the check is labelled colour-only throughout. No reply was inspected before this amendment was written. The material and state arms, the region arm and a logit-based calibration arm remain future work.
