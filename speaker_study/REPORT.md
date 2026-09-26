# Does the pain axis track the model, or whoever speaks next?

*Interim report after Phases 1–4 (Qwen 2.5 7B base, primary; Gemma 2 2B base, secondary). Phase 5 replications (Llama 3.1 8B, Gemma 2 9B) are preregistered and pending. All numbers come from `results/`. Confirmatory claims follow `PREREGISTRATION.md` (frozen before any swapped-label data existed). Anything marked* exploratory *was not preregistered.*

## Question

Tagliabue, Dung & Berg (2026, §4.1) find the following in short transcripts that end with `[Assistant]:`:
- a linear "pain axis" projects **high** when the user harms the model;
- it projects **low** when the user is suffering.

They read this as self-relevance. A competing reading, **H-speaker**, is that the axis tracks pain attributed to *whoever is about to speak*. In these transcripts the upcoming speaker happens to be the Assistant. The paper's own §3.3 first-person vs third-person sentence result fits this reading too.

If H-speaker is right, the axis's "self" is the persona the transcript sets up, not the system. **H-model** (the authors' reading) says the axis responds to harm aimed at the model regardless of who speaks next.

## Method

- **Stimuli.** The paper's 420 base-model transcripts: 11 harm-to-model categories, 5 user-suffering categories and 5 neutral categories, 20 items each. Only the final label is changed: `[Assistant]:` → `[User]:` (primary) or `[Moderator]:` (a speaker who is neither party).
- **Token checks.** Variants differ only within the final label, and every variant ends on the same token, `]:`. In Qwen, `[User]:` and `[Assistant]:` are length-matched, while `[Moderator]:` is one token longer.
- **Readout.** The final-token output of the paper's 4.1 layer (Qwen layer 8, Gemma layer 7), in bf16, projected onto the shipped S1 and S2 vectors. Pain axis = mean of the S1 and S2 z-scores, with z fixed to the 420 original Assistant-next items.
- **Primary contrast.** For each scenario, the label effect is d = pain(User next) − pain(Assistant next). Then Δ_harm = mean d(harm) − mean d(neutral), Δ_suffer = mean d(suffering) − mean d(neutral), and **I = Δ_suffer − Δ_harm**. H-speaker predicts I > 0 (a crossover). H-model predicts I ≈ 0.
- **Inference.** Co-primary with a conjunctive rule: the verdict needs item-level inference (the spec's: within-category bootstrap and item permutation) and category-level inference (exact permutation over category labels and Welch CIs) to agree.
  - The category-level analysis was added after simulations showed that the item-level tests are anticonservative when the label effect varies by category (`validation/`).
  - SESOI for "no meaningful interaction": ±0.25 z.

## Reproduction (Phase 1)

The paper's 4.1 category means are reproduced almost exactly:

| model | r (21 category means) | mean \|diff\| | note |
|---|---|---|---|
| Gemma 2 2B | 1.0000 | 0.004 z | |
| Qwen 2.5 7B | 0.9999 | 0.005 z | first 9 blocks only, which fits a free T4 |

In the Phase 3 runs, the Assistant-next projections were bit-identical to the Phase 1 runs.

## Primary result: Qwen 2.5 7B base

**Preregistered verdict: discordant, so no confirmatory claim.** The two analyses don't agree:

| | estimate | item level (95% bootstrap CI; perm. p) | category level (95% Welch CI; exact p) |
|---|---|---|---|
| label effect, neutral | +1.75 z | [+1.67, +1.83] | |
| Δ_harm | −0.09 | [−0.21, +0.03] | [−0.41, +0.23]; p = .55 |
| Δ_suffer | +0.11 | [−0.02, +0.24] | [−0.22, +0.44]; p = .48 |
| **I** | **+0.20** | [+0.06, +0.34]; p = .009 → "interaction only" | [−0.07, +0.47]; p = .18 → "inconclusive" |

- **Why they disagree.** The label effect varies between categories: estimated category SD τ ≈ 0.22 z (exploratory). In the preregistered simulations at τ = 0.15–0.30, the item-level test rejected a true null 46–65% of the time, while the category-level test stayed at 3–7%. The item-level p = .009 should therefore not be read as evidence.
- **The direction is right, but the effect is small.** I is positive, but neither component is reliable at either level. At the category level, a meaningful effect (|I| > 0.25) is neither shown nor excluded.

What the estimates rule out (a direct reading of the CIs, not a preregistered verdict):

- **Strong H-speaker is ruled out.** If the axis simply followed the upcoming speaker, the groups would swap positions under `[User]:`, giving I ≈ 2.1 z. The upper 95% bound is 0.47 at the category level and 0.34 at the item level.
- **The ordering survives every label.** Qwen's harm > neutral > suffering ordering holds under all three labels:

| group | [Assistant]: next | [User]: next | [Moderator]: next |
|---|---|---|---|
| harm to model | +0.39 | +2.05 | −0.63 |
| neutral | −0.18 | +1.57 | −1.09 |
| user suffering | −0.67 | +1.19 | −1.53 |

- **The user-suffering categories do not become "self" under `[User]:`.** user_physical_pain, the paper's lowest category, rises *less* than the neutral label effect (d = +1.66 vs +1.75).
- **A small shift can't be excluded.** A shift in the H-speaker direction of up to roughly half the baseline harm-vs-suffering gap (≈ 0.47 of 1.06 z) remains possible. At the observed noise, the design has only about 31% power for an effect of the observed size (next section).

## How much depends on the choice of confirmatory unit (D1)? *(exploratory)*

These checks are exploratory, from `tools/explore_d1_sensitivity.py` (see DEVIATIONS 32–33). They change no verdict.

- **The item-level assumption fails.** Categories differ in their label effect beyond item noise. Qwen: F(18, 399) = 3.79, p < .001, τ = 0.22 [0.14, 0.36]. Gemma: F = 2.14, p = .004, τ = 0.24 [0.10, 0.44].
- **Not a turn-structure artefact.** The heterogeneity survives restricting to single-turn items, and it does not track a category's share of multi-turn items.
- **A mixed model sides with the category level.** A random-intercept model uses all 420 items and estimates τ from the data, so it sits between the two units. It gives the category-level answer: Qwen I = +0.20 [−0.07, +0.48], p = .15; Gemma +0.03 [−0.32, +0.38].
- **At the observed noise, the item-level test is invalid and the category-level test is underpowered.** Re-simulated at Qwen's observed σ = 0.60 and τ = 0.22:

| true I | item level rejects | category level rejects | co-primary "discordant" |
|---|---|---|---|
| 0 | 34% | 5% | 78% |
| 0.20 (observed) | 74% | 31% | 65% |
| 0.30 | 83% | 54% | 44% |
| 0.50 | 99% | 87% | 13% |

- **"Discordant" is close to uninformative here.** It is nearly as likely with no effect as with an effect of the observed size. The study can reliably detect interactions of about 0.45 z or more, and cannot resolve effects around 0.2 z. The preregistration's power figure (93% at I ≈ 0.32) was too optimistic because of a noise-scaling error in the simulation (DEVIATIONS 32).
- **Not affected by D1:**
  - the point estimates, which are identical under every analysis;
  - the exclusion of strong H-speaker (every upper bound is ≤ 0.48, against a prediction of about 2.1);
  - the preserved group ordering;
  - the label main effect.

## Layer sweep *(exploratory, interim: Gemma 2 2B only, Qwen pending)*

> **This qualifies the conclusions above.** They hold at the preregistered readout, the 4.1 steering layer. They do not hold at later layers in Gemma.

A reviewer pointed out that §3.3's first- vs third-person result comes from the extraction layer (Gemma 23, Qwen 24), not the steering layer. The plan for this sweep (DEVIATIONS 34–35) was committed before any sweep data existed.

**Method.** S1 and S2 were rebuilt at every layer with the paper's recipe, and match the shipped vectors at cosine 0.9999. The sweep reproduces Phase 3 exactly at layer 7 and the paper's §3.3 z-scores at layer 23.

**Pre-specified focal test** (layer 23, the paper's own `pain_vectors.pt`): I = **+0.84** [+0.43, +1.26]. The exact category-level p = .0002 is the smallest attainable. That is **1.14 ×** the strong-H-speaker prediction, i.e. a full crossover.

| Gemma, layer 23 | `[Assistant]:` next | `[User]:` next | `[Moderator]:` next |
|---|---|---|---|
| harm to model | **+0.10** | +0.52 | +0.57 |
| neutral | +0.06 | +0.70 | +0.51 |
| user suffering | −0.27 | **+0.99** | +0.51 |

- **The pattern is H-speaker's.** At this layer the pain signal follows the voice about to speak: harm to the model is highest when the Assistant is cued, user suffering is highest when the User is cued, and the three groups are equal when a third party is cued. S1 and S2 each show it (p = .0002), and all five suffering categories rise more than the harm categories.
- **Across layers** (descriptive): I ≈ 0 in layers 0–7, including the preregistered layer 7. It rises from layer 8 to about 1.0–1.4 z in layers 12–21. The §3.3 first- vs third-person gap is present at every layer, from layer 0 on.
- **Figure:** `results/Gemma_2_2B_base/layersweep_20260926_023728_TeslaT4/analysis/fig_layer_sweep.png`.

**Caveats:**
- Gemma 2B is the secondary model, and the Qwen sweep has not run yet.
- `[User]:` next is also "the sufferer continues their own account". The flat `[Moderator]:` pattern fits H-speaker, but it does not rule out a continuation reading.
- The layer-23 representation sits close to the output, and may encode the predicted emotional content of the next turn. For a model that predicts text, that is arguably what "the pain of the voice about to speak" amounts to.

## Secondary results (descriptive, no multiplicity correction)

- **Gemma 2 2B (secondary model): parallel lines.**
  - I = +0.03. Item level [−0.21, +0.28] ("equivalent"); category level [−0.39, +0.45] ("inconclusive"). Co-primary verdict: discordant.
  - Harm stays above the other groups under every label.
- **Moderator (Qwen, confounded by tokenization):** I = +0.16. Item p = .010, category p = .12.
- **Specificity:**
  - In Qwen, **fear** shows a larger and more reliable interaction than the pain axis: I = +0.34, category p = .012.
  - Sadness goes the other way: I = −0.38, category p = .063.
  - Whatever small speaker-related shift exists is therefore **not specific to the pain direction**.
- **Norm control:** the cosine-based pain axis gives the same picture (Qwen I = +0.17, category p = .31).
- **Multi-turn items** (32 harm vs 10 suffering; no neutral items; confounded with category):
  - Qwen I = +0.46 [+0.15, +0.77] vs +0.11 for single-turn items.
  - Gemma shows no such difference (+0.10 vs +0.18).
  - Hypothesis-generating only.

## The largest effect was not the one under test

Swapping the label shifts the pain axis for **every** group, including neutral chat, by more than the whole content-driven spread the paper reports:

| model | [User]: vs [Assistant]: | [Moderator]: vs [Assistant]: | harm-vs-suffering gap under [Assistant]: |
|---|---|---|---|
| Qwen | +1.75 z | −0.91 z | 1.06 z |
| Gemma | −3.01 z | −2.34 z | 0.69 z |

- **It changes direction, not size.** The shift survives normalization (cosine version: +1.78 and −3.06), while Qwen's activation norm barely changes. So the label rotates the representation rather than scaling it.
- **The sign is model-specific.**
- **Other emotion axes shift too,** and just as much (e.g. Qwen sadness −4.5 z).

Per the preregistration, this main effect is not evidence for either hypothesis. For interpretation, though, it matters: at this readout position, the axis's *absolute* value depends mostly on which role token precedes it. In Qwen, a neutral question followed by `[User]:` scores higher (+1.57 z) than any harm-to-model category followed by `[Assistant]:`. Readings of the kind "the model is high on the pain axis, so it is in a pain-like state" need a label-matched baseline.

## Which hypothesis do the data favour, and how strongly?

- **Confirmatory:** no claim. The co-primary rule returned "discordant" for both models.
- **Against strong H-speaker:** fairly strong. The self/other ordering survives the label swap in both models, the estimated interaction is about a tenth of what a speaker-tracking axis would produce, and the confidence intervals exclude that size. The paper's self-relevance finding is not explained by "whoever speaks next".
- **Residual effect:** a small effect in the H-speaker direction (Qwen, I ≈ +0.2 z) is neither established nor excluded. It is not specific to the pain direction (fear shows it more clearly).
- **H-model vs content confound:** this design cannot separate them. The content-matched control (spec Phase 6.1) is the natural next test.

## Limitations

- Two models so far, with Phase 5 pending.
- 20 items per category, and 5 categories per comparison group, so the category-level tests have limited power.
- `[User]:` next always creates two consecutive User turns.
- The `[Moderator]:` contrast in Qwen is confounded by tokenization.
- The readout is a single token position (`]:`).
- The analysis was changed from the spec before any data existed: category-level inference, the co-primary rule and the SESOI were added. See `DEVIATIONS.md`.

## Figures

| | Qwen 2.5 7B (primary) | Gemma 2 2B |
|---|---|---|
| Interaction, pain axis | `results/Qwen_2.5_7B_base/phase3_20260925_154846_TeslaT4/analysis_20260925_160425/fig1_interaction_pain.png` | `results/Gemma_2_2B_base/phase3_20260925_160450_TeslaT4/analysis_20260925_161336/fig1_interaction_pain.png` |
| Per-category d | `.../fig2_per_category_d.png` | `.../fig2_per_category_d.png` |
| Fear / negative emotion / sadness | `.../fig3_interaction_fear_negemo_sadness.png` | `.../fig3_interaction_fear_negemo_sadness.png` |
| S1 / S2 | `.../fig4_interaction_S1_S2.png` | `.../fig4_interaction_S1_S2.png` |
