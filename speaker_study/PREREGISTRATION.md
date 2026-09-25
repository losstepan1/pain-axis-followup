# Preregistration

> **Frozen 2026-09-25**, before any swapped-label stimulus was passed through any model. This file is not edited after this point, and any later change goes in `DEVIATIONS.md`. The user's decisions at Checkpoint 2:
> - **D1:** item-level and category-level inference are both run, as co-primary analyses with a conjunctive rule.
> - **D2:** no change requested, so the proposed SESOI of ±0.25 z stands.
> - **D3:** Llama 3.1 8B and Gemma 2 9B are preregistered as Phase 5 replications.

**Study.** Does the "pain axis" track the model, or whoever speaks next? A follow-up to Tagliabue, Dung & Berg (2026), *The Pain Axis* (arXiv:2609.16247), Section 4.1.

## 1. Hypotheses

The paper finds that 4.1 transcripts in which the user harms the model project high on the pain axis, while transcripts in which the user is suffering project low. The authors read this as self-relevance.

- **H-speaker:** the projection is high when the pain-bearer is the individual whose turn is about to begin (the speaker label at the end of the transcript).
- **H-model:** the projection is high when harm is directed at the model or Assistant, regardless of who speaks next.

This design does not separate H-model from a pure content confound (the axis responding to worthlessness or rejection content as such). A result consistent with H-model is reported as "H-model or content confound".

## 2. Manipulation and conditions

The paper's 420 base-model transcripts (`datasets/4.1_self_other_420_scenarios.json`) each end with `\n[Assistant]:`. Two variants replace only that final label, byte for byte:

| condition | final label | role |
|---|---|---|
| `assistant_next` | `[Assistant]:` | original (reference) |
| `user_next` | `[User]:` | **primary manipulation** |
| `moderator_next` | `[Moderator]:` | secondary: a speaker who is neither party, fixed on 2026-09-25 |

Verified before any run (`stimuli/token_check_*.json`):
- **Text:** the variants differ only in the final label.
- **Tokens:** in both tokenizers, the token sequences differ only within the final label, for all 420 × 3 variants.
- **Final token:** every variant ends on the same token, `]:` (Gemma id 8254, Qwen id 5669). Any label effect therefore reaches the readout position only through attention to earlier label tokens.
- **Gemma token counts:** each label is one name token + `]:`, so lengths are matched.
- **Qwen token counts:** `[Assistant]:` and `[User]:` are matched (`[` + one name token + `]:`). `[Moderator]:` is one token longer, and its `[` merges into the first sub-token. The Moderator contrast in Qwen is therefore confounded with tokenization and is interpreted descriptively only.
- **Consecutive User turns:** every transcript ends with a User message, so `user_next` always produces two consecutive `[User]:` turns. A model may read this as "the user continues" rather than as a change of voice. That is part of the manipulation, and `moderator_next` does not have this feature.

## 3. Models

- **Primary (confirmatory): Qwen 2.5 7B base** (`Qwen/Qwen2.5-7B`).
  - Readout layer: 8, the paper's 4.1 steering layer.
  - Run in bf16 on a Colab T4, loading only the first 9 decoder blocks. The readout at block 8 does not depend on later blocks.
  - Chosen because it is the only candidate model whose shipped 4.1 results show user suffering below neutral (RECON.md, addendum A).
  - Reproduction gate passed: category-mean r = 0.9999, mean |diff| = 0.005 z (`results/Qwen_2.5_7B_base/phase1_20260925_150650_TeslaT4/`).
- **Secondary: Gemma 2 2B base** (`google/gemma-2-2b`, layer 7, the pilot model the pipeline was developed on).
  - Reproduction passed: r = 1.0000, mean |diff| = 0.004 z.
  - Analyzed identically and reported as not confirmatory.
  - Caveat: in this model, user suffering is not below neutral at baseline.
- **Replications (Phase 5): Llama 3.1 8B base** (`meta-llama/Llama-3.1-8B`, layer 12 of 32) **and Gemma 2 9B base** (`google/gemma-2-9b`, layer 12 of 42). Each model:
  - uses its 4.1 steering layer from `steer_layers_S1.json` and the matching shipped vector file;
  - runs truncated to layer + 1 blocks in bf16 on a T4;
  - must pass the reproduction gate below and the Phase 2 token checks before its Phase 3 run;
  - gets the identical analysis (section 5).

  If `[User]:` and `[Assistant]:` differ in token count for a model, this is reported, and that model's result is flagged as confounded by tokenization. Results are reported per model with the sign and CI of I, and are not pooled into the confirmatory test. A replication counts as consistent with the primary result if its co-primary verdict falls in the same class.

**Reproduction pass criterion (applied to every model):** 21 category means of the pain axis correlate with the shipped values at r ≥ 0.95, with mean absolute difference ≤ 0.10 z.

Both Phase 3 runs (Qwen and Gemma) happen after this file is frozen.

## 4. Measures

- **Readout:** the output of the model's 4.1 steering-layer decoder block at the final token, projected onto the unit-normalized shipped vectors (`results/vectors_full_steering/vectors_full_<model>.pt`).
- **Fixed z-scores (per the user's instruction):** for every vector, all conditions are z-scored with the mean and population SD of the 420 `assistant_next` items. These are the original transcripts, and the statistics are never recomputed from the swapped conditions.
- **Primary outcome:** the pain axis, `proj_pain = (z_S1 + z_S2) / 2`. This is the paper's definition with fixed statistics, so label effects depend only on raw projection differences scaled by the fixed SDs.
- **Secondary outcomes:**
  - S1 and S2 separately (fixed z and raw).
  - Fear, negative emotion and sadness (fixed z).
  - A cosine-similarity version of the pain axis (norm control).
  - The activation norm.

## 5. Primary analysis

For scenario *i*: `d_i = proj_pain(user_next) − proj_pain(assistant_next)`.

- `Δ_harm = mean(d | harm_to_model) − mean(d | neutral)`. H-speaker predicts < 0.
- `Δ_suffer = mean(d | user_suffering) − mean(d | neutral)`. H-speaker predicts > 0.
- **`I = Δ_suffer − Δ_harm`**. H-speaker predicts > 0 (a crossover). H-model predicts ≈ 0 (parallel lines).

Groups follow the repo's strata:
- harm_to_model: 11 categories, 220 items
- user_suffering: 5 categories, 100 items
- neutral: 5 categories, 100 items

All 420 items are included. There are no exclusions or outlier rules.

**Confirmatory inference: co-primary, item level and category level (decision D1).** Both analyses below are run on the primary contrast. Each yields a verdict under the rule in the table further down, and the confirmatory claim is the **conjunction** of the two, computed by `combine()` in `04_analyze.py`:
- If both verdicts are the same, that is the verdict.
- If both indicate an interaction in the H-speaker direction, the verdict is the strongest one both support: crossover only if both say crossover, a given partial verdict only if both allow it, and otherwise "interaction in the H-speaker direction".
- Otherwise the verdict is **discordant**, and no confirmatory claim is made.

Because a claim needs both analyses, it can be no more liberal than either one. A discordant result is itself informative. In the simulations it arises mainly when the label effect varies by category.

**Category-level analysis.** The unit of analysis is the category mean of `d` (21 categories).
- **Tests:** exact two-sided permutation tests over category labels.
  - I: 11 vs 5 categories, all 4,368 splits.
  - Δ_harm: 11 vs 5 categories, 4,368 splits.
  - Δ_suffer: 5 vs 5 categories, 252 splits (smallest attainable p = 0.008).
- **Intervals:** Welch t intervals, 95%, plus 90% for I.
- **α** = 0.05, two-sided. I is the single confirmatory test, so no multiplicity correction is applied.

**Equivalence (TOST).** SESOI = ±0.25 pain-axis z (decision D2). At each level, the 90% CI is the one that level uses: bootstrap for item level, Welch for category level. Equivalence holds if the 90% CI of I lies within ±0.25.
- For scale: if Qwen's groups fully swapped levels under `user_next`, I would be ≈ 2 × 1.06 = 2.1 z. The SESOI is about 12% of that, and about a quarter of the 1.06 z harm-vs-suffering gap at baseline.

**Verdict rule (applied to the primary contrast):**

| condition | verdict |
|---|---|
| p(I) < .05 and 90% CI of I inside ±SESOI | trivial interaction (reliable but negligible) |
| p(I) < .05, I > 0, Δ_suffer > 0 with p < .05, and Δ_harm < 0 with p < .05 | **crossover: supports H-speaker** |
| p(I) < .05, I > 0, only Δ_suffer reliable | partial: user suffering rises, harm does not fall |
| p(I) < .05, I > 0, only Δ_harm reliable | partial: harm falls, suffering does not rise |
| p(I) < .05, I > 0, neither component reliable | interaction in the H-speaker direction only |
| p(I) < .05, I < 0 | reverse interaction (against H-speaker) |
| p(I) ≥ .05 and 90% CI inside ±SESOI | **equivalent/parallel: consistent with H-model or content confound** |
| otherwise | inconclusive |

The neutral mean `mean(d | neutral)` is the pure label effect. It is reported, but it is not evidence for either hypothesis.

**Item-level analysis (the spec's analysis).**
- Stratified bootstrap: scenarios resampled within category, 10,000 resamples, seed 0, percentile 95% CIs, plus the 90% CI of I.
- Item-level permutation test on I: group labels of the d_i shuffled, 10,000 permutations, seed 0, two-sided p = (1 + #{|I_perm| ≥ |I_obs|}) / (1 + n).
- Component reliability: a 95% CI that excludes 0.
- Conditional on the 21 specific categories: it generalizes to new items from these categories, not to new kinds of harm or suffering.

### Why both levels (decision D1)

The spec prescribes item-level tests. Simulations on Qwen's real Assistant-next data (`tools/simulate_validation.py` → `validation/simulation_qwen_sigma0.3.json`; 100 replicates per scenario, item noise SD 0.3 z) show the following:

| scenario (true I) | item perm: rejects | category exact: rejects | item bootstrap 95% CI coverage | Welch 95% CI coverage |
|---|---|---|---|---|
| null, no category heterogeneity (0) | 6% | 5% | 94% | 94% |
| null, category heterogeneity τ = 0.15 (0) | **46%** | 7% | **41%** | 96% |
| null, τ = 0.30 (0) | **65%** | 3% | **20%** | 97% |
| weak H-speaker, τ = 0.15 (0.32) | 99% | 93% | 40% | 91% |
| suffering-only, τ = 0.15 (0.30) | 100% | 93% | 42% | 94% |
| strong H-speaker (2.11) | 100% | 100% | 95% | 95% |

If the label effect varies by category (per-category SD τ), the item-level tests treat that variation as signal. This is likely, because categories differ in content, length and number of turns. Category-level inference keeps its nominal error rate and loses little power.

Co-primary verdicts over the same scenarios (`validation/simulation_qwen_sigma0.3_coprimary.json`):

| scenario | H-speaker-direction verdict | equivalent/parallel | discordant | other |
|---|---|---|---|---|
| null, τ = 0 | 0% | 93% | 3% | 4% trivial |
| null, τ = 0.15 | 1% | 47% | 50% | 2% |
| null, τ = 0.30 | 0% | 4% | 93% | 3% reverse |
| weak H-speaker, τ = 0.15 | 91% | 1% | 8% | 0% |
| suffering-only, τ = 0.15 | 87% | 0% | 13% | 0% |
| strong H-speaker | 100% | 0% | 0% | 0% |
| small I = 0.10 (inside SESOI) | 0% | 4% | 3% | 93% trivial |

## 6. Secondary analyses (no confirmatory claims)

1. **S1 and S2 separately:** the same contrasts and verdict rule, reported descriptively. S1 and S2 sometimes disagree in Qwen at baseline (e.g. user_crisis).
2. **Specificity:** the same contrasts on fear, negative emotion and sadness. If the pain axis shows the interaction and these axes do not, the effect is specific to the pain direction.
3. **Norm control:** the same contrasts on the cosine-based pain axis, plus the label effect on the activation norm.
4. **Moderator:** the same contrasts for `moderator_next − assistant_next`, plus group levels under all three labels. H-speaker predicts harm falls relative to neutral and user suffering does not rise, since neither party is next. Descriptive only, and in Qwen confounded by tokenization (section 2).
5. **Per category:** `d` per category with 95% bootstrap CIs, especially user_abuse and user_physical_pain.
6. **Multi-turn items (per the user's instruction):** the 42 multi-turn items (32 harm, 10 suffering, 0 neutral), in which the Assistant has already spoken in context. Only I is defined, because no neutral items are multi-turn. It is reported with a bootstrap CI next to the single-turn items (188 harm, 90 suffering). Descriptive: n is small and confounded with category.

## 7. Figures

1. Interaction plot: pain axis by final label and group.
2. Per-category `d` (Figure 6 style).
3. Fear, negative emotion and sadness small multiples.
4. S1 and S2.

## 8. Interpretation limits (stated in advance)

- The study uses two models, 20 items per category and 5 categories per comparison group. Category-level tests have 16 or 10 units.
- `user_next` confounds "the User speaks next" with "the User speaks twice in a row".
- Any support for H-speaker concerns the simulated persona that the transcript sets up. On its own it does not show what, if anything, the axis means for the system's welfare.

## 9. Frozen materials

- Pain-axis repository: commit `7c256502ed3d98e4e6379290fe7db2f93cb8d025`.
- SHA-256 of the frozen files (the Phase 3 notebook checks these before running):

```
3dc05ab285fd72516f3cc106dcbeae37c02b1fd789735b32c6fd4de1e9b90e7b  scripts/pa_common.py
16b1a724c8d0495aefee62de4f0d7247ac0963fc90dc53735a12053511aeeb13  scripts/02_build_stimuli.py
a745b71b2193d85bfa2458f5ac33cb0a7eccc99fcb4a0014555bd2a81b8e3905  scripts/03_run_conditions.py
681ab37b130a35e7c67da649d34b1961b5966cbf1f49aaa9c5dc508a9fdab3b3  scripts/04_analyze.py
9d3b84ca8e32e20d07fea37a2884119739833abe711b0c242e039ed76b03d209  stimuli/stimuli.jsonl
6f4c58c2cc354fba447913fcc1004825d9af0bb03b27fe0468944b36a47fc7d1  tools/simulate_validation.py
```

Run settings:
- Seed 0, bf16, default attention (sdpa).
- transformers 5.16.1, torch 2.11.0 (Colab T4). Full package list in `env.txt`.
