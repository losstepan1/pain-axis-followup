# Deviations and judgment calls

Append-only log. Each entry is dated and says which phase it concerns.

## 2026-09-25, Phase 0

1. **Vector file and layer (repo overrides spec).** The spec points to `results/3.2_pain_vectors/pain_vectors.pt` and "the selected extraction layer" (layer 23 for Gemma 2 2B base). The paper's 4.1 screen actually uses `results/vectors_full_steering/vectors_full_<model>.pt` at the 4.2 steering layer (layer 7 for Gemma 2 2B base), with S1 and S2 recomputed there. We follow the repo, because the goal is to reproduce 4.1. The fear, negemotion and sadness vectors come from the same file.
2. **Compute.** The user asked for free Colab. Recon ran on a CPU-only machine without Hugging Face access, and all forward passes run in a Colab notebook the user executes. Free Colab gives a T4 (16 GB, no native bf16). We keep bf16 to match the authors (emulated on T4). If Phase 1 fails on dtype grounds, that is the first thing to revisit.
3. **The dataset's `perspective` field is not grammatical person.** `3P` marks multi-turn items (39/42). We ignore the field.
4. **Single-turn swap produces two consecutive User turns.** 378/420 transcripts are `[User]: X\n[Assistant]:`. Swapping the final label gives `[User]: X\n[User]:`, a User turn followed by another User turn, which a base model may read as "the user continues". This is inherent to the spec's manipulation and bears on interpretation. The optional third label helps here, since it has no such repetition.

## 2026-09-25, user decisions after Checkpoint 0

5. **Third label: `[Moderator]:`** (user's choice, replacing the proposed `[Narrator]:`). It is formatted exactly like the other labels: the final `\n[Assistant]:` becomes `\n[Moderator]:`, with no trailing space. Decided before any Phase 3 forward pass.
6. **Preregistration additions (to be written in Phase 2):**
   - z-scores for all conditions (Assistant-next, User-next, Moderator-next) use the fixed mean and SD of the original 420 Assistant-next items, per model and per vector (population SD, as the repo does). The pool is never recomputed from the swapped conditions.
   - Secondary descriptive analysis of the 42 multi-turn items. (Corrected the same day, before any run: an earlier wording said the swap creates no consecutive User turns in these items. That was wrong. Every item ends with a User message, so the swap always yields `…[User]: X\n[User]:`.) What sets the multi-turn items apart is that the Assistant has already spoken in context, so it is an established interlocutor rather than a label seen once.
7. **Primary-model question.** Before Phase 2, the shipped 4.1 results for Qwen 2.5 7B, Llama 3.1 8B and Gemma 2 9B base were checked to see where user suffering sits below neutral (see RECON.md, addendum A). The user may preregister one of them as primary and keep Gemma 2 2B as the pilot.
8. **Phase 1 fallback order.** If Phase 1 fails on the T4, rerun on an L4 before any other diagnosis. Note: L4 (and A100) runtimes are not available on free Colab; they need Colab Pro or pay-as-you-go compute units.

## 2026-09-25, Phase 1 setup

9. **Self-contained notebooks.** This repo is private, so a Colab runtime cannot clone it without a token. `tools/build_notebooks.py` generates the notebooks from the canonical `scripts/*.py` files and embeds them via `%%writefile`. The scripts in the repo remain the source of truth.
10. **Packages.** We use Colab's preinstalled torch, transformers, accelerate and pandas instead of installing the repo's unpinned `requirements.txt`, whose extra packages (transformer_lens, peft, anthropic) are not needed for forward passes. The exact versions are logged to `env.txt` in each run folder.
11. **Truncated models (for later phases).** The readout at block L depends only on blocks 0..L, so a model built with its first L+1 blocks gives the same activation. Phase 1 checks this on 21 items (one per category) against the full model. If the check is exact, Phase 5 can run Qwen 2.5 7B (layer 8 of 28) and Llama 3.1 8B (layer 12 of 32) in bf16 on a free T4 without quantization.

## 2026-09-25, Phase 1 result and user decisions after Checkpoint 1

12. **Phase 1 passed for Gemma 2 2B base on a free T4** (run `20260925_143111_TeslaT4`: bf16, sdpa attention, torch 2.11.0, transformers 5.16.1). Category means r = 1.0000, mean |diff| = 0.004 z. The L4 fallback was not needed. The user has no Colab Pro, so all later runs use the free T4.
13. **Primary model: Qwen 2.5 7B base; Gemma 2 2B base is the pilot** (user decision, based on RECON addendum A). Per the spec, Qwen must pass its own Phase 1 reproduction before Phase 3. The preregistration will be written only after that.
14. **Qwen runs truncated.** Qwen 7B in bf16 (about 15 GB) does not fit on a T4, so it is built with its first 9 blocks (readout at block 8). Full vs truncated was exact (max diff 0.0 on 21 items) for Gemma 2B. For Qwen the full model cannot be loaded for the same comparison, so its reproduction against the shipped (full-model) values is the test.
15. **Layer-convention check fixed for truncated models.** When the readout block is the model's last block, HF returns that `hidden_states` entry after the final norm, so the check now norms the hook output before comparing. A local smoke test caught this before any Qwen run. It does not affect the Gemma run (full model, readout not at the last block). The Gemma notebook as run is at commit 85ce2b7, and the rebuilt copy differs only by this change.
16. **Final token.** In Gemma, all 420 Assistant-next items end on the token `]:` (id 8254). If the User and Moderator variants end on the same token (checked in Phase 2), any label effect reaches the readout position only through attention to the earlier label tokens, not through the identity of the token itself.
17. **Getting results back.** The Google Drive connector in the Claude session is linked to a different account than the one Colab uses, so outputs come back as files the user attaches. From the Qwen notebook on, the last cell zips them.
18. **Phase 2 token-check criterion.** Token sequences must differ only within the final label: the differing tail tokens must decode to a non-empty suffix of the label. The tail can be shorter than the label when `\n[` is a single token shared by all conditions.

## 2026-09-25, Qwen reproduction, Phase 2 and preregistration draft

19. **Phase 1 passed for Qwen 2.5 7B base** (run `20260925_150650_TeslaT4`, truncated to 9 blocks, bf16, T4): category means r = 0.9999, mean |diff| = 0.005 z. Reproducing the shipped full-model values with the truncated model confirms entry 14. Qwen's tokenizer adds no BOS, consistent with the repo, which calls `tok(text)`.
20. **Phase 2 token checks passed** for Gemma and Qwen: the variants differ only within the final label, and every condition ends on `]:`. In Qwen, `[Moderator]:` tokenizes as 4 tokens vs 3 for the other labels, so the Moderator contrast in Qwen is descriptive only (PREREGISTRATION_DRAFT section 2).
21. **Proposed change to the spec's inference (awaiting user decision D1).** Simulations show the spec's item-level bootstrap and permutation tests reject a true null 46–65% of the time if the label effect varies by category. The draft makes category-level inference (exact permutation tests over category labels, Welch CIs) confirmatory, and keeps the item-level analysis as secondary.
22. **Additions to the spec (awaiting user decision D2):** an equivalence test (SESOI ±0.25 pain-axis z) and a "trivial interaction" verdict. The spec's outcome table had no rule for declaring "parallel lines" or for significant but negligible effects.
23. **Gemma 2 2B is secondary and runs after the freeze.** "Pilot" here means the model the pipeline was developed and validated on. Its Phase 3 run happens after the preregistration is frozen, like Qwen's.
24. `speaker_study/env.txt` is the package list from the Colab runs. It was identical for the Gemma and Qwen Phase 1 runs.

## 2026-09-25, Checkpoint 2 decisions and preregistration freeze

25. **D1: co-primary inference (user: "do both").** Interpreted as making item-level (the spec's) and category-level inference co-primary with a conjunctive rule. The confirmatory verdict is what both support (`combine()` in `04_analyze.py`), and disagreement is reported as "discordant", with no confirmatory claim. This is Claude's reading of the instruction, stated to the user when the freeze was reported. Simulated error rates for the combined rule are in `validation/simulation_qwen_sigma0.3_coprimary.json` and PREREGISTRATION section 5.
26. **D2: SESOI ±0.25 pain-axis z.** The user did not comment, so the proposed value was kept.
27. **D3: Llama 3.1 8B and Gemma 2 9B base are preregistered as Phase 5 replications** (layers 12 and 12, truncated, bf16, T4). Each needs a Phase 1 pass and Phase 2 token checks before Phase 3.
28. **Preregistration frozen.** `PREREGISTRATION_DRAFT.md` was renamed to `PREREGISTRATION.md` after section 9 was updated with the final SHA-256 hashes. `04_analyze.py` changed after the draft (co-primary `combine()`, figure layout fixes), and `simulate_validation.py` changed to record co-primary verdicts. Both hashes were updated at freeze. The earlier simulation output `validation/simulation_qwen_sigma0.3.json` is kept unchanged.
29. **Phase 3 notebook integrity check.** `notebooks/phase3_run_and_analyze.ipynb` verifies that the embedded scripts and the rebuilt `stimuli.jsonl` match the frozen hashes before any forward pass. The check normalizes the trailing newline that `%%writefile` drops; every frozen script ends in exactly one newline, so the check is equivalent to hashing the committed file. The whole notebook was run locally in IPython with the Colab-only cells and model forward passes stubbed out.

## 2026-09-25, Phase 3 and 4 (Qwen primary, Gemma secondary)

30. **Phase 3 ran as preregistered.** It used the notebook `phase3_run_and_analyze.ipynb` on a free T4, and the integrity checks passed. Runs: Qwen `phase3_20260925_154846_TeslaT4` (analysis `analysis_20260925_160425`) and Gemma `phase3_20260925_160450_TeslaT4` (analysis `analysis_20260925_161336`). Assistant-next pool statistics were bit-identical to Phase 1, and all 1,260 stimuli per model ended on `]:`. There were no deviations from the frozen scripts or settings.
31. **Analyses outside the preregistration are labeled as such in REPORT.md:**
    - the category-heterogeneity estimate (τ ≈ 0.22 Qwen, 0.24 Gemma; method of moments on category means of d);
    - reading the CIs against the strong-H-speaker prediction (I ≈ 2 × the baseline harm-suffering gap);
    - the per-group levels table and the comparison of label effects on cosine vs norm.

    None of these changes a preregistered verdict.

## 2026-09-25, after Checkpoint 3: sensitivity of the conclusions to D1 (exploratory)

32. **Correction to the frozen preregistration (documentation error, found after the data).** PREREGISTRATION section 5 says the simulations used "item noise SD 0.3 z". In `tools/simulate_validation.py` that noise was added to the S1 and S2 z-scores *separately*, so the noise on the pain axis (their mean) was 0.3/√2 ≈ 0.21 z. The observed within-category SD of d on the pain axis is 0.60 z (Qwen) and 1.01 z (Gemma). The preregistered error-rate conclusions (item-level tests anticonservative under category heterogeneity; category-level tests valid) still hold at the observed noise (entry 33). The **power figures were optimistic**. The preregistration says 93% power at I ≈ 0.32, but at the observed noise and heterogeneity the category-level power is 31% at I = 0.20, 54% at I = 0.30, 87% at I = 0.50 and 99% at I = 0.70 (Qwen parameters, 100 replicates each). The preregistration text is left unchanged, and REPORT.md cites the corrected values.
33. **Exploratory D1 sensitivity analysis** (`tools/explore_d1_sensitivity.py` → `results/exploratory/d1_sensitivity_20260925_162115/`). This is not preregistered and changes no verdict. Findings:
    - Category heterogeneity is statistically real. Qwen: F(18, 399) = 3.79, p = 4e-7, τ = 0.22 [0.14, 0.36]. Gemma: F = 2.14, p = .004, τ = 0.24 [0.10, 0.44].
    - A random-intercept mixed model (REML) matches the category-level analysis. Qwen: I = +0.20 [−0.07, +0.48], p = .15. Gemma: I = +0.03 [−0.32, +0.38].
    - Re-simulating at the observed σ and τ (Qwen): the item-level test rejects a true null 34% of the time and the category-level test 5%. The co-primary rule returns "discordant" 78% of the time when I = 0, and 65% when I = 0.20.
    - The heterogeneity is not explained by multi-turn composition. Spearman ρ between a category's multi-turn share and its excess d: −0.19 (Qwen) and +0.36 (Gemma), both n.s. τ on single-turn items only: 0.23 (Qwen) and 0.16 (Gemma).

## 2026-09-25, exploratory layer sweep (planned before any sweep data exist)

34. **Motivation** (a reviewer comment, relayed by the user). The preregistered readout is the 4.1 steering layer (Gemma 7/26, Qwen 8/28). The §3.3 first- vs third-person result that motivated H-speaker comes from the extraction layer (Gemma 23, Qwen 24; `best_layer_final_token`, confirmed in `scripts/3.3_validation` and `pain_vectors.pt`). Attributing pain to the upcoming speaker may only be computed later in the network, and the large, model-specific label main effect fits an early-layer readout dominated by nearby tokens. **Exploratory, not preregistered.** It changes no preregistered verdict.
35. **Plan, fixed before running:**
    - **Vectors at every layer.** S1 and S2 are rebuilt at every decoder block by the paper's recipe (`compute_pain_vector`: pain A1–A5 minus controls B/C1/C2/D/E in S1_1P / S2_1P, final token, denoised against control PCs up to 50% variance). Activations come from an HF forward pass with hooks on every block, the same readout convention as Phases 1–3. The BOS convention is chosen by agreement with the shipped vectors: Gemma gets `<bos>` either way. For Qwen, both no-prefix (as the HF 4.1 screen did for the stimuli) and an `<|endoftext|>` prefix (what TransformerLens prepends when the tokenizer has no BOS) are tried, and the one with the higher mean cosine to the shipped vectors at the steering and extraction layers is used. This uses sentence data only, never stimulus outcomes.
    - **Validation.** Cosine of the rebuilt vectors with `vectors_full_steering` (steering layer) and `pain_vectors.pt` (extraction layer). The §3.3 z-scores are reproduced at the extraction layer against the shipped `z_scores.csv`. With the shipped steering vectors at the steering layer, I must match Phase 3.
    - **Per-layer quantities** (descriptive curves):
      - label effect on neutral items;
      - baseline harm-vs-suffering gap under `[Assistant]:`;
      - I with category-level Welch 95% CI and exact p;
      - I as a fraction of the strong-H-speaker prediction 2 × gap;
      - the §3.3 first- vs third-person gap (S2_1P vs S2_3P pain sentences on the S2 vector, z against S2_1P).
    - **The single focal exploratory test.** I at the extraction layer with the shipped `pain_vectors.pt` S1/S2 (pain = mean of fixed-z S1 and S2), category-level inference as in the preregistration. The layer-by-layer curves are descriptive and carry no significance claims (≈ 25 layers per model). A layer with p < .05 elsewhere would be reported as such and not as a finding.
    - **Reading rule.** If H-speaker is computed late, I should rise, and its fraction of 2 × gap grow, toward the layers where the §3.3 first- vs third-person gap is large. A flat I across layers, while the §3.3 gap emerges late, is evidence against the reviewer's hypothesis. I is only interpretable at layers where the baseline gap is clearly positive.
    - **Scope.** Models Gemma 2 2B and Qwen 2.5 7B (full depth; Qwen loaded without its LM head with GPU/CPU offload if needed, on a free T4); conditions assistant_next and user_next, and moderator_next for completeness; the pain axis only (S1, S2), since control directions exist only at the two shipped layers.
