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
