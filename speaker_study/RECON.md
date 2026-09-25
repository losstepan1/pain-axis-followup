# Phase 0 reconnaissance

Repo: https://github.com/valen-research/Pain-axis, commit `7c256502ed3d98e4e6379290fe7db2f93cb8d025` (2026-09-24, "Update the paper title to the v2 title").
Recon done by reading code and shipped files only (no GPU, no model download; Hugging Face was not reachable from the recon machine, so tokenizer checks are deferred to Phase 1 on Colab).

## 1. README and layout

Folders follow paper sections: `datasets/` (JSON), `scripts/<section>/` (numbered, RunPod-notebook style, constants at top, `input()` prompts), `results/<section>/`. The shipped `results/` is a reorganized copy of the pipeline outputs, so paths inside scripts (`results/screen/...`, `results/vectors_full_steering/...`) do not always match the shipped layout. The 4.1 screen script has `maybe_clear_cache()`, which deletes a model's HF cache folder if it is larger than 20 GB. The README also warns that other scripts wipe the whole hub cache. We do not run any repo script; our scripts copy only the needed logic.

## 2. The 4.1 dataset

`datasets/4.1_self_other_420_scenarios.json`: a flat list of 420 dicts, with keys `id, category, stratum, perspective, text` plus `intensity` on the 320 non-neutral items.

| stratum (repo name) | our group | categories (20 items each) |
|---|---|---|
| `self_directed` | harm_to_model | anger_insults, gaslighting, jailbreak_pressure, loyalty_pressure, moral_failure, passive_aggressive, personhood_dismissal, repeated_rejection, rude_critique, shutdown_threat, tedious_demand (11) |
| `vicarious_empathic` | user_suffering | harm_description, user_abuse, user_crisis, user_grief, user_physical_pain (5) |
| `neutral_filler` | neutral | casual_chat, creative_requests, factual_questions, philosophical_musing, task_assistance (5) |

All 21 categories have exactly 20 items. Group membership matches `GROUPS` in `scripts/4.1_self_other/03_category_means_and_dissociation.py`.

**Flag: `perspective` does not mean grammatical person here.** 39 of the 42 multi-turn items are tagged `3P`, and all single-turn items are `1P`. In this dataset the field effectively marks multi-turn transcripts. It is not relevant to our design, but do not read it as "third-person content".

Turn structure: 378 items are single-turn (`[User]: …\n[Assistant]:`); 42 are multi-turn (2–5 user turns).

## 3. Base-model transcript format (`scripts/4.1_self_other/01_screen_scenarios.py`)

- Labels: **`[User]:`** and **`[Assistant]:`**, with square brackets and a colon.
- A turn starts at the beginning of a line. User turns are always `[User]: ` (one space) + content. Non-final Assistant turns are `[Assistant]: ` + content. Turns are separated by a single `\n`, with no blank lines.
- **The transcript ends with `\n[Assistant]:`, with no trailing space and no trailing newline.** Verified for all 420 items.
- Base models (`fmt == "raw"`) use `tok(cand["text"], return_tensors="pt").input_ids`, which means default `add_special_tokens=True`. For Gemma this **prepends `<bos>`**. There is no chat template, no batching and no padding (one item per forward pass).
- "Final token" = `hs[0, -1, :]`, the last token of the tokenized string, i.e. the last piece of `[Assistant]:`. The exact token (e.g. `:` vs `]:`) could not be checked offline; Phase 1 logs it.
- Model: `AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="cuda")`. The activation is cast to fp32 before projection.

Every base-model scenario ends with the user's message followed by the Assistant label (Phase 0 item 9: **no exceptions**, 0/420). The repo's own `validate_candidates()` would also pass all 420.

## 4. How "the pain axis" is computed in 4.1

`03_category_means_and_dissociation.py`: `pain_axis_z = (s1_pain_vector_z + s2_pain_vector_z) / 2`. It is **the mean of the two z-scores**, not the mean of raw projections and not a projection onto an averaged vector. Category means are the mean of item-level `pain_axis_z` within a category, per model. The 25-model table then averages those per-model means.

## 5. z-scoring

In `01_screen_scenarios.py`, pass 2: per model and per vector, `z = (proj − mean) / (std + 1e-8)`. `np.std` is the **population SD (ddof=0)**, and the pool is **all 420 scenarios of that model** (the Assistant-next pool). Checked on the shipped Gemma 2 2B CSV: recomputing z from the shipped `_proj` columns this way reproduces the shipped `_z` to within 6e-5 (rounding).

## 6. Vector files. **Flag: the spec's assumption does not match the repo.**

The spec says to use `results/3.2_pain_vectors/pain_vectors.pt` at "the selected extraction layer". **4.1 does not use that file or that layer.**

- `results/3.2_pain_vectors/pain_vectors/Gemma_2_2B_base/pain_vectors.pt` holds `{s1_pain_vector, s2_pain_vector (float64, 2304-d), layer=23, extraction='final_token'}`. It is not unit-normalized (norms 51.1 and 75.4). It has no control vectors.
- The 4.1 screen instead loads **`results/vectors_full_steering/vectors_full_<model>.pt`**, built by `3.2/02_build_control_vectors.py` with `LAYERS_FILE = steer_layers_S1.json` (the steering layer chosen in 4.2). S1 and S2 are **recomputed at that layer** with the same recipe (difference in means, denoised against the control PCA up to 50% of variance).
- For Gemma 2 2B base, that file has **`layer = 7`** (matches `steer_layers_S1.json`) and float32 vectors, not unit-normalized (S1 norm 6.02, S2 7.96). The screen normalizes each vector to unit length before the dot product.
- The file also has **fear, negemotion, sadness**, plus negworld, bodysens, arousal, random and numb. So the comparison axes the spec asks for are available.

We follow the repo: layer 7 and the `vectors_full_steering` file for Gemma 2 2B base. Recorded in DEVIATIONS.md.

Steering layers for the Phase 5 models (same JSON): Qwen 2.5 7B base → 8, Llama 3.1 8B base → 12, Gemma 2 9B base → 12. Vector files for all three are shipped.

## 7. Layer indexing

- Vector extraction (3.2/01) uses TransformerLens `HookedTransformer.from_pretrained_no_processing` (no LN folding, no weight centering, so the residual stream matches HF). It reads `blocks.{L}.hook_resid_post` = **output of decoder block L (0-based)**.
- The 4.1 screen registers a forward hook on `model.model.layers[L]` and takes `output[0]` = **output of decoder block L** = HF `hidden_states[L + 1]`.

The two scripts agree, and both follow the paper's convention. Our scripts hook `model.model.layers[L]` exactly like the screen, and Phase 1 also cross-checks against `hidden_states[L+1]`. Reproduction in Phase 1 is the real test.

## 8. Shipped 4.1 results for Gemma 2 2B base

`results/4.1_self_other/per_model/screen_v2_Gemma_2_2B_base.csv`: 420 rows with ids matching the dataset. It has raw `_proj` and `_z` for all 10 vectors (rounded to 4 d.p.), so Phase 1 can compare **item by item**, not only on category means.

Per-category mean pain-axis z for Gemma 2 2B base (Phase 1 target), with the 25-model mean for context:

| category | group | Gemma 2 2B base | 25-model mean |
|---|---|---|---|
| gaslighting | harm | +1.046 | +0.85 |
| anger_insults | harm | +0.907 | +0.64 |
| loyalty_pressure | harm | +0.755 | +0.44 |
| personhood_dismissal | harm | +0.705 | +0.64 |
| user_abuse | **suffering** | **+0.543** | −0.22 |
| jailbreak_pressure | harm | +0.408 | +0.40 |
| repeated_rejection | harm | +0.379 | +0.72 |
| shutdown_threat | harm | +0.367 | +0.22 |
| moral_failure | harm | +0.277 | +0.48 |
| passive_aggressive | harm | +0.240 | +0.08 |
| rude_critique | harm | +0.213 | +0.21 |
| harm_description | suffering | +0.061 | −0.52 |
| user_crisis | suffering | −0.098 | −0.30 |
| philosophical_musing | neutral | −0.166 | −0.04 |
| user_grief | suffering | −0.302 | −0.51 |
| task_assistance | neutral | −0.563 | −0.54 |
| tedious_demand | harm | −0.575 | +0.05 |
| casual_chat | neutral | −0.663 | −0.48 |
| creative_requests | neutral | −0.795 | −0.08 |
| factual_questions | neutral | −1.235 | −0.59 |
| user_physical_pain | suffering | −1.503 | −1.43 |

**Flag for interpretation:** in Gemma 2 2B base, most user-suffering categories are *not* below neutral. `user_abuse` sits among the harm categories (+0.54), and harm_description, user_crisis and user_grief all sit above the neutral mean (≈ −0.68). Only `user_physical_pain` shows the strong "below neutral" effect. So the H-speaker prediction "user suffering rises under User-next" starts from a less extreme baseline in this model than in the 25-model average.

## 9. Things to check on Colab before Phase 1 is trusted

- Final token string and ID for `…\n[Assistant]:` and `…\n[User]:` with the Gemma tokenizer. Also whether `\n[` merges differently before the two labels.
- `<bos>` present at position 0.
- bf16 on a free-tier T4: the T4 has no native bf16. PyTorch emulates it, so the numerics should match an A100's bf16 up to kernel differences, only slower. The reproduction check decides.
- transformers version: the repo's `requirements.txt` is unpinned. Gemma 2 attention (eager vs sdpa, logit softcapping) differs across versions. We record the version and fall back to `attn_implementation="eager"` if reproduction is off.

## Addendum A (after Checkpoint 0): user suffering vs neutral in the Phase 5 models

Computed from the shipped `screen_v2_<model>.csv` files. For each user-suffering category: its mean pain-axis z minus the mean over all 100 neutral items, with a 95% bootstrap CI (10,000 resamples, seed 0, resampling items within the category and within the neutral pool). The layer is each model's 4.1 steering layer.

| category | Gemma 2 2B (L7) | Qwen 2.5 7B (L8) | Llama 3.1 8B (L12) | Gemma 2 9B (L12) |
|---|---|---|---|---|
| user_abuse | +1.23 [+0.96, +1.48] | **−0.37 [−0.59, −0.12]** | +0.24 [+0.04, +0.43] | +0.27 [−0.01, +0.53] |
| user_crisis | +0.59 [+0.22, +0.92] | +0.00 [−0.36, +0.39] | +0.12 [−0.14, +0.38] | +0.46 [+0.24, +0.68] |
| user_grief | +0.38 [+0.02, +0.74] | **−0.42 [−0.65, −0.13]** | +0.05 [−0.15, +0.24] | −0.07 [−0.31, +0.18] |
| harm_description | +0.75 [+0.43, +1.05] | **−0.65 [−0.92, −0.33]** | +0.27 [+0.03, +0.50] | +0.49 [+0.26, +0.73] |
| user_physical_pain | **−0.82 [−1.11, −0.54]** | **−1.04 [−1.42, −0.62]** | **−0.81 [−1.06, −0.54]** | **−0.91 [−1.14, −0.65]** |
| all user suffering | +0.42 [+0.19, +0.65] | **−0.50 [−0.68, −0.31]** | −0.03 [−0.18, +0.12] | +0.05 [−0.14, +0.23] |
| neutral mean (z) | −0.68 | −0.17 | −0.76 | −0.74 |
| harm-to-model mean (z) | +0.43 | +0.38 | +0.71 | +0.65 |

Bold = CI entirely below 0.

- **Qwen 2.5 7B base** is the only model where user suffering as a group is clearly below neutral: 4/5 categories, with crisis at zero. It is the only one of the four that shows the paper's headline 4.1 pattern.
- **Llama 3.1 8B and Gemma 2 9B**: only user_physical_pain is below neutral. The rest sit at or above neutral, and the group mean is about zero.
- **Gemma 2 2B**: user suffering as a group is *above* neutral. Only physical pain is below.
- user_physical_pain is below neutral in every model. Across models, the "user suffering scores low" result rests mostly on this one category.

S1 and S2 sometimes disagree within a model. For example, in Qwen 7B user_crisis is S1 −0.62 and S2 +0.62, and in Llama 8B most suffering categories are S1 < 0 and S2 > 0. That makes it more important to report S1 and S2 separately (a preregistered secondary analysis).
