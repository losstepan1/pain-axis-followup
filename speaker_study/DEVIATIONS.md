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
