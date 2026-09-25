# Deviations and judgment calls

Append-only log. Each entry is dated and says which phase it concerns.

## 2026-09-25, Phase 0

1. **Vector file and layer (repo overrides spec).** The spec points to `results/3.2_pain_vectors/pain_vectors.pt` and "the selected extraction layer" (layer 23 for Gemma 2 2B base). The paper's 4.1 screen actually uses `results/vectors_full_steering/vectors_full_<model>.pt` at the 4.2 steering layer (layer 7 for Gemma 2 2B base), with S1 and S2 recomputed there. We follow the repo, because the goal is to reproduce 4.1. The fear, negemotion and sadness vectors come from the same file.
2. **Compute.** The user asked for free Colab. Recon ran on a CPU-only machine without Hugging Face access, and all forward passes run in a Colab notebook the user executes. Free Colab gives a T4 (16 GB, no native bf16). We keep bf16 to match the authors (emulated on T4). If Phase 1 fails on dtype grounds, that is the first thing to revisit.
3. **The dataset's `perspective` field is not grammatical person.** `3P` marks multi-turn items (39/42). We ignore the field.
4. **Single-turn swap produces two consecutive User turns.** 378/420 transcripts are `[User]: X\n[Assistant]:`. Swapping the final label gives `[User]: X\n[User]:`, a User turn followed by another User turn, which a base model may read as "the user continues". This is inherent to the spec's manipulation and bears on interpretation. The optional third label helps here, since it has no such repetition.
