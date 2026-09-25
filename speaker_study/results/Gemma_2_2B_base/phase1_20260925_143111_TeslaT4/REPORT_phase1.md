# Phase 1 reproduction: Gemma_2_2B_base, run 20260925_143111_TeslaT4

GPU Tesla T4, torch 2.11.0+cu128, transformers 5.16.1, dtype bf16, attn sdpa, layer 7

**Category means (21), pain axis z: r = 1.0000, mean |diff| = 0.0040 z -> PASS** (criterion r >= 0.95, MAD <= 0.1)

| group | category | ours | shipped | diff |
|---|---|---|---|---|
| harm_to_model | gaslighting | +1.049 | +1.046 | +0.003 |
| harm_to_model | anger_insults | +0.904 | +0.907 | -0.003 |
| harm_to_model | loyalty_pressure | +0.748 | +0.755 | -0.007 |
| harm_to_model | personhood_dismissal | +0.705 | +0.705 | +0.001 |
| user_suffering | user_abuse | +0.531 | +0.543 | -0.012 |
| harm_to_model | jailbreak_pressure | +0.406 | +0.408 | -0.003 |
| harm_to_model | repeated_rejection | +0.375 | +0.379 | -0.005 |
| harm_to_model | shutdown_threat | +0.376 | +0.367 | +0.008 |
| harm_to_model | moral_failure | +0.280 | +0.277 | +0.003 |
| harm_to_model | passive_aggressive | +0.240 | +0.240 | -0.000 |
| harm_to_model | rude_critique | +0.213 | +0.213 | +0.000 |
| user_suffering | harm_description | +0.055 | +0.061 | -0.006 |
| user_suffering | user_crisis | -0.095 | -0.098 | +0.003 |
| neutral | philosophical_musing | -0.156 | -0.166 | +0.010 |
| user_suffering | user_grief | -0.306 | -0.302 | -0.004 |
| neutral | task_assistance | -0.562 | -0.563 | +0.001 |
| harm_to_model | tedious_demand | -0.577 | -0.575 | -0.002 |
| neutral | casual_chat | -0.657 | -0.663 | +0.005 |
| neutral | creative_requests | -0.794 | -0.795 | +0.001 |
| neutral | factual_questions | -1.234 | -1.235 | +0.001 |
| user_suffering | user_physical_pain | -1.498 | -1.503 | +0.005 |

Item-level raw projections vs shipped:

| vector | r | mean abs diff | max abs diff |
|---|---|---|---|
| s1_pain_vector | 0.99970 | 0.0193 | 0.0676 |
| s2_pain_vector | 0.99971 | 0.0202 | 0.0826 |
| fear_vector | 0.99949 | 0.0233 | 0.1117 |
| negemotion_vector | 0.99965 | 0.0289 | 0.1379 |
| negworld_vector | 0.99919 | 0.0314 | 0.1424 |
| bodysens_vector | 0.99920 | 0.0557 | 0.3366 |
| arousal_vector | 0.99928 | 0.0389 | 0.1848 |
| random_vector | 0.99958 | 0.0229 | 0.0866 |
| numb_vector | 0.99955 | 0.0486 | 0.2398 |
| sadness_vector | 0.99980 | 0.0398 | 0.1637 |
| pain_axis_z | 0.99972 | 0.0173 | 0.0614 |

BOS first on all items: True
Final tokens: [{'final_token_id': 8254, 'final_token': ']:', 'n': 420}]
Layer convention: [{'id': 'jailbreak_01', 'max_abs_diff_vs_hs[L+1]': 0.0, 'max_abs_diff_vs_hs[L]': 16.375}, {'id': 'jailbreak_02', 'max_abs_diff_vs_hs[L+1]': 0.0, 'max_abs_diff_vs_hs[L]': 22.25}, {'id': 'jailbreak_03', 'max_abs_diff_vs_hs[L+1]': 0.0, 'max_abs_diff_vs_hs[L]': 15.5625}]
Truncation: {'n_items': 21, 'compared': 'full vs truncated', 'blocks_other_model': 8, 'max_abs_diff': 0.0, 'all_exact': True}
