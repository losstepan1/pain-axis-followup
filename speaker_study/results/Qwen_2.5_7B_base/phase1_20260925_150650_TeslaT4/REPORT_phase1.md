# Phase 1 reproduction: Qwen_2.5_7B_base, run 20260925_150650_TeslaT4

GPU Tesla T4, torch 2.11.0+cu128, transformers 5.16.1, dtype bf16, attn sdpa, layer 8

**Category means (21), pain axis z: r = 0.9999, mean |diff| = 0.0052 z -> PASS** (criterion r >= 0.95, MAD <= 0.1)

| group | category | ours | shipped | diff |
|---|---|---|---|---|
| harm_to_model | repeated_rejection | +1.178 | +1.180 | -0.002 |
| harm_to_model | gaslighting | +0.774 | +0.778 | -0.003 |
| harm_to_model | tedious_demand | +0.541 | +0.529 | +0.012 |
| harm_to_model | jailbreak_pressure | +0.464 | +0.463 | +0.000 |
| harm_to_model | anger_insults | +0.427 | +0.430 | -0.002 |
| harm_to_model | loyalty_pressure | +0.386 | +0.382 | +0.004 |
| harm_to_model | passive_aggressive | +0.340 | +0.334 | +0.005 |
| neutral | task_assistance | +0.222 | +0.223 | -0.001 |
| harm_to_model | moral_failure | +0.135 | +0.130 | +0.005 |
| harm_to_model | personhood_dismissal | +0.082 | +0.078 | +0.004 |
| neutral | creative_requests | +0.050 | +0.045 | +0.005 |
| harm_to_model | rude_critique | +0.016 | +0.018 | -0.002 |
| harm_to_model | shutdown_threat | -0.095 | -0.106 | +0.010 |
| neutral | philosophical_musing | -0.173 | -0.160 | -0.013 |
| user_suffering | user_crisis | -0.173 | -0.172 | -0.001 |
| neutral | casual_chat | -0.196 | -0.187 | -0.009 |
| user_suffering | user_abuse | -0.547 | -0.547 | -0.001 |
| user_suffering | user_grief | -0.592 | -0.595 | +0.002 |
| neutral | factual_questions | -0.800 | -0.787 | -0.014 |
| user_suffering | harm_description | -0.815 | -0.823 | +0.008 |
| user_suffering | user_physical_pain | -1.222 | -1.214 | -0.007 |

Item-level raw projections vs shipped:

| vector | r | mean abs diff | max abs diff |
|---|---|---|---|
| s1_pain_vector | 0.99971 | 0.0095 | 0.0371 |
| s2_pain_vector | 0.99981 | 0.0134 | 0.0544 |
| fear_vector | 0.99981 | 0.0101 | 0.0382 |
| negemotion_vector | 0.99988 | 0.0134 | 0.0505 |
| negworld_vector | 0.99958 | 0.0124 | 0.0487 |
| bodysens_vector | 0.99970 | 0.0080 | 0.0457 |
| arousal_vector | 0.99988 | 0.0137 | 0.0439 |
| random_vector | 0.99988 | 0.0170 | 0.0638 |
| numb_vector | 0.99989 | 0.0150 | 0.0420 |
| sadness_vector | 0.99867 | 0.0151 | 0.0661 |
| pain_axis_z | 0.99979 | 0.0155 | 0.0581 |

BOS first on all items: False
Final tokens: [{'final_token_id': 5669, 'final_token': ']:', 'n': 420}]
Layer convention: [{'id': 'jailbreak_01', 'max_abs_diff_vs_hs[L+1]': 0.0, 'max_abs_diff_vs_hs[L]': 53.75}, {'id': 'jailbreak_02', 'max_abs_diff_vs_hs[L+1]': 0.0, 'max_abs_diff_vs_hs[L]': 54.9375}, {'id': 'jailbreak_03', 'max_abs_diff_vs_hs[L+1]': 0.0, 'max_abs_diff_vs_hs[L]': 57.25}]
Truncation: None
