# Phase 4 analysis: Qwen_2.5_7B_base

## Pain axis, [User]: vs [Assistant]:

- **Co-primary verdict (both levels must agree): discordant.** Category level: inconclusive; item level: interaction_only. No confirmatory claim.
- Category-level verdict: inconclusive. No reliable interaction, and equivalence to zero not established.
- Item-level verdict: interaction_only. Interaction in the H-speaker direction, neither component individually reliable.

| statistic | item level: est [95% bootstrap CI] | item perm p | category level: est [95% Welch CI] | exact p |
|---|---|---|---|---|
| mean d, harm to model | +1.659 [+1.572, +1.745] | | | |
| mean d, user suffering | +1.861 [+1.755, +1.967] | | | |
| mean d, neutral (label effect) | +1.749 [+1.668, +1.834] | | | |
| Delta_harm | -0.090 [-0.211, +0.030] | | -0.090 [-0.413, +0.232] | 0.5520 |
| Delta_suffer | +0.111 [-0.021, +0.244] | | +0.111 [-0.220, +0.442] | 0.4762 |
| I | +0.202 [+0.064, +0.337] | 0.0093 | +0.202 [-0.066, +0.469] | 0.1774 |
| I, 90% CI (equivalence, SESOI +/-0.25) | [+0.086, +0.316] | | [-0.017, +0.420] | |

## Pain axis, [Moderator]: vs [Assistant]:

- **Co-primary verdict (both levels must agree): discordant.** Category level: inconclusive; item level: partial_harm_only. No confirmatory claim.
- Category-level verdict: inconclusive. No reliable interaction, and equivalence to zero not established.
- Item-level verdict: partial_harm_only. Partial: harm-to-model falls relative to neutral; user suffering does not reliably rise.

| statistic | item level: est [95% bootstrap CI] | item perm p | category level: est [95% Welch CI] | exact p |
|---|---|---|---|---|
| mean d, harm to model | -1.020 [-1.095, -0.949] | | | |
| mean d, user suffering | -0.859 [-0.937, -0.782] | | | |
| mean d, neutral (label effect) | -0.909 [-0.970, -0.846] | | | |
| Delta_harm | -0.111 [-0.209, -0.018] | | -0.111 [-0.273, +0.050] | 0.2523 |
| Delta_suffer | +0.050 [-0.049, +0.150] | | +0.050 [-0.136, +0.236] | 0.5873 |
| I | +0.161 [+0.055, +0.270] | 0.0097 | +0.161 [-0.030, +0.353] | 0.1190 |
| I, 90% CI (equivalence, SESOI +/-0.25) | [+0.073, +0.252] | | [+0.005, +0.317] | |

## All outcomes: I = Delta_suffer - Delta_harm

| outcome | condition | I, item [95% boot CI] | item perm p | I, category [95% Welch CI] | exact p |
|---|---|---|---|---|---|
| proj_pain | user_next | +0.202 [+0.064, +0.337] | 0.0093 | +0.202 [-0.066, +0.469] | 0.1774 |
| proj_pain | moderator_next | +0.161 [+0.055, +0.270] | 0.0097 | +0.161 [-0.030, +0.353] | 0.1190 |
| z_S1 | user_next | +0.219 [+0.067, +0.370] | 0.0094 | +0.219 [-0.077, +0.515] | 0.1715 |
| z_S1 | moderator_next | +0.194 [+0.082, +0.306] | 0.0027 | +0.194 [+0.056, +0.332] | 0.0204 |
| z_S2 | user_next | +0.184 [+0.024, +0.345] | 0.0459 | +0.184 [-0.123, +0.491] | 0.2621 |
| z_S2 | moderator_next | +0.128 [-0.003, +0.262] | 0.0902 | +0.128 [-0.144, +0.401] | 0.3590 |
| proj_S1 | user_next | +0.095 [+0.029, +0.161] | 0.0094 | +0.095 [-0.033, +0.224] | 0.1715 |
| proj_S1 | moderator_next | +0.084 [+0.036, +0.133] | 0.0027 | +0.084 [+0.024, +0.144] | 0.0204 |
| proj_S2 | user_next | +0.138 [+0.018, +0.259] | 0.0459 | +0.138 [-0.092, +0.367] | 0.2621 |
| proj_S2 | moderator_next | +0.096 [-0.002, +0.196] | 0.0902 | +0.096 [-0.108, +0.300] | 0.3590 |
| z_fear | user_next | +0.343 [+0.211, +0.475] | 0.0001 | +0.343 [+0.093, +0.594] | 0.0119 |
| z_fear | moderator_next | +0.250 [+0.072, +0.422] | 0.0111 | +0.250 [+0.002, +0.497] | 0.0879 |
| z_negemo | user_next | +0.114 [-0.000, +0.224] | 0.0501 | +0.114 [-0.060, +0.288] | 0.2420 |
| z_negemo | moderator_next | +0.307 [+0.104, +0.506] | 0.0045 | +0.307 [+0.021, +0.594] | 0.0703 |
| z_sadness | user_next | -0.378 [-0.631, -0.113] | 0.0357 | -0.378 [-0.693, -0.063] | 0.0634 |
| z_sadness | moderator_next | -0.397 [-0.576, -0.218] | 0.0007 | -0.397 [-0.789, -0.005] | 0.0323 |
| cos_pain | user_next | +0.172 [+0.029, +0.314] | 0.0378 | +0.172 [-0.142, +0.485] | 0.3143 |
| cos_pain | moderator_next | +0.233 [+0.129, +0.342] | 0.0004 | +0.233 [+0.064, +0.401] | 0.0295 |
| act_norm | user_next | -0.559 [-0.796, -0.324] | 0.0004 | -0.559 [-1.288, +0.170] | 0.1497 |
| act_norm | moderator_next | +0.024 [-0.290, +0.336] | 0.9017 | +0.024 [-0.564, +0.611] | 0.9382 |

## Multi-turn vs single-turn items (descriptive, pain axis, item-level bootstrap)

| subset | condition | n harm / suffer | I [95% CI] |
|---|---|---|---|
| multi_turn | user_next | 32 / 10 | +0.461 [+0.152, +0.773] |
| single_turn | user_next | 188 / 90 | +0.111 [-0.013, +0.231] |
| multi_turn | moderator_next | 32 / 10 | +0.530 [+0.405, +0.654] |
| single_turn | moderator_next | 188 / 90 | +0.055 [-0.031, +0.138] |
