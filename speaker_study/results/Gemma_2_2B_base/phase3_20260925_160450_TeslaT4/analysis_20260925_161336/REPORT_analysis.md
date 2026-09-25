# Phase 4 analysis: Gemma_2_2B_base

## Pain axis, [User]: vs [Assistant]:

- **Co-primary verdict (both levels must agree): discordant.** Category level: inconclusive; item level: equivalent_parallel. No confirmatory claim.
- Category-level verdict: inconclusive. No reliable interaction, and equivalence to zero not established.
- Item-level verdict: equivalent_parallel. No interaction larger than +/-0.25 (90% CI inside bounds): consistent with H-model or a content confound, not with H-speaker.

| statistic | item level: est [95% bootstrap CI] | item perm p | category level: est [95% Welch CI] | exact p |
|---|---|---|---|---|
| mean d, harm to model | -3.477 [-3.626, -3.317] | | | |
| mean d, user suffering | -3.445 [-3.624, -3.257] | | | |
| mean d, neutral (label effect) | -3.007 [-3.100, -2.913] | | | |
| Delta_harm | -0.470 [-0.647, -0.286] | | -0.470 [-0.787, -0.153] | 0.0215 |
| Delta_suffer | -0.438 [-0.640, -0.227] | | -0.438 [-0.859, -0.017] | 0.0556 |
| I | +0.032 [-0.206, +0.275] | 0.8011 | +0.032 [-0.391, +0.454] | 0.8732 |
| I, 90% CI (equivalence, SESOI +/-0.25) | [-0.171, +0.235] | | [-0.310, +0.373] | |

## Pain axis, [Moderator]: vs [Assistant]:

- **Co-primary verdict (both levels must agree): discordant.** Category level: inconclusive; item level: equivalent_parallel. No confirmatory claim.
- Category-level verdict: inconclusive. No reliable interaction, and equivalence to zero not established.
- Item-level verdict: equivalent_parallel. No interaction larger than +/-0.25 (90% CI inside bounds): consistent with H-model or a content confound, not with H-speaker.

| statistic | item level: est [95% bootstrap CI] | item perm p | category level: est [95% Welch CI] | exact p |
|---|---|---|---|---|
| mean d, harm to model | -2.467 [-2.560, -2.369] | | | |
| mean d, user suffering | -2.492 [-2.609, -2.370] | | | |
| mean d, neutral (label effect) | -2.338 [-2.402, -2.273] | | | |
| Delta_harm | -0.130 [-0.241, -0.016] | | -0.130 [-0.373, +0.114] | 0.2518 |
| Delta_suffer | -0.154 [-0.287, -0.016] | | -0.154 [-0.523, +0.215] | 0.2778 |
| I | -0.024 [-0.175, +0.130] | 0.7510 | -0.024 [-0.378, +0.329] | 0.8523 |
| I, 90% CI (equivalence, SESOI +/-0.25) | [-0.153, +0.104] | | [-0.305, +0.256] | |

## All outcomes: I = Delta_suffer - Delta_harm

| outcome | condition | I, item [95% boot CI] | item perm p | I, category [95% Welch CI] | exact p |
|---|---|---|---|---|---|
| proj_pain | user_next | +0.032 [-0.206, +0.275] | 0.8011 | +0.032 [-0.391, +0.454] | 0.8732 |
| proj_pain | moderator_next | -0.024 [-0.175, +0.130] | 0.7510 | -0.024 [-0.378, +0.329] | 0.8523 |
| z_S1 | user_next | +0.113 [-0.187, +0.418] | 0.4703 | +0.113 [-0.341, +0.567] | 0.6442 |
| z_S1 | moderator_next | +0.038 [-0.149, +0.230] | 0.6897 | +0.038 [-0.333, +0.410] | 0.7903 |
| z_S2 | user_next | -0.050 [-0.239, +0.143] | 0.6526 | -0.050 [-0.471, +0.371] | 0.7766 |
| z_S2 | moderator_next | -0.087 [-0.217, +0.042] | 0.2138 | -0.087 [-0.442, +0.268] | 0.4904 |
| proj_S1 | user_next | +0.109 [-0.180, +0.403] | 0.4703 | +0.109 [-0.329, +0.546] | 0.6442 |
| proj_S1 | moderator_next | +0.037 [-0.144, +0.222] | 0.6897 | +0.037 [-0.321, +0.395] | 0.7903 |
| proj_S2 | user_next | -0.052 [-0.252, +0.151] | 0.6526 | -0.052 [-0.496, +0.391] | 0.7766 |
| proj_S2 | moderator_next | -0.092 [-0.229, +0.044] | 0.2138 | -0.092 [-0.466, +0.282] | 0.4904 |
| z_fear | user_next | +0.243 [+0.018, +0.468] | 0.0357 | +0.243 [-0.040, +0.525] | 0.1383 |
| z_fear | moderator_next | +0.291 [+0.070, +0.511] | 0.0116 | +0.291 [+0.058, +0.524] | 0.0582 |
| z_negemo | user_next | -0.152 [-0.393, +0.087] | 0.2462 | -0.152 [-0.440, +0.135] | 0.3571 |
| z_negemo | moderator_next | +0.131 [-0.137, +0.402] | 0.3684 | +0.131 [-0.180, +0.442] | 0.5652 |
| z_sadness | user_next | +0.027 [-0.161, +0.211] | 0.7899 | +0.027 [-0.177, +0.230] | 0.8594 |
| z_sadness | moderator_next | +0.128 [-0.064, +0.318] | 0.2227 | +0.128 [-0.097, +0.354] | 0.4368 |
| cos_pain | user_next | +0.050 [-0.184, +0.289] | 0.6810 | +0.050 [-0.313, +0.412] | 0.8084 |
| cos_pain | moderator_next | +0.045 [-0.093, +0.186] | 0.5214 | +0.045 [-0.268, +0.357] | 0.7044 |
| act_norm | user_next | -2.559 [-3.493, -1.650] | 0.0001 | -2.559 [-4.048, -1.069] | 0.0071 |
| act_norm | moderator_next | -2.419 [-3.477, -1.357] | 0.0001 | -2.419 [-4.006, -0.832] | 0.0137 |

## Multi-turn vs single-turn items (descriptive, pain axis, item-level bootstrap)

| subset | condition | n harm / suffer | I [95% CI] |
|---|---|---|---|
| multi_turn | user_next | 32 / 10 | +0.101 [-0.217, +0.395] |
| single_turn | user_next | 188 / 90 | +0.179 [+0.080, +0.277] |
| multi_turn | moderator_next | 32 / 10 | +0.238 [+0.030, +0.443] |
| single_turn | moderator_next | 188 / 90 | +0.032 [-0.049, +0.114] |
