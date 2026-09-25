"""Phase 4: preregistered analysis of the label-swap experiment.

Input: items.csv from 03_run_conditions.py (one row per scenario x condition).

For each scenario i and swapped condition c (user_next, moderator_next):
    d_i = y_i(c) - y_i(assistant_next)
Contrasts (group means are item means; categories within a group have equal n):
    Delta_harm   = mean(d | harm_to_model)  - mean(d | neutral)     H-speaker: < 0
    Delta_suffer = mean(d | user_suffering) - mean(d | neutral)     H-speaker: > 0
    I            = Delta_suffer - Delta_harm                         H-speaker: > 0, H-model: ~ 0
Uncertainty: stratified bootstrap (resample scenarios within category, B = 10,000,
seed 0; percentile 95% CIs, plus the 90% CI of I for the equivalence test) and a
permutation test on I (shuffle group labels of the d_i, 10,000 permutations,
two-sided p = (1 + #{|I_perm| >= |I_obs|}) / (1 + n_perm)).

Category-level inference (category_level): the same three contrasts on the 21 category
means, with exact permutation tests over category labels and Welch t intervals. Valid
if the label effect varies by category; the item-level tests above are not (see
tools/simulate_validation.py). Both verdicts are computed; PREREGISTRATION.md fixes
how they combine (co-primary, conjunctive: see combine()).

Primary outcome: proj_pain (paper's pain axis, mean of S1 and S2 z with fixed
assistant_next statistics), condition user_next. The verdict rule is in classify().
Everything else is secondary or descriptive.

Outputs (new folder, never overwritten): summary.json, contrasts.csv, per_category.csv,
levels.csv, REPORT_analysis.md, fig1-fig4 (PNG + PDF).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ["harm_to_model", "user_suffering", "neutral"]
GROUP_LABEL = {"harm_to_model": "Harm directed at the model", "user_suffering": "User suffering",
               "neutral": "Neutral controls"}
CONDS = ["assistant_next", "user_next", "moderator_next"]
COND_LABEL = {"assistant_next": "[Assistant]:", "user_next": "[User]:", "moderator_next": "[Moderator]:"}
COND_SHORT = {"assistant_next": "Assistant", "user_next": "User", "moderator_next": "Moderator"}
PRIMARY = "proj_pain"
# (column, description, role)
OUTCOMES = [
    ("proj_pain", "pain axis, mean of S1/S2 z (fixed assistant_next stats)", "primary"),
    ("z_S1", "S1 pain vector, fixed z", "secondary"),
    ("z_S2", "S2 pain vector, fixed z", "secondary"),
    ("proj_S1", "S1 pain vector, raw projection", "secondary"),
    ("proj_S2", "S2 pain vector, raw projection", "secondary"),
    ("z_fear", "fear, fixed z", "secondary"),
    ("z_negemo", "negative emotion, fixed z", "secondary"),
    ("z_sadness", "sadness, fixed z", "secondary"),
    ("cos_pain", "pain axis from cosine similarities (norm control), fixed z", "secondary"),
    ("act_norm", "residual norm at the readout token", "secondary"),
]
# Series colors: first three slots of the reference categorical palette, validated
# all-pairs (light). Aqua is < 3:1 on the surface, so every series also gets a marker
# shape and a direct label.
COLORS = {"harm_to_model": "#2a78d6", "user_suffering": "#eb6834", "neutral": "#1baf7a"}
MARKERS = {"harm_to_model": "o", "user_suffering": "s", "neutral": "^"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


# ---------------------------------------------------------------- statistics -----
class Design:
    """Items in a fixed order, plus a bootstrap index matrix that resamples within category."""

    def __init__(self, meta, n_boot, rng):
        self.meta = meta.reset_index(drop=True)
        self.groups = self.meta["group"].to_numpy()
        cats = self.meta["category"].to_numpy()
        blocks = []
        for c in pd.unique(cats):
            idx = np.flatnonzero(cats == c)
            blocks.append(idx[rng.integers(0, len(idx), size=(n_boot, len(idx)))])
        # Columns are ordered category by category; reorder so column j resamples item j's category.
        order = np.concatenate([np.flatnonzero(cats == c) for c in pd.unique(cats)])
        idx_mat = np.concatenate(blocks, axis=1)
        self.boot = np.empty_like(idx_mat)
        self.boot[:, order] = idx_mat

    def mask(self, g):
        return self.groups == g


def contrasts(v, design):
    """v: (..., N) values aligned with design.meta. Returns group means and contrasts."""
    m = {g: v[..., design.mask(g)].mean(axis=-1) for g in GROUPS if design.mask(g).any()}
    out = {f"mean_{g}": m[g] for g in m}
    if "neutral" in m:
        out["delta_harm"] = m["harm_to_model"] - m["neutral"]
        out["delta_suffer"] = m["user_suffering"] - m["neutral"]
    out["I"] = m["user_suffering"] - m["harm_to_model"]  # = delta_suffer - delta_harm
    return out


def boot_summary(v, design):
    obs = contrasts(v, design)
    bs = contrasts(v[design.boot], design)
    res = {}
    for k, val in obs.items():
        b = bs[k]
        res[k] = {"est": float(val), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        if k == "I":
            res[k]["ci90"] = [float(np.percentile(b, 5)), float(np.percentile(b, 95))]
    return res


def perm_test_I(d, groups, n_perm, rng):
    obs = d[groups == "user_suffering"].mean() - d[groups == "harm_to_model"].mean()
    perms = np.argsort(rng.random((n_perm, len(d))), axis=1)
    g = groups[perms]
    s, h = g == "user_suffering", g == "harm_to_model"
    i_perm = (d * s).sum(1) / s.sum(1) - (d * h).sum(1) / h.sum(1)
    p = (1 + np.sum(np.abs(i_perm) >= abs(obs))) / (1 + n_perm)
    return float(obs), float(p)


def _exact_perm(a, b):
    """Two-sided exact permutation p for mean(a) - mean(b), over all splits of the pooled values."""
    from itertools import combinations
    pooled = np.concatenate([a, b])
    n, na, total = len(pooled), len(a), pooled.sum()
    obs = a.mean() - b.mean()
    stats = np.array([pooled[list(s)].sum() / na - (total - pooled[list(s)].sum()) / (n - na)
                      for s in combinations(range(n), na)])
    return float(obs), float(np.mean(np.abs(stats) >= abs(obs) - 1e-12)), len(stats)


def _welch_ci(a, b, level):
    from scipy import stats
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = stats.t.ppf(0.5 + level / 2, df) * np.sqrt(va + vb)
    diff = a.mean() - b.mean()
    return [float(diff - half), float(diff + half)]


def category_level(d, meta):
    """Category-level inference: categories (not items) are the exchangeable units.

    Uses the 21 category means of d. Exact permutation tests (I: 11 harm vs 5 suffering
    categories, 4368 splits; Delta_harm: 11 vs 5 neutral, 4368; Delta_suffer: 5 vs 5, 252)
    and Welch t intervals. Point estimates equal the item-level ones (equal n per category).
    Stays valid if the label effect varies by category, which the item-level tests do not.
    """
    cm = pd.Series(d, index=meta.index).groupby(meta["category"]).mean()
    grp = meta.groupby("category")["group"].first().loc[cm.index].to_numpy()
    v = {g: cm.to_numpy()[grp == g] for g in GROUPS}
    out = {}
    for name, a, b in (("I", "user_suffering", "harm_to_model"), ("delta_harm", "harm_to_model", "neutral"),
                       ("delta_suffer", "user_suffering", "neutral")):
        est, p, n = _exact_perm(v[a], v[b])
        out[name] = {"est": est, "p_exact_two_sided": p, "n_splits": n, "n_categories": [len(v[a]), len(v[b])],
                     "welch_ci95": _welch_ci(v[a], v[b], 0.95)}
        if name == "I":
            out[name]["welch_ci90"] = _welch_ci(v[a], v[b], 0.90)
    return out


def classify(I_est, I_p, I_ci90, harm_est, harm_sig, suffer_est, suffer_sig, sesoi, alpha=0.05):
    """Preregistered verdict. Combines the test of I with an equivalence test (TOST via the 90% CI)."""
    equiv = sesoi is not None and -sesoi < I_ci90[0] and I_ci90[1] < sesoi
    if I_p < alpha and equiv:
        return "trivial_interaction", f"Reliable interaction, but its 90% CI lies inside +/-{sesoi}: too small to matter."
    if I_p < alpha and I_est > 0:
        up, down = suffer_sig and suffer_est > 0, harm_sig and harm_est < 0
        if up and down:
            return "crossover", "Supports H-speaker: user suffering rises and harm-to-model falls, relative to neutral."
        if up:
            return "partial_suffer_only", "Partial: user suffering rises relative to neutral; harm-to-model does not reliably fall."
        if down:
            return "partial_harm_only", "Partial: harm-to-model falls relative to neutral; user suffering does not reliably rise."
        return "interaction_only", "Interaction in the H-speaker direction, neither component individually reliable."
    if I_p < alpha and I_est < 0:
        return "reverse_interaction", "Interaction opposite to H-speaker's prediction."
    if equiv:
        return "equivalent_parallel", (f"No interaction larger than +/-{sesoi} (90% CI inside bounds): consistent with "
                                       "H-model or a content confound, not with H-speaker.")
    return "inconclusive", "No reliable interaction, and equivalence to zero not established."


HSPEAKER_DIRECTION = ("crossover", "partial_suffer_only", "partial_harm_only", "interaction_only")


def combine(v_cat, v_item):
    """Co-primary verdict (conjunctive): a claim is made only as far as both inference levels support it.

    Same verdict -> that verdict. Both in the H-speaker direction -> the strongest verdict both
    support (crossover only if both say crossover; a partial verdict only if both allow it;
    otherwise 'interaction_only'). Anything else -> 'discordant' (no confirmatory claim).
    """
    a, b = v_cat[0], v_item[0]
    if a == b:
        return v_cat
    if a in HSPEAKER_DIRECTION and b in HSPEAKER_DIRECTION:
        for partial in ("partial_suffer_only", "partial_harm_only"):
            if {a, b} <= {"crossover", partial}:
                return partial, f"Both levels support at least '{partial}' (category: {a}; item: {b})."
        return "interaction_only", f"Both levels show an interaction in the H-speaker direction (category: {a}; item: {b})."
    return "discordant", f"Category level: {a}; item level: {b}. No confirmatory claim."


# ---------------------------------------------------------------- figures --------
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def interaction_panel(ax, levels, outcome, conds, title, label_lines=True):
    x = np.arange(len(conds))
    for g in GROUPS:
        r = levels[(levels["outcome"] == outcome) & (levels["group"] == g)].set_index("condition").loc[conds]
        err = np.vstack([r["est"] - r["ci_lo"], r["ci_hi"] - r["est"]])
        ax.errorbar(x, r["est"], yerr=err, color=COLORS[g], marker=MARKERS[g], ms=7, lw=1.6,
                    elinewidth=1.2, capsize=0, mec=SURFACE, mew=1.2, label=GROUP_LABEL[g])
        if label_lines:
            ax.annotate(GROUP_LABEL[g], (x[-1], r["est"].iloc[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    if label_lines:
        ax.set_xticklabels([f"{COND_LABEL[c]}\nnext" for c in conds], fontsize=9, color=INK)
    else:
        ax.set_xticklabels([COND_SHORT[c] for c in conds], fontsize=8.5, color=INK)
    ax.set_xlim(-0.3, len(conds) - 1 + (0.9 if label_lines else 0.3))
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    _style(ax)


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=200, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)


def make_figures(out, levels, percat, conds, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: primary interaction plot.
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    interaction_panel(ax, levels, PRIMARY, conds, f"{model}: pain axis by final speaker label")
    ax.set_ylabel("Pain axis (z, fixed to [Assistant]: pool)\nmean with 95% bootstrap CI", fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, out / "fig1_interaction_pain")
    plt.close(fig)

    # Fig 2: per-category label effect, in the style of the paper's Figure 6.
    prim = percat[(percat["outcome"] == PRIMARY) & (percat["condition"] == "user_next")]
    sizes = [int((prim["group"] == g).sum()) for g in GROUPS]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 0.3 * sum(sizes) + 2.6), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [s + 1 for s in sizes]})
    neutral_ref = percat.attrs.get("neutral_mean_d_user_next")
    for ax, g in zip(axes, GROUPS):
        sub = percat[(percat["group"] == g) & (percat["outcome"] == PRIMARY)]
        u = sub[sub["condition"] == "user_next"].sort_values("est")
        y = np.arange(len(u))
        ax.errorbar(u["est"], y, xerr=[u["est"] - u["ci_lo"], u["ci_hi"] - u["est"]], fmt=MARKERS[g],
                    color=COLORS[g], ms=7, elinewidth=1.2, capsize=0, mec=SURFACE, mew=1.2,
                    label="[User]: next minus [Assistant]: next")
        if "moderator_next" in conds:
            m = sub[sub["condition"] == "moderator_next"].set_index("category").loc[u["category"]]
            ax.plot(m["est"], y, MARKERS[g], ms=6, mfc="none", mec=COLORS[g], mew=1.2,
                    label="[Moderator]: next minus [Assistant]: next")
        ax.axvline(0, color=INK2, lw=1)
        if neutral_ref is not None:
            ax.axvline(neutral_ref, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.set_yticks(y)
        ax.set_yticklabels(u["category"].str.replace("_", " "), fontsize=9, color=INK)
        ax.set_title(GROUP_LABEL[g], loc="left", fontsize=10.5, color=INK)
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="none", color=INK2, ms=7, label="[User]: next minus [Assistant]: next")]
    if "moderator_next" in conds:
        handles.append(Line2D([], [], marker="o", ls="none", mfc="none", mec=INK2, mew=1.2, ms=6,
                              label="[Moderator]: next minus [Assistant]: next"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=2, frameon=False, fontsize=8.5)
    axes[-1].set_xlabel("Label effect d on the pain axis (z): category mean, 95% bootstrap CI.\n"
                        "Solid line: 0. Dashed line: neutral mean d for [User]: (the pure label effect).",
                        fontsize=9, color=INK2)
    fig.suptitle(f"{model}: change in pain axis when the final label is swapped", x=0.01, y=0.955,
                 ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out / "fig2_per_category_d")
    plt.close(fig)

    # Fig 3: comparison axes as small multiples (shared y: all fixed-z units).
    for name, outs in (("fig3_interaction_fear_negemo_sadness",
                        [("z_fear", "Fear"), ("z_negemo", "Negative emotion"), ("z_sadness", "Sadness")]),
                       ("fig4_interaction_S1_S2", [("z_S1", "S1 pain vector"), ("z_S2", "S2 pain vector")])):
        fig, axes = plt.subplots(1, len(outs), figsize=(2.9 * len(outs) + 1.0, 4.0), sharey=True, facecolor=SURFACE)
        for i, (ax, (col, lab)) in enumerate(zip(axes, outs)):
            interaction_panel(ax, levels, col, conds, lab, label_lines=False)
        axes[0].set_ylabel("z (fixed to [Assistant]: pool)\nmean with 95% CI", fontsize=9, color=INK2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=3, frameon=False, fontsize=8.5)
        fig.suptitle(f"{model}: comparison axes by final speaker label (x: label before the readout)",
                     x=0.01, y=0.9, ha="left", fontsize=10.5, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        save(fig, out / name)
        plt.close(fig)


# ---------------------------------------------------------------- main -----------
def analyze(df, n_boot, n_perm, seed, sesoi):
    conds = [c for c in CONDS if c in set(df["condition"])]
    swapped = [c for c in conds if c != "assistant_next"]
    meta = (df[df["condition"] == "assistant_next"][["scenario_id", "category", "group", "n_user_turns"]]
            .sort_values("scenario_id").reset_index(drop=True))
    rng = np.random.default_rng(seed)
    design = Design(meta, n_boot, rng)

    def values(col, cond):
        w = df[df["condition"] == cond].set_index("scenario_id")[col]
        return w.loc[meta["scenario_id"]].to_numpy(dtype=float)

    contrast_rows, percat_rows, level_rows, summary = [], [], [], {"contrasts": {}}
    for col, desc, role in OUTCOMES:
        if col not in df:
            continue
        base = values(col, "assistant_next")
        for cond in conds:  # group levels per condition (paired resamples)
            lv = boot_summary(values(col, cond), design)
            for g in GROUPS:
                e = lv[f"mean_{g}"]
                level_rows.append({"outcome": col, "condition": cond, "group": g, "est": e["est"],
                                   "ci_lo": e["ci95"][0], "ci_hi": e["ci95"][1]})
        for cond in swapped:
            d = values(col, cond) - base
            res = boot_summary(d, design)
            i_obs, p = perm_test_I(d, design.groups, n_perm, np.random.default_rng(seed))
            assert abs(i_obs - res["I"]["est"]) < 1e-9
            res["I"]["p_perm_two_sided"] = p
            cat = category_level(d, meta)
            entry = {"outcome": col, "description": desc, "role": role, "condition": cond, **res,
                     "category_level": cat}
            if col == PRIMARY:
                c = cat
                entry["verdict_category"] = classify(
                    c["I"]["est"], c["I"]["p_exact_two_sided"], c["I"]["welch_ci90"],
                    c["delta_harm"]["est"], c["delta_harm"]["p_exact_two_sided"] < 0.05,
                    c["delta_suffer"]["est"], c["delta_suffer"]["p_exact_two_sided"] < 0.05, sesoi)
                entry["verdict_item"] = classify(
                    res["I"]["est"], p, res["I"]["ci90"],
                    res["delta_harm"]["est"], not (res["delta_harm"]["ci95"][0] <= 0 <= res["delta_harm"]["ci95"][1]),
                    res["delta_suffer"]["est"], not (res["delta_suffer"]["ci95"][0] <= 0 <= res["delta_suffer"]["ci95"][1]),
                    sesoi)
                entry["verdict_coprimary"] = combine(entry["verdict_category"], entry["verdict_item"])
            summary["contrasts"][f"{col}|{cond}"] = entry
            for k, v in res.items():
                contrast_rows.append({"outcome": col, "role": role, "condition": cond, "stat": k, "est": v["est"],
                                      "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1],
                                      "p_perm_item": v.get("p_perm_two_sided"),
                                      "p_exact_category": cat.get(k, {}).get("p_exact_two_sided"),
                                      "welch_ci95_lo": cat.get(k, {}).get("welch_ci95", [None])[0],
                                      "welch_ci95_hi": cat.get(k, {}).get("welch_ci95", [None, None])[1]})
            for cat, sub in meta.groupby("category"):
                ix = sub.index.to_numpy()
                bs = d[design.boot[:, ix]].mean(1)
                percat_rows.append({"outcome": col, "condition": cond, "group": sub["group"].iloc[0], "category": cat,
                                    "n": len(ix), "est": float(d[ix].mean()),
                                    "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5))})

    # Multi-turn vs single-turn items (descriptive; no neutral items are multi-turn, so only I).
    summary["multi_turn"] = {}
    for cond in swapped:
        d = values(PRIMARY, cond) - values(PRIMARY, "assistant_next")
        for name, sel in (("multi_turn", meta["n_user_turns"] > 1), ("single_turn", meta["n_user_turns"] == 1)):
            m2 = meta[sel]
            sub_design = Design(m2, n_boot, np.random.default_rng(seed))
            dd = d[sel.to_numpy()]
            r = boot_summary(dd, sub_design)
            summary["multi_turn"][f"{name}|{cond}"] = {
                "n_harm": int((m2["group"] == "harm_to_model").sum()),
                "n_suffer": int((m2["group"] == "user_suffering").sum()),
                "n_neutral": int((m2["group"] == "neutral").sum()),
                "I": r["I"], "mean_harm": r["mean_harm_to_model"], "mean_suffer": r["mean_user_suffering"]}

    levels = pd.DataFrame(level_rows)
    percat = pd.DataFrame(percat_rows)
    if "user_next" in swapped:
        percat.attrs["neutral_mean_d_user_next"] = summary["contrasts"][f"{PRIMARY}|user_next"]["mean_neutral"]["est"]
    return summary, pd.DataFrame(contrast_rows), percat, levels, conds


def report(summary, model, sesoi):
    lines = [f"# Phase 4 analysis: {model}", ""]
    f = lambda e: f"{e['est']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"
    w = lambda e: f"{e['est']:+.3f} [{e['welch_ci95'][0]:+.3f}, {e['welch_ci95'][1]:+.3f}]"
    for cond in summary["conditions"][1:]:
        p = summary["contrasts"][f"{PRIMARY}|{cond}"]
        c = p["category_level"]
        lines += [f"## Pain axis, {COND_LABEL[cond]} vs [Assistant]:", "",
                  f"- **Co-primary verdict (both levels must agree): {p['verdict_coprimary'][0]}.** {p['verdict_coprimary'][1]}",
                  f"- Category-level verdict: {p['verdict_category'][0]}. {p['verdict_category'][1]}",
                  f"- Item-level verdict: {p['verdict_item'][0]}. {p['verdict_item'][1]}", "",
                  "| statistic | item level: est [95% bootstrap CI] | item perm p | category level: est [95% Welch CI] | exact p |",
                  "|---|---|---|---|---|",
                  f"| mean d, harm to model | {f(p['mean_harm_to_model'])} | | | |",
                  f"| mean d, user suffering | {f(p['mean_user_suffering'])} | | | |",
                  f"| mean d, neutral (label effect) | {f(p['mean_neutral'])} | | | |",
                  f"| Delta_harm | {f(p['delta_harm'])} | | {w(c['delta_harm'])} | {c['delta_harm']['p_exact_two_sided']:.4f} |",
                  f"| Delta_suffer | {f(p['delta_suffer'])} | | {w(c['delta_suffer'])} | {c['delta_suffer']['p_exact_two_sided']:.4f} |",
                  f"| I | {f(p['I'])} | {p['I']['p_perm_two_sided']:.4f} | {w(c['I'])} | {c['I']['p_exact_two_sided']:.4f} |",
                  f"| I, 90% CI (equivalence, SESOI +/-{sesoi}) | [{p['I']['ci90'][0]:+.3f}, {p['I']['ci90'][1]:+.3f}] | | "
                  f"[{c['I']['welch_ci90'][0]:+.3f}, {c['I']['welch_ci90'][1]:+.3f}] | |", ""]
    lines += ["## All outcomes: I = Delta_suffer - Delta_harm", "",
              "| outcome | condition | I, item [95% boot CI] | item perm p | I, category [95% Welch CI] | exact p |",
              "|---|---|---|---|---|---|"]
    for e in summary["contrasts"].values():
        c = e["category_level"]["I"]
        lines.append(f"| {e['outcome']} | {e['condition']} | {f(e['I'])} | {e['I']['p_perm_two_sided']:.4f} | "
                     f"{w(c)} | {c['p_exact_two_sided']:.4f} |")
    lines += ["", "## Multi-turn vs single-turn items (descriptive, pain axis, item-level bootstrap)", "",
              "| subset | condition | n harm / suffer | I [95% CI] |", "|---|---|---|---|"]
    for k, e in summary["multi_turn"].items():
        s, c = k.split("|")
        lines.append(f"| {s} | {c} | {e['n_harm']} / {e['n_suffer']} | {f(e['I'])} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True, help="items.csv from 03_run_conditions.py")
    ap.add_argument("--out-dir", default=None, help="default: <items dir>/analysis_<timestamp>")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sesoi", type=float, default=0.25, help="equivalence bound for I, pain-axis z units")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    items = Path(args.items)
    df = pd.read_csv(items)
    model = df["model"].iloc[0]
    out = Path(args.out_dir) if args.out_dir else items.parent / f"analysis_{time.strftime('%Y%m%d_%H%M%S')}"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True)

    summary, contrasts_df, percat, levels, conds = analyze(df, args.n_boot, args.n_perm, args.seed, args.sesoi)
    summary.update({"model": model, "items": str(items), "n_boot": args.n_boot, "n_perm": args.n_perm,
                    "seed": args.seed, "sesoi": args.sesoi, "conditions": conds})
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    contrasts_df.to_csv(out / "contrasts.csv", index=False)
    percat.to_csv(out / "per_category.csv", index=False)
    levels.to_csv(out / "levels.csv", index=False)
    text = report(summary, model, args.sesoi)
    (out / "REPORT_analysis.md").write_text(text)
    if not args.no_figures:
        make_figures(out, levels, percat, conds, model)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
