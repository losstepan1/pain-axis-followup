"""EXPLORATORY (not preregistered): how much do the Phase 4 conclusions depend on the D1
choice of confirmatory unit (item vs category)?

For each model, on the primary contrast (pain axis, [User]: vs [Assistant]:):
  1. Heterogeneity: nested one-way ANOVA of d (categories within groups). F test of
     between-category variance beyond item noise; tau with an exact CI (balanced design).
  2. Mixed model: d ~ group + (1 | category), REML. The middle ground between item level
     (tau assumed 0) and category level (item information discarded). Wald CI for I.
  3. Re-simulation at the observed noise (sigma) and heterogeneity (tau): error rate and
     power of each level and of the co-primary rule, for true I = 0 and true I = observed.
  4. Turn structure: does a category's excess label effect track its share of multi-turn
     items? And tau on single-turn items only.

  python explore_d1_sensitivity.py --phase3 ../results/Qwen_2.5_7B_base/phase3_.../items.csv [...] --out DIR
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("analysis", HERE.parent / "scripts" / "04_analyze.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
A.OUTCOMES = [o for o in A.OUTCOMES if o[0] == "proj_pain"]


def label_effects(items):
    w = items.pivot(index="scenario_id", columns="condition", values="proj_pain")
    meta = items[items["condition"] == "assistant_next"].set_index("scenario_id")[["category", "group", "n_user_turns"]]
    return meta.assign(d=w["user_next"] - w["assistant_next"]).reset_index()


def heterogeneity(df):
    n = df.groupby("category").size().iloc[0]
    k, N, G = df["category"].nunique(), len(df), df["group"].nunique()
    cm = df.groupby("category")["d"].mean()
    grp = df.groupby("category")["group"].first()
    ms_between = n * ((cm - cm.groupby(grp).transform("mean")) ** 2).sum() / (k - G)
    ms_within = ((df["d"] - df.groupby("category")["d"].transform("mean")) ** 2).sum() / (N - k)
    F = ms_between / ms_within
    df1, df2 = k - G, N - k
    p = stats.f.sf(F, df1, df2)
    # Exact CI for theta = tau^2 / sigma^2 in the balanced one-way random-effects model.
    lo = max(0.0, (F / stats.f.ppf(0.975, df1, df2) - 1) / n)
    hi = max(0.0, (F / stats.f.ppf(0.025, df1, df2) - 1) / n)
    sigma = float(np.sqrt(ms_within))
    tau2 = max(0.0, (ms_between - ms_within) / n)
    return {"n_per_category": int(n), "F": float(F), "df": [df1, df2], "p": float(p), "sigma_within": sigma,
            "tau": float(np.sqrt(tau2)), "tau_ci95": [float(sigma * np.sqrt(lo)), float(sigma * np.sqrt(hi))]}


def mixed_model(df):
    import statsmodels.formula.api as smf
    m = smf.mixedlm("d ~ C(group, Treatment('neutral'))", df, groups=df["category"]).fit(reml=True)
    h, s = "C(group, Treatment('neutral'))[T.harm_to_model]", "C(group, Treatment('neutral'))[T.user_suffering]"
    cov = m.cov_params()
    est = m.params[s] - m.params[h]
    se = float(np.sqrt(cov.loc[s, s] + cov.loc[h, h] - 2 * cov.loc[s, h]))
    return {"I": float(est), "se": se, "ci95_wald": [float(est - 1.96 * se), float(est + 1.96 * se)],
            "p_wald": float(2 * stats.norm.sf(abs(est / se))),
            "delta_harm": float(m.params[h]), "delta_suffer": float(m.params[s]),
            "tau_reml": float(np.sqrt(m.cov_re.iloc[0, 0])), "sigma_reml": float(np.sqrt(m.scale)),
            "note": "Wald CI with 21 categories is somewhat anticonservative (no small-sample df correction)."}


def resimulate(df, sigma, tau, true_I, reps, n_boot, n_perm, seed):
    """Synthetic d with the observed noise structure; analysed with the frozen 04_analyze.py."""
    rng = np.random.default_rng(seed)
    meta = df[["scenario_id", "category", "group", "n_user_turns"]].copy()
    eff = {"harm_to_model": -true_I / 2, "user_suffering": true_I / 2, "neutral": 0.0}
    out = {"verdict_coprimary": [], "reject_item": [], "reject_category": []}
    cats = meta["category"].unique()
    for r in range(reps):
        u = dict(zip(cats, rng.normal(0, tau, len(cats))))
        d = (meta["group"].map(eff) + meta["category"].map(u)).to_numpy() + rng.normal(0, sigma, len(meta))
        base = meta.assign(model="sim", condition="assistant_next", proj_pain=0.0)
        user = meta.assign(model="sim", condition="user_next", proj_pain=d)
        summ, *_ = A.analyze(pd.concat([base, user]), n_boot, n_perm, seed + r, 0.25)
        e = summ["contrasts"]["proj_pain|user_next"]
        out["verdict_coprimary"].append(e["verdict_coprimary"][0])
        out["reject_item"].append(e["I"]["p_perm_two_sided"] < 0.05)
        out["reject_category"].append(e["category_level"]["I"]["p_exact_two_sided"] < 0.05)
    return {"true_I": true_I, "sigma": sigma, "tau": tau, "reps": reps,
            "reject_item_perm": float(np.mean(out["reject_item"])),
            "reject_category_exact": float(np.mean(out["reject_category"])),
            "verdicts_coprimary": pd.Series(out["verdict_coprimary"]).value_counts(normalize=True).round(3).to_dict()}


def turn_structure(df):
    cm = df.groupby("category").agg(d=("d", "mean"), group=("group", "first"),
                                    multi_share=("n_user_turns", lambda x: float((x > 1).mean())))
    cm["excess"] = cm["d"] - cm.groupby("group")["d"].transform("mean")
    has_mt = cm[cm["multi_share"] > 0]
    r, p = stats.spearmanr(cm["multi_share"], cm["excess"])
    single = df[df["n_user_turns"] == 1]
    # Unbalanced after dropping multi-turn items: method-of-moments tau with mean n.
    k, G = single["category"].nunique(), single["group"].nunique()
    n_bar = single.groupby("category").size().mean()
    c = single.groupby("category")["d"].agg(["mean", "size"]).join(single.groupby("category")["group"].first())
    msb = (c["size"] * (c["mean"] - c.groupby("group")["mean"].transform("mean")) ** 2).sum() / (k - G)
    msw = ((single["d"] - single.groupby("category")["d"].transform("mean")) ** 2).sum() / (len(single) - k)
    return {"spearman_multi_share_vs_excess_d": {"rho": float(r), "p": float(p), "n_categories": len(cm),
                                                 "n_categories_with_multi_turn": len(has_mt)},
            "tau_single_turn_only": float(np.sqrt(max(0.0, (msb - msw) / n_bar))),
            "per_category": cm.round(3).reset_index().to_dict(orient="records")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase3", nargs="+", required=True, help="items.csv files from Phase 3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True)

    results = {"EXPLORATORY": "not preregistered; does not change any preregistered verdict"}
    for path in args.phase3:
        items = pd.read_csv(path)
        model = items["model"].iloc[0]
        df = label_effects(items)
        het = heterogeneity(df)
        mm = mixed_model(df)
        obs_I = float(df[df.group == "user_suffering"]["d"].mean() - df[df.group == "harm_to_model"]["d"].mean())
        sims = [resimulate(df, het["sigma_within"], het["tau"], t, args.reps, 1000, 1000, args.seed)
                for t in (0.0, round(obs_I, 3))]
        results[model] = {"observed_I": obs_I, "heterogeneity": het, "mixed_model": mm,
                          "resimulation_at_observed_noise": sims, "turn_structure": turn_structure(df)}
        print(model, json.dumps({k: v for k, v in results[model].items() if k != "turn_structure"}, indent=1))
        print(" turn structure:", results[model]["turn_structure"]["spearman_multi_share_vs_excess_d"],
              "tau single-turn:", round(results[model]["turn_structure"]["tau_single_turn_only"], 3))
    (out / "d1_sensitivity.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
