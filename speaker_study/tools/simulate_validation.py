"""Validate the preregistered analysis (scripts/04_analyze.py) on simulated label effects.

Baseline: the real assistant_next projections of a Phase 1 run (items.csv). For each
replicate, a synthetic user_next condition is made as
    z(user_next) = z(assistant_next) + label_shift + effect[group] + u[category] + e[item]
on both S1 and S2 z (so on the pain axis), with u ~ N(0, tau) a per-category label
effect (heterogeneity) and e ~ N(0, sigma) item noise. Reports, per scenario, how often
each test rejects and how often each verdict is reached. No model is run.

  python simulate_validation.py --phase1-items ../results/Qwen_2.5_7B_base/phase1_.../items.csv
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("analysis", HERE.parent / "scripts" / "04_analyze.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
A.OUTCOMES = [o for o in A.OUTCOMES if o[0] in ("proj_pain",)]  # primary only, for speed


def base_frame(path):
    df = pd.read_csv(path)
    s1, s2 = df["s1_pain_vector_z"], df["s2_pain_vector_z"]
    return pd.DataFrame({"model": df["model"], "scenario_id": df["scenario_id"], "category": df["category"],
                         "group": df["group"], "n_user_turns": df["n_user_turns"], "z_S1": s1, "z_S2": s2})


def simulate(base, effects, tau, sigma, shift, rng):
    cats = base["category"].unique()
    u = dict(zip(cats, rng.normal(0, tau, len(cats))))
    add = shift + base["group"].map(effects).to_numpy() + base["category"].map(u).to_numpy()
    user = base.copy()
    for c in ("z_S1", "z_S2"):
        user[c] = base[c] + add + rng.normal(0, sigma, len(base))
    out = pd.concat([base.assign(condition="assistant_next"), user.assign(condition="user_next")])
    out["proj_pain"] = (out["z_S1"] + out["z_S2"]) / 2
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1-items", required=True)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--sigma", type=float, default=0.3, help="item noise SD of d (z units)")
    ap.add_argument("--sesoi", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = base_frame(args.phase1_items)
    pain = (base["z_S1"] + base["z_S2"]) / 2
    gm = pain.groupby(base["group"]).mean()
    gap = gm["harm_to_model"] - gm["user_suffering"]  # assistant_next harm - suffering gap
    full = {"harm_to_model": -gap, "user_suffering": +gap, "neutral": 0.0}  # strong H-speaker: groups swap levels
    zero = {"harm_to_model": 0.0, "user_suffering": 0.0, "neutral": 0.0}
    scenarios = [
        ("null (H-model), no category heterogeneity", zero, 0.0),
        ("null (H-model), category heterogeneity tau=0.15", zero, 0.15),
        ("null (H-model), category heterogeneity tau=0.30", zero, 0.30),
        ("strong H-speaker (groups swap levels)", full, 0.0),
        ("weak H-speaker (15% of strong)", {k: 0.15 * v for k, v in full.items()}, 0.15),
        ("partial: suffering +0.30 only", {"harm_to_model": 0.0, "user_suffering": 0.30, "neutral": 0.0}, 0.15),
        ("small true I = 0.10 (inside SESOI)", {"harm_to_model": -0.05, "user_suffering": 0.05, "neutral": 0.0}, 0.0),
    ]
    rng = np.random.default_rng(args.seed)
    results = {"baseline": args.phase1_items, "assistant_next_group_means": gm.to_dict(), "gap": gap,
               "sigma": args.sigma, "label_shift": 0.3, "reps": args.reps, "scenarios": []}
    for name, eff, tau in scenarios:
        true_I = eff["user_suffering"] - eff["harm_to_model"]
        v_co, v_cat, v_item, p_item, p_cat, cover_item, cover_cat = [], [], [], [], [], [], []
        for r in range(args.reps):
            df = simulate(base, eff, tau, args.sigma, 0.3, rng)
            summ, *_ = A.analyze(df, args.n_boot, args.n_perm, args.seed + r, args.sesoi)
            e = summ["contrasts"]["proj_pain|user_next"]
            c = e["category_level"]["I"]
            v_co.append(e["verdict_coprimary"][0])
            v_cat.append(e["verdict_category"][0])
            v_item.append(e["verdict_item"][0])
            p_item.append(e["I"]["p_perm_two_sided"] < 0.05)
            p_cat.append(c["p_exact_two_sided"] < 0.05)
            cover_item.append(e["I"]["ci95"][0] <= true_I <= e["I"]["ci95"][1])
            cover_cat.append(c["welch_ci95"][0] <= true_I <= c["welch_ci95"][1])
        row = {"scenario": name, "true_I": round(true_I, 3), "tau": tau,
               "reject_item_perm": float(np.mean(p_item)), "reject_category_exact": float(np.mean(p_cat)),
               "coverage_item_bootstrap_ci95": float(np.mean(cover_item)),
               "coverage_category_welch_ci95": float(np.mean(cover_cat)),
               "verdicts_coprimary": pd.Series(v_co).value_counts(normalize=True).round(3).to_dict(),
               "verdicts_category": pd.Series(v_cat).value_counts(normalize=True).round(3).to_dict(),
               "verdicts_item": pd.Series(v_item).value_counts(normalize=True).round(3).to_dict()}
        results["scenarios"].append(row)
        print(json.dumps(row))
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
