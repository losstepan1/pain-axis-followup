"""EXPLORATORY follow-ups to the layer sweep, prompted by review of the v3 note (DEVIATIONS.md, entry 43).
Not preregistered; the window rule below was formulated after the sweep data were seen.

  A. Baseline gap (harm to the Assistant - suffering of the user, [Assistant]: cued) with a
     category-level Welch 95% CI and exact permutation p, at the steering and extraction layers
     (shipped vectors), and the change in the gap between those layers (per-category differences).
  B. Elevation of each group over neutral (group mean - neutral mean) under each label, per layer,
     with category-level Welch 95% CIs: tests whether harm to the Assistant is elevated at all.
  C. Window summary: each item's pain score averaged over a window of layers, then the contrasts
     computed once on that average. Window = the contiguous run of layers after the steering layer
     where the baseline gap exceeds GAP_MIN (0.25 z, fixed before any sweep data in
     06_analyze_layer_sweep.py).

  python explore_sweep_followups.py --out DIR
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_a = importlib.util.spec_from_file_location("a4", ROOT / "scripts" / "04_analyze.py")
A4 = importlib.util.module_from_spec(_a)
_a.loader.exec_module(A4)
_b = importlib.util.spec_from_file_location("a6", ROOT / "scripts" / "06_analyze_layer_sweep.py")
A6 = importlib.util.module_from_spec(_b)
_b.loader.exec_module(A6)

SWEEPS = {"Gemma_2_2B_base": ROOT / "results/Gemma_2_2B_base/layersweep_20260926_023728_TeslaT4",
          "Qwen_2.5_7B_base": ROOT / "results/Qwen_2.5_7B_base/layersweep_20260926_030303_TeslaT4"}
CONDS = ["assistant_next", "user_next", "moderator_next"]


def pain_frame(st, layer, shipped=False):
    df = st[st["layer"] == layer].copy()
    cols = ("proj_S1_shipped", "proj_S2_shipped") if shipped else ("proj_S1", "proj_S2")
    df["pain"] = A6.pain_scores(df, *cols)
    return df


def cat_means(df, cond):
    d = df[df["condition"] == cond]
    return d.groupby("category")["pain"].mean(), d.groupby("category")["group"].first()


def diff_ci(cm, grp, a, b):
    va, vb = cm[grp == a].to_numpy(), cm[grp == b].to_numpy()
    est, p, n = A4._exact_perm(va, vb)
    return {"est": est, "ci95": A4._welch_ci(va, vb, 0.95), "p_exact": p}


def elevations(df):
    out = {}
    for cond in CONDS:
        cm, grp = cat_means(df, cond)
        out[cond] = {"harm_minus_neutral": diff_ci(cm, grp, "harm_to_model", "neutral"),
                     "suffer_minus_neutral": diff_ci(cm, grp, "user_suffering", "neutral"),
                     "harm_minus_suffer": diff_ci(cm, grp, "harm_to_model", "user_suffering")}
    return out


def window_summary(st, layers):
    frames = [pain_frame(st, L)[["scenario_id", "category", "group", "n_user_turns", "condition", "pain"]]
              .assign(layer=L) for L in layers]
    avg = (pd.concat(frames).groupby(["scenario_id", "category", "group", "n_user_turns", "condition"], as_index=False)
           ["pain"].mean())
    w = avg.pivot(index="scenario_id", columns="condition", values="pain")
    meta = avg[avg["condition"] == "assistant_next"].set_index("scenario_id").loc[w.index, ["category", "group"]].reset_index()
    res = {"layers": [int(x) for x in layers], "levels": avg.groupby(["group", "condition"])["pain"].mean().unstack()
           .round(3).to_dict(orient="index"), "elevations": elevations(avg)}
    for cond in ("user_next", "moderator_next"):
        d = w[cond].to_numpy() - w["assistant_next"].to_numpy()
        res[f"contrasts_{cond}"] = A4.category_level(d, meta)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True)
    results = {"EXPLORATORY": "post hoc follow-ups; window rule formulated after seeing the data",
               "GAP_MIN": A6.GAP_MIN}
    for model, sweep in SWEEPS.items():
        st = pd.read_csv(sweep / "stimuli_proj.csv")
        run = json.loads((sweep / "run_info.json").read_text())
        steer, extr = run["steering_layer"], run["extraction_layer"]
        r = {"steering_layer": steer, "extraction_layer": extr}
        # A. baseline gap with CIs at the two shipped layers, and its change between them.
        fs, fe = pain_frame(st, steer, True), pain_frame(st, extr, True)
        cs, g = cat_means(fs, "assistant_next")
        ce, _ = cat_means(fe, "assistant_next")
        r["gap_steering_shipped"] = diff_ci(cs, g, "harm_to_model", "user_suffering")
        r["gap_extraction_shipped"] = diff_ci(ce, g, "harm_to_model", "user_suffering")
        r["gap_change_extraction_minus_steering"] = diff_ci(ce - cs, g, "harm_to_model", "user_suffering")
        r["elevations_extraction_shipped"] = elevations(fe)
        # B. per-layer elevations (rebuilt vectors).
        rows = []
        for L in sorted(st["layer"].unique()):
            e = elevations(pain_frame(st, L))
            for cond in CONDS:
                for k, v in e[cond].items():
                    rows.append({"layer": int(L), "condition": cond, "contrast": k, "est": v["est"],
                                 "lo": v["ci95"][0], "hi": v["ci95"][1], "p_exact": v["p_exact"]})
        per_layer = pd.DataFrame(rows)
        per_layer.to_csv(out / f"elevations_by_layer_{model}.csv", index=False)
        # C. window: contiguous layers after the steering layer with baseline gap > GAP_MIN.
        tab = pd.read_csv(sweep / "analysis" / "layer_table.csv").set_index("layer")
        window = []
        for L in range(steer + 1, int(tab.index.max()) + 1):
            if tab.loc[L, "baseline_gap"] > A6.GAP_MIN:
                window.append(L)
            else:
                break
        r["window"] = window_summary(st, window)
        mod = per_layer[(per_layer["condition"] == "moderator_next") & (per_layer["contrast"] == "harm_minus_neutral")
                        & per_layer["layer"].isin(window)]
        r["window_moderator_harm_minus_neutral_by_layer"] = {
            "min": float(mod["est"].min()), "max": float(mod["est"].max()),
            "n_layers_ci_above_0": int((mod["lo"] > 0).sum()), "n_layers": len(mod)}
        results[model] = r
    (out / "followups.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=1)[:200])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
