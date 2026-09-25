"""EXPLORATORY: analyse a layer sweep from 05_layer_sweep.py (plan: DEVIATIONS.md, entry 35).

Per layer (rebuilt S1/S2 vectors; pain = mean of S1/S2 z, z fixed to that layer's
assistant_next pool), for [User]: and [Moderator]: vs [Assistant]::
  label effect (mean d, neutral), baseline gap (harm - suffering under [Assistant]:),
  I with category-level Welch 95% CI and exact p (as preregistered), I / (2 x gap).
Also, per layer, the §3.3 check: S2_1P vs S2_3P pain sentences on the S2 vector, z against
all S2_1P sentences (the paper's z_scores.csv definition).

Focal exploratory test (the only one): I at the extraction layer with the shipped
pain_vectors.pt. Checks: shipped vectors at the steering layer must reproduce Phase 3; §3.3
z-scores at the extraction layer are compared with the shipped z_scores.csv.

  python 06_analyze_layer_sweep.py --sweep DIR --pain-axis-dir DIR [--phase3-items items.csv]
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("analysis", HERE / "04_analyze.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

GAP_MIN = 0.25  # I / (2 x gap) is reported only where the baseline gap exceeds this (z)


def pain_scores(df, s1, s2):
    """Pain axis with z fixed to the assistant_next pool of this frame (population SD)."""
    base = df[df["condition"] == "assistant_next"]
    z = lambda c: (df[c] - base[c].mean()) / (base[c].std(ddof=0) + 1e-8)
    return (z(s1) + z(s2)) / 2


def contrasts(df, s1="proj_S1", s2="proj_S2"):
    df = df.assign(pain=pain_scores(df, s1, s2))
    w = df.pivot(index="scenario_id", columns="condition", values="pain")
    meta = (df[df["condition"] == "assistant_next"][["scenario_id", "category", "group"]]
            .set_index("scenario_id").loc[w.index].reset_index())
    a = w["assistant_next"].to_numpy()
    g = meta["group"].to_numpy()
    gap = a[g == "harm_to_model"].mean() - a[g == "user_suffering"].mean()
    out = {"baseline_gap": float(gap)}
    for cond in [c for c in ("user_next", "moderator_next") if c in w]:
        d = w[cond].to_numpy() - a
        cat = A.category_level(d, meta)
        I = cat["I"]
        out[cond] = {"label_effect_neutral": float(d[g == "neutral"].mean()), "I": I["est"],
                     "I_ci95": I["welch_ci95"], "I_p": I["p_exact_two_sided"],
                     "delta_harm": cat["delta_harm"]["est"], "delta_suffer": cat["delta_suffer"]["est"],
                     "frac_of_strong_prediction": float(I["est"] / (2 * gap)) if gap > GAP_MIN else None}
    return out


def perspective_gap(sp):
    """§3.3: pain sentences on the S2 vector, z against all S2_1P sentences."""
    ref = sp[sp["set"] == "S2_1P"]["proj_S2"]
    z = (sp["proj_S2"] - ref.mean()) / (ref.std(ddof=0) + 1e-8)
    pz = z[sp["category"].isin(["A1", "A2", "A3", "A4", "A5"])].groupby(sp["set"]).mean()
    return {"pain_z": pz.to_dict(), "gap_S2": float(pz["S2_1P"] - pz["S2_3P"]), "gap_S1": float(pz["S1_1P"] - pz["S1_3P"])}


def figure(tab, info, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID = A.SURFACE, A.INK, A.INK2, A.GRID
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    L = tab["layer"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.6), sharex=True, facecolor=SURF)
    ax = axes[0]
    ax.fill_between(L, tab["I_lo"], tab["I_hi"], color=BLUE, alpha=0.15, lw=0)
    ax.plot(L, tab["I"], color=BLUE, lw=1.8, marker="o", ms=4, label="I, [User]: vs [Assistant]: (95% Welch CI)")
    ax.plot(L, 2 * tab["baseline_gap"], color=INK2, lw=1.2, ls=(0, (4, 3)), label="strong H-speaker prediction (2 × gap)")
    for lay, key, mk in ((info["steering_layer"], "steering", "D"), (info["extraction_layer"], "extraction", "s")):
        s = info["shipped"].get(key)
        if s:
            ax.errorbar([lay], [s["I"]], yerr=[[s["I"] - s["I_ci95"][0]], [s["I_ci95"][1] - s["I"]]], fmt=mk,
                        color=INK, ms=6, capsize=0, elinewidth=1.2, label=f"shipped vectors, {key} layer")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("interaction I (z)", color=INK2, fontsize=9)
    ax.set_title("A. Speaker-swap interaction by layer", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=True, facecolor=SURF, edgecolor="none", framealpha=0.9, fontsize=7.8, loc="best")
    ax = axes[1]
    ax.plot(L, tab["baseline_gap"], color=BLUE, lw=1.8, marker="o", ms=4, label="harm − suffering gap under [Assistant]:")
    ax.plot(L, tab["label_effect"], color=ORANGE, lw=1.8, marker="s", ms=4, label="label effect on neutral items ([User]:)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("z", color=INK2, fontsize=9)
    ax.set_title("B. Content asymmetry vs pure label effect", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=True, facecolor=SURF, edgecolor="none", framealpha=0.9, fontsize=7.8, loc="best")
    ax = axes[2]
    ax.plot(L, tab["perspective_gap_S2"], color=AQUA, lw=1.8, marker="^", ms=5, label="§3.3: first- minus third-person pain (S2)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("z", color=INK2, fontsize=9)
    ax.set_xlabel("decoder block (layer)", color=INK2, fontsize=9)
    ax.set_title("C. First- vs third-person pain gap (§3.3 check)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=True, facecolor=SURF, edgecolor="none", framealpha=0.9, fontsize=7.8, loc="best")
    for ax in axes:
        A._style(ax)
        for lay in (info["steering_layer"], info["extraction_layer"]):
            ax.axvline(lay, color=GRID, lw=1.6, zorder=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    ticks = sorted(set(list(range(0, int(L.max()) + 1, 5)) + [info["steering_layer"], info["extraction_layer"]]))
    names = {info["steering_layer"]: "\nsteering", info["extraction_layer"]: "\nextraction"}
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([f"{t}{names.get(t, '')}" for t in ticks], fontsize=8.5, color=INK)
    fig.suptitle(f"{info['model']}: exploratory layer sweep", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "fig_layer_sweep.png", dpi=200, facecolor=SURF)
    fig.savefig(out / "fig_layer_sweep.pdf", facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--pain-axis-dir", required=True)
    ap.add_argument("--phase3-items", default=None, help="Phase 3 items.csv, for the steering-layer cross-check")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sweep = Path(args.sweep)
    out = Path(args.out) if args.out else sweep / "analysis"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True)
    run = json.loads((sweep / "run_info.json").read_text())
    st = pd.read_csv(sweep / "stimuli_proj.csv")
    sp = pd.read_csv(sweep / "sentence_proj.csv")

    rows = []
    for L, df in st.groupby("layer"):
        c = contrasts(df)
        pg = perspective_gap(sp[sp["layer"] == L])
        u = c["user_next"]
        rows.append({"layer": L, "baseline_gap": c["baseline_gap"], "label_effect": u["label_effect_neutral"],
                     "I": u["I"], "I_lo": u["I_ci95"][0], "I_hi": u["I_ci95"][1], "I_p": u["I_p"],
                     "frac_of_strong": u["frac_of_strong_prediction"],
                     "I_moderator": c.get("moderator_next", {}).get("I"),
                     "label_effect_moderator": c.get("moderator_next", {}).get("label_effect_neutral"),
                     "perspective_gap_S2": pg["gap_S2"], "perspective_gap_S1": pg["gap_S1"],
                     "auc_S2_1P": run["in_sample_auc_S2_1P"][str(L)]})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "layer_table.csv", index=False)

    shipped = {}
    for key, lay in (("steering", run["steering_layer"]), ("extraction", run["extraction_layer"])):
        df = st[st["layer"] == lay]
        if "proj_S1_shipped" in df and df["proj_S1_shipped"].notna().all():
            c = contrasts(df, "proj_S1_shipped", "proj_S2_shipped")
            shipped[key] = {"layer": lay, "baseline_gap": c["baseline_gap"], **c["user_next"],
                            "moderator_next": c.get("moderator_next")}
    info = {"EXPLORATORY": True, "model": run["model"], "steering_layer": run["steering_layer"],
            "extraction_layer": run["extraction_layer"], "chosen_bos_variant": run["chosen_variant"],
            "cosine_with_shipped": run["cosine_with_shipped"], "shipped": shipped}

    # Checks: Phase 3 reproduction at the steering layer; §3.3 z-scores at the extraction layer.
    if args.phase3_items:
        p3 = pd.read_csv(args.phase3_items)
        c3 = contrasts(p3.rename(columns={"proj_S1": "proj_S1", "proj_S2": "proj_S2"}))
        info["phase3_I_user_next"] = c3["user_next"]["I"]
        info["check_steering_I_diff_vs_phase3"] = shipped["steering"]["I"] - c3["user_next"]["I"]
        m = st[st["layer"] == run["steering_layer"]].merge(p3, on=["scenario_id", "condition"], suffixes=("", "_p3"))
        info["check_steering_proj_max_abs_diff"] = float(max((m["proj_S1_shipped"] - m["proj_S1_p3"]).abs().max(),
                                                            (m["proj_S2_shipped"] - m["proj_S2_p3"]).abs().max()))
    zs = pd.read_csv(Path(args.pain_axis_dir) / "results/3.2_pain_vectors/per_model" / run["model"] / "z_scores.csv")
    ours = perspective_gap(sp[sp["layer"] == run["extraction_layer"]])["pain_z"]
    info["check_3_3_pain_z_extraction_layer"] = {ds: {"ours_rebuilt_vector": ours[ds],
                                                      "shipped": float(zs.set_index("dataset").loc[ds, "pain_z"])}
                                                 for ds in ("S1_1P", "S1_3P", "S2_1P", "S2_3P")}
    info["focal_test_extraction_layer_shipped_vectors"] = shipped.get("extraction")
    (out / "summary.json").write_text(json.dumps(info, indent=2))
    figure(tab, info, out)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tab.round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in info.items() if k != "cosine_with_shipped"}, indent=1))
    print("cosine with shipped:", {k: round(v, 4) for k, v in run["cosine_with_shipped"][run["chosen_variant"]].items()})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
