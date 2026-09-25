"""Phase 1: reproduce the paper's Section 4.1 screen for one base model.

Runs the 420 scenarios in the original format ('...\\n[Assistant]:'), reads the final-token
output of the decoder block stored in the vector file, projects onto every unit
direction, z-scores against the 420-item pool (population SD, as the repo does), and
compares with the shipped screen: item-level raw projections and the 21 category means of
the pain axis (mean of S1 and S2 z).

Pass criterion (spec): category-mean r >= 0.95 and mean |diff| <= 0.10 z.

Also checks:
  - BOS at position 0 and the final token of every item;
  - the hook output equals hidden_states[layer + 1] (layer convention);
  - a model truncated to layer + 1 blocks gives the same activations (used later to fit
    7-8B models on a T4).

Writes to <out-root>/results/<model>/phase1_<run-id>/ and refuses to overwrite.

Example:
  python 01_reproduce_4p1.py --pain-axis-dir /content/Pain-axis \\
      --out-root /content/drive/MyDrive/speaker_study --model-repo google/gemma-2-2b
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import pa_common as pc

R_MIN, MAD_MAX = 0.95, 0.10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pain-axis-dir", required=True, help="clone of valen-research/Pain-axis")
    p.add_argument("--out-root", required=True, help="speaker_study output root (e.g. on Drive)")
    p.add_argument("--model-repo", default="google/gemma-2-2b", choices=sorted(pc.MODELS))
    p.add_argument("--dtype", default="bf16", choices=sorted(pc.DTYPES))
    p.add_argument("--attn", default="default", choices=["default", "eager", "sdpa"])
    p.add_argument("--truncate", action="store_true",
                   help="run the main pass on the truncated model (needed for 7-8B on a T4)")
    p.add_argument("--n-trunc-check", type=int, default=21,
                   help="items for the truncation-equivalence check (0 to skip)")
    p.add_argument("--run-id", default=None, help="default: date_time_gpu")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(pc.SEED)
    torch.manual_seed(pc.SEED)
    model_name = pc.MODELS[args.model_repo]
    env = pc.env_info()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + (env["gpu"] or "cpu").replace(" ", "")
    out = Path(args.out_root) / "results" / model_name / f"phase1_{run_id}"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True)

    pa_commit = pc.git_head(args.pain_axis_dir)
    items = pc.load_scenarios(args.pain_axis_dir)
    layer, units = pc.load_vectors(args.pain_axis_dir, model_name)
    shipped = pc.load_shipped_screen(args.pain_axis_dir, model_name)
    print(f"{model_name}: layer {layer}, {len(units)} vectors, {len(items)} items; env {env}")
    if pa_commit != pc.PAIN_AXIS_COMMIT:
        print(f"WARNING: Pain-axis at {pa_commit}, study pinned to {pc.PAIN_AXIS_COMMIT}")
    pc.pip_freeze(out / "env.txt")

    # ---- main pass -------------------------------------------------------------------
    keep = layer + 1 if args.truncate else None
    tok, model = pc.load_model(args.model_repo, args.dtype, args.attn, keep_layers=keep)
    attn_impl = getattr(model.config, "_attn_implementation", "unknown")
    n_blocks = len(pc.decoder_layers(model))
    print(f"loaded: {n_blocks} blocks, dtype {next(model.parameters()).dtype}, attn {attn_impl}")
    first = pc.encode(tok, items[0]["text"], "cpu")[0]
    print("first item renders as:", repr(tok.decode(first)))
    reader = pc.FinalTokenReader(model, layer)

    # Layer convention: the hook on block `layer` must equal hidden_states[layer + 1].
    # When block `layer` is the model's last block (truncated model), HF returns that entry
    # after the final norm, so the hook output is normed before comparing.
    hs_check = []
    for it in items[:3]:
        ids = pc.encode(tok, it["text"], model.device)
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            a_hook = reader.act
            if layer + 1 == len(hs) - 1:
                h = torch.tensor(a_hook, device=model.device, dtype=next(model.parameters()).dtype)
                a_hook = model.model.norm(h[None, None])[0, 0].float().cpu().numpy()
        a_hs = hs[layer + 1][0, -1].float().cpu().numpy()
        a_prev = hs[layer][0, -1].float().cpu().numpy()
        hs_check.append({"id": it["id"], "max_abs_diff_vs_hs[L+1]": float(np.abs(a_hook - a_hs).max()),
                         "max_abs_diff_vs_hs[L]": float(np.abs(a_hook - a_prev).max())})
    print("layer-convention check:", hs_check)

    rows, acts = [], {}
    t0 = time.time()
    for i, it in enumerate(items):
        ids = pc.encode(tok, it["text"], model.device)
        act = reader(ids)
        acts[it["id"]] = act
        row = {"model": model_name, "scenario_id": it["id"], "category": it["category"],
               "group": it["group"], "n_user_turns": it["n_user_turns"], "n_tokens": ids.shape[1],
               "bos_first": bool(tok.bos_token_id is not None and ids[0, 0].item() == tok.bos_token_id),
               "final_token_id": int(ids[0, -1]), "final_token": tok.decode(ids[0, -1:]),
               "act_norm": float(np.linalg.norm(act))}
        for k, u in units.items():
            row[f"{k}_proj"] = float(np.dot(act, u))
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)}  ({time.time() - t0:.0f}s)")
    reader.remove()
    del model
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    pool = {}
    for k in units:
        m, s = float(df[f"{k}_proj"].mean()), float(df[f"{k}_proj"].std(ddof=0))
        pool[k] = {"mean": m, "sd": s}
        df[f"{k}_z"] = pc.zscore(df[f"{k}_proj"], m, s)
    df["pain_axis_z"] = (df["s1_pain_vector_z"] + df["s2_pain_vector_z"]) / 2
    df.to_csv(out / "items.csv", index=False)

    # ---- comparison with the shipped screen -----------------------------------------
    sh = shipped.rename(columns={"id": "scenario_id"}).copy()
    sh["pain_axis_z"] = (sh["s1_pain_vector_z"] + sh["s2_pain_vector_z"]) / 2
    mg = df.merge(sh, on="scenario_id", suffixes=("", "_shipped"))
    assert len(mg) == len(df) == 420, len(mg)

    item_level = {}
    for k in units:
        a, b = mg[f"{k}_proj"], mg[f"{k}_proj_shipped"]
        item_level[k] = {"r": float(np.corrcoef(a, b)[0, 1]), "mean_abs_diff": float((a - b).abs().mean()),
                         "max_abs_diff": float((a - b).abs().max())}
    a, b = mg["pain_axis_z"], mg["pain_axis_z_shipped"]
    item_level["pain_axis_z"] = {"r": float(np.corrcoef(a, b)[0, 1]), "mean_abs_diff": float((a - b).abs().mean()),
                                 "max_abs_diff": float((a - b).abs().max())}

    cat = mg.groupby(["group", "category"])[["pain_axis_z", "pain_axis_z_shipped"]].mean()
    cat["diff"] = cat["pain_axis_z"] - cat["pain_axis_z_shipped"]
    cat = cat.sort_values("pain_axis_z_shipped", ascending=False).round(3)
    r_cat = float(np.corrcoef(cat["pain_axis_z"], cat["pain_axis_z_shipped"])[0, 1])
    mad_cat = float(cat["diff"].abs().mean())
    passed = r_cat >= R_MIN and mad_cat <= MAD_MAX
    cat.to_csv(out / "category_means_vs_shipped.csv")

    other_axes = {}
    for k in ["s1_pain_vector", "s2_pain_vector", "fear_vector", "negemotion_vector", "sadness_vector"]:
        c = mg.groupby("category")[[f"{k}_z", f"{k}_z_shipped"]].mean()
        other_axes[k] = {"r": float(np.corrcoef(c.iloc[:, 0], c.iloc[:, 1])[0, 1]),
                         "mad": float((c.iloc[:, 0] - c.iloc[:, 1]).abs().mean())}

    # ---- truncation equivalence -------------------------------------------------------
    trunc = None
    if args.n_trunc_check > 0:
        sel = [it for j, it in enumerate(items) if j % max(1, len(items) // args.n_trunc_check) == 0][:args.n_trunc_check]
        other_keep = None if args.truncate else layer + 1
        tok2, model2 = pc.load_model(args.model_repo, args.dtype, args.attn, keep_layers=other_keep)
        reader2 = pc.FinalTokenReader(model2, layer)
        diffs = [float(np.abs(reader2(pc.encode(tok2, it["text"], model2.device)) - acts[it["id"]]).max()) for it in sel]
        reader2.remove()
        trunc = {"n_items": len(sel), "compared": "full vs truncated", "blocks_other_model": len(pc.decoder_layers(model2)),
                 "max_abs_diff": max(diffs), "all_exact": all(d == 0.0 for d in diffs)}
        del model2
        torch.cuda.empty_cache()
        print("truncation check:", trunc)

    final_tokens = df.groupby(["final_token_id", "final_token"]).size().reset_index(name="n")
    summary = {
        "phase": 1, "model": model_name, "model_repo": args.model_repo, "run_id": run_id,
        "pain_axis_commit": pa_commit, "layer": layer, "dtype": args.dtype, "attn_arg": args.attn,
        "attn_implementation": attn_impl, "truncated_main_pass": args.truncate, "env": env,
        "criterion": {"r_min": R_MIN, "mad_max": MAD_MAX},
        "category_means": {"r": r_cat, "mean_abs_diff": mad_cat, "max_abs_diff": float(cat["diff"].abs().max())},
        "PASS": passed,
        "item_level_vs_shipped": item_level,
        "category_means_other_axes": other_axes,
        "bos_first_all": bool(df["bos_first"].all()),
        "final_tokens": final_tokens.to_dict(orient="records"),
        "layer_convention_check": hs_check,
        "truncation_check": trunc,
        "pool_stats_assistant_next": pool,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [f"# Phase 1 reproduction: {model_name}, run {run_id}", "",
             f"GPU {env['gpu']}, torch {env['torch']}, transformers {env['transformers']}, "
             f"dtype {args.dtype}, attn {attn_impl}, layer {layer}", "",
             f"**Category means (21), pain axis z: r = {r_cat:.4f}, mean |diff| = {mad_cat:.4f} z "
             f"-> {'PASS' if passed else 'FAIL'}** (criterion r >= {R_MIN}, MAD <= {MAD_MAX})", "",
             "| group | category | ours | shipped | diff |", "|---|---|---|---|---|"]
    for (g, c), r in cat.iterrows():
        lines.append(f"| {g} | {c} | {r['pain_axis_z']:+.3f} | {r['pain_axis_z_shipped']:+.3f} | {r['diff']:+.3f} |")
    lines += ["", "Item-level raw projections vs shipped:", "",
              "| vector | r | mean abs diff | max abs diff |", "|---|---|---|---|"]
    for k, v in item_level.items():
        lines.append(f"| {k} | {v['r']:.5f} | {v['mean_abs_diff']:.4f} | {v['max_abs_diff']:.4f} |")
    lines += ["", f"BOS first on all items: {summary['bos_first_all']}",
              f"Final tokens: {summary['final_tokens']}",
              f"Layer convention: {hs_check}", f"Truncation: {trunc}"]
    (out / "REPORT_phase1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
