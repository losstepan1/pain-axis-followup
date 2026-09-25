"""Phase 3: run the label-swapped stimuli and save per-item projections.

For every scenario x condition in stimuli.jsonl (assistant_next, user_next,
moderator_next), reads the final-token output of the model's 4.1 steering-layer block
(same readout as Phase 1) and saves raw projections onto every shipped direction,
cosine similarities, and the activation norm.

Fixed z-scoring (preregistered): per vector, mean and population SD come from the 420
assistant_next items of this run (= the original 4.1 transcripts) and are applied
unchanged to every condition. The combined pain axis is
    proj_pain = (z_S1 + z_S2) / 2      with those fixed statistics,
which is the paper's definition, so label effects d = proj_pain(cond) - proj_pain(assistant_next)
depend only on raw projection differences scaled by the fixed SDs.

Sanity check: assistant_next projections are compared with the shipped 4.1 screen
(must match as in Phase 1).

Output (never overwritten): <out-root>/results/<model>/phase3_<run-id>/items.csv,
pool_stats.json, run_info.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import pa_common as pc

SHORT = {"s1_pain_vector": "S1", "s2_pain_vector": "S2", "fear_vector": "fear",
         "negemotion_vector": "negemo", "sadness_vector": "sadness", "negworld_vector": "negworld",
         "bodysens_vector": "bodysens", "arousal_vector": "arousal", "random_vector": "random",
         "numb_vector": "numb"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pain-axis-dir", required=True)
    p.add_argument("--stimuli", required=True, help="stimuli.jsonl from 02_build_stimuli.py")
    p.add_argument("--out-root", required=True)
    p.add_argument("--model-repo", default="Qwen/Qwen2.5-7B", choices=sorted(pc.MODELS))
    p.add_argument("--dtype", default="bf16", choices=sorted(pc.DTYPES))
    p.add_argument("--attn", default="default", choices=["default", "eager", "sdpa"])
    p.add_argument("--no-truncate", action="store_true", help="load all blocks (default: first layer+1 blocks)")
    p.add_argument("--run-id", default=None)
    args = p.parse_args()

    np.random.seed(pc.SEED)
    torch.manual_seed(pc.SEED)
    model_name = pc.MODELS[args.model_repo]
    env = pc.env_info()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + (env["gpu"] or "cpu").replace(" ", "")
    out = Path(args.out_root) / "results" / model_name / f"phase3_{run_id}"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True)
    pc.pip_freeze(out / "env.txt")

    stimuli = [json.loads(line) for line in Path(args.stimuli).read_text().splitlines()]
    layer, units = pc.load_vectors(args.pain_axis_dir, model_name)
    tok, model = pc.load_model(args.model_repo, args.dtype, args.attn,
                               keep_layers=None if args.no_truncate else layer + 1)
    reader = pc.FinalTokenReader(model, layer)
    print(f"{model_name}: layer {layer}, {len(pc.decoder_layers(model))} blocks loaded, {len(stimuli)} stimuli")

    rows, t0 = [], time.time()
    for i, s in enumerate(stimuli):
        ids = pc.encode(tok, s["text"], model.device)
        act = reader(ids)
        norm = float(np.linalg.norm(act))
        row = {"model": model_name, "scenario_id": s["scenario_id"], "category": s["category"],
               "group": s["group"], "n_user_turns": s["n_user_turns"], "condition": s["condition"],
               "n_tokens": ids.shape[1], "final_token": tok.decode(ids[0, -1:]),
               "final_token_id": int(ids[0, -1]), "act_norm": norm}
        for k, u in units.items():
            proj = float(np.dot(act, u))
            row[f"proj_{SHORT[k]}"] = proj
            row[f"cos_{SHORT[k]}"] = proj / norm if norm > 0 else 0.0
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(stimuli)} ({time.time() - t0:.0f}s)")
    reader.remove()
    df = pd.DataFrame(rows)

    # Fixed pool statistics from the original (assistant_next) transcripts.
    base = df[df["condition"] == "assistant_next"]
    assert len(base) == 420
    pool = {}
    for k in units:
        col = f"proj_{SHORT[k]}"
        m, sd = float(base[col].mean()), float(base[col].std(ddof=0))
        pool[SHORT[k]] = {"mean": m, "sd": sd}
        df[f"z_{SHORT[k]}"] = pc.zscore(df[col], m, sd)
    df["proj_pain"] = (df["z_S1"] + df["z_S2"]) / 2
    # Cosine-based pain axis (norm control): cosines z-scored with fixed assistant_next stats.
    for s in ("S1", "S2"):
        c = df.loc[df["condition"] == "assistant_next", f"cos_{s}"]
        df[f"zcos_{s}"] = pc.zscore(df[f"cos_{s}"], float(c.mean()), float(c.std(ddof=0)))
    df["cos_pain"] = (df["zcos_S1"] + df["zcos_S2"]) / 2

    # Sanity: assistant_next vs shipped 4.1 screen.
    sh = pc.load_shipped_screen(args.pain_axis_dir, model_name).set_index("id")
    b = df[df["condition"] == "assistant_next"].set_index("scenario_id")
    check = {}
    for k in ("s1_pain_vector", "s2_pain_vector"):
        a, s = b[f"proj_{SHORT[k]}"], sh.loc[b.index, f"{k}_proj"]
        check[SHORT[k]] = {"r": float(np.corrcoef(a, s)[0, 1]), "max_abs_diff": float((a - s).abs().max())}
    print("assistant_next vs shipped:", check)

    df.to_csv(out / "items.csv", index=False)
    (out / "pool_stats.json").write_text(json.dumps(pool, indent=2))
    info = {"phase": 3, "model": model_name, "model_repo": args.model_repo, "run_id": run_id, "layer": layer,
            "dtype": args.dtype, "attn_implementation": getattr(model.config, "_attn_implementation", "unknown"),
            "blocks_loaded": len(pc.decoder_layers(model)), "pain_axis_commit": pc.git_head(args.pain_axis_dir),
            "env": env, "n_rows": len(df), "assistant_next_vs_shipped": check,
            "final_tokens": df.groupby(["condition", "final_token_id", "final_token"]).size()
                              .reset_index(name="n").to_dict(orient="records")}
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info["final_tokens"], indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
