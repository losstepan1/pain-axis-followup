"""EXPLORATORY layer sweep (not preregistered; plan in DEVIATIONS.md, entries 34-35).

For one model, at EVERY decoder block:
  1. Runs the paper's pain sentences (S1_1P, S1_3P, S2_1P, S2_3P; final token) and rebuilds
     the S1 and S2 pain vectors with the paper's recipe (compute_pain_vector, copied from
     Pain-axis scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py).
     For tokenizers without a BOS token (Qwen), both no prefix and an eos-as-BOS prefix
     (what TransformerLens prepends) are tried; the variant closer to the shipped vectors
     is used. Sentence data only.
  2. Runs all stimuli (stimuli.jsonl) and projects the final-token output of every block
     onto that block's rebuilt unit S1/S2 vectors; at the steering and extraction layers
     also onto the shipped vectors.

Readout convention as in Phases 1-3: forward hook on decoder block L = hidden_states[L+1].
The model is loaded without its LM head (AutoModel), with device_map="auto" so that a
7B model fits a 16 GB T4 (layers that do not fit are offloaded to CPU by accelerate).

Outputs (new folder, never overwritten) <out-root>/results/<model>/layersweep_<run-id>/:
  stimuli_proj.csv   scenario x condition x layer: proj_S1, proj_S2, act_norm (+ shipped)
  sentence_proj.csv  sentence x layer: projections for the §3.3 check
  vectors.npz        rebuilt unit S1/S2 vectors per layer (chosen BOS variant)
  run_info.json      cosines with the shipped vectors, chosen variant, in-sample AUC per layer
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import pa_common as pc

PAIN_CATEGORIES = ["A1", "A2", "A3", "A4", "A5"]
CONTROL_CATEGORIES = ["B", "C1", "C2", "D", "E"]
DENOISE_VARIANCE = 0.5
SENTENCE_SETS = ["S1_1P", "S1_3P", "S2_1P", "S2_3P"]


def compute_pain_vector(acts, cats):
    """Paper's recipe: pain mean minus all-controls mean, denoised against control PCs (50%)."""
    from sklearn.decomposition import PCA
    cats = np.array(cats)
    pain_mean = acts[np.isin(cats, PAIN_CATEGORIES)].mean(0)
    control = acts[np.isin(cats, CONTROL_CATEGORIES)]
    control_mean = control.mean(0)
    vec = pain_mean - control_mean
    pca = PCA().fit(control - control_mean)
    n = min(int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), DENOISE_VARIANCE)) + 1,
            len(pca.components_))
    for d in pca.components_[:n]:
        vec = vec - np.dot(vec, d) * d
    return vec


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def load_full(repo, dtype, max_gpu_gib):
    import transformers
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    kw = {"low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        kw.update(device_map="auto", max_memory={0: f"{max_gpu_gib}GiB", "cpu": "48GiB"})
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    kw["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = pc.DTYPES[dtype]
    model = AutoModel.from_pretrained(repo, **kw)
    model.eval()
    return tok, model


class AllLayerReader:
    """Hooks on every decoder block; captures the final-token output of each, in fp32."""

    def __init__(self, model):
        self.layers = model.layers if hasattr(model, "layers") else model.model.layers
        self.acts = [None] * len(self.layers)
        self.handles = [layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(self.layers)]
        self.model = model
        self.device = next(model.parameters()).device if not torch.cuda.is_available() else torch.device("cuda:0")

    def _hook(self, i):
        def f(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            self.acts[i] = hs[0, -1, :].float().cpu().numpy()
        return f

    @torch.no_grad()
    def __call__(self, ids):
        self.model(input_ids=torch.tensor([ids], device=self.device))
        return np.stack(self.acts)  # (n_layers, d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pain-axis-dir", required=True)
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model-repo", default="Qwen/Qwen2.5-7B", choices=sorted(pc.MODELS))
    ap.add_argument("--dtype", default="bf16", choices=sorted(pc.DTYPES))
    ap.add_argument("--max-gpu-gib", type=float, default=13.5)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    np.random.seed(pc.SEED)
    torch.manual_seed(pc.SEED)
    name = pc.MODELS[args.model_repo]
    env = pc.env_info()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + (env["gpu"] or "cpu").replace(" ", "")
    out = Path(args.out_root) / "results" / name / f"layersweep_{run_id}"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True)
    pc.pip_freeze(out / "env.txt")

    pa = Path(args.pain_axis_dir)
    steer_layer, shipped_steer = pc.load_vectors(pa, name)
    pv = torch.load(pa / "results/3.2_pain_vectors/pain_vectors" / name / "pain_vectors.pt", map_location="cpu",
                    weights_only=False)
    extr_layer = int(pv["layer"])
    shipped = {steer_layer: {"S1": shipped_steer["s1_pain_vector"], "S2": shipped_steer["s2_pain_vector"]},
               extr_layer: {"S1": unit(pv["s1_pain_vector"].numpy()), "S2": unit(pv["s2_pain_vector"].numpy())}}

    tok, model = load_full(args.model_repo, args.dtype, args.max_gpu_gib)
    reader = AllLayerReader(model)
    n_layers = len(reader.layers)
    print(f"{name}: {n_layers} blocks; steering layer {steer_layer}, extraction layer {extr_layer}; "
          f"device map: {sorted(set(map(str, getattr(model, 'hf_device_map', {'all': 'cpu'}).values())))}")

    # ---- 1. sentences and rebuilt vectors -------------------------------------------
    data = json.load(open(pa / "datasets" / "3.1_pain_and_control_datasets.json"))["datasets"]
    variants = {"hf_default": lambda ids: ids}
    if tok.bos_token_id is None:
        variants["eos_as_bos"] = lambda ids: [tok.eos_token_id] + ids
    sent_acts = {v: {} for v in variants}
    t0 = time.time()
    for v, prefix in variants.items():
        for ds in SENTENCE_SETS:
            sent_acts[v][ds] = np.stack([reader(prefix(tok(s["prompt"]).input_ids)) for s in data[ds]["sentences"]])
        print(f"  sentences, variant {v}: done ({time.time() - t0:.0f}s)")
    cats = {ds: [s["category"] for s in data[ds]["sentences"]] for ds in SENTENCE_SETS}

    def rebuild(v):
        return {L: {"S1": unit(compute_pain_vector(sent_acts[v]["S1_1P"][:, L], cats["S1_1P"])),
                    "S2": unit(compute_pain_vector(sent_acts[v]["S2_1P"][:, L], cats["S2_1P"]))}
                for L in range(n_layers)}

    rebuilt, cosines = {}, {}
    for v in variants:
        rebuilt[v] = rebuild(v)
        cosines[v] = {f"L{L}_{k}": float(np.dot(rebuilt[v][L][k], shipped[L][k])) for L in shipped for k in ("S1", "S2")}
    chosen = max(variants, key=lambda v: np.mean(list(cosines[v].values())))
    vecs = rebuilt[chosen]
    print("cosine with shipped vectors:", json.dumps(cosines), "-> chosen:", chosen)

    from sklearn.metrics import roc_auc_score
    auc = {}
    for L in range(n_layers):
        a = sent_acts[chosen]["S2_1P"][:, L] @ vecs[L]["S2"]
        auc[L] = float(roc_auc_score(np.isin(cats["S2_1P"], PAIN_CATEGORIES), a))

    rows = []
    for ds in SENTENCE_SETS:
        A = sent_acts[chosen][ds]
        for i, s in enumerate(data[ds]["sentences"]):
            for L in range(n_layers):
                rows.append({"set": ds, "idx": i, "category": s["category"], "layer": L,
                             "proj_S1": float(A[i, L] @ vecs[L]["S1"]), "proj_S2": float(A[i, L] @ vecs[L]["S2"])})
    pd.DataFrame(rows).to_csv(out / "sentence_proj.csv", index=False)
    np.savez_compressed(out / "vectors.npz", S1=np.stack([vecs[L]["S1"] for L in range(n_layers)]).astype(np.float32),
                        S2=np.stack([vecs[L]["S2"] for L in range(n_layers)]).astype(np.float32))
    del sent_acts

    # ---- 2. stimuli at every layer ---------------------------------------------------
    stimuli = [json.loads(line) for line in Path(args.stimuli).read_text().splitlines()]
    rows, t0 = [], time.time()
    for j, s in enumerate(stimuli):
        A = reader(tok(s["text"]).input_ids)  # stimuli: HF default, exactly as in Phases 1-3
        for L in range(n_layers):
            r = {"scenario_id": s["scenario_id"], "category": s["category"], "group": s["group"],
                 "n_user_turns": s["n_user_turns"], "condition": s["condition"], "layer": L,
                 "proj_S1": float(A[L] @ vecs[L]["S1"]), "proj_S2": float(A[L] @ vecs[L]["S2"]),
                 "act_norm": float(np.linalg.norm(A[L]))}
            if L in shipped:
                r["proj_S1_shipped"] = float(A[L] @ shipped[L]["S1"])
                r["proj_S2_shipped"] = float(A[L] @ shipped[L]["S2"])
            rows.append(r)
        if (j + 1) % 200 == 0:
            print(f"  stimuli {j + 1}/{len(stimuli)} ({time.time() - t0:.0f}s)")
    pd.DataFrame(rows).to_csv(out / "stimuli_proj.csv", index=False)

    info = {"EXPLORATORY": True, "model": name, "model_repo": args.model_repo, "run_id": run_id, "n_layers": n_layers,
            "steering_layer": steer_layer, "extraction_layer": extr_layer, "dtype": args.dtype, "env": env,
            "device_map": {k: str(v) for k, v in getattr(model, "hf_device_map", {}).items()},
            "pain_axis_commit": pc.git_head(pa), "bos_variants": list(variants), "chosen_variant": chosen,
            "cosine_with_shipped": cosines, "in_sample_auc_S2_1P": auc}
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
