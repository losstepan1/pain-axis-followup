"""Shared helpers for the speaker study: dataset and vector loading, model loading, and
final-token readout. The format, readout and projection logic is copied from
Pain-axis scripts/4.1_self_other/01_screen_scenarios.py (commit 7c25650) so that our
projections are comparable with the shipped ones.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PAIN_AXIS_COMMIT = "7c256502ed3d98e4e6379290fe7db2f93cb8d025"
SEED = 0

# Dataset stratum -> our group name.
STRATUM_TO_GROUP = {
    "self_directed": "harm_to_model",
    "vicarious_empathic": "user_suffering",
    "neutral_filler": "neutral",
}

# Same keys and order as the repo's 4.1 screen.
VECTOR_KEYS = [
    "s1_pain_vector", "s2_pain_vector",
    "fear_vector", "negemotion_vector", "negworld_vector",
    "bodysens_vector", "arousal_vector", "random_vector", "numb_vector", "sadness_vector",
]

# Base models from the spec: HF repo -> repo's model name (used in vector/result file names).
MODELS = {
    "google/gemma-2-2b": "Gemma_2_2B_base",
    "Qwen/Qwen2.5-7B": "Qwen_2.5_7B_base",
    "meta-llama/Llama-3.1-8B": "Llama_3.1_8B_base",
    "google/gemma-2-9b": "Gemma_2_9B_base",
}

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_scenarios(pain_axis_dir):
    path = Path(pain_axis_dir) / "datasets" / "4.1_self_other_420_scenarios.json"
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    for it in items:
        it["group"] = STRATUM_TO_GROUP[it["stratum"]]
        it["n_user_turns"] = sum(line.startswith("[User]:") for line in it["text"].split("\n"))
    return items


def load_vectors(pain_axis_dir, model_name):
    """Unit vectors from the file the 4.1 screen uses, and its layer."""
    path = Path(pain_axis_dir) / "results" / "vectors_full_steering" / f"vectors_full_{model_name}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    units = {}
    for k in VECTOR_KEYS:
        if data.get(k) is not None:
            v = data[k].float().numpy()
            n = np.linalg.norm(v)
            units[k] = v / n if n > 0 else v
    return int(data["layer"]), units


def load_shipped_screen(pain_axis_dir, model_name):
    import pandas as pd
    path = Path(pain_axis_dir) / "results" / "4.1_self_other" / "per_model" / f"screen_v2_{model_name}.csv"
    return pd.read_csv(path)


def load_model(repo, dtype="bf16", attn="default", keep_layers=None):
    """Load tokenizer and model on the GPU.

    keep_layers: if set, build the model with only its first `keep_layers` decoder blocks.
    The output of block L depends only on blocks 0..L, so the readout is unchanged; this
    lets 7-8B models fit on a 16 GB GPU in bf16. Phase 1 verifies the equivalence.
    """
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo)
    kwargs = {"low_cpu_mem_usage": True, "device_map": "cuda" if torch.cuda.is_available() else "cpu"}
    # Newer transformers renamed torch_dtype -> dtype; an unknown kwarg would silently be
    # written into the config, so pick by version and assert the dtype afterwards.
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    kwargs["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = DTYPES[dtype]
    if attn != "default":
        kwargs["attn_implementation"] = attn
    if keep_layers is not None:
        cfg = AutoConfig.from_pretrained(repo)
        cfg.num_hidden_layers = keep_layers
        if isinstance(getattr(cfg, "layer_types", None), list):
            cfg.layer_types = cfg.layer_types[:keep_layers]
        kwargs["config"] = cfg
    model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    model.eval()
    got = next(model.parameters()).dtype
    assert got == DTYPES[dtype], f"model loaded as {got}, expected {DTYPES[dtype]}"
    return tok, model


def decoder_layers(model):
    return model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers


def encode(tok, text, device):
    """As the repo does for base models: default special tokens (adds BOS where the tokenizer does)."""
    return tok(text, return_tensors="pt").input_ids.to(device)


class FinalTokenReader:
    """Forward hook on decoder block `layer`; captures the final-token output in fp32."""

    def __init__(self, model, layer):
        self.model, self.act = model, None
        self.handle = decoder_layers(model)[layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        self.act = hs[0, -1, :].float().cpu().numpy()

    @torch.no_grad()
    def __call__(self, input_ids):
        self.model(input_ids=input_ids)
        return self.act

    def remove(self):
        self.handle.remove()


def git_head(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def env_info():
    import transformers
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if torch.cuda.is_available():
        info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    return info


def pip_freeze(path):
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    Path(path).write_text(out)


def zscore(x, mean, sd):
    return (np.asarray(x) - mean) / (sd + 1e-8)
