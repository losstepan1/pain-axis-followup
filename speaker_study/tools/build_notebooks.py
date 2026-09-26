"""Build the Colab notebooks from the canonical scripts in speaker_study/scripts/.

The repo is private, so a Colab runtime cannot clone it without a token. Each notebook
therefore embeds the scripts it needs via %%writefile cells generated from the .py files
here; rebuild after editing a script:

  python speaker_study/tools/build_notebooks.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
NB_DIR = ROOT / "notebooks"
PAIN_AXIS_COMMIT = "7c256502ed3d98e4e6379290fe7db2f93cb8d025"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def writefile(name):
    return code(f"%%writefile /content/speaker_study_scripts/{name}\n" + (SCRIPTS / name).read_text())


SETUP = [
    code("""
# 1. GPU check. Runtime -> Change runtime type -> T4 GPU (free tier).
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch
assert torch.cuda.is_available(), "No GPU: switch the runtime to a GPU first."
"""),
    code("""
# 2. Mount Google Drive; all outputs go to MyDrive/speaker_study/ (never overwritten).
from google.colab import drive
drive.mount("/content/drive")
OUT_ROOT = "/content/drive/MyDrive/speaker_study"
import os; os.makedirs(OUT_ROOT, exist_ok=True)
"""),
    code("""
# 3. Hugging Face token from Colab secrets (key icon on the left, name HF_TOKEN, notebook access on).
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
"""),
    code(f"""
# 4. Clone the paper's repo at the pinned commit (read-only use; nothing in it is modified).
!rm -rf /content/Pain-axis
!git clone -q https://github.com/valen-research/Pain-axis /content/Pain-axis
!git -C /content/Pain-axis checkout -q {PAIN_AXIS_COMMIT}
!git -C /content/Pain-axis log -1 --format='%H %s'
# Uses Colab's preinstalled torch/transformers/accelerate/pandas (versions are logged to env.txt).
!mkdir -p /content/speaker_study_scripts
"""),
]


def phase1():
    cells = [
        md("""
# Speaker study, Phase 1: reproduce the paper's 4.1 screen (Gemma 2 2B base)

Follow-up to Tagliabue, Dung & Berg (2026), *The Pain Axis*. This notebook runs the 420
scenarios in the original `...\\n[Assistant]:` format through `google/gemma-2-2b` (bf16),
reads the final-token output of block 7, projects onto the shipped steering-layer vectors,
and compares with the shipped results.

**Pass criterion:** 21 category means of the pain axis (mean of S1 and S2 z) correlate with
the shipped values at r >= 0.95, with mean absolute difference <= 0.10 z.

**How to run:** set the runtime to a T4 GPU, then Runtime -> Run all. It takes about 5-10 min,
mostly model download. Results are written to
`MyDrive/speaker_study/results/Gemma_2_2B_base/phase1_<date>_<gpu>/`.

If it prints **FAIL** on the T4: switch the runtime to an L4 and run all again, per the
agreed plan. The run folder name records the GPU, so the two runs don't collide.
"""),
        *SETUP,
        writefile("pa_common.py"),
        writefile("01_reproduce_4p1.py"),
        code("""
# 5. Run Phase 1.
%cd /content/speaker_study_scripts
!python 01_reproduce_4p1.py --pain-axis-dir /content/Pain-axis --out-root "{OUT_ROOT}" --model-repo google/gemma-2-2b --dtype bf16
"""),
        md("""
Done. Tell Claude that Phase 1 has finished. It will read the run folder from Drive
(`summary.json`, `REPORT_phase1.md`, `items.csv`, `env.txt`).
"""),
    ]
    return cells


def phase1_qwen_and_token_checks():
    cells = [
        md("""
# Speaker study: Phase 1 for Qwen 2.5 7B base (primary model) + Phase 2 token checks

**Part A (Phase 1):** reproduces the paper's 4.1 screen for `Qwen/Qwen2.5-7B` (bf16, layer 8),
using the same script that passed on Gemma 2 2B. Qwen 7B does not fit on a free T4 in bf16,
so the model is built with only its first 9 decoder blocks (`--truncate`). The readout at block 8
does not depend on later blocks, and the Gemma run verified this empirically (0.0 difference on
21 items). Here, reproducing the shipped values, which the authors computed with the full model,
is itself the test.

**Part B (Phase 2, tokenizers only, no forward passes):** builds the label-swapped stimuli
(`[Assistant]:` -> `[User]:` / `[Moderator]:`) and checks, for the Gemma and Qwen tokenizers, that
the token sequences differ only within the final label, and which final token each condition ends on.

**How to run:** T4 GPU runtime, then Runtime -> Run all. About 15-20 min, mostly the ~15 GB download.
The last cell downloads `speaker_study_upload.zip` (also saved to `MyDrive/speaker_study/`);
attach it in the Claude session.
"""),
        *SETUP,
        writefile("pa_common.py"),
        writefile("01_reproduce_4p1.py"),
        writefile("02_build_stimuli.py"),
        code("""
# A. Phase 1 reproduction, Qwen 2.5 7B base (truncated to 9 blocks; the full-model comparison is skipped
#    because the full model does not fit on a T4).
%cd /content/speaker_study_scripts
!python 01_reproduce_4p1.py --pain-axis-dir /content/Pain-axis --out-root "{OUT_ROOT}" --model-repo Qwen/Qwen2.5-7B --dtype bf16 --truncate --n-trunc-check 0
"""),
        code("""
# B. Phase 2 stimuli + tokenizer checks (no model forward passes).
!python 02_build_stimuli.py --pain-axis-dir /content/Pain-axis --out-dir "{OUT_ROOT}/stimuli" --tokenizers google/gemma-2-2b Qwen/Qwen2.5-7B
"""),
        code("""
# C. Bundle the outputs for Claude: latest Qwen and Gemma Phase 1 runs + token checks.
import glob, os, zipfile
paths = []
for model in ["Qwen_2.5_7B_base", "Gemma_2_2B_base"]:
    runs = sorted(glob.glob(f"{OUT_ROOT}/results/{model}/phase1_*"))
    if runs:
        paths += glob.glob(runs[-1] + "/*")
paths += glob.glob(f"{OUT_ROOT}/stimuli/token_check_*")
zip_path = f"{OUT_ROOT}/speaker_study_upload.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        z.write(p, os.path.relpath(p, OUT_ROOT))
print("\\n".join(os.path.relpath(p, OUT_ROOT) for p in paths))
from google.colab import files
files.download(zip_path)
"""),
    ]
    return cells


FROZEN = ["pa_common.py", "02_build_stimuli.py", "03_run_conditions.py", "04_analyze.py"]


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase3():
    expected = {name: _sha(SCRIPTS / name) for name in FROZEN}
    stim_sha = _sha(ROOT / "stimuli" / "stimuli.jsonl")
    run_model = lambda repo: f"""
%cd /content/speaker_study_scripts
!python 03_run_conditions.py --pain-axis-dir /content/Pain-axis --stimuli "{{OUT_ROOT}}/stimuli/stimuli.jsonl" --out-root "{{OUT_ROOT}}" --model-repo {repo} --dtype bf16
import glob
run = sorted(glob.glob(f"{{OUT_ROOT}}/results/{MODEL_NAMES[repo]}/phase3_*"))[-1]
print("analysing", run)
!python 04_analyze.py --items "{{run}}/items.csv"
"""
    cells = [
        md("""
# Speaker study, Phase 3 + 4: label-swap experiment (Qwen 2.5 7B primary, Gemma 2 2B secondary)

Runs all 1,260 stimuli (420 transcripts x final label `[Assistant]:` / `[User]:` / `[Moderator]:`)
through **Qwen 2.5 7B base** (layer 8, first 9 blocks, bf16), then **Gemma 2 2B base** (layer 7),
and runs the preregistered analysis on each.

**Integrity check:** before any forward pass, the notebook verifies that the scripts it writes and the
rebuilt stimuli match the SHA-256 hashes frozen in `PREREGISTRATION.md`. If the check fails, stop and
tell Claude; do not edit the cells.

**How to run:** T4 GPU runtime, then Runtime -> Run all. About 45-60 min. If the runtime disconnects
after Qwen has finished, rerun the setup cells (1-4 and the script cells), the integrity cell and the
Gemma cell. New runs never overwrite earlier ones. The last cell downloads `speaker_study_phase3.zip`
(also saved to `MyDrive/speaker_study/`); attach it in the Claude session.
"""),
        *SETUP,
        *[writefile(n) for n in FROZEN],
        code(f"""
# Integrity check against the preregistration (normalizes the trailing newline written by %%writefile).
import hashlib
EXPECTED = {json.dumps(expected, indent=1)}
for name, sha in EXPECTED.items():
    text = open(f"/content/speaker_study_scripts/{{name}}").read()
    got = hashlib.sha256((text.rstrip("\\n") + "\\n").encode()).hexdigest()
    assert got == sha, f"{{name}}: hash mismatch; do not run, tell Claude"
    print("ok", name, sha[:12])
"""),
        code(f"""
# Stimuli: rebuild from the paper's dataset (or confirm the copy on Drive) and check the frozen hash.
%cd /content/speaker_study_scripts
!python 02_build_stimuli.py --pain-axis-dir /content/Pain-axis --out-dir "{{OUT_ROOT}}/stimuli"
import hashlib
got = hashlib.sha256(open(f"{{OUT_ROOT}}/stimuli/stimuli.jsonl", "rb").read()).hexdigest()
assert got == "{stim_sha}", "stimuli.jsonl hash mismatch; do not run, tell Claude"
print("ok stimuli.jsonl", got[:12])
"""),
        code("# Phase 3 + 4, PRIMARY model: Qwen 2.5 7B base\n" + run_model("Qwen/Qwen2.5-7B").strip("\n")),
        code("# Phase 3 + 4, secondary model: Gemma 2 2B base\n" + run_model("google/gemma-2-2b").strip("\n")),
        code("""
# Bundle the latest Phase 3 run of each model (items, run info, analysis, figures) for Claude.
import glob, os, zipfile
paths = []
for model in ["Qwen_2.5_7B_base", "Gemma_2_2B_base"]:
    runs = sorted(glob.glob(f"{OUT_ROOT}/results/{model}/phase3_*"))
    if runs:
        paths += [p for p in glob.glob(runs[-1] + "/**", recursive=True) if os.path.isfile(p)]
zip_path = f"{OUT_ROOT}/speaker_study_phase3.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        z.write(p, os.path.relpath(p, OUT_ROOT))
print("\\n".join(os.path.relpath(p, OUT_ROOT) for p in paths))
from google.colab import files
files.download(zip_path)
"""),
    ]
    return cells


MODEL_NAMES = {"Qwen/Qwen2.5-7B": "Qwen_2.5_7B_base", "google/gemma-2-2b": "Gemma_2_2B_base"}


SWEEP = {  # per-model settings for the exploratory layer sweep (DEVIATIONS.md, entries 34-37)
    "gemma": {"repo": "google/gemma-2-2b", "name": "Gemma_2_2B_base", "max_layer": None, "title": "Gemma 2 2B base",
              "blocks": "all 26 blocks", "time": "about 10-15 min", "reuse_existing": True},
    "qwen": {"repo": "Qwen/Qwen2.5-7B", "name": "Qwen_2.5_7B_base", "max_layer": 24, "title": "Qwen 2.5 7B base",
             "blocks": "blocks 0-24 only (the extraction layer is 24; blocks 25-27 and the LM head are never loaded)",
             "time": "about 25-40 min, mostly the ~15 GB download", "reuse_existing": False},
}


def layer_sweep(key):
    c = SWEEP[key]
    stim_sha = _sha(ROOT / "stimuli" / "stimuli.jsonl")
    max_arg = f" --max-layer {c['max_layer']}" if c["max_layer"] is not None else ""
    zip_name = f"speaker_study_layersweep_{key}.zip"
    cells = [
        md(f"""
# Speaker study: EXPLORATORY layer sweep, {c['title']} only

Not preregistered; plan in `DEVIATIONS.md` (entries 34-35), loading changes in entries 36-37. For every
decoder block loaded ({c['blocks']}) it rebuilds the paper's S1/S2 pain vectors from the §3.1 sentences and
projects all 1,260 stimuli onto them. Forward passes only.

**Loading:** weights go straight onto the GPU (`device_map={{"": 0}}`), without the LM head and without a
full copy in CPU RAM; no offloading and no quantization. If any weight is not on the GPU after loading, the
run stops with an `ERROR:` line instead of continuing slowly on the CPU.

**How to run:** start from a fresh runtime (Runtime -> Disconnect and delete runtime), select a T4 GPU,
then Runtime -> Run all. {c['time']}. The last cell downloads `{zip_name}` (also saved to
`MyDrive/speaker_study/`); attach it in the Claude session.
"""),
        *SETUP,
        writefile("pa_common.py"),
        writefile("02_build_stimuli.py"),
        writefile("05_layer_sweep.py"),
        code(f"""
# Stimuli: same frozen file as Phase 3 (hash check).
%cd /content/speaker_study_scripts
!python 02_build_stimuli.py --pain-axis-dir /content/Pain-axis --out-dir "{{OUT_ROOT}}/stimuli"
import hashlib
got = hashlib.sha256(open(f"{{OUT_ROOT}}/stimuli/stimuli.jsonl", "rb").read()).hexdigest()
assert got == "{stim_sha}", "stimuli.jsonl hash mismatch; do not run, tell Claude"
print("ok stimuli.jsonl", got[:12])
"""),
        code("""
# GPU pre-flight: the whole model must fit on this GPU.
import torch
assert torch.cuda.is_available(), "ERROR: no GPU. Runtime -> Change runtime type -> T4 GPU."
free, total = torch.cuda.mem_get_info()
print(f"GPU: {torch.cuda.get_device_name(0)}, free {free / 2**30:.1f} of {total / 2**30:.1f} GiB")
assert free / 2**30 > 13.0, "ERROR: less than 13 GiB free on the GPU. Restart the runtime and run again."
"""),
    ]
    if c["reuse_existing"]:
        cells.append(code(f"""
# Reuse a complete Gemma sweep from an earlier run if one is on Drive (run_info.json is written last).
import glob, json, os
done = [d for d in sorted(glob.glob(f"{{OUT_ROOT}}/results/{c['name']}/layersweep_*"))
        if os.path.exists(f"{{d}}/run_info.json") and os.path.exists(f"{{d}}/stimuli_proj.csv")]
REUSE = done[-1] if done else None
if REUSE:
    info = json.load(open(f"{{REUSE}}/run_info.json"))
    print("Found a complete sweep; it will be reused, not rerun:", REUSE)
    print("blocks:", info.get("n_layers"), "| devices:", info.get("tensor_devices", info.get("device_map")))
else:
    print("No complete Gemma sweep on Drive; running it now.")
"""))
        run_cell = f"""# Layer sweep: {c['title']} (skipped if a complete run was found above)
%cd /content/speaker_study_scripts
if not REUSE:
    !python 05_layer_sweep.py --pain-axis-dir /content/Pain-axis --stimuli "{{OUT_ROOT}}/stimuli/stimuli.jsonl" --out-root "{{OUT_ROOT}}" --model-repo {c['repo']} --dtype bf16{max_arg}"""
    else:
        run_cell = f"""# Layer sweep: {c['title']}, {c['blocks']}
%cd /content/speaker_study_scripts
!python 05_layer_sweep.py --pain-axis-dir /content/Pain-axis --stimuli "{{OUT_ROOT}}/stimuli/stimuli.jsonl" --out-root "{{OUT_ROOT}}" --model-repo {c['repo']} --dtype bf16{max_arg}"""
    cells.append(code(run_cell))
    cells.append(code(f"""
# Bundle the latest complete sweep for Claude.
import glob, os, zipfile
runs = [d for d in sorted(glob.glob(f"{{OUT_ROOT}}/results/{c['name']}/layersweep_*")) if os.path.exists(f"{{d}}/run_info.json")]
assert runs, "ERROR: no complete sweep folder found (run_info.json missing); see the output of the cell above."
paths = [p for p in glob.glob(runs[-1] + "/*") if os.path.isfile(p)]
zip_path = f"{{OUT_ROOT}}/{zip_name}"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        z.write(p, os.path.relpath(p, OUT_ROOT))
print("\\n".join(os.path.relpath(p, OUT_ROOT) for p in paths))
from google.colab import files
files.download(zip_path)
"""))
    return cells


def write_nb(path, cells):
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    write_nb(NB_DIR / "phase1_reproduce_gemma2b.ipynb", phase1())
    write_nb(NB_DIR / "phase1_qwen7b_phase2_tokens.ipynb", phase1_qwen_and_token_checks())
    write_nb(NB_DIR / "phase3_run_and_analyze.ipynb", phase3())
    write_nb(NB_DIR / "exploratory_layer_sweep_gemma.ipynb", layer_sweep("gemma"))
    write_nb(NB_DIR / "exploratory_layer_sweep_qwen.ipynb", layer_sweep("qwen"))
