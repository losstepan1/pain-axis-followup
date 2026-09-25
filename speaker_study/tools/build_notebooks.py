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
