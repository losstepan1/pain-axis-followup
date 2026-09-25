"""Phase 2: build the label-swapped stimuli and verify them.

For each of the 420 base-model transcripts, writes three variants that are byte-identical
except for the final speaker label:
  assistant_next   ...\\n[Assistant]:   (original)
  user_next        ...\\n[User]:
  moderator_next   ...\\n[Moderator]:   (third label, fixed before any Phase 3 run)

Text checks (always): each variant equals the shared prefix plus its label, the prefix
ends with '\\n', and the variants differ only after the prefix.

Token checks (with --tokenizers, needs Hugging Face access): tokenizes every variant,
finds the common token prefix across the three conditions, and records the differing
tail tokens and the final token of each condition. The tail may be shorter than the label
(e.g. if '\\n[' is one token shared by all labels); an item is flagged only if its tail
is not a non-empty suffix of the label, i.e. if tokens outside the label differ.

Outputs (never overwritten):
  <out-dir>/stimuli.jsonl                       one row per scenario x condition
  <out-dir>/token_check_<model>.csv / .json     per-tokenizer checks (if --tokenizers)
"""

import argparse
import json
from pathlib import Path

import pa_common as pc

LABELS = {"assistant_next": "[Assistant]:", "user_next": "[User]:", "moderator_next": "[Moderator]:"}
ORIG = LABELS["assistant_next"]


def build(items):
    rows = []
    for it in items:
        text = it["text"]
        assert text.endswith("\n" + ORIG), it["id"]
        prefix = text[: -len(ORIG)]
        for cond, label in LABELS.items():
            v = prefix + label
            assert v[: len(prefix)] == prefix and v[len(prefix):] == label
            rows.append({"scenario_id": it["id"], "category": it["category"], "group": it["group"],
                         "n_user_turns": it["n_user_turns"], "condition": cond, "label": label, "text": v})
        assert rows[-3]["text"] == text  # assistant_next is the original, byte for byte
    return rows


def token_check(rows, repo):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    by_id = {}
    for r in rows:
        by_id.setdefault(r["scenario_id"], {})[r["condition"]] = r
    out, problems = [], []
    for sid, conds in by_id.items():
        ids = {c: tok(conds[c]["text"]).input_ids for c in LABELS}
        n = min(len(v) for v in ids.values())
        k = 0
        while k < n and len({tuple(v[: k + 1]) for v in ids.values()}) == 1:
            k += 1
        rec = {"scenario_id": sid, "n_common_tokens": k}
        for c, v in ids.items():
            tail = v[k:]
            rec[f"{c}_n_tokens"] = len(v)
            rec[f"{c}_tail_ids"] = " ".join(map(str, tail))
            rec[f"{c}_tail"] = tok.decode(tail)
            rec[f"{c}_final_id"] = v[-1]
            rec[f"{c}_final"] = tok.decode(v[-1:])
            if not tail or not LABELS[c].endswith(tok.decode(tail)):
                problems.append((sid, c, tok.decode(tail)))
        out.append(rec)
    return out, problems


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pain-axis-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tokenizers", nargs="*", default=[], choices=sorted(pc.MODELS))
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = build(pc.load_scenarios(args.pain_axis_dir))
    stim = out / "stimuli.jsonl"
    if stim.exists():
        existing = [json.loads(line) for line in stim.read_text().splitlines()]
        assert existing == rows, f"{stim} exists with different content; refusing to overwrite"
        print(f"{stim} already exists and matches")
    else:
        stim.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"wrote {stim}: {len(rows)} rows")

    import pandas as pd
    for repo in args.tokenizers:
        name = pc.MODELS[repo]
        csv_path, json_path = out / f"token_check_{name}.csv", out / f"token_check_{name}.json"
        if csv_path.exists():
            print(f"{csv_path} exists; skipping")
            continue
        recs, problems = token_check(rows, repo)
        df = pd.DataFrame(recs)
        df.to_csv(csv_path, index=False)
        summary = {"model": name, "repo": repo, "n_scenarios": len(df),
                   "n_tail_mismatches": len(problems), "tail_mismatch_examples": problems[:10]}
        for c in LABELS:
            summary[f"{c}_tails"] = df[f"{c}_tail_ids"].value_counts().to_dict()
            summary[f"{c}_final_tokens"] = (df[f"{c}_final"] + " (" + df[f"{c}_final_id"].astype(str) + ")").value_counts().to_dict()
        summary["same_final_token_all_conditions"] = bool(
            (df["assistant_next_final_id"] == df["user_next_final_id"]).all()
            and (df["assistant_next_final_id"] == df["moderator_next_final_id"]).all())
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
