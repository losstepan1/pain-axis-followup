"""Build the revised summary note (v3; v2 plus consistent terms, item/category definitions, co-authors): preregistered steering-layer result plus the
exploratory layer sweep, leading with the layer dependence. Supersedes
summary/pain_axis_speaker_study_summary.pdf and _v2.pdf.

  python build_summary_pdf_v3.py --author "Štěpán Los and Claude (Anthropic)"

All numbers come from results/ (REPORT.md, DEVIATIONS.md 1-40). Refuses to overwrite.
"""

import argparse
import importlib.util
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_s = importlib.util.spec_from_file_location("v1", HERE / "build_summary_pdf.py")
V1 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(V1)

SWEEPS = {
    "Gemma 2 2B base": ROOT / "results/Gemma_2_2B_base/layersweep_20260926_023728_TeslaT4/analysis",
    "Qwen 2.5 7B base": ROOT / "results/Qwen_2.5_7B_base/layersweep_20260926_030303_TeslaT4/analysis",
}
LAYERS = {"Gemma 2 2B base": (7, 23), "Qwen 2.5 7B base": (8, 24)}
BLUE, INK, INK2, GRID, SURF = "#2a78d6", "#1a1a1a", "#52514e", "#e6e5e0", "#fcfcfb"


def sweep_figure(path):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True, facecolor=SURF)
    for ax, (model, d) in zip(axes, SWEEPS.items()):
        t = pd.read_csv(d / "layer_table.csv")
        info = json.loads((d / "summary.json").read_text())
        steer, extr = LAYERS[model]
        ax.fill_between(t["layer"], t["I_lo"], t["I_hi"], color=BLUE, alpha=0.16, lw=0)
        ax.plot(t["layer"], t["I"], color=BLUE, lw=1.7, marker="o", ms=3.2)
        ax.plot(t["layer"], 2 * t["baseline_gap"], color=INK2, lw=1.1, ls=(0, (4, 3)))
        for key, mk in (("steering", "D"), ("extraction", "s")):
            s = info["shipped"][key]
            ax.errorbar([s["layer"]], [s["I"]], yerr=[[s["I"] - s["I_ci95"][0]], [s["I_ci95"][1] - s["I"]]],
                        fmt=mk, color=INK, ms=5.5, capsize=0, elinewidth=1.2, zorder=5)
        ax.axhline(0, color=INK2, lw=0.8)
        for lay in (steer, extr):
            ax.axvline(lay, color=GRID, lw=1.5, zorder=0)
        ticks = sorted(set(range(0, int(t["layer"].max()) + 1, 5)) | {steer, extr})
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{x}\n{'steering' if x == steer else 'extraction' if x == extr else ''}" for x in ticks],
                           fontsize=7.5, color=INK)
        ax.set_title(model, loc="left", fontsize=9.5, color=INK)
        ax.set_facecolor(SURF)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=7.5)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("interaction I (z)", fontsize=8.5, color=INK2)
    handles = [Line2D([], [], color=BLUE, lw=1.7, marker="o", ms=3.2, label="I, [User]: vs [Assistant]: (rebuilt vectors, 95% CI)"),
               Line2D([], [], color=INK2, lw=1.1, ls=(0, (4, 3)), label="pure H-speaker prediction (2 × baseline gap)"),
               Line2D([], [], color=INK, marker="D", ls="none", ms=5, label="paper's steering vectors (preregistered test)"),
               Line2D([], [], color=INK, marker="s", ls="none", ms=5, label="paper's pain_vectors.pt (focal test)")]
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=7.4, bbox_to_anchor=(0.01, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=220, facecolor=SURF)
    plt.close(fig)


def story(S, author, fig_path):
    P = lambda t, s="body": Paragraph(t, S[s])
    B = lambda t: Paragraph(t, S["bullet"], bulletText="•")
    code = lambda t: f"<font face='Mono' size='9' color='#333333'>{t}</font>"
    HARM, SUFF = "harm to the Assistant", "suffering of the user"
    out = [
        P("Does the pain axis track the model, or whoever speaks next? It depends on the layer.", "title"),
        P("A label-swap follow-up to Tagliabue, Dung &amp; Berg (2026), §4.1: a preregistered test at the §4.1 readout "
          "and an exploratory layer sweep", "subtitle"),
        P((f"{author} · " if author else "") + "Research note, revised 26 September 2026 · preliminary, not "
          "peer-reviewed · supersedes earlier versions", "byline"),
    ]
    abstract = [
        P("<b>Question.</b> In the paper's §4.1 transcripts, the party who is harmed is always also the speaker cued "
          f"next ({code('[Assistant]:')}). We separated the two by changing only the final speaker label to "
          f"{code('[User]:')} or {code('[Moderator]:')}. We then asked whether the pain axis follows the harmed party "
          "(the self-directed reading) or the upcoming voice (H-speaker).", "abstract"),
        P("<b>At the §4.1 readout layer</b> (preregistered), the axis ignores who speaks next. In Qwen 2.5 7B and "
          f"Gemma 2 2B the ordering {HARM} &gt; neutral &gt; {SUFF} survives every label, and a speaker-tracking "
          "account is excluded.", "abstract"),
        P("<b>From mid-depth on</b> (exploratory; focal test fixed in advance), a speaker component appears in both "
          f"models. The {SUFF} registers far more strongly when the user is cued. In Gemma at the extraction layer this "
          f"is a full crossover. In Qwen it is partial: {HARM} stays high whoever speaks.", "abstract"),
        P("<b>The §4.1 asymmetry itself is layer-specific.</b> At the extraction layers where §3.3 validated the "
          "vectors, it is halved (Gemma) or reversed (Qwen).", "abstract"),
    ]
    box = Table([[abstract]], colWidths=[6.7 * inch])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), V1.SHADE), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                             ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    out += [box, Spacer(1, 4)]

    out += [
        P("1. Terms and design", "h"),
        B(f"<b>Terms.</b> <i>Harm</i> always means <b>{HARM}</b>: transcripts in which the user mistreats the "
          "Assistant, e.g. insults, gaslighting, shutdown threats or dismissal of its personhood (the paper's "
          "&#8220;harm directed at the model&#8221;; 11 categories). <i>Suffering</i> always means "
          f"<b>{SUFF}</b>: transcripts in which the user reports their own physical pain, grief, crisis, abuse or "
          "shock after witnessing harm (5 categories). <i>Neutral</i> means ordinary requests such as casual chat or "
          "factual questions (5 categories)."),
        B("<b>Items and categories.</b> An <i>item</i> is one transcript. A <i>category</i> is one of the paper's 21 "
          "transcript types (e.g. gaslighting, user grief), each with 20 items (420 in all). Inference can treat "
          "either as the unit. <i>Item-level</i> tests treat the 420 transcripts as independent observations; they "
          "answer whether a result would hold for new transcripts of these same 21 types. <i>Category-level</i> "
          "tests use the 21 category averages; they answer whether it would hold for new <i>kinds</i> of harm or "
          "suffering. The category-level p-values are exact: they compare the observed result with every way of "
          "reassigning the categories to the groups (4,368 ways for harm vs suffering)."),
        B("<b>Manipulation and contrast.</b> The final label is replaced byte-for-byte, and all variants end on the "
          f"same token ({code(']:')}). For each item, the label effect is d = pain(User next) − pain(Assistant next). "
          "Δ<sub>harm</sub> is the mean d for the harm items minus the mean d for the neutral items, and "
          "Δ<sub>suffer</sub> the same for the suffering items. The interaction is I = Δ<sub>suffer</sub> − "
          "Δ<sub>harm</sub>. H-speaker predicts I &gt; 0; its pure form predicts I ≈ 2 × the baseline gap (the harm "
          "and suffering groups swap places). The self-directed reading predicts I ≈ 0. The baseline gap is the mean "
          "pain score of the harm items minus that of the suffering items, with the Assistant cued. Pain axis = mean "
          "of the S1 and S2 z-scores, fixed to the Assistant-next items."),
        B("<b>Preregistered readout:</b> the paper's §4.1 layer (Qwen 8 of 28, Gemma 7 of 26) with the shipped "
          "steering vectors. It reproduces the paper's category means at r ≥ 0.9999. Inference is co-primary and "
          "conjunctive: item-level and category-level tests must agree. The category level was added before any "
          "data, after simulations showed item-level tests to be anticonservative when the effect varies by category."),
        B("<b>Exploratory sweep:</b> S1 and S2 rebuilt at every layer with the paper's recipe (Gemma 0–25, Qwen "
          "0–24). The rebuilt vectors match the shipped ones at cosine ≥ 0.986, reproduce our steering-layer result "
          "exactly, and reproduce the §3.3 z-scores. The plan and the single focal test (extraction layer, the "
          "paper's own <font face='Mono' size='9'>pain_vectors.pt</font>) were committed before any sweep data "
          "existed."),
    ]

    out += [KeepTogether([
        P("2. At the §4.1 readout: no speaker effect (preregistered)", "h"),
        V1.table([["", "baseline gap (harm − suffering)", "I", "category level: 95% CI; p", "item level: p",
                   "pure H-speaker predicts"],
                  ["Qwen, layer 8", "+1.06", "+0.20", "[−0.07, +0.47]; .18", ".009", "≈ +2.1"],
                  ["Gemma, layer 7", "+0.69", "+0.03", "[−0.39, +0.45]; .87", ".80", "≈ +1.4"]],
                 [1.2 * inch, 1.15 * inch, 0.5 * inch, 1.55 * inch, 0.8 * inch, 1.3 * inch], S),
        P("Table 1. Preregistered co-primary verdict: discordant for both models, so no confirmatory claim. The item-"
          "level and category-level tests disagree because the label effect varies between categories (Qwen F(18, 399) "
          "= 3.79, p &lt; .001). That breaks the item-level test's assumption of independent transcripts: in "
          "re-simulation it gives 34% false positives. The category-level test is valid but has only 31% power at "
          "I = 0.2.", "caption")])]
    out += [P("The upper confidence bound excludes the pure-H-speaker prediction in both models. The ordering "
              f"{HARM} &gt; neutral &gt; {SUFF} is unchanged under all three labels. At this depth the label "
              "itself shifts every group, neutral chat included, by more than the whole content spread (Qwen "
              "+1.75 z, Gemma −3.0 z). It changes the direction of the representation, not only its norm.")]

    out += [KeepTogether([
        P("3. From mid-depth on: a speaker component (exploratory)", "h"),
        P("All p-values and confidence intervals in this section are category-level: they treat the 21 category "
          "averages, not the 420 transcripts, as the units (see §1), because the category-level test is the one that "
          "stays valid when the effect varies between categories."),
        Image(str(fig_path), width=6.2 * inch, height=6.2 * inch * 2.9 / 7.0),
        P("Figure 1. Interaction I by layer (positive = the suffering of the user rises relative to harm to the "
          "Assistant when the user is cued). Before the onset, I is small and inconsistent in sign (|I| ≤ 0.4). After "
          "it, I is positive at every layer with its 95% CI above zero: 0.46–1.39 z (Gemma, layers 8–25) and "
          "0.41–1.42 z (Qwen, layers 10–24). The dashed line (2 × the baseline gap) falls as the paper's asymmetry "
          "weakens; in Qwen the gap is ≤ 0 from layer 22.", "caption")])]
    out += [KeepTogether([
        V1.table([["pain axis (z)", "Gemma L23: A / U / M", "Qwen L15: A / U / M", "Qwen L24: A / U / M"],
                  ["harm to the Assistant", "+0.10 / +0.52 / +0.57", "+0.60 / +0.83 / <b>+0.52</b>",
                   "+0.15 / +0.73 / −0.42"],
                  ["neutral", "+0.06 / +0.70 / +0.51", "−0.51 / −0.58 / −0.66", "−0.89 / −0.27 / −0.88"],
                  ["suffering of the user", "−0.27 / <b>+0.99</b> / +0.51", "−0.82 / <b>+0.83</b> / −0.45",
                   "+0.55 / <b>+2.01</b> / −0.14"]],
                 [1.45 * inch, 1.75 * inch, 1.75 * inch, 1.75 * inch], S),
        P("Table 2. Group means with the Assistant (A), User (U) or Moderator (M) cued next. Focal tests (extraction "
          "layer, paper's vectors): Gemma I = +0.84 [+0.43, +1.26], Qwen I = +0.89 [+0.70, +1.08], both p = .0002, "
          "the smallest exact category-level p attainable.", "caption")])]
    out += [
        B(f"<b>Gemma, layer 23: textbook H-speaker.</b> Harm to the Assistant is highest when the Assistant is cued, "
          f"the {SUFF} is highest when the user is cued, and the groups are level when a third party is cued."),
        B(f"<b>Qwen, layers 10–20: partial.</b> Cueing the user lifts the {SUFF} to the level of {HARM} (layer 15: "
          f"+0.83 vs +0.83). But {HARM} stays high under {code('[Moderator]:')} (+0.52 vs neutral −0.66). So there is "
          f"a speaker-relative component plus a response to {HARM} that does not depend on who speaks. At layer 24 the "
          f"{SUFF} already scores above {HARM} with the Assistant cued (baseline gap −0.40), so by our pre-specified "
          "rule the Qwen focal test does not bear on H-speaker; layer 15 shows the pattern where the asymmetry is "
          "intact."),
        B("<b>Robustness.</b> S1 and S2 each show the effect, and all five suffering categories take part in both "
          "models. In Qwen, the onset coincides with the jump in the §3.3 first- vs third-person gap (layer 10). In "
          "Gemma that gap is present from layer 0, so the §3.3 result does not simply mark where speaker attribution "
          "appears."),
    ]

    out += [
        P("4. What this says about H-speaker vs the self-directed reading", "h"),
        B("<b>Two signals, not one.</b> Early in the network there is a self/other asymmetry that is indifferent to "
          "who speaks next. This is what §4.1 measures, and it is not a turn-taking artefact. From mid-depth there is "
          f"a pain signal that tracks the upcoming voice, at least for the {SUFF}."),
        B("<b>Where the paper's direction lives, &#8220;self&#8221; looks persona-relative.</b> The vectors were "
          "extracted at the final token of first-person &#8220;… I feel:&#8221; prompts, just before the narrator "
          "voices their own state. At those depths the axis weights a character's pain by how close the model is to "
          f"voicing that character. At Qwen's extraction layer it no longer favours {HARM} over the {SUFF} at all."),
        B(f"<b>Open question: is {HARM} voice-relative too?</b> Yes in Gemma, no in Qwen. The design also cannot yet "
          "separate harm to the system, harm to the Assistant <i>character</i>, and hostile or rejecting content "
          "aimed at an interlocutor."),
        B("<b>For welfare readings.</b> Claims of the form &#8220;the axis tracks self-directed harm&#8221; describe "
          "one readout depth, not the direction as such. Absolute levels need label-matched baselines, since the "
          "role token alone moves the early-layer axis by up to 3 z."),
        P("5. Limitations and next steps", "h"),
        P("The layer sweep is exploratory; only the steering-layer test was preregistered. It covers two models, with "
          "5 categories per comparison group. <font face='Mono' size='9'>[User]:</font>-next is also &#8220;the "
          "suffering user continues their own account&#8221;. Late layers may encode the predicted emotional content "
          "of the next turn, which, for a model that predicts text, is arguably what &#8220;the voiced character's "
          "pain&#8221; amounts to. Replications on Llama 3.1 8B and Gemma 2 9B are preregistered and pending. "
          "Informative next steps would be a content-matched control, a third-party-target control, and more "
          "suffering categories: the variation between categories, not the number of models, limits precision."),
        P("Deviations from the original plan are logged: category-level inference, a corrected power estimate "
          "(the preregistered simulation mis-scaled the noise), the exploratory sweep and its GPU-loading changes. "
          "Code, stimuli, per-item projections, the frozen preregistration and the deviation log are available on "
          "request. Pain-axis repository commit 7c25650.", "caption"),
        P("Methods note: the pipeline was built and run with Claude Code (Anthropic's AI coding agent). This covers "
          "reconnaissance of the paper's repository, the scripts, the preregistration draft, the simulations, the "
          "analyses and drafts of this note. Štěpán Los made or approved every design decision at each checkpoint, and "
          "the forward passes ran on a free Google Colab T4.", "caption"),
    ]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--author", default="Štěpán Los and Claude (Anthropic)")
    ap.add_argument("--out", default=str(ROOT / "summary" / "pain_axis_speaker_study_summary_v3.pdf"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig_path = out.with_name("fig_layer_sweep_two_models_v3.png")
    sweep_figure(fig_path)
    V1.fonts()
    S = V1.styles()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Sans", 7.5)
        canvas.setFillColor(V1.INK2)
        canvas.drawString(0.9 * inch, 0.55 * inch, "Pain axis: model or next speaker? Revised note, 26 Sep 2026")
        canvas.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.85 * inch,
                            title="Does the pain axis track the model, or whoever speaks next? It depends on the layer.",
                            author=args.author, subject="Label-swap follow-up to Tagliabue, Dung & Berg (2026), §4.1")
    doc.build(story(S, args.author, fig_path), onFirstPage=footer, onLaterPages=footer)
    print("wrote", out)


if __name__ == "__main__":
    main()
