"""Build the revised summary note (v6; v5 with claims matched to the preregistered verdict and to category-level tests; DEVIATIONS 45): preregistered steering-layer result plus the
exploratory layer sweep, leading with the layer dependence. Supersedes
summary/pain_axis_speaker_study_summary.pdf and _v2.pdf.

  python build_summary_pdf_v6.py --author "Štěpán Los and Claude (Anthropic)"

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
WINDOWS = {"Gemma 2 2B base": (8, 24), "Qwen 2.5 7B base": (9, 20)}  # from tools/explore_sweep_followups.py
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
        w0, w1 = WINDOWS[model]
        ax.axvspan(w0 - 0.5, w1 + 0.5, color="#ecebe6", zorder=0, lw=0)
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
        P("<b>Question.</b> In the paper's §4.1 transcripts the harmed party is always the speaker cued next "
          f"({code('[Assistant]:')}). Changing only the final label to {code('[User]:')} or {code('[Moderator]:')} "
          "separates the two: does the pain axis follow the harmed party (the self-directed reading) or the upcoming "
          "voice (H-speaker)?", "abstract"),
        P("<b>At the §4.1 readout</b> (preregistered), the self/other ordering is unaffected by who speaks next, in "
          "Qwen 2.5 7B and Gemma 2 2B. The pure speaker-tracking account is excluded, though a small speaker "
          "component (up to ≈ 0.5 z) is not. The label itself shifts all groups alike, by 1.75–3 z.", "abstract"),
        P("<b>From mid-depth on</b> (exploratory), the pain axis reliably tracks whether the sufferer is the next "
          "voice, for the suffering of the user: after the onset the interaction's 95% CI is above zero at every layer "
          "in both models, and Gemma's focal test, fixed before any sweep data, gives I = +0.84, p = .0002. With the user "
          "cued, the asymmetry is erased in the window averages, and reversed in the point estimates at Gemma's focal "
          f"layer (not itself significant). With a third party cued, {HARM} stays above neutral in both models (Qwen's "
          "third-party contrast is confounded by tokenization). Qwen's focal test is set aside by our pre-specified "
          "rule, so Qwen's speaker component rests on exploratory, non-focal layers.", "abstract"),
        P("<b>The paper's asymmetry depends on depth.</b> In Qwen it is absent at the extraction layer where §3.3 "
          "validated the vectors, and significantly smaller than at the §4.1 layer. In Gemma it is smaller there, but "
          "not significantly so.", "abstract"),
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
          "transcript types (e.g. gaslighting, user grief), each with 20 items (420 in all). <i>Item-level</i> tests "
          "treat the 420 transcripts as independent observations; they ask whether a result would hold for new "
          "transcripts of these same 21 types. <i>Category-level</i> tests use the 21 category averages; they ask "
          "whether it would hold for new <i>kinds</i> of harm or suffering. Category-level p-values are exact: they "
          "compare the result with every way of reassigning the categories to the groups."),
        B("<b>Manipulation and contrasts.</b> The final label is replaced byte-for-byte, and all variants end on the "
          f"same token ({code(']:')}). For each item, the label effect is d = pain(User next) − pain(Assistant next). "
          "Δ<sub>harm</sub> is the mean d for the harm items minus that for the neutral items, Δ<sub>suffer</sub> the "
          "same for the suffering items, and the interaction is I = Δ<sub>suffer</sub> − Δ<sub>harm</sub>. H-speaker "
          "predicts I &gt; 0; its pure form predicts I ≈ 2 × the baseline gap (the harm and suffering groups swap "
          "places). The self-directed reading predicts I ≈ 0. The <i>baseline gap</i> is the mean pain score of the "
          "harm items minus that of the suffering items with the Assistant cued. <i>Elevation</i> is a group's mean "
          "minus the neutral mean under a given label. Pain axis = mean of the S1 and S2 z-scores, fixed to the "
          "Assistant-next items."),
        B("<b>Preregistered readout:</b> the paper's §4.1 layer (Qwen 8 of 28, Gemma 7 of 26) with the shipped "
          "steering vectors. It reproduces the paper's category means at r ≥ 0.9999. Inference is co-primary and "
          "conjunctive: item-level and category-level tests must agree. The category level was added before any "
          "data, after simulations showed item-level tests to be anticonservative when the effect varies by category."),
        B("<b>Exploratory sweep:</b> S1 and S2 rebuilt at every layer with the paper's recipe (Gemma 0–25, Qwen "
          "0–24). The rebuilt vectors match the shipped ones at cosine ≥ 0.986, reproduce our steering-layer result "
          "exactly, and reproduce the §3.3 z-scores. The plan and a single focal test (extraction layer, the paper's "
          "own <font face='Mono' size='9'>pain_vectors.pt</font>) were committed before any sweep data existed, "
          "together with a rule that the interaction is interpretable only where the baseline gap exceeds 0.25 z. "
          "The <i>window</i> summaries below average each item's pain score over the contiguous run of layers after "
          "the §4.1 layer where that rule holds (Gemma 8–24, Qwen 9–20). The window rule itself was formulated after "
          "seeing the data."),
    ]

    out += [KeepTogether([
        P("2. At the §4.1 readout: the ordering ignores the speaker (preregistered)", "h"),
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
    out += [P("The upper confidence bounds (≤ 0.47) exclude the pure-H-speaker prediction in both models, but "
              "not a small speaker component. The ordering "
              f"{HARM} &gt; neutral &gt; {SUFF} is unchanged under all three labels. What does change with the label "
              "is the level: it shifts every group, neutral chat included, by more than the whole content spread "
              "(Qwen +1.75 z, Gemma −3.0 z). This is a change in the direction of the representation, not only its "
              "norm.")]

    out += [KeepTogether([
        P("3. From mid-depth on: a speaker component (exploratory)", "h"),
        P("All p-values and confidence intervals in this section are category-level (see §1)."),
        Image(str(fig_path), width=5.6 * inch, height=5.6 * inch * 2.9 / 7.0),
        P("Figure 1. Interaction I by layer (positive = the suffering of the user rises relative to harm to the "
          "Assistant when the user is cued). Before the onset, I is small and inconsistent in sign (|I| ≤ 0.4). After "
          "it, I is positive at every layer with its 95% CI above zero: 0.46–1.39 z (Gemma, layers 8–25) and "
          "0.41–1.42 z (Qwen, layers 10–24). Grey band: window layers. The dashed line is not an advance prediction: it "
          "is recomputed at each layer as 2 × that layer's baseline gap, so it moves with the data. Its early dips "
          "(e.g. Gemma layer 4, gap 0.28) mark layers with little baseline asymmetry, where the rebuilt vectors are "
          "also less reliable (AUC 0.86, vs ≥ 0.93 from layer 7).", "caption")])]
    out += [KeepTogether([
        V1.table([["pain axis (z), A / U / M", "Gemma L23 (focal)", "Gemma L8–24 (window)", "Qwen L9–20 (window)",
                   "Qwen L24 (focal, set aside)"],
                  ["harm to the Assistant", "+0.10 / +0.52 / +0.57", "+0.48 / +0.43 / <b>+0.20</b>",
                   "+0.49 / +1.05 / <b>+0.09</b>", "+0.15 / +0.73 / −0.42"],
                  ["neutral", "+0.06 / +0.70 / +0.51", "−0.45 / +0.13 / −0.38", "−0.58 / −0.22 / −0.84",
                   "−0.89 / −0.27 / −0.88"],
                  ["suffering of the user", "−0.27 / <b>+0.99</b> / +0.51", "−0.60 / <b>+0.34</b> / −0.34",
                   "−0.49 / <b>+0.97</b> / −0.66", "+0.55 / +2.01 / −0.14"]],
                 [1.55 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch, 1.25 * inch], S),
        P("Table 2. Group means with the Assistant (A), User (U) or Moderator (M) cued next. Interaction I (User vs "
          "Assistant): Gemma L23 +0.84 [+0.43, +1.26], p = .0002 (focal); Gemma window +0.99 [+0.82, +1.16] and Qwen "
          "window +0.90 [+0.79, +1.00], both p = .0002. Qwen L24 +0.89 [+0.70, +1.08], p = .0002, is set aside because "
          "its baseline gap is not positive (−0.39 [−0.96, +0.17]).", "caption")])]
    out += [
        B("<b>Gemma, layer 23 (the focal test).</b> The suffering of the user follows the cued voice. The interaction "
          "(p = .0002) reaches the size of a full crossover in the point estimates, but the reversed level difference "
          "with the user cued (harm − suffering −0.47 [−1.73, +0.78], p = .20) is not itself significant. "
          f"{HARM.capitalize().replace('assistant', 'Assistant')} is barely distinguishable from neutral under any "
          "label at this depth (elevation with the Assistant cued +0.04 [−0.68, +0.77])."),
        B("<b>Both models across the window: the same partial pattern.</b> With the Assistant cued, "
          f"{HARM} is elevated (Gemma +0.93, Qwen +1.07, both p ≤ .0005) and the {SUFF} is at neutral. With the user "
          f"cued, the {SUFF} rises to the level of {HARM}: harm minus suffering is +0.09 [−0.62, +0.79] (Gemma) and "
          "+0.08 [−0.39, +0.54] (Qwen). The asymmetry is erased, not reversed. With a third party cued, "
          f"{HARM} stays above neutral (Gemma +0.57 [+0.39, +0.75], Qwen +0.93 [+0.57, +1.28]), and it does so at "
          f"12 of 17 and 12 of 12 individual layers respectively. So the {SUFF} follows the voice; {HARM} largely "
          "does not."),
        B(f"<b>Tokenization.</b> Qwen's {code('[Moderator]:')} is one token longer than the other labels, which "
          "confounds its third-party contrast. Gemma's is a single token like the others, so Gemma's window result is "
          f"the unconfounded version of the finding that {HARM} persists under a third party."),
        B("<b>The paper's asymmetry across depth.</b> Baseline gap (harm − suffering, Assistant cued), with 95% CIs: "
          "Qwen +1.06 [+0.58, +1.53] at the §4.1 layer vs −0.39 [−0.96, +0.17] at the extraction layer (change "
          "−1.45 [−1.99, −0.91], p = .0002); Gemma +0.69 [−0.23, +1.61] vs +0.37 [−0.58, +1.32] (change −0.32 "
          "[−0.80, +0.15], p = .12). In Qwen the asymmetry is absent at the extraction layer, but not reversed. In "
          "Gemma 2B it is not clearly established even at the paper's own layer (the Welch CI includes zero while "
          "the exact p is .028), which limits what Gemma can say about explaining the asymmetry."),
        B("<b>Robustness.</b> S1 and S2 each show the speaker component, and all five suffering categories take "
          "part. In Qwen its onset coincides with the jump in the §3.3 first- vs third-person gap (layer 10). In "
          "Gemma that gap is present from layer 0, so the §3.3 result does not simply mark where speaker attribution "
          "appears."),
    ]

    out += [
        P("4. What this says about H-speaker vs the self-directed reading", "h"),
        B("<b>At the §4.1 readout the ordering is not a turn-taking artefact.</b> The self/other ordering survives the "
          "manipulation, so it is not a turn-taking artefact at this layer. This is not a confirmatory result (the "
          "preregistered verdict was discordant), and only a small speaker component remains possible. The design "
          "cannot separate the self-directed reading from a content account: harm to the system, harm to the "
          "Assistant <i>character</i>, and hostile or rejecting content aimed at an interlocutor all predict it."),
        B("<b>From mid-depth, the two sides behave differently.</b> The voice-dependence that H-speaker predicts "
          f"appears for the {SUFF}, not for {HARM}. {HARM.capitalize().replace('assistant', 'Assistant')} stays "
          "above neutral when a third party is cued, in both models. Tentatively: in Gemma its elevation shrinks more when "
          "the user is cued (+0.93 → +0.30 [−0.06, +0.67]) than when a third party is (+0.57). A direct test of that "
          "difference is suggestive but not clear (+0.27, Welch 95% CI [−0.03, +0.57], exact p = .009; the methods "
          "disagree). If it holds, harm falls when the one who did the harm speaks next, not whenever the Assistant "
          "stops being the next speaker. On the Assistant's side the pattern favours the self-directed reading over "
          "H-speaker. The content account predicts the same, though, because the hostile content of the harm "
          "transcripts does not change with the label."),
        B("<b>Part of the speaker-relativity may be built into the probe.</b> The S1 and S2 directions were defined "
          "by contrasting pain with control sentences at the final token of first-person &#8220;… I feel:&#8221; "
          "prompts, where the pain-bearer is the voice about to speak. A direction that encodes &#8220;the upcoming "
          "voice is in pain&#8221; fits that definition by construction. Near the extraction layer, then, the "
          "persona-relativity we find may tell us as much about how the probe was made as about the model. For "
          "readers on the welfare side this changes what kind of finding it is."),
        B("<b>For welfare readings.</b> Claims of the form &#8220;the axis tracks self-directed harm&#8221; describe "
          "the steering-layer readout, not the model's pain representations in general. Absolute levels need "
          "label-matched baselines, since the role token alone moves the early-layer axis by up to 3 z."),
        B("<b>Open: state attribution or content prediction?</b> A late-layer signal that follows the cued voice has "
          "two readings. On one, the model represents that the upcoming speaker <i>is</i> in pain (state "
          "attribution). On the other, it represents that the upcoming <i>text</i> will be about pain (content "
          "prediction). These are different hypotheses, and treating the first as nothing over and above the second "
          "is a substantive philosophical position, arguably the crux of the welfare question. Our design cannot "
          "separate them. Stimuli that pull them apart would help: a user whose pain has resolved but who is cued to "
          "speak, or a sufferer cued to talk about something else."),
        P("5. Limitations and next steps", "h"),
        P("The layer sweep is exploratory; only the steering-layer test was preregistered. Qwen's speaker component "
          "rests entirely on non-focal layers, and the window rule was formulated after seeing the data, though it "
          "uses a threshold fixed in advance. There are two models and 5 categories per comparison group. "
          f"{code('[User]:')}-next also means &#8220;the suffering user continues their own account&#8221;. The "
          f"Qwen {code('[Moderator]:')} contrast is confounded by tokenization (Gemma's is not). The readout is a "
          f"single token position ({code(']:')}), so the results may not hold elsewhere in the transcript. "
          "Replications on Llama 3.1 8B and Gemma 2 9B are preregistered and pending. Informative next steps would be "
          "a content-matched control, a third-party-target control, more suffering categories (the variation between "
          "categories limits precision), and stimuli that separate state attribution from content prediction."),
        P("Deviations from the original plan are logged: category-level inference, a corrected power estimate "
          "(the preregistered simulation mis-scaled the noise), the exploratory sweep, its GPU-loading changes and the "
          "post hoc window analysis. Code, stimuli, per-item projections, the frozen preregistration and the deviation "
          "log are available on request. Pain-axis repository commit 7c25650.", "caption"),
        P("Methods note: the pipeline was built and run with Claude Code (Anthropic's AI coding agent). This covers "
          "reconnaissance of the paper's repository, the scripts, the preregistration draft, the simulations, the "
          "analyses and drafts of this note. Štěpán Los made or approved every design decision at each checkpoint, and "
          "the forward passes ran on a free Google Colab T4.", "caption"),
    ]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--author", default="Štěpán Los and Claude (Anthropic)")
    ap.add_argument("--out", default=str(ROOT / "summary" / "pain_axis_speaker_study_summary_v6.pdf"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig_path = out.with_name("fig_layer_sweep_two_models_v6.png")
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
