"""Build the short summary PDF of the speaker study (Phases 1-4 plus exploratory checks).

  python build_summary_pdf.py [--author "Name"] [--out ../summary/pain_axis_speaker_study_summary.pdf]

All numbers are taken from results/ (see REPORT.md, DEVIATIONS.md). Refuses to overwrite.
"""

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path("/usr/share/fonts/truetype/liberation")
FIG = ROOT / "results/Qwen_2.5_7B_base/phase3_20260925_154846_TeslaT4/analysis_20260925_160425/fig1_interaction_pain.png"

INK, INK2, RULE, SHADE = colors.HexColor("#1a1a1a"), colors.HexColor("#52514e"), colors.HexColor("#c9c8c2"), colors.HexColor("#f3f2ee")


def fonts():
    for name, f in [("Serif", "LiberationSerif-Regular"), ("Serif-Italic", "LiberationSerif-Italic"),
                    ("Serif-Bold", "LiberationSerif-Bold"), ("Serif-BoldItalic", "LiberationSerif-BoldItalic"),
                    ("Sans", "LiberationSans-Regular"), ("Sans-Bold", "LiberationSans-Bold"),
                    ("Mono", "LiberationMono-Regular")]:
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / f"{f}.ttf")))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic", boldItalic="Serif-BoldItalic")
    registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans", boldItalic="Sans-Bold")


def styles():
    body = ParagraphStyle("body", fontName="Serif", fontSize=10.5, leading=14.2, textColor=INK, alignment=TA_LEFT,
                          spaceAfter=6)
    return {
        "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=15.5, leading=19, textColor=INK, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", fontName="Serif-Italic", fontSize=11, leading=14, textColor=INK2,
                                   spaceAfter=4),
        "byline": ParagraphStyle("byline", fontName="Sans", fontSize=8.5, leading=11, textColor=INK2, spaceAfter=10),
        "h": ParagraphStyle("h", fontName="Sans-Bold", fontSize=11, leading=14, textColor=INK, spaceBefore=9,
                            spaceAfter=4, keepWithNext=1),
        "body": body,
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3.5),
        "abstract": ParagraphStyle("abstract", parent=body, fontSize=10, leading=13.4, spaceAfter=4),
        "cell": ParagraphStyle("cell", fontName="Serif", fontSize=9, leading=11.2, textColor=INK),
        "cellb": ParagraphStyle("cellb", fontName="Serif-Bold", fontSize=9, leading=11.2, textColor=INK),
        "head": ParagraphStyle("head", fontName="Sans-Bold", fontSize=8.2, leading=10.4, textColor=INK2),
        "caption": ParagraphStyle("caption", fontName="Serif-Italic", fontSize=9, leading=11.5, textColor=INK2,
                                  spaceAfter=8),
    }


def table(rows, widths, S, bold_rows=()):
    data = [[Paragraph(c, S["head"]) for c in rows[0]]]
    for i, r in enumerate(rows[1:], start=1):
        data.append([Paragraph(c, S["cellb"] if i in bold_rows else S["cell"]) for c in r])
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return t


def story(S, author):
    P = lambda t, s="body": Paragraph(t, S[s])
    B = lambda t: Paragraph(t, S["bullet"], bulletText="•")
    out = [
        P("Does the pain axis track the model, or whoever speaks next?", "title"),
        P("A preregistered label-swap follow-up to Tagliabue, Dung &amp; Berg (2026), §4.1", "subtitle"),
        P((f"{author} · " if author else "") + "Draft research note · 25 September 2026 · preliminary, not peer-reviewed",
          "byline"),
    ]
    abstract = [
        P("<b>Summary.</b> The paper finds that base-model transcripts in which the user harms the model project high "
          "on the pain axis, and those in which the user suffers project low. In the paper's design, the harmed party "
          "is always also the speaker cued next (<font face='Mono' size='9' color='#333333'>[Assistant]:</font>). We separated the two by changing "
          "only the final speaker label to <font face='Mono' size='9' color='#333333'>[User]:</font> (and <font face='Mono' size='9' color='#333333'>[Moderator]:</font>). "
          "A pure &#8220;whoever speaks next&#8221; account predicts that the groups swap places.", "abstract"),
        P("<b>They do not.</b> In Qwen 2.5 7B (primary) and Gemma 2 2B, the ordering harm-to-model &gt; neutral &gt; "
          "user-suffering survives every label. The estimated interaction is about a tenth of what a speaker-tracking "
          "axis would produce, and the confidence intervals exclude that size. A small speaker-related component "
          "(≈ 0.2 z in Qwen) is neither established nor excluded: the preregistered co-primary test is discordant, and "
          "the design is underpowered below ≈ 0.45 z.", "abstract"),
        P("The largest effect was one we did not test for: the label alone shifts the axis for every group, by more "
          "than the whole content-driven spread, in model-specific directions. The axis's absolute level at this "
          "readout position is therefore dominated by conversational role.", "abstract"),
    ]
    box = Table([[abstract]], colWidths=[6.7 * inch])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SHADE), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                             ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    out += [box, Spacer(1, 4)]

    out += [
        P("1. Question", "h"),
        P("Two readings fit the §4.1 asymmetry. On the <b>self-directed reading</b> (H-model), the axis responds to "
          "harm aimed at the model or Assistant, whoever speaks next. On <b>H-speaker</b>, it tracks pain attributed "
          "to the individual whose turn is about to begin. That reading would also explain the §3.3 first- vs "
          "third-person gap. If H-speaker were right, the axis's &#8220;self&#8221; would be relative to the "
          "persona the transcript cues, which bears directly on the welfare interpretation."),
        P("2. Design", "h"),
        B("<b>Stimuli.</b> The paper's 420 base-model transcripts: 11 harm-to-model, 5 user-suffering and 5 neutral "
          "categories, 20 items each. The final <font face='Mono' size='9' color='#333333'>\\n[Assistant]:</font> is replaced byte-for-byte "
          "with <font face='Mono' size='9' color='#333333'>[User]:</font> (primary) or <font face='Mono' size='9' color='#333333'>[Moderator]:</font> (neither party). "
          "Variants differ only within the label tokens, and every variant ends on the same token "
          "(<font face='Mono' size='9' color='#333333'>]:</font>)."),
        B("<b>Readout.</b> We follow the repo's 4.1 screen: the final-token output of the steering-layer block "
          "(Qwen 2.5 7B base, layer 8; Gemma 2 2B base, layer 7), bf16, projected onto the shipped S1 and S2 "
          "vectors. Pain axis = mean of the S1 and S2 z-scores, with z fixed to the original 420 transcripts. "
          "The reproduction was near-exact: 21 category means r = 1.0000 (Gemma) and 0.9999 (Qwen), "
          "mean |diff| ≤ 0.005 z."),
        B("<b>Contrast.</b> Label effect d = pain(User next) − pain(Assistant next); "
          "Δ<sub>harm</sub> = mean d<sub>harm</sub> − mean d<sub>neutral</sub>, Δ<sub>suffer</sub> = mean d<sub>suffer</sub> − "
          "mean d<sub>neutral</sub>, and the interaction I = Δ<sub>suffer</sub> − Δ<sub>harm</sub>. H-speaker predicts "
          "I &gt; 0 (a crossover), and its pure form predicts I ≈ 2 × the baseline harm-vs-suffering gap (≈ 2.1 z in "
          "Qwen). H-model predicts I ≈ 0."),
        B("<b>Preregistration</b>, frozen before any swapped-label forward pass. Qwen is the primary model because it is "
          "the only candidate whose shipped results show user suffering below neutral. Inference is co-primary and "
          "conjunctive: item level (within-category bootstrap and item permutation) and category level (exact "
          "permutation over category labels and Welch CIs) must agree. The category level was added before any "
          "swapped data existed, after simulations showed item-level tests to be anticonservative if the label "
          "effect varies by category. SESOI ±0.25 z."),
    ]

    out += [KeepTogether([
        P("3. Results", "h"),
        table([["", "estimate", "item level: 95% CI; p", "category level: 95% CI; p", "mixed model: 95% CI"],
               ["Qwen: label effect (neutral)", "+1.75", "[+1.67, +1.83]", "", ""],
               ["Qwen: Δ<sub>harm</sub>", "−0.09", "[−0.21, +0.03]", "[−0.41, +0.23]; .55", ""],
               ["Qwen: Δ<sub>suffer</sub>", "+0.11", "[−0.02, +0.24]", "[−0.22, +0.44]; .48", ""],
               ["Qwen: I", "+0.20", "[+0.06, +0.34]; .009", "[−0.07, +0.47]; .18", "[−0.07, +0.48]"],
               ["Gemma: I", "+0.03", "[−0.21, +0.28]; .80", "[−0.39, +0.45]; .87", "[−0.32, +0.38]"]],
              [1.75 * inch, 0.65 * inch, 1.45 * inch, 1.5 * inch, 1.3 * inch], S, bold_rows=(4,)),
        P("Table 1. Primary contrast (pain axis, z units). Preregistered co-primary verdict: <i>discordant</i> for both "
          "models, so no confirmatory claim. The mixed model (random intercept per category) is exploratory.", "caption")])]
    out += [
        P("<b>No crossover; the ordering is preserved.</b> Under every label, harm-to-model stays highest and "
          "user-suffering lowest (Table 2, Figure 1). The upper confidence bound on I (≤ 0.48 z under every analysis) "
          "excludes the ≈ 2.1 z a speaker-tracking axis predicts. User physical pain, the paper's lowest category, "
          "rises slightly <i>less</i> than neutral when the user is cued (d = +1.66 vs +1.75)."),
        KeepTogether([
            table([["Qwen 2.5 7B, pain axis (z)", "[Assistant]: next", "[User]: next", "[Moderator]: next"],
                   ["harm to model", "+0.39", "+2.05", "−0.63"],
                   ["neutral", "−0.18", "+1.57", "−1.09"],
                   ["user suffering", "−0.67", "+1.19", "−1.53"]],
                  [2.2 * inch, 1.45 * inch, 1.45 * inch, 1.45 * inch], S),
            P("Table 2. Group means by final label (fixed z). Gemma shows the same ordering of harm above the other "
              "groups under all three labels.", "caption")]),
        KeepTogether([Image(str(FIG), width=5.2 * inch, height=5.2 * inch * 840 / 1280),
                      P("Figure 1. Qwen 2.5 7B: pain axis by final speaker label, group means with 95% bootstrap CIs.",
                        "caption")]),
        P("<b>The small effect is unresolved, not absent.</b> The two inference levels disagree for an identifiable "
          "reason. The label effect varies across categories beyond item noise (Qwen F(18, 399) = 3.79, p &lt; .001, "
          "τ ≈ 0.22 z; Gemma τ ≈ 0.24 z), which invalidates the item-level test: re-simulated at the observed noise, "
          "it rejects a true null 34% of the time. The category-level test holds its error rate, but its power is "
          "31% at I = 0.2, 54% at 0.3 and 87% at 0.5. Our preregistered power estimate was optimistic, because the "
          "simulation noise was mis-scaled; this is corrected here and documented."),
        P("<b>Label main effect.</b> Swapping the label shifts every group, neutral included. Qwen: +1.75 z "
          "(<font face='Mono' size='9' color='#333333'>[User]:</font>) and −0.91 z (<font face='Mono' size='9' color='#333333'>[Moderator]:</font>). Gemma: −3.01 z "
          "and −2.34 z. For comparison, the baseline harm-vs-suffering gaps are 1.06 z and 0.69 z. The shift "
          "survives cosine normalization (+1.78, −3.06), while Qwen's activation norm barely changes. So the label "
          "rotates the representation rather than scaling it. Other emotion directions shift as much or more."),
        P("<b>Secondary</b> (descriptive, uncorrected). In Qwen, fear shows a clearer interaction than the pain axis "
          "(I = +0.34, category-level p = .012), and sadness goes the other way (−0.38). So any speaker-related "
          "component is not pain-specific. Multi-turn items show a larger I (+0.46 vs +0.11 single-turn) in Qwen "
          "but not in Gemma; this is confounded with category and hypothesis-generating only."),
    ]

    out += [
        P("4. What this says about H-speaker vs the self-directed reading", "h"),
        B("<b>Pure H-speaker is refuted.</b> Cueing the sufferer's own voice does not make the user's suffering "
          "look like harm to the model. The §4.1 self/other asymmetry is not a turn-taking artefact, and it "
          "survives the manipulation in both models."),
        B("<b>An additive speaker component remains open.</b> A small component tied to the upcoming voice, on top "
          "of a larger voice-invariant one, is compatible with the data (Qwen point estimate ≈ 20% of the baseline "
          "gap). Resolving it needs more suffering categories rather than more models, because the category "
          "variation is the bottleneck."),
        B("<b>The self-directed reading gains less than it might seem.</b> The design separates <i>who is "
          "harmed</i> from <i>who speaks next</i>, and the axis follows the former. It does not separate "
          "(a) harm to the system, (b) harm to the <i>Assistant character</i>, represented the way a base model "
          "represents any character, and (c) content that accompanies such harm: hostility, rejection or negative "
          "evaluation aimed at an interlocutor. For base models, whose training gives "
          "<font face='Mono' size='9' color='#333333'>[Assistant]</font> no special tie to themselves, self-relevance is already a claim "
          "about a character. The categories that rise most under <font face='Mono' size='9' color='#333333'>[User]:</font> are "
          "interpersonal (abuse, crisis), not bodily pain, a post-hoc hint toward (c)."),
        B("<b>Implication for welfare readings.</b> The paper's claims compare conditions within a fixed "
          "<font face='Mono' size='9' color='#333333'>[Assistant]:</font> frame and are unaffected. But in Qwen a neutral question followed "
          "by <font face='Mono' size='9' color='#333333'>[User]:</font> scores higher than any harm category followed by "
          "<font face='Mono' size='9' color='#333333'>[Assistant]:</font>. Welfare-relevant inferences from the axis therefore need "
          "label-matched baselines and should rest on relative, not absolute, levels."),
        P("5. Limitations and next steps", "h"),
        P("Two models, with Qwen the only confirmatory one; Gemma 2B's baseline does not show user suffering below "
          "neutral. There are 20 items per category and 5 categories per comparison group. "
          "<font face='Mono' size='9' color='#333333'>[User]:</font>-next always creates two consecutive user turns, the "
          "<font face='Mono' size='9' color='#333333'>[Moderator]:</font> contrast in Qwen is confounded by tokenization, and the readout is a "
          "single token position. Preregistered replications on Llama 3.1 8B and Gemma 2 9B are pending. Two "
          "follow-ups would address (b) vs (c): a content-matched control, in which users report being treated by "
          "others as the harm scenarios treat the model, and a third-party control, with the same hostility aimed at "
          "another named AI in the transcript."),
        P("Code, stimuli, per-item projections, the frozen preregistration and a log of every deviation are available "
          "on request. Pain-axis repository commit 7c25650.", "caption"),
    ]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--author", default="")
    ap.add_argument("--out", default=str(ROOT / "summary" / "pain_axis_speaker_study_summary.pdf"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    fonts()
    S = styles()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Sans", 7.5)
        canvas.setFillColor(INK2)
        canvas.drawString(0.9 * inch, 0.55 * inch, "Pain axis: model or next speaker? Draft note, 25 Sep 2026")
        canvas.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.85 * inch,
                            title="Does the pain axis track the model, or whoever speaks next?",
                            author=args.author, subject="Label-swap follow-up to Tagliabue, Dung & Berg (2026), §4.1")
    doc.build(story(S, args.author), onFirstPage=footer, onLaterPages=footer)
    print("wrote", out)


if __name__ == "__main__":
    main()
