#!/usr/bin/env python3
"""Render the gold-standard worked example as a SAGE-WACV-style reasoning trace:
a breakable tcolorbox with a header table (model / KB / budget / prediction /
ground truth / outcome), the query thumbnail, a contrasting-reference contact
sheet, and numbered Step 1..N reasoning steps. Emits figs/gold_trace_box.tex.
"""
import json, re, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

GOLD = Path("/work/mech-ai-scratch/tirtho/SoybeanContrastKB/outputs/gold/Frogeye_Leaf_Spot__Soybean_PNAS_268")
FIGS = Path("/work/mech-ai-scratch/tirtho/overleaf-6a7d2bd/figs")
IMGDIR = FIGS / "traces" / "images"; IMGDIR.mkdir(parents=True, exist_ok=True)

S = json.loads((GOLD / "summary.json").read_text())
recs = [json.loads(l) for l in (GOLD / "trace.jsonl").open()]
meta = next(r for r in recs if r["stage"] == "meta")
s1 = next(r for r in recs if r["stage"] == "stage1")
s2 = next(r for r in recs if r["stage"] == "stage2")
s3 = {r["condition"]: r for r in recs if r["stage"] == "stage3"}
s4 = {r["condition"]: r for r in recs if r["stage"] == "stage4"}
kc_refs = s3["KC8"]["refs"]; edge = S["contrasting_edge"]
dk = s3["KC8"].get("diff_knowledge", "")


def esc(t):
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(t)).replace("**", "")
    for c, r in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        t = t.replace(c, r)
    return t


# query thumbnail
q = Image.open(meta["query_path"]).convert("RGB"); q.thumbnail((260, 260))
q.save(IMGDIR / "gold_query.jpg", quality=85)

# contrasting-reference contact sheet (2 rows x 4), labelled by class colour
cell, pad, lab = 150, 6, 20
cols = 4
a_refs = [r for r in kc_refs if r["class"] == edge["a"]][:4]
b_refs = [r for r in kc_refs if r["class"] == edge["b"]][:4]
rows = [(a_refs, (217, 138, 43), edge["a"]), (b_refs, (106, 90, 205), edge["b"])]
sheet = Image.new("RGB", (cols*cell + (cols+1)*pad, 2*(cell+lab) + 3*pad), "white")
d = ImageDraw.Draw(sheet)
try: font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 13)
except Exception: font = ImageFont.load_default()
for ri, (refs, col, name) in enumerate(rows):
    y0 = pad + ri*(cell+lab+pad)
    d.text((pad+2, y0), name.replace("_", " "), fill=col, font=font)
    for ci, r in enumerate(refs):
        im = Image.open(r["path"]).convert("RGB"); im = im.resize((cell, cell))
        x0 = pad + ci*(cell+pad); yy = y0+lab
        sheet.paste(im, (x0, yy))
        d.rectangle([x0, yy, x0+cell-1, yy+cell-1], outline=col, width=3)
sheet.save(IMGDIR / "gold_refs.jpg", quality=85)

# ---- numbered reasoning steps -----------------------------------------------
obs = (s1.get("parsed") or {}).get("observed_features", [])
obs_txt = "; ".join(obs[:3])
c2 = (s2.get("parsed") or {}).get("candidates", [])
top = c2[0] if c2 else {}
steps = []
steps.append(("Observe query leaf",
    f"{esc(obs_txt)[:240]}"))
steps.append(("Stage 1 — Broad KB screening (recall-first)",
    f"Screen the query against all {len(meta['kb_classes'])} soybean classes using only the "
    f"frozen source-grounded descriptions. Top-5 shortlist: "
    f"\\textit{{{esc(', '.join(S['shortlist']))}}}. "
    f"\\textcolor{{green!55!black}}{{Truth retained in shortlist.}}"))
steps.append(("Stage 2 — Focused re-read of candidates",
    f"Re-read only the shortlisted descriptions and weigh supporting vs.\\ contradicting "
    f"evidence. Re-ranked top-1: \\textbf{{{esc(S['ranked'][0])}}}. "
    f"Support: {esc(top.get('support_evidence',''))[:150]}"))
steps.append(("Stage 3 — Look-alike reference routing",
    f"Induce the frozen look-alike subgraph over the shortlist and select the highest-priority "
    f"contrasting edge \\textbf{{{esc(edge['a'])} $\\leftrightarrow$ {esc(edge['b'])}}}. "
    f"Retrieve 8 \\emph{{symmetric}} references from the local CyAg pool "
    f"(4 + 4); the ground-truth label is never used in retrieval."))
steps.append(("View contrasting references",
    f"Inspect the 8 reference images (see contact sheet): four confirmed "
    f"\\textit{{{esc(edge['a'])}}} and four confirmed \\textit{{{esc(edge['b'])}}} exemplars."))
steps.append(("Read source-grounded differentiating knowledge",
    f"{esc(dk[:260].strip())}\\ldots"))
steps.append(("Stage 4 — Final diagnosis (proposed, KC8)",
    f"Prediction: \\textbf{{{esc(S['KC8']['prediction'])}}} "
    f"(\\textcolor{{green!55!black}}{{\\textbf{{Correct}}}}). "
    f"Decisive cue: {esc(S['KC8']['decisive_cue'])[:220]}"))
steps.append(("Matched control (KG8, general references)",
    f"Same query and budget with general references gives "
    f"\\textbf{{{esc(S['KG8']['prediction'])}}} "
    f"(\\textcolor{{green!55!black}}{{Correct}}); the contrast makes the cited evidence "
    f"explicitly reject both look-alikes rather than merely asserting the label."))

steps_tex = "\n    ".join(
    rf"\item[\textcolor{{blue!45!black}}{{Step {i}:}}] \textbf{{{esc(t)}.}} {b}"
    for i, (t, b) in enumerate(steps, 1))

outcome = r"\textcolor{green!60!black}{\textbf{Correct}}"
tex = textwrap.dedent(rf"""
% Auto-generated SAGE-WACV-style reasoning trace (gold standard)
\begin{{tcolorbox}}[breakable, colback=gray!3, colframe=gray!55, boxrule=0.6pt,
  arc=2pt, left=3pt, right=3pt, top=3pt, bottom=3pt,
  title={{\small\textbf{{Reasoning Trace (gold-standard worked example):}}
  Soybean --- \texttt{{{esc(meta['query_id'])}}}}}, fonttitle=\small]
\footnotesize
\begin{{minipage}}[t]{{0.60\linewidth}}
\begin{{tabular}}{{@{{}}ll@{{}}}}
  \textbf{{Model:}} & frozen VLM via \texttt{{claude -p}} (headless) \\
  \textbf{{KB source:}} & SAGE source-grounded visual KB \\
  \textbf{{Pipeline:}} & KB screen $\rightarrow$ focus $\rightarrow$ route $\rightarrow$ diagnose \\
  \textbf{{Reference budget:}} & {S['budget']} images (matched) \\
  \textbf{{Contrasting edge:}} & {esc(edge['a'])} $\leftrightarrow$ {esc(edge['b'])} \\
  \textbf{{Prediction (KC8):}} & {esc(S['KC8']['prediction'])} \\
  \textbf{{Ground truth:}} & {esc(S['truth'])} \\
  \textbf{{Outcome:}} & {outcome} \\
\end{{tabular}}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.36\linewidth}}\centering
\fbox{{\includegraphics[height=2.4cm]{{figs/traces/images/gold_query.jpg}}}}\\[1pt]
{{\scriptsize\textit{{Query leaf (held out, local CyAg)}}}}
\end{{minipage}}

\vspace{{4pt}}\hrule\vspace{{5pt}}
\begin{{description}}[style=nextline, leftmargin=1.35cm, font=\footnotesize]
    {steps_tex}
\end{{description}}

\vspace{{2pt}}
\begin{{center}}
\includegraphics[width=0.86\linewidth]{{figs/traces/images/gold_refs.jpg}}\\[1pt]
{{\scriptsize\textit{{Stage 3 contrasting references (top: {esc(edge['a'])};
bottom: {esc(edge['b'])}) --- drawn from local CyAg, ground-truth label never used.}}}}
\end{{center}}
\end{{tcolorbox}}
""").strip()
(FIGS / "gold_trace_box.tex").write_text(tex)
print("wrote", FIGS / "gold_trace_box.tex")
print("wrote thumbnails in", IMGDIR)
