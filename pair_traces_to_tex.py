#!/usr/bin/env python3
"""Render ALL pair-experiment reasoning traces as SAGE-WACV-style tcolorboxes
(one per test image) + a shared reference panel, for the paper appendix.
Emits figs/pair_traces.tex and thumbnails under figs/traces/images/pair/.
"""
import json, re, glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import yaml, sys
sys.path.insert(0, str(Path(__file__).parent))
import kb_loader as K, pipeline as P, pair_disambiguation as PD, run_pair_experiment as RPE

A, B = "Bacterial_Blight", "Bacterial_Pustule"
cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
EXP = Path(cfg["experiment_output_dir"]) / "pair_experiment" / f"{A}__vs__{B}"
FIGS = Path("/work/mech-ai-scratch/tirtho/overleaf-6a7d2bd/figs")
IMG = FIGS / "traces" / "images" / "pair"; IMG.mkdir(parents=True, exist_ok=True)

recs = [json.loads(l) for l in (EXP / "records.jsonl").open()]
checklist = json.loads((EXP / "checklist.json").read_text())


def esc(t):
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(t)).replace("**", "")
    for c, r in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        t = t.replace(c, r)
    return t


def snap_path(qid, truth):
    stem = qid.split("__", 1)[1]
    for ext in ("png", "jpg", "jpeg"):
        hits = glob.glob(str(Path(cfg["experiment_output_dir"]) / "snapshot" / "queries" / truth / f"{stem}.{ext}"))
        if hits:
            return hits[0]
    hits = [str(p) for p in P.class_images(cfg, truth) if p.stem == stem]
    return hits[0] if hits else None


def thumb(src, name, box=240):
    im = Image.open(src).convert("RGB"); im.thumbnail((box, box))
    out = IMG / f"{name}.jpg"; im.save(out, quality=85); return out


# shared reference panel (same 4+4 refs used for every test image)
graph = K.load_lookalike_graph(cfg)
edge = PD.select_pair(graph, A, shortlist={A, B})
tests = []
for cls in (A, B):
    tests += [{**t, "truth": cls} for t in RPE.load_test_queries(cfg, cls)]
held = {Path(t["path"]).name for t in tests}
refs = PD.retrieve_pair_refs(cfg, edge, 4, held, cfg["seed"])
cell, pad, lab = 150, 6, 22
sheet = Image.new("RGB", (4*cell+5*pad, 2*(cell+lab)+3*pad), "white")
d = ImageDraw.Draw(sheet)
try: font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 13)
except Exception: font = ImageFont.load_default()
for ri, (cls, col) in enumerate([(A, (191, 80, 77)), (B, (106, 90, 205))]):
    y0 = pad + ri*(cell+lab+pad); d.text((pad+2, y0), cls.replace("_", " "), fill=col, font=font)
    for ci, p in enumerate(refs[cls]):
        im = Image.open(p).convert("RGB").resize((cell, cell)); x0 = pad+ci*(cell+pad); yy = y0+lab
        sheet.paste(im, (x0, yy)); d.rectangle([x0, yy, x0+cell-1, yy+cell-1], outline=col, width=3)
sheet.save(IMG / "shared_refs.jpg", quality=88)


def trace_box(rec):
    truth = rec["truth"]; qid = rec["query_id"]
    sp = snap_path(qid, truth)
    thumb(sp, qid) if sp else None
    good = rec["C_correct"]
    outcome = (r"\textcolor{green!60!black}{\textbf{Correct}}" if good
               else r"\textcolor{red!75!black}{\textbf{Incorrect}}")
    def mark(ok): return (r"\textcolor{green!55!black}{Y}" if ok else r"\textcolor{red!70!black}{n}")
    rows = []
    zoom_tag = r" \textit{[zoom]}"
    eol = r" \\"
    for t in rec["C_table"]:
        obs = esc(t.get('query_observation', '')[:44]) + (zoom_tag if t.get('zoomed') else "")
        row = (esc(t.get('feature', '')[:24]) + " & " + esc(t.get('class_a_refs', '')[:30]) +
               " & " + esc(t.get('class_b_refs', '')[:30]) + " & " + obs + " & " +
               esc(t.get('supports', '')) + eol)
        rows.append(row)
    table = "\n".join(rows)
    imgtex = (rf"\includegraphics[height=2.3cm]{{figs/traces/images/pair/{esc(qid)}.jpg}}"
              if sp else "")
    return rf"""
\begin{{tcolorbox}}[breakable, colback=gray!3, colframe=gray!55, boxrule=0.6pt, arc=2pt,
  left=3pt,right=3pt,top=3pt,bottom=3pt,
  title={{\footnotesize\textbf{{Reasoning Trace:}} {esc(A)} vs {esc(B)} --- \texttt{{{esc(qid)}}}}},
  fonttitle=\footnotesize]
\footnotesize
\begin{{minipage}}[t]{{0.63\linewidth}}
\begin{{tabular}}{{@{{}}ll@{{}}}}
 \textbf{{Ground truth:}} & {esc(truth)} \\
 \textbf{{A (image+KB):}} & {esc(rec['A_pred'])}\; {mark(rec['A_correct'])} \\
 \textbf{{B (KB+refs):}} & {esc(rec['B_pred'])}\; {mark(rec['B_correct'])} \\
 \textbf{{C (feature comparison):}} & {esc(rec['C_pred'])}\; {mark(rec['C_correct'])} \\
 \textbf{{C decision:}} & {esc(rec['C_decision'])} (supports A={rec['C_supA']}, B={rec['C_supB']}) \\
 \textbf{{Outcome (C):}} & {outcome} \\
\end{{tabular}}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.34\linewidth}}\centering
\fbox{{{imgtex}}}\\[1pt]{{\scriptsize\textit{{query (held out)}}}}
\end{{minipage}}

\vspace{{3pt}}
{{\scriptsize
\begin{{tabular}}{{p{{2.0cm}}p{{2.5cm}}p{{2.5cm}}p{{3.6cm}}c}}
\toprule
Feature & Class A refs & Class B refs & Query observation & Supp. \\ \midrule
{table}
\bottomrule
\end{{tabular}}}}

\vspace{{2pt}}{{\scriptsize\textit{{Rationale:}} {esc(rec['C_explanation'][:300])}}}
\end{{tcolorbox}}
"""


header = rf"""% Auto-generated pair-experiment reasoning traces (appendix)
\subsection{{Structured feature-comparison reasoning traces: {esc(A)} vs {esc(B)}}}
\label{{app:pair_traces}}
Every held-out test image is diagnosed under three matched conditions
(A: image$+$KB; B: $+$ordinary references; C: $+$the same references with the
structured distinguishing-feature comparison). The shared reference set below is
identical across B and C and disjoint from all test images. A not-assessable
feature is never counted as evidence. These traces are descriptive; per-class
$n{{=}}5$ is a pilot.

\begin{{figure}}[H]\centering
\includegraphics[width=0.86\linewidth]{{figs/traces/images/pair/shared_refs.jpg}}
\caption{{Shared held-out reference set (top: {esc(A)}; bottom: {esc(B)}), four per
class, drawn from the local CyAg pool and used identically in conditions B and C.}}
\label{{fig:pair_refs}}
\end{{figure}}
"""
body = "\n".join(trace_box(r) for r in recs)
(FIGS / "pair_traces.tex").write_text(header + body)
print("wrote", FIGS / "pair_traces.tex", "with", len(recs), "trace boxes")
