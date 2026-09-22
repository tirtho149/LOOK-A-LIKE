#!/usr/bin/env python3
"""Gold-standard worked-example figure — styled to match construction_pipeline.png:
a clean single-row flow of rounded boxes on a light panel, straight arrows with
small edge labels, and a reference-evidence banner below. Publication quality.
"""
import json, sys, textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

GOLD = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/work/mech-ai-scratch/tirtho/SoybeanContrastKB/outputs/gold/Frogeye_Leaf_Spot__Soybean_PNAS_268")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "/work/mech-ai-scratch/tirtho/overleaf-6a7d2bd/figs/gold_worked_example.png")

S = json.loads((GOLD / "summary.json").read_text())
recs = [json.loads(l) for l in (GOLD / "trace.jsonl").open()]
meta = next(r for r in recs if r["stage"] == "meta")
s3 = {r["condition"]: r for r in recs if r["stage"] == "stage3"}
kc_refs = s3["KC8"]["refs"]
dk = s3["KC8"].get("diff_knowledge", "")
edge = S["contrasting_edge"]

# palette matched to construction_pipeline.png
INK   = "#20303f"
GREEN = "#3b8f5a"; BLUE = "#2f6f8f"; PURP = "#6a5acd"
ORNG  = "#d98a2b"; GOLD_ = "#c79a1e"; GREY = "#9aa7b3"; RED = "#c0504d"
PANEL = "#eef2f5"
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(15.5, 8.4), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# light rounded background panel behind the flow
ax.add_patch(FancyBboxPatch((0.015, 0.44), 0.97, 0.46,
             boxstyle="round,pad=0.004,rounding_size=0.02",
             fc=PANEL, ec="none", zorder=0))

ax.text(0.5, 0.955, "Worked example  —  KB-guided contrasting-reference diagnosis of one held-out soybean leaf",
        ha="center", va="center", fontsize=16, fontweight="bold", color=INK)
ax.text(0.5, 0.915, f"query ground truth:  {S['truth']}      matched 8-image budget      frozen VLM via claude ‑p",
        ha="center", va="center", fontsize=10.5, color=GREY)

def rbox(x, y, w, h, ec, title, sub, tfs=11.5, sfs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.03",
                 fc="white", ec=ec, lw=2.2, zorder=2))
    ax.text(x + w/2, y + h*0.72, title, ha="center", va="center",
            fontsize=tfs, fontweight="bold", color=INK, zorder=3)
    ax.text(x + w/2, y + h*0.34, sub, ha="center", va="center",
            fontsize=sfs, color="#54636f", zorder=3, linespacing=1.35)

def arrow(x0, x1, y, label, lc=RED):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=17, color=GREY, lw=2.0, zorder=2,
                 shrinkA=0, shrinkB=0))
    ax.text((x0+x1)/2, y + 0.028, label, ha="center", va="bottom",
            fontsize=8.2, color=lc, fontweight="bold", zorder=3)

def place_img(path, x, y, w, h, ec, label=None, lc=None):
    iax = fig.add_axes([x, y, w, h], zorder=4)
    try: iax.imshow(Image.open(path).convert("RGB"))
    except Exception: iax.text(.5, .5, "img", ha="center")
    iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values(): s.set_edgecolor(ec); s.set_linewidth(2.0)
    if label:
        ax.text(x + w/2, y - 0.018, label, ha="center", va="top",
                transform=fig.transFigure, fontsize=8.4, color=lc or ec, fontweight="bold")

# ---- Flow row (single, aligned, well spaced) --------------------------------
row_y, bh = 0.60, 0.20
# 1) query thumbnail box
qx, qw = 0.035, 0.135
ax.add_patch(FancyBboxPatch((qx, row_y), qw, bh, boxstyle="round,pad=0.006,rounding_size=0.03",
             fc="white", ec=GREEN, lw=2.2, zorder=2))
place_img(meta["query_path"], qx+0.026, row_y+0.042, qw-0.052, bh-0.098, GREEN)
ax.text(qx+qw/2, row_y+bh-0.022, "QUERY LEAF", ha="center", fontsize=9.5,
        fontweight="bold", color=GREEN, zorder=5)
ax.text(qx+qw/2, row_y+0.022, "local CyAg, held out", ha="center", fontsize=7.6,
        color="#54636f", zorder=5)

boxes = [
    (0.205, 0.155, BLUE,  "Stage 1", "Broad KB screening\n(recall-first, full label set)"),
    (0.390, 0.150, PURP,  "Stage 2", "Focused re-read\nof shortlisted candidates"),
    (0.560, 0.175, ORNG,  "Stage 3", "Look-alike routing\n+ symmetric references"),
    (0.760, 0.170, GOLD_, "Stage 4", "Final diagnosis\nquery vs. references"),
]
for x, w, ec, t, s in boxes:
    rbox(x, row_y, w, bh, ec, t, s)

# arrows with edge labels (like 'cosine >= 0.80')
ay = row_y + bh/2
arrow(qx+qw,            0.205,       ay, "leaf image")
arrow(0.205+0.155,      0.390,       ay, "top-5 shortlist  ✓ truth kept")
arrow(0.390+0.150,      0.560,       ay, "re-ranked top-1")
arrow(0.560+0.175,      0.760,       ay, "8 refs, matched budget")

# annotate stage bodies with the concrete outputs
ax.text(0.205+0.155/2, row_y-0.028, "  ".join(S["shortlist"][:3]) + " ...",
        ha="center", fontsize=7.4, color=BLUE, style="italic")
ax.text(0.390+0.150/2, row_y-0.028, S["ranked"][0], ha="center",
        fontsize=8.0, color=PURP, fontweight="bold")
ax.text(0.560+0.175/2, row_y-0.028, f"edge: {edge['a']} ↔ {edge['b']}",
        ha="center", fontsize=7.2, color=ORNG, fontweight="bold")

# Stage-4 verdict chips
def chip(cx, cy, text, ok):
    c = GREEN if ok else RED
    ax.add_patch(FancyBboxPatch((cx, cy), 0.15, 0.036, boxstyle="round,pad=0.002,rounding_size=0.02",
                 fc=("#eaf6ee" if ok else "#fbeaea"), ec=c, lw=1.6, zorder=5))
    ax.text(cx+0.075, cy+0.018, text, ha="center", va="center", fontsize=8.4,
            color=c, fontweight="bold", zorder=6)
chip(0.767, row_y-0.030, f"KG8 general → {S['KG8']['prediction'].split('_')[0]} ✓", S["KG8"]["correct"])
chip(0.767, row_y-0.076, f"KC8 contrast → {S['KC8']['prediction'].split('_')[0]} ✓", S["KC8"]["correct"])

# ---- Reference-evidence banner (below panel) --------------------------------
ax.text(0.035, 0.395, "Stage 3 contrasting references  (drawn from local CyAg; ground-truth label never used in retrieval)",
        fontsize=10.5, fontweight="bold", color=INK)

a_refs = [r for r in kc_refs if r["class"] == edge["a"]][:4]
b_refs = [r for r in kc_refs if r["class"] == edge["b"]][:4]
tx0, tw, tgap = 0.055, 0.088, 0.012
for j, (refs, col, name) in enumerate([(a_refs, ORNG, edge["a"]), (b_refs, PURP, edge["b"])]):
    ry = 0.235 - j*0.155
    ax.text(0.045, ry+0.055, name.replace("_", " "), rotation=90, ha="center", va="center",
            fontsize=8.6, fontweight="bold", color=col)
    for i, r in enumerate(refs):
        place_img(r["path"], tx0 + i*(tw+tgap), ry, tw, 0.115, col)

# differentiating knowledge + decisive cue box (right side)
kx = 0.475
ax.add_patch(FancyBboxPatch((kx, 0.055), 0.50, 0.32, boxstyle="round,pad=0.006,rounding_size=0.02",
             fc="#fff9ee", ec=GOLD_, lw=1.8, zorder=2))
ax.text(kx+0.02, 0.345, "Source-grounded differentiating knowledge (frozen)", fontsize=9.6,
        fontweight="bold", color=INK, zorder=3)
ax.text(kx+0.02, 0.235, textwrap.fill(dk[:300].strip(), 78), fontsize=8.1, color="#3b4a55",
        va="top", zorder=3, linespacing=1.4)
ax.text(kx+0.02, 0.150, "KC8 decisive cue  (sharpened by the contrast):", fontsize=9.2,
        fontweight="bold", color=GREEN, zorder=3)
ax.text(kx+0.02, 0.115, textwrap.fill(S["KC8"]["decisive_cue"][:230].strip(), 82), fontsize=8.1,
        color="#234", va="top", zorder=3, linespacing=1.4)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
print("wrote", OUT)
