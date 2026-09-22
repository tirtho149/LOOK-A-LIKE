#!/usr/bin/env python3
"""
make_plots.py — the 9 required figures (protocol §7).
====================================================
Fig1 data-flow schematic          Fig2 shot-efficiency curves
Fig3 top-k candidate recall        Fig4 forest plot of paired deltas
Fig5 confusion matrices Z0/G8/K0/KC8   Fig6 look-alike edge heatmap
Fig7 counterfactual flip plot      Fig8 per-class delta waterfall
Fig9 accuracy-cost Pareto

Saves PDF+PNG to outputs/figures/. MUST run on synthetic/partial data without
crashing: every figure guards empty inputs and draws an explanatory placeholder
instead of raising.
"""
from __future__ import annotations
import json, csv, sys
from collections import defaultdict
from pathlib import Path
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent


def _save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _empty(out, name, msg):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True, fontsize=11)
    ax.axis("off")
    _save(fig, out, name)


def load_preds(cfg):
    root = Path(cfg["experiment_output_dir"]) / "outputs"
    recs = []
    for p in [root / "preds" / "final_predictions.jsonl",
              root / "baselines" / "baseline_predictions.jsonl"]:
        if p.exists():
            for l in p.open():
                try:
                    recs.append(json.loads(l))
                except Exception:
                    pass
    by_q = defaultdict(dict)
    for r in recs:
        by_q[r["query_id"]][r["condition"]] = r
    return by_q, recs


def load_summary(cfg):
    p = Path(cfg["experiment_output_dir"]) / "outputs" / "stats" / "results_summary.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def load_graph(cfg):
    p = Path(cfg["experiment_output_dir"]) / "graph.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def acc_of(by_q, cond):
    rows = [cm[cond] for cm in by_q.values() if cond in cm]
    return (sum(int(r["correct"]) for r in rows) / len(rows)) if rows else None


# ── Fig 1: data-flow schematic ──
def fig1(out, cfg):
    fig, ax = plt.subplots(figsize=(10, 3))
    boxes = ["Full soybean\ntaxonomy", "Stage 1\nKB broad screen",
             "Stage 2\nfocused re-read", "Stage 3\npair routing",
             "Stage 4\nfinal diagnosis"]
    x = np.linspace(0.05, 0.85, len(boxes))
    for i, (xi, b) in enumerate(zip(x, boxes)):
        ax.add_patch(plt.Rectangle((xi, 0.35), 0.13, 0.3, fill=True,
                                   facecolor="#dfeaf5", edgecolor="black"))
        ax.text(xi + 0.065, 0.5, b, ha="center", va="center", fontsize=8)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x[i + 1], 0.5), xytext=(xi + 0.13, 0.5),
                        arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Figure 1: Method schematic (taxonomy → KB shortlist → pair routing → decision)")
    _save(fig, out, "fig1_dataflow")


# ── Fig 2: shot-efficiency curves ──
def fig2(out, by_q):
    families = {"Z/G": ["Z0", "G2", "G4", "G8"], "KG": ["K0", "KG2", "KG4", "KG8"],
                "KS": ["K0", "KS2", "KS4", "KS8"], "KC": ["K0", "KC2", "KC4", "KC8"]}
    shots = [0, 2, 4, 8]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    plotted = False
    for fam, conds in families.items():
        ys = [acc_of(by_q, c) for c in conds]
        xs = [s for s, y in zip(shots, ys) if y is not None]
        yy = [y for y in ys if y is not None]
        if yy:
            ax.plot(xs, yy, marker="o", label=fam)
            plotted = True
    if not plotted:
        return _empty(out, "fig2_shot_efficiency", "Fig 2: no data yet")
    ax.set_xlabel("Reference images (B)"); ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Figure 2: Shot-efficiency curves"); ax.legend()
    _save(fig, out, "fig2_shot_efficiency")


# ── Fig 3: top-k candidate recall ──
def fig3(out, cfg, by_q):
    root = Path(cfg["experiment_output_dir"]) / "outputs"
    s1 = root / "stage1.jsonl"
    ks = [1, 3, 5, 8, 10, 15]
    if not s1.exists():
        return _empty(out, "fig3_topk_recall", "Fig 3: no stage1.jsonl yet")
    # truth retained at rank<=k across queries
    truths = {}
    for cm in by_q.values():
        for r in cm.values():
            truths.setdefault(r["query_id"], r["truth"])
    recalls = {k: [0, 0] for k in ks}
    for l in s1.open():
        try:
            d = json.loads(l)
        except Exception:
            continue
        qid = d["query_id"]
        truth = truths.get(qid)
        shortlist = d.get("shortlist", [])
        if truth is None:
            continue
        rank = shortlist.index(truth) + 1 if truth in shortlist else 999
        for k in ks:
            recalls[k][1] += 1
            if rank <= k:
                recalls[k][0] += 1
    xs = [k for k in ks if recalls[k][1] > 0]
    ys = [recalls[k][0] / recalls[k][1] for k in xs]
    if not xs:
        return _empty(out, "fig3_topk_recall", "Fig 3: no shortlist data")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="s")
    ax.axhline(cfg.get("recall_target", 0.95), ls="--", color="grey",
               label=f"target {cfg.get('recall_target',0.95)}")
    ax.set_xlabel("k"); ax.set_ylabel("Top-k truth-retention recall")
    ax.set_title("Figure 3: Candidate recall curve"); ax.legend()
    _save(fig, out, "fig3_topk_recall")


# ── Fig 4: forest plot of paired deltas ──
def fig4(out, summary):
    contrasts = (summary or {}).get("contrasts", {})
    if not contrasts:
        return _empty(out, "fig4_forest", "Fig 4: no contrasts computed yet")
    names, deltas, los, his = [], [], [], []
    for name, c in contrasts.items():
        names.append(name)
        deltas.append(c["delta_acc"])
        ci = c.get("delta_boot_ci", [c["delta_acc"], c["delta_acc"]])
        los.append(ci[0]); his.append(ci[1])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(names) + 2))
    ax.errorbar(deltas, y, xerr=[np.array(deltas) - np.array(los),
                                 np.array(his) - np.array(deltas)],
                fmt="o", capsize=3)
    ax.axvline(0, color="grey", ls="--")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("Paired Δ top-1 accuracy (95% CI)")
    ax.set_title("Figure 4: Forest plot of planned paired deltas")
    _save(fig, out, "fig4_forest")


# ── Fig 5: confusion matrices Z0/G8/K0/KC8 ──
def fig5(out, by_q):
    conds = ["Z0", "G8", "K0", "KC8"]
    labels = sorted({r["truth"] for cm in by_q.values() for r in cm.values()})
    if not labels:
        return _empty(out, "fig5_confusion", "Fig 5: no data yet")
    idx = {l: i for i, l in enumerate(labels)}
    fig, axes = plt.subplots(1, len(conds), figsize=(4 * len(conds), 4))
    if len(conds) == 1:
        axes = [axes]
    any_data = False
    for ax, cond in zip(axes, conds):
        M = np.zeros((len(labels), len(labels)))
        rows = [cm[cond] for cm in by_q.values() if cond in cm]
        for r in rows:
            t = idx.get(r["truth"]); p = idx.get(r.get("prediction"))
            if t is not None and p is not None:
                M[t, p] += 1
        rs = M.sum(axis=1, keepdims=True)
        Mn = np.divide(M, rs, out=np.zeros_like(M), where=rs > 0)
        if rows:
            any_data = True
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{cond} (n={len(rows)})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Figure 5: Normalized confusion matrices (identical class order)")
    if not any_data:
        plt.close(fig)
        return _empty(out, "fig5_confusion", "Fig 5: conditions not present yet")
    _save(fig, out, "fig5_confusion")


# ── Fig 6: look-alike edge heatmap (baseline vs proposed confusion) ──
def fig6(out, by_q, graph):
    edges = (graph or {}).get("edges", [])
    if not edges:
        return _empty(out, "fig6_lookalike_heatmap", "Fig 6: no look-alike edges")
    rows = []
    for e in edges:
        a, b = e["a"], e["b"]
        def confuse(cond):
            n = m = 0
            for cm in by_q.values():
                if cond in cm:
                    r = cm[cond]
                    if r["truth"] in (a, b):
                        n += 1
                        if r.get("prediction") in (a, b) and r.get("prediction") != r["truth"]:
                            m += 1
            return m / n if n else 0.0
        rows.append((f"{a[:10]}/{b[:10]}", confuse("G8"), confuse("KC8")))
    if not rows:
        return _empty(out, "fig6_lookalike_heatmap", "Fig 6: no covered edges")
    labels = [r[0] for r in rows]
    data = np.array([[r[1], r[2], r[1] - r[2]] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(rows) + 2))
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["G8 confuse", "KC8 confuse", "reduction"])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax); ax.set_title("Figure 6: Look-alike edge confusion")
    _save(fig, out, "fig6_lookalike_heatmap")


# ── Fig 7: counterfactual/contrasting flip plot ──
def fig7(out, summary):
    net = (summary or {}).get("net_correction_KC8_vs_KG8")
    if not net:
        return _empty(out, "fig7_flip", "Fig 7: no KG8/KC8 transition data")
    cats = ["wrong→right", "right→wrong", "unchanged", "wrong→diff-wrong"]
    vals = [net["wrong_to_right"], net["right_to_wrong"],
            net["unchanged"], net["wrong_to_diff_wrong"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(cats, vals, color=["#2ca02c", "#d62728", "#7f7f7f", "#ff7f0e"])
    ax.set_ylabel("Queries")
    ax.set_title(f"Figure 7: KG8→KC8 flips (net correction = {net['net_correction']:+d})")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, out, "fig7_flip")


# ── Fig 8: per-class delta waterfall (KC8 - G8) ──
def fig8(out, by_q, graph):
    a, b = "KC8", "G8"
    by_class = defaultdict(lambda: {"a": [0, 0], "b": [0, 0]})
    for cm in by_q.values():
        if a in cm and b in cm:
            cls = cm[a]["truth"]
            by_class[cls]["a"][1] += 1; by_class[cls]["a"][0] += int(cm[a]["correct"])
            by_class[cls]["b"][1] += 1; by_class[cls]["b"][0] += int(cm[b]["correct"])
    items = []
    edges = (graph or {}).get("edges", [])
    edge_count = defaultdict(int)
    for e in edges:
        edge_count[e["a"]] += 1; edge_count[e["b"]] += 1
    for cls, d in by_class.items():
        if d["a"][1] and d["b"][1]:
            delta = d["a"][0] / d["a"][1] - d["b"][0] / d["b"][1]
            items.append((cls, delta, edge_count.get(cls, 0)))
    if not items:
        return _empty(out, "fig8_waterfall", "Fig 8: no paired KC8/G8 classes")
    items.sort(key=lambda x: x[1])
    labels = [f"{c} (e={e})" for c, _, e in items]
    vals = [v for _, v, _ in items]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(items) + 2))
    ax.barh(range(len(items)), vals,
            color=["#2ca02c" if v >= 0 else "#d62728" for v in vals])
    ax.axvline(0, color="grey")
    ax.set_yticks(range(len(items))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Per-class Δ accuracy (KC8 − G8)")
    ax.set_title("Figure 8: Per-class delta waterfall")
    _save(fig, out, "fig8_waterfall")


# ── Fig 9: accuracy-cost Pareto ──
def fig9(out, summary):
    pc = (summary or {}).get("per_condition", {})
    if not pc:
        return _empty(out, "fig9_pareto", "Fig 9: no per-condition metrics")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for cond, m in pc.items():
        cost = (m.get("mean_n_refs") or 0) + (m.get("mean_latency_s") or 0)
        ax.scatter(cost, m["accuracy"])
        ax.annotate(cond, (cost, m["accuracy"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Cost proxy (mean refs + latency s)")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Figure 9: Accuracy–cost Pareto")
    _save(fig, out, "fig9_pareto")


def main():
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    out = Path(cfg["experiment_output_dir"]) / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    by_q, recs = load_preds(cfg)
    summary = load_summary(cfg)
    graph = load_graph(cfg)

    fig1(out, cfg)
    fig2(out, by_q)
    fig3(out, cfg, by_q)
    fig4(out, summary)
    fig5(out, by_q)
    fig6(out, by_q, graph)
    fig7(out, summary)
    fig8(out, by_q, graph)
    fig9(out, summary)
    print(f"[plots] wrote 9 figures (PDF+PNG) to {out}  "
          f"[{len(by_q)} queries, {len(recs)} records]")


if __name__ == "__main__":
    main()
