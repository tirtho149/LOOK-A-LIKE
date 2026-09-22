#!/usr/bin/env python3
"""
build_lookalike_graph.py — freeze the validated look-alike graph (protocol §2,§4).
=================================================================================
Freezes kb_loader.load_lookalike_graph into outputs/graph.json with edge weights
derived from VALIDATION-ONLY statistics (n_pairs from the frozen look-alike
dataset). The held-out TEST labels are NEVER used to build or tune the graph.

Restricted to the selected/eligible soybean classes (read from class_manifest if
present, otherwise every KB class with enough images).

Usage:  python build_lookalike_graph.py
Output: outputs/graph.json  { nodes, edges[{pair_id,a,b,weight,n_pairs,...}] }
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import kb_loader as K
import pipeline as P


def selected_classes(cfg, kb) -> list[str]:
    out = Path(cfg["experiment_output_dir"]) / "class_manifest.csv"
    if out.exists():
        with open(out) as f:
            return [r["class"] for r in csv.DictReader(f)]
    minimg = int(cfg["min_images_per_eligible_class"])
    return [c for c in sorted(kb) if len(P.class_images(cfg, c)) >= minimg]


def main():
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    kb = K.load_kb(cfg)
    graph = K.load_lookalike_graph(cfg)
    sel = set(selected_classes(cfg, kb))

    edges = [e for e in graph if e["a"] in sel and e["b"] in sel]
    if not edges:
        print("[warn] no look-alike edges induced on selected classes.")

    max_n = max((e["n_pairs"] for e in edges), default=1) or 1
    out_edges = []
    for e in edges:
        out_edges.append({
            "pair_id": e["pair_id"], "a": e["a"], "b": e["b"],
            "display_a": e["display_a"], "display_b": e["display_b"],
            "n_pairs": e["n_pairs"],
            # weight = validation-only normalized co-confusion evidence in [0,1]
            "weight": round(e["n_pairs"] / max_n, 6),
            "difficulty": e.get("difficulty", ""),
            "source": e.get("source", ""),
            "validated": e.get("validated", True),
            "diagnostic_knowledge": e.get("diagnostic_knowledge", "")[:1500],
            "provenance": "validation-only n_pairs (no test labels)",
        })
    # deterministic priority order (freeze): higher weight, then pair_id
    out_edges.sort(key=lambda x: (-x["weight"], str(x["pair_id"])))

    graph_obj = {
        "nodes": sorted(sel),
        "n_nodes": len(sel),
        "n_edges": len(out_edges),
        "weight_definition": "n_pairs (validation co-confusion) / max_n_pairs; "
                             "NEVER built or tuned on held-out test labels",
        "edges": out_edges,
    }
    dst = Path(cfg["experiment_output_dir"]) / "graph.json"
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(graph_obj, indent=2))
    tmp.replace(dst)
    print(f"[graph] froze {len(out_edges)} edges over {len(sel)} nodes -> {dst}")
    for e in out_edges[:8]:
        print(f"   {e['a']} <-> {e['b']}  w={e['weight']:.3f}  n={e['n_pairs']}")


if __name__ == "__main__":
    main()
