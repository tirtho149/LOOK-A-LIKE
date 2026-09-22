#!/usr/bin/env python3
"""
GOLD-STANDARD single-image run (protocol §8 pilot gate / execution checklist).
=============================================================================
Runs the WHOLE pipeline on ONE held-out soybean query image, twice under a
matched 8-image budget:
    KG8  = KB + GENERAL references
    KC8  = KB + CONTRASTING (look-alike) references   <- proposed
Shared Stage-1/Stage-2 (cached) so only the reference policy differs — this is
the counterfactual/contrasting mechanism test the professor asked to see first.

Outputs (outputs/gold/<query_id>/):
    trace.jsonl   one record per stage/condition (auditable)
    trace.md      human-readable reasoning trace
    trace.tex     LaTeX snippet for the paper (Figure: worked example)
    summary.json  machine-readable verdict + transition
Usage:  python run_gold_example.py [--truth Frogeye_Leaf_Spot] [--budget 8]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import kb_loader as K
import pipeline as P
from vlm_backend import VLMBackend


def pick_query(cfg, truth: str, seed: int) -> Path:
    imgs = P.class_images(cfg, truth)
    if not imgs:
        sys.exit(f"No images for truth class {truth}")
    import random
    rnd = random.Random(f"gold:{seed}")
    rnd.shuffle(imgs)
    return imgs[0]


def emit(fp, rec):
    fp.write(json.dumps(rec) + "\n"); fp.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="Frogeye_Leaf_Spot")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    seed = cfg["seed"]
    k = args.k or cfg["candidate_k"]
    kb = K.load_kb(cfg)
    graph = K.load_lookalike_graph(cfg)
    vlm = VLMBackend(cfg)

    query = pick_query(cfg, args.truth, seed)
    qid = f"{args.truth}__{query.stem}"
    outdir = Path(cfg["experiment_output_dir"]) / "gold" / qid
    outdir.mkdir(parents=True, exist_ok=True)
    query_ids = {query.name}   # hold this image out of the support pool
    trace = (outdir / "trace.jsonl").open("w")

    print(f"[gold] query = {query}")
    print(f"[gold] truth = {args.truth}   budget = {args.budget}   k = {k}")
    emit(trace, {"stage": "meta", "query_id": qid, "query_path": str(query),
                 "truth": args.truth, "budget": args.budget, "k": k,
                 "kb_classes": sorted(kb), "model_id": vlm.model_id})

    # Stage 1 — broad KB screening
    print("[gold] Stage 1: broad KB screening ...")
    s1 = P.run_stage1(vlm, kb, query, k)
    emit(trace, {"stage": "stage1", **{x: s1[x] for x in ("shortlist",)},
                 "parsed": s1["parsed"], "latency_s": s1["raw"]["latency_s"]})
    shortlist = s1["shortlist"] or sorted(kb)[:k]
    truth_in_shortlist = args.truth in shortlist
    print(f"          shortlist = {shortlist}  (truth retained: {truth_in_shortlist})")

    # Stage 2 — focused candidate reasoning
    print("[gold] Stage 2: focused candidate comparison ...")
    s2 = P.run_stage2(vlm, kb, query, shortlist)
    ranked = s2["ranked"]
    emit(trace, {"stage": "stage2", "ranked": ranked, "top1": s2["top1"],
                 "parsed": s2["parsed"], "latency_s": s2["raw"]["latency_s"]})
    print(f"          ranked = {ranked}   stage2 top1 = {s2['top1']}")

    # Stages 3+4 under both reference policies (matched budget)
    conditions = {"KG8": "general", "KC8": "contrasting"}
    results = {}
    for code, policy in conditions.items():
        print(f"[gold] {code}: routing ({policy}) + final diagnosis ...")
        routing = P.route_references(cfg, graph, kb, ranked, args.budget,
                                     policy, query_ids, seed)
        emit(trace, {"stage": "stage3", "condition": code, "policy": routing["policy"],
                     "edge": (routing["edge"] and
                              {"a": routing["edge"]["a"], "b": routing["edge"]["b"],
                               "n_pairs": routing["edge"]["n_pairs"]}),
                     "rationale": routing["rationale"],
                     "refs": [{"class": r["class"], "role": r["role"],
                               "path": r["path"]} for r in routing["refs"]],
                     "diff_knowledge": routing.get("diff_knowledge", "")})
        s4 = P.run_stage4(vlm, kb, query, ranked, routing, policy)
        correct = (s4["prediction"] == args.truth)
        emit(trace, {"stage": "stage4", "condition": code,
                     "prediction": s4["prediction"], "correct": correct,
                     "decisive_cue": s4["decisive_cue"], "n_refs": s4["n_refs"],
                     "parsed": s4["parsed"], "latency_s": s4["raw"]["latency_s"]})
        results[code] = {"policy": routing["policy"], "prediction": s4["prediction"],
                         "correct": correct, "decisive_cue": s4["decisive_cue"],
                         "edge": routing["edge"], "routing": routing, "s4": s4}
        print(f"          {code} -> prediction = {s4['prediction']}  correct = {correct}")
    trace.close()

    # transition (counterfactual mechanism)
    kg, kc = results["KG8"], results["KC8"]
    transition = ("wrong->right" if (not kg["correct"] and kc["correct"]) else
                  "right->wrong" if (kg["correct"] and not kc["correct"]) else
                  "both-correct" if kg["correct"] else "both-wrong")
    summary = {"query_id": qid, "query_path": str(query), "truth": args.truth,
               "budget": args.budget, "k": k, "shortlist": shortlist,
               "truth_in_shortlist": truth_in_shortlist, "ranked": ranked,
               "KG8": {k2: kg[k2] for k2 in ("policy", "prediction", "correct", "decisive_cue")},
               "KC8": {k2: kc[k2] for k2 in ("policy", "prediction", "correct", "decisive_cue")},
               "contrasting_edge": (kc["edge"] and {"a": kc["edge"]["a"], "b": kc["edge"]["b"]}),
               "transition_general_to_contrasting": transition}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_markdown(outdir, cfg, kb, query, args, shortlist, ranked, s1, s2, results, summary)
    _write_tex(outdir, summary, results)
    print(f"\n[gold] DONE. transition = {transition}")
    print(f"[gold] outputs in {outdir}")
    print(json.dumps(summary, indent=2))


def _write_markdown(outdir, cfg, kb, query, args, shortlist, ranked, s1, s2, results, summary):
    L = []
    A = L.append
    A(f"# Gold-standard worked example — {args.truth}\n")
    A(f"**Query image:** `{query}`  ")
    A(f"**Ground truth:** `{args.truth}`  |  **Budget:** {args.budget} images  |  "
      f"**Model:** {summary.get('model_id','claude -p headless')}\n")
    A("## Stage 1 — Broad KB screening (recall-first)\n")
    of = (s1["parsed"] or {}).get("observed_features", [])
    A("Observed features: " + "; ".join(of[:6]) + "\n")
    A(f"**Top-{args.k or cfg['candidate_k']} candidates (shortlist):** "
      + ", ".join(f"`{c}`" for c in shortlist))
    A(f"\nTruth retained in shortlist: **{summary['truth_in_shortlist']}**\n")
    A("## Stage 2 — Focused candidate comparison\n")
    for c in (s2["parsed"] or {}).get("candidates", [])[:5]:
        A(f"- **{c.get('label')}** — support: {c.get('support_evidence','')[:140]}"
          f"  | contradicting: {c.get('contradicting_evidence','')[:120]}")
    A(f"\nStage-2 top-1: **{s2['top1']}**\n")
    for code in ("KG8", "KC8"):
        r = results[code]; rt = r["routing"]
        A(f"## Stage 3+4 — {code}  ({r['policy']})\n")
        if rt.get("edge"):
            e = rt["edge"]
            A(f"Look-alike edge selected: **{e['a']} ↔ {e['b']}** "
              f"(validation weight n={e['n_pairs']}).")
            A(f"\nDifferentiating knowledge (source-grounded):\n\n> {rt.get('diff_knowledge','')[:600]}\n")
        A(f"References ({len(rt['refs'])}): " +
          ", ".join(f"{Path(x['path']).name}→`{x['class']}`" for x in rt["refs"]) + "\n")
        A(f"**Prediction:** `{r['prediction']}`  |  correct: **{r['correct']}**")
        A(f"\nDecisive cue: {r['decisive_cue']}\n")
        es = (r["s4"]["parsed"] or {}).get("evidence_summary", "")
        if es:
            A(f"Evidence summary: {es[:400]}\n")
    A("## Verdict\n")
    A(f"General→Contrasting transition: **{summary['transition_general_to_contrasting']}**  ")
    A(f"(KG8 `{results['KG8']['prediction']}` → KC8 `{results['KC8']['prediction']}`, "
      f"truth `{args.truth}`).")
    (outdir / "trace.md").write_text("\n".join(L))


def _write_tex(outdir, summary, results):
    def esc(s): return str(s).replace("_", r"\_").replace("&", r"\&")
    kg, kc = results["KG8"], results["KC8"]
    tex = rf"""% Auto-generated worked-example trace (gold standard)
\begin{{table}}[t]\centering\small
\caption{{Worked example: KB-guided contrasting-reference diagnosis on one
held-out soybean query (ground truth: \texttt{{{esc(summary['truth'])}}}).
Stage~1/2 are shared; only the reference policy differs between KG8 and KC8.}}
\label{{tab:worked-example}}
\begin{{tabular}}{{ll}}
\toprule
Stage & Output \\ \midrule
S1 shortlist (top-{summary['k']}) & {esc(', '.join(summary['shortlist']))} \\
S1 truth retained & {summary['truth_in_shortlist']} \\
S2 top-1 & \texttt{{{esc(summary['ranked'][0] if summary['ranked'] else '')}}} \\
S3 contrasting edge & {esc((kc['edge'] or {}).get('a',''))} $\leftrightarrow$ {esc((kc['edge'] or {}).get('b',''))} \\
KG8 (general) prediction & \texttt{{{esc(kg['prediction'])}}} (correct: {kg['correct']}) \\
KC8 (contrasting) prediction & \texttt{{{esc(kc['prediction'])}}} (correct: {kc['correct']}) \\
Transition & {esc(summary['transition_general_to_contrasting'])} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (outdir / "trace.tex").write_text(tex)


if __name__ == "__main__":
    main()
