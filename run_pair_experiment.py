#!/usr/bin/env python3
"""
Focused look-alike pair experiment: does explicit distinguishing-feature
comparison reduce confusion BEYOND simply providing reference images?
====================================================================
Pair: bacterial_blight vs bacterial_pustule (top look-alike edge).
Three matched conditions per held-out test image (same test set, same references
between B and C):
    A  image + KB                        (initial prediction)
    B  image + KB + ordinary references  (references shown)
    C  image + KB + same references + structured feature comparison (proposed)

Reports, per disease: accuracy(A,B,C); errors FIXED and INTRODUCED by C vs B and
vs A; and a failure taxonomy (irrelevant refs / unsupported feature claim /
feature not visible). Writes evidence tables for inspection.

Usage:
    python run_pair_experiment.py [--a Bacterial_Blight --b Bacterial_Pustule]
                                  [--n-refs 4] [--limit N] [--seed 42]
"""
from __future__ import annotations
import argparse, csv, json, sys, collections
from pathlib import Path
import yaml

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import kb_loader as K
import pipeline as P
import pair_disambiguation as PD
from vlm_backend import VLMBackend


def load_test_queries(cfg, cls):
    """Held-out test images for `cls` from the frozen snapshot query_manifest."""
    mani = Path(cfg["experiment_output_dir"]) / "query_manifest.csv"
    out = []
    if mani.exists():
        for r in csv.DictReader(mani.open()):
            rc = r.get("class") or r.get("label")
            if rc == cls:
                path = r.get("snapshot_path") or r.get("path") or r.get("dst") or r.get("src_path")
                out.append({"query_id": r.get("query_id") or Path(path).stem, "path": path})
    if not out:  # fallback: sample from CyAg (deterministic), held out later
        for p in P.class_images(cfg, cls)[:5]:
            out.append({"query_id": f"{cls}__{p.stem}", "path": str(p)})
    return out


def taxonomy_for_error(rec):
    """Classify a condition-C error: irrelevant_refs | unsupported_claim | feature_not_visible."""
    table = rec.get("C_table", []) or []
    if not table:
        return "no_table"
    n = len(table)
    not_assessable = sum(1 for t in table if not t.get("assessable", True))
    if not_assessable >= max(1, n // 2):
        return "feature_not_visible"
    # supports that disagree with a clearly-stated KB signature would be unsupported;
    # heuristic: if the decision was a flip on a thin (<=1) margin, call it unsupported_claim
    if rec["C_changed"] and abs(rec["C_supA"] - rec["C_supB"]) <= 1:
        return "unsupported_claim"
    return "irrelevant_refs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="Bacterial_Blight")
    ap.add_argument("--b", default="Bacterial_Pustule")
    ap.add_argument("--n-refs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    seed = args.seed or cfg["seed"]
    kb = K.load_kb(cfg); graph = K.load_lookalike_graph(cfg)
    a, b = args.a, args.b
    for c in (a, b):
        if c not in kb:
            sys.exit(f"{c} has no frozen KB description; cannot run pair experiment.")
    vlm = VLMBackend(cfg)

    outdir = Path(cfg["experiment_output_dir"]) / "pair_experiment" / f"{a}__vs__{b}"
    outdir.mkdir(parents=True, exist_ok=True)
    crop_dir = outdir / "crops"

    # 1. relevant, documented pair (else skip)
    edge = PD.select_pair(graph, a, shortlist={a, b})
    if edge is None:
        sys.exit(f"No documented look-alike edge between {a} and {b}; skip comparison.")
    print(f"[pair] documented edge {edge['a']} <-> {edge['b']}  n={edge['n_pairs']}")

    # 3. distinguishing-feature checklist (once, cached)
    checklist = PD.extract_checklist(vlm, edge)["features"]
    (outdir / "checklist.json").write_text(json.dumps(checklist, indent=2))
    print(f"[pair] checklist features: {[f.get('feature') for f in checklist]}")

    # test queries (5/class) + held-out set = every test image (never used as ref)
    tests = []
    for cls in (a, b):
        tests += [{**t, "truth": cls} for t in load_test_queries(cfg, cls)]
    if args.limit:
        # balanced limit: first limit//2 per class
        per = collections.Counter()
        keep = []
        for t in tests:
            if per[t["truth"]] < max(1, args.limit // 2):
                keep.append(t); per[t["truth"]] += 1
        tests = keep
    held_out = {Path(t["path"]).name for t in tests}
    print(f"[pair] test images: {len(tests)}  (held out from references)")

    refs = PD.retrieve_pair_refs(cfg, edge, args.n_refs, held_out, seed)
    recf = (outdir / "records.jsonl").open("w")
    records = []
    for t in tests:
        q, truth = t["path"], t["truth"]
        A = PD.cond_kb_only(vlm, kb, q, a, b)
        init = A["prediction"] if A["prediction"] in (a, b) else a
        B = PD.cond_refs_only(vlm, kb, q, a, b, refs[a], refs[b])
        C = PD.cond_feature_compare(vlm, kb, q, a, b, refs[a], refs[b],
                                    checklist, edge, init, crop_dir=crop_dir)
        rec = {"query_id": t["query_id"], "truth": truth,
               "A_pred": init, "A_correct": init == truth,
               "B_pred": B["prediction"], "B_correct": B["prediction"] == truth,
               "C_pred": C["prediction"], "C_correct": C["prediction"] == truth,
               "C_decision": C["decision"], "C_changed": C["changed_from_initial"],
               "C_supA": C["supports_A"], "C_supB": C["supports_B"],
               "C_table": C["evidence_table"], "C_explanation": C["explanation"]}
        rec["C_taxonomy"] = None if rec["C_correct"] else taxonomy_for_error(rec)
        records.append(rec); recf.write(json.dumps(rec) + "\n"); recf.flush()
        print(f"  {t['query_id'][:34]:34s} truth={truth[:14]:14s} "
              f"A={init==truth} B={B['prediction']==truth} C={C['prediction']==truth} "
              f"({'chg' if C['changed_from_initial'] else 'keep'})")
    recf.close()

    summary = summarize(records, a, b)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_report(outdir, a, b, edge, checklist, records, summary, refs, args)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\n[pair] outputs in {outdir}")


def _acc(records, key, cls=None):
    R = [r for r in records if (cls is None or r["truth"] == cls)]
    return (sum(r[key] for r in R), len(R))


def summarize(records, a, b):
    s = {"n": len(records), "pair": [a, b], "accuracy": {}, "transitions": {}, "taxonomy": {}}
    for cond in ("A", "B", "C"):
        s["accuracy"][cond] = {
            "overall": _acc(records, f"{cond}_correct"),
            a: _acc(records, f"{cond}_correct", a),
            b: _acc(records, f"{cond}_correct", b)}
    # errors fixed / introduced by C relative to B and to A, per disease
    for base in ("A", "B"):
        fixed = collections.Counter(); intro = collections.Counter()
        for r in records:
            if not r[f"{base}_correct"] and r["C_correct"]:
                fixed[r["truth"]] += 1
            if r[f"{base}_correct"] and not r["C_correct"]:
                intro[r["truth"]] += 1
        s["transitions"][f"C_vs_{base}"] = {
            "fixed": {a: fixed[a], b: fixed[b], "total": sum(fixed.values())},
            "introduced": {a: intro[a], b: intro[b], "total": sum(intro.values())},
            "net": sum(fixed.values()) - sum(intro.values())}
    tax = collections.Counter(r["C_taxonomy"] for r in records if r["C_taxonomy"])
    s["taxonomy"] = dict(tax)
    # central question verdict
    bB = s["accuracy"]["B"]["overall"]; cC = s["accuracy"]["C"]["overall"]
    s["central_question"] = {
        "B_refs_only_acc": round(bB[0]/max(1, bB[1]), 3),
        "C_feature_compare_acc": round(cC[0]/max(1, cC[1]), 3),
        "structured_comparison_helps_beyond_refs":
            (cC[0]/max(1, cC[1])) > (bB[0]/max(1, bB[1]))}
    return s


def _write_report(outdir, a, b, edge, checklist, records, summary, refs, args):
    L = []; add = L.append
    add(f"# Look-alike pair disambiguation: {a} vs {b}\n")
    add(f"Documented edge weight n={edge['n_pairs']}. Test images: {summary['n']} "
        f"(held out from references). References: {args.n_refs}/class, same between B and C.\n")
    add("## Distinguishing-feature checklist (source-grounded)\n")
    for f in checklist:
        add(f"- **{f.get('feature')}** — {a}: {f.get('class_a_signature')} | "
            f"{b}: {f.get('class_b_signature')}")
    add("\n## Accuracy by condition\n")
    add("| Condition | Overall | " + f"{a} | {b} |")
    add("|---|---|---|---|")
    for cond, name in [("A", "A: image+KB"), ("B", "B: +ordinary refs"),
                       ("C", "C: +feature comparison")]:
        ac = summary["accuracy"][cond]
        def f(t): return f"{t[0]}/{t[1]}"
        add(f"| {name} | {f(ac['overall'])} | {f(ac[a])} | {f(ac[b])} |")
    add("\n## Errors fixed vs introduced by C (structured comparison)\n")
    for base in ("B", "A"):
        tr = summary["transitions"][f"C_vs_{base}"]
        add(f"- **vs {base}**: fixed {tr['fixed']['total']} "
            f"(A:{tr['fixed'][a]}, B:{tr['fixed'][b]}), introduced {tr['introduced']['total']} "
            f"(A:{tr['introduced'][a]}, B:{tr['introduced'][b]}), net {tr['net']:+d}")
    add(f"\n## Failure taxonomy (C errors)\n{summary['taxonomy']}\n")
    cq = summary["central_question"]
    add("## Central question\n")
    add(f"Does explicit feature comparison beat merely showing references? "
        f"**B (refs only) = {cq['B_refs_only_acc']}, C (feature comparison) = "
        f"{cq['C_feature_compare_acc']}** → "
        f"{'YES' if cq['structured_comparison_helps_beyond_refs'] else 'not on this sample'}.")
    (outdir / "report.md").write_text("\n".join(L))

    # one worked evidence table (first correct C disambiguation) as markdown + tex
    ex = next((r for r in records if r["C_correct"] and r["C_table"]), None)
    if ex:
        _write_evidence_table(outdir, a, b, ex)


def _write_evidence_table(outdir, a, b, rec):
    T = rec["C_table"]
    md = [f"# Evidence table — {rec['query_id']} (truth {rec['truth']})\n",
          "| Distinguishing feature | Class A refs | Class B refs | Query observation | Supports |",
          "|---|---|---|---|---|"]
    for t in T:
        md.append(f"| {t.get('feature','')} | {t.get('class_a_refs','')[:60]} | "
                  f"{t.get('class_b_refs','')[:60]} | {t.get('query_observation','')[:70]}"
                  f"{' [zoom]' if t.get('zoomed') else ''} | {t.get('supports','')} |")
    md.append(f"\nSupports A={rec['C_supA']}, B={rec['C_supB']} → decision "
              f"{rec['C_decision']} → **{rec['C_pred']}**. {rec['C_explanation'][:300]}")
    (outdir / "evidence_table_example.md").write_text("\n".join(md))

    def esc(s): return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    rows = "\n".join(
        rf"{esc(t.get('feature',''))[:26]} & {esc(t.get('class_a_refs',''))[:34]} & "
        rf"{esc(t.get('class_b_refs',''))[:34]} & {esc(t.get('query_observation',''))[:40]} & "
        rf"{esc(t.get('supports',''))} \\" for t in T)
    tex = rf"""% Structured distinguishing-feature evidence table (worked example)
\begin{{table*}}[t]\centering\footnotesize
\caption{{Structured distinguishing-feature comparison for {esc(a)} vs {esc(b)}
on a held-out query (truth: \texttt{{{esc(rec['truth'])}}}). A not-assessable
feature is not counted as evidence. Supports A={rec['C_supA']}, B={rec['C_supB']}
$\Rightarrow$ \texttt{{{esc(rec['C_pred'])}}}.}}
\label{{tab:evidence}}
\begin{{tabular}}{{p{{2.6cm}}p{{3.4cm}}p{{3.4cm}}p{{3.8cm}}c}}
\toprule
Distinguishing feature & Class A references & Class B references & Query observation & Supports \\ \midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table*}}"""
    (outdir / "evidence_table_example.tex").write_text(tex)


if __name__ == "__main__":
    main()
