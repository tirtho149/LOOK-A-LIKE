#!/usr/bin/env python3
"""
make_report.py — assemble the README-style results report (protocol §8, final).
==============================================================================
Reads the frozen manifests + results_summary.json and writes a human-readable
report (outputs/report/REPORT.md) plus a machine-readable results_summary.json
copy. Guards missing pieces so it runs on partial data. Labels everything
SOYBEAN-ONLY / PRELIMINARY.
"""
from __future__ import annotations
import json, csv
from pathlib import Path
import yaml

HERE = Path(__file__).parent


def read_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def read_csv(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []


def main():
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    out = Path(cfg["experiment_output_dir"])
    rep_dir = out / "outputs" / "report"
    rep_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = read_json(out / "run_manifest.json")
    split_audit = read_json(out / "split_audit.json")
    class_rows = read_csv(out / "class_manifest.csv")
    pair_rows = read_csv(out / "pair_manifest.csv")
    summary = read_json(out / "outputs" / "stats" / "results_summary.json")

    L, A = [], None
    A = L.append
    A("# Soybean Contrasting-Reference KB — Results Report")
    A("\n> **PRELIMINARY / SOYBEAN-ONLY prototype.** Five held-out queries per "
      "class; per-class p-values are not credible. All formal tests pool paired "
      "query outcomes while preserving class structure (protocol §6).\n")

    A("## Reproducibility contract")
    A(f"- git commit: `{run_manifest.get('git_commit','?')}`")
    A(f"- config sha256: `{run_manifest.get('config_sha256','?')}`")
    A(f"- model id: `{run_manifest.get('model_id','?')}`  | seed: "
      f"`{run_manifest.get('seed','?')}`")
    A(f"- snapshot dir: `{run_manifest.get('snapshot_dir','?')}`")
    if run_manifest.get("manifest_sha256"):
        A("- frozen manifest checksums:")
        for f, h in run_manifest["manifest_sha256"].items():
            A(f"  - `{f}`: `{h[:16]}…`")

    A("\n## Dataset & split audit")
    A(f"- classes: {split_audit.get('n_classes','?')}  | queries total: "
      f"{split_audit.get('n_queries_total','?')}  | queries/class: "
      f"{split_audit.get('queries_per_class','?')}")
    A(f"- support general: {split_audit.get('n_support_general','?')}  | "
      f"look-alike: {split_audit.get('n_support_lookalike','?')}  | "
      f"pairs: {split_audit.get('n_pairs_selected','?')}  | "
      f"exclusions: {split_audit.get('n_exclusions','?')}")
    if class_rows:
        A("\n| class | queries | gen | look-alike | edges | source |")
        A("|---|---|---|---|---|---|")
        for r in class_rows:
            A(f"| {r['class']} | {r['n_queries']} | {r['n_support_general']} | "
              f"{r['n_support_lookalike']} | {r['n_lookalike_edges']} | "
              f"{'yes' if r.get('source_url') else '—'} |")

    A("\n## Look-alike pairs (frozen, validation-only weights)")
    if pair_rows:
        A("\n| pair_id | a | b | n_pairs (val) | validated |")
        A("|---|---|---|---|---|")
        for r in pair_rows[:30]:
            A(f"| {r['pair_id']} | {r['source_class']} | {r['target_class']} | "
              f"{r['n_pairs_validation']} | {r['validated']} |")
    else:
        A("_(no pairs induced on the selected classes)_")

    A("\n## Recognition results per condition")
    pc = summary.get("per_condition", {})
    if pc:
        A("\n| condition | n | accuracy | 95% CI | macro-F1 | bal.acc | mean refs |")
        A("|---|---|---|---|---|---|---|")
        for cond, m in sorted(pc.items()):
            ci = m.get("accuracy_boot_ci", [0, 0])
            A(f"| {cond} | {m['n']} | {m['accuracy']:.3f} | "
              f"[{ci[0]:.3f}, {ci[1]:.3f}] | {m['macro_f1']:.3f} | "
              f"{m['balanced_accuracy']:.3f} | {m.get('mean_n_refs',0):.1f} |")
    else:
        A("_(no predictions yet — run run_conditions.py / run_baselines.py, then "
          "compute_statistics.py)_")

    A("\n## Primary & confirmatory contrasts (Holm-adjusted)")
    contrasts = summary.get("contrasts", {})
    if contrasts:
        A("\n| contrast | Δacc | b | c | exact p | Holm p | Δ 95% CI |")
        A("|---|---|---|---|---|---|---|")
        for name, c in contrasts.items():
            ci = c.get("delta_boot_ci", [0, 0])
            A(f"| {name} | {c['delta_acc']:+.3f} | {c['discordant_b']} | "
              f"{c['discordant_c']} | {c['p_value']:.4f} | "
              f"{c.get('p_holm','—') if isinstance(c.get('p_holm'),str) else (('%.4f'%c['p_holm']) if c.get('p_holm') is not None else '—')} | "
              f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    else:
        A("_(no contrasts computed yet)_")

    cq = summary.get("cochran_q")
    if cq and "Q" in cq:
        A(f"\n**Cochran's Q** over {cq.get('conditions')}: Q={cq['Q']:.3f}, "
          f"df={cq['df']}, p={cq['p_value']:.4f}")

    net = summary.get("net_correction_KC8_vs_KG8")
    if net:
        A(f"\n**Counterfactual net correction (KG8→KC8):** wrong→right="
          f"{net['wrong_to_right']}, right→wrong={net['right_to_wrong']}, "
          f"net={net['net_correction']:+d}, exact p={net['p_value']:.4f}")

    A("\n## Figures")
    figdir = out / "outputs" / "figures"
    if figdir.exists():
        for f in sorted(figdir.glob("*.png")):
            A(f"- `{f.name}`")

    A("\n---\n_Generated by make_report.py. Findings are preliminary and "
      "soybean-only; a larger evaluation is required for agricultural claims._")

    (rep_dir / "REPORT.md").write_text("\n".join(L))
    # machine-readable copy
    (rep_dir / "results_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[report] wrote {rep_dir/'REPORT.md'} and results_summary.json")


if __name__ == "__main__":
    main()
