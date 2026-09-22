#!/usr/bin/env python3
"""
run_baselines.py — AgEval-style closed-list baselines (protocol §3).
===================================================================
Implements Z0 and G2/G4/G8 EXACTLY in AgEval's style: reuse the verbatim
`universal_prompt` from third_party_AgEval/inference.py, a closed candidate list
= full soybean taxonomy, and the {"prediction": ...} JSON contract. Few-shot
support images are labeled example images drawn ONLY from the local CyAg support
pool (never a query image, never the query's group).

For a B-shot baseline, B is the TOTAL number of reference images (protocol §3),
allocated round-robin across the taxonomy (AgEval samples random labeled shots;
we draw balanced labeled shots deterministically from the frozen support pool).

Same held-out query set as run_conditions.py. Resume-safe JSONL + AgEval-style CSV.

Flags: --conditions Z0,G8 --classes A,B --limit N --dry-run
Output: outputs/baselines/baseline_predictions.jsonl + baseline_results.csv
"""
from __future__ import annotations
import argparse, csv, json, sys, random
from pathlib import Path
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import kb_loader as K
import pipeline as P
from vlm_backend import VLMBackend
sys.path.insert(0, str(HERE / "third_party_AgEval"))
from inference import universal_prompt  # verbatim AgEval prompt

from run_conditions import DeterministicMockVLM, append_jsonl, load_done_keys, load_queries


def build_shots(cfg, taxonomy, budget, query_ids, seed):
    """Balanced labeled shots across the closed taxonomy from the CyAg support
    pool. Returns list of (path, label) example pairs, total == budget."""
    if budget == 0:
        return []
    rnd = random.Random(f"{seed}:baseline:{budget}")
    order = list(taxonomy)
    rnd.shuffle(order)
    shots, i = [], 0
    # round-robin one image per class until budget filled
    per_class_iter = {c: iter(P.sample_support(cfg, c, budget, query_ids, seed)) for c in order}
    while len(shots) < budget:
        progressed = False
        for c in order:
            if len(shots) >= budget:
                break
            try:
                p = next(per_class_iter[c])
                shots.append((str(p), c))
                progressed = True
            except StopIteration:
                continue
        if not progressed:
            break
    return shots[:budget]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="Z0,G2,G4,G8")
    ap.add_argument("--classes", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    seed = int(cfg["seed"])
    kb = K.load_kb(cfg)
    taxonomy = sorted(kb)   # closed candidate list = full eligible soybean taxonomy
    expected_classes = taxonomy

    want = set(args.conditions.split(","))
    baseline_specs = {c: v for c, v in cfg["conditions"].items()
                      if c in want and v["policy"] in ("none", "general") and not v["kb"]}
    if not baseline_specs:
        sys.exit(f"No AgEval baseline conditions matched {want} "
                 f"(valid: Z0,G2,G4,G8).")

    classes_filter = set(args.classes.split(",")) if args.classes else None
    queries = load_queries(cfg, classes_filter, args.limit)

    vlm = DeterministicMockVLM(cfg, kb) if args.dry_run else VLMBackend(cfg)

    out = Path(cfg["experiment_output_dir"]) / "outputs" / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    preds_p = out / "baseline_predictions.jsonl"
    done = load_done_keys(preds_p, lambda r: (r["query_id"], r["condition"], r["seed"]))

    prompt_tmpl = universal_prompt.format(expected_classes=expected_classes)

    rows = []
    for q in queries:
        qid, truth = q["query_id"], q["class"]
        qpath = q["_path"]
        query_ids = {qpath.name}
        for code, spec in baseline_specs.items():
            if (qid, code, str(seed)) in done:
                continue
            budget = spec["shots"]
            shots = build_shots(cfg, taxonomy, budget, query_ids, seed)
            # AgEval order: examples first, then the query prompt + query image
            specs = [(p, f"EXAMPLE labeled '{lab}'") for (p, lab) in shots]
            specs.append((str(qpath), "QUERY IMAGE (classify this)"))
            # shot labels are injected as text into the prompt for the frozen VLM
            shot_txt = ""
            if shots:
                shot_txt = "\nLabeled example images (in order):\n" + "\n".join(
                    f"  example {i+1}: {lab}" for i, (_, lab) in enumerate(shots)) + "\n"
            full_prompt = prompt_tmpl + shot_txt
            r = vlm.call(full_prompt, specs, tag=f"baseline_{code}")
            parsed = r["parsed"] or {}
            pred = parsed.get("prediction", "NA")
            rec = {"query_id": qid, "condition": code, "seed": seed, "truth": truth,
                   "prediction": pred, "correct": bool(pred == truth),
                   "shots": budget, "policy": spec["policy"], "kb": False,
                   "n_refs": len(shots), "latency_s": r["latency_s"],
                   "model_id": vlm.model_id}
            append_jsonl(preds_p, rec)
            done.add((qid, code, str(seed)))
            rows.append(rec)
        print(f"[baseline] {qid}: {len(baseline_specs)} conditions done")

    # AgEval-style CSV (wide by shot count)
    csv_p = out / "baseline_results.csv"
    all_rows = [json.loads(l) for l in preds_p.open()] if preds_p.exists() else rows
    fields = ["query_id", "condition", "seed", "truth", "prediction", "correct",
              "shots", "policy", "n_refs", "latency_s", "model_id"]
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\n[baseline] wrote {len(rows)} new records; CSV -> {csv_p}")
    print(f"[baseline] total model calls: {vlm.calls}")


if __name__ == "__main__":
    main()
