#!/usr/bin/env python3
"""
run_conditions.py — main proposed-system driver (protocol §3, §4).
==================================================================
Iterates config.conditions × held-out queries, composing:
    Stage 1 (broad KB screening)  ->  Stage 2 (focused re-read)
    ->  route_references (Stage 3) ->  Stage 4 (final diagnosis).

Stage 1/2 are CACHED per query (KB two-stage conditions share them) so that
comparisons across reference policies ISOLATE the reference effect only
(protocol §3 "cache identical first-stage outputs").

Resume-safe: one JSONL record per (query, condition, seed) with atomic append;
already-completed records are skipped on re-run.

Outputs (under {experiment_output_dir}/outputs/):
    preds/final_predictions.jsonl   stage1.jsonl   stage2.jsonl   retrieval.jsonl

Flags: --conditions Z0,K0,KC8 --classes A,B --limit N --dry-run
  --dry-run uses a DeterministicMockVLM (no `claude -p` calls) so the whole
  composition, routing, budgets, schema and JSONL plumbing can be exercised
  with zero model/GPU cost.
"""
from __future__ import annotations
import argparse, csv, json, sys, hashlib
from pathlib import Path
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import kb_loader as K
import pipeline as P
from vlm_backend import VLMBackend


# ─────────────────────────────────────────────────────────────────────────────
#  Deterministic mock VLM for --dry-run (no API/GPU)
# ─────────────────────────────────────────────────────────────────────────────
class DeterministicMockVLM:
    """Mirrors VLMBackend.call() contract, returns schema-valid parsed dicts
    deterministically from a hash of the prompt+images. NO external calls."""
    def __init__(self, cfg, kb):
        self.model_id = "mock-deterministic"
        self.kb = kb
        self.labels = sorted(kb)
        self.calls = 0

    def _rng(self, prompt, image_specs):
        import random
        h = hashlib.sha256(prompt.encode())
        for p, c in image_specs:
            h.update(str(p).encode())
        return random.Random(h.hexdigest())

    def call(self, prompt, image_specs=None, expect_json=True, tag="call"):
        image_specs = image_specs or []
        self.calls += 1
        rng = self._rng(prompt, image_specs)
        if tag == "stage1":
            cands = rng.sample(self.labels, min(len(self.labels), 5))
            parsed = {"observed_features": ["mock lesion", "mock chlorosis"],
                      "candidates": [{"label": c, "rank": i + 1,
                                      "support_evidence": "mock", "missing_evidence": "mock"}
                                     for i, c in enumerate(cands)],
                      "missing_organs": []}
        elif tag == "stage2":
            # shortlist labels appear in prompt; re-rank a plausible subset
            present = [l for l in self.labels if f"- {l}:" in prompt]
            present = present or self.labels[:5]
            rng.shuffle(present)
            parsed = {"candidates": [{"label": c, "rank": i + 1,
                                      "support_evidence": "mock", "contradicting_evidence": "mock",
                                      "missing_evidence": "mock"} for i, c in enumerate(present)],
                      "top1": present[0], "note": "mock"}
        else:  # stage4 / baseline
            present = [l for l in self.labels if l in prompt]
            present = present or self.labels
            parsed = {"prediction": rng.choice(present), "evidence_summary": "mock",
                      "decisive_cue": "mock", "alternatives": [], "confidence": 0.5}
        return {"tag": tag, "prompt_chars": len(prompt), "n_images": len(image_specs),
                "raw": json.dumps(parsed), "parsed": parsed, "cache_key": "mock",
                "latency_s": 0.01, "model_id": self.model_id, "attempts": 1, "cached": False}


# ─────────────────────────────────────────────────────────────────────────────
#  Resume-safe JSONL
# ─────────────────────────────────────────────────────────────────────────────
def append_jsonl(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()


def load_done_keys(path: Path, keyfn) -> set:
    done = set()
    if path.exists():
        for line in path.open():
            try:
                done.add(keyfn(json.loads(line)))
            except Exception:
                continue
    return done


# ─────────────────────────────────────────────────────────────────────────────
#  Query loading (from frozen manifest)
# ─────────────────────────────────────────────────────────────────────────────
def load_queries(cfg, classes_filter=None, limit=None):
    out = Path(cfg["experiment_output_dir"])
    man = out / "query_manifest.csv"
    if not man.exists():
        sys.exit("query_manifest.csv not found — run preprocess_data.py first.")
    rows = list(csv.DictReader(man.open()))
    if classes_filter:
        rows = [r for r in rows if r["class"] in classes_filter]
    if limit:
        rows = rows[:limit]
    # prefer snapshot path if it exists, else src path (dry-run has no copies)
    for r in rows:
        p = Path(r["snapshot_path"])
        r["_path"] = p if p.exists() else Path(r["src_path"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default=None, help="comma list; default=all")
    ap.add_argument("--classes", default=None, help="comma list of classes")
    ap.add_argument("--limit", type=int, default=None, help="cap #queries")
    ap.add_argument("--dry-run", action="store_true",
                    help="use deterministic mock VLM (no claude -p / GPU calls)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    seed = int(cfg["seed"])
    k = int(cfg["candidate_k"])
    kb = K.load_kb(cfg)
    graph = K.load_lookalike_graph(cfg)

    conditions = cfg["conditions"]
    if args.conditions:
        want = set(args.conditions.split(","))
        conditions = {c: v for c, v in conditions.items() if c in want}
    classes_filter = set(args.classes.split(",")) if args.classes else None
    queries = load_queries(cfg, classes_filter, args.limit)

    vlm = DeterministicMockVLM(cfg, kb) if args.dry_run else VLMBackend(cfg)

    out = Path(cfg["experiment_output_dir"]) / "outputs"
    preds_p = out / "preds" / "final_predictions.jsonl"
    s1_p, s2_p, ret_p = out / "stage1.jsonl", out / "stage2.jsonl", out / "retrieval.jsonl"
    (out / "preds").mkdir(parents=True, exist_ok=True)

    done_pred = load_done_keys(preds_p, lambda r: (r["query_id"], r["condition"], r["seed"]))
    done_s1 = load_done_keys(s1_p, lambda r: r["query_id"])
    done_s2 = load_done_keys(s2_p, lambda r: r["query_id"])

    # in-memory Stage1/2 cache for this run (shared across KB conditions)
    stage_cache: dict[str, dict] = {}

    def kb_stages(qid, qpath):
        """Return cached (shortlist, ranked) for a KB two-stage query."""
        if qid in stage_cache:
            return stage_cache[qid]
        s1 = P.run_stage1(vlm, kb, qpath, k)
        shortlist = s1["shortlist"] or sorted(kb)[:k]
        s2 = P.run_stage2(vlm, kb, qpath, shortlist)
        ranked = s2["ranked"]
        if qid not in done_s1:
            append_jsonl(s1_p, {"query_id": qid, "shortlist": shortlist,
                                "parsed": s1["parsed"], "latency_s": s1["raw"]["latency_s"],
                                "model_id": vlm.model_id})
            done_s1.add(qid)
        if qid not in done_s2:
            append_jsonl(s2_p, {"query_id": qid, "shortlist": shortlist, "ranked": ranked,
                                "top1": s2["top1"], "parsed": s2["parsed"],
                                "latency_s": s2["raw"]["latency_s"], "model_id": vlm.model_id})
            done_s2.add(qid)
        stage_cache[qid] = (shortlist, ranked, s1, s2)
        return stage_cache[qid]

    n_written = 0
    for q in queries:
        qid, truth, qpath = q["query_id"], q["class"], q["_path"]
        query_ids = {qpath.name}   # hold out this exact image from support pool
        for code, spec in conditions.items():
            if (qid, code, str(seed)) in done_pred:
                continue
            shots, policy, use_kb = spec["shots"], spec["policy"], spec["kb"]

            if use_kb:
                shortlist, ranked, s1, s2 = kb_stages(qid, qpath)
            else:
                # non-KB conditions (Z0, G*) diagnose over full taxonomy, no shortlist
                shortlist = sorted(kb)
                ranked = sorted(kb)

            routing = P.route_references(cfg, graph, kb, ranked, shots, policy,
                                         query_ids, seed)
            append_jsonl(ret_p, {"query_id": qid, "condition": code, "seed": seed,
                                 "policy": routing["policy"],
                                 "edge": (routing.get("edge") and
                                          {"a": routing["edge"]["a"], "b": routing["edge"]["b"],
                                           "n_pairs": routing["edge"]["n_pairs"]}),
                                 "n_refs": len(routing["refs"]),
                                 "refs": [{"class": r["class"], "role": r["role"]}
                                          for r in routing["refs"]],
                                 "rationale": routing["rationale"]})

            s4 = P.run_stage4(vlm, kb, qpath, ranked, routing, policy)
            pred = s4["prediction"]
            rec = {"query_id": qid, "condition": code, "seed": seed, "truth": truth,
                   "prediction": pred, "correct": bool(pred == truth),
                   "shots": shots, "policy": routing["policy"], "kb": use_kb,
                   "shortlist": shortlist if use_kb else None,
                   "truth_in_shortlist": (truth in shortlist) if use_kb else None,
                   "n_refs": s4["n_refs"], "decisive_cue": s4.get("decisive_cue", ""),
                   "edge": (routing.get("edge") and
                            {"a": routing["edge"]["a"], "b": routing["edge"]["b"]}),
                   "latency_s": s4["raw"]["latency_s"], "model_id": vlm.model_id}
            append_jsonl(preds_p, rec)
            done_pred.add((qid, code, str(seed)))
            n_written += 1
        print(f"[run] {qid}: {len(conditions)} conditions done")

    print(f"\n[run] wrote {n_written} new prediction records to {preds_p}")
    print(f"[run] total model calls: {vlm.calls}")


if __name__ == "__main__":
    main()
