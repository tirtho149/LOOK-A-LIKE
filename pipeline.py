#!/usr/bin/env python3
"""
Four-stage KB-guided contrasting-reference diagnosis pipeline (protocol §4).
============================================================================
Stage 1  Broad screening      query + ALL soybean KB descriptions -> top-k candidates
Stage 2  Focused comparison   query + shortlist descriptions       -> re-ranked + evidence
Stage 3  Reference routing     look-alike subgraph + budget B        -> selected references
Stage 4  Final diagnosis       query + candidate desc + exactly B refs -> final class

Reference IMAGES are always drawn from the LOCAL CyAg curated dataset
(supports/ pool), never from the query and never using the ground-truth label.
Built on AgEval's closed-list JSON-prediction convention (third_party_AgEval).
"""
from __future__ import annotations
import json, random, hashlib
from pathlib import Path
import kb_loader as K

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
#  Local CyAg support pool  (protocol §2: support = SAGE curated dataset)
# ─────────────────────────────────────────────────────────────────────────────
def class_images(cfg, cls: str) -> list[Path]:
    d = Path(cfg["sage_curated_dataset_dir"]) / cfg["sage_curated_images_subdir"] / cls
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def sample_support(cfg, cls: str, n: int, exclude: set[str], seed: int) -> list[Path]:
    """n support images for `cls` from local CyAg, disjoint from `exclude` (query ids)."""
    pool = [p for p in class_images(cfg, cls) if p.name not in exclude]
    rnd = random.Random(f"{seed}:{cls}")
    rnd.shuffle(pool)
    return pool[:n]


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt builders  (auditable structured evidence, not hidden chain-of-thought)
# ─────────────────────────────────────────────────────────────────────────────
def _kb_block(kb: dict, classes=None) -> str:
    classes = classes or sorted(kb)
    out = []
    for c in classes:
        e = kb[c]
        src = f"  (source: {e['source_url']})" if e.get("source_url") else ""
        out.append(f"- {c}: {e['description']}{src}")
    return "\n".join(out)


def stage1_prompt(kb: dict, k: int) -> str:
    labels = sorted(kb)
    return f"""You are a soybean disease diagnostician performing BROAD SCREENING.
You are shown ONE query image. Using ONLY the source-grounded visual knowledge
base below, return the most plausible candidate diseases — favour RECALL (do not
prematurely discard). The full soybean label set:
{', '.join(labels)}

SOURCE-GROUNDED VISUAL KNOWLEDGE BASE (frozen; never derived from this image):
{_kb_block(kb)}

Report the observed features in the query, then rank the top {k} candidates.
JSON schema:
{{"observed_features": ["..."],
  "candidates": [{{"label": "<one of the list>", "rank": 1,
                   "support_evidence": "...", "missing_evidence": "..."}}],
  "missing_organs": ["..."]}}"""


def stage2_prompt(kb: dict, shortlist: list[str]) -> str:
    return f"""You are performing FOCUSED COMPARISON on a shortlist of candidate
soybean diseases for the SAME query image. Re-read ONLY these candidate
descriptions and re-rank them, giving supporting AND contradicting evidence
for each from what is visible in the query.

CANDIDATE DESCRIPTIONS (source-grounded):
{_kb_block(kb, shortlist)}

JSON schema:
{{"candidates": [{{"label": "...", "rank": 1,
   "support_evidence": "...", "contradicting_evidence": "...",
   "missing_evidence": "..."}}],
  "top1": "...", "note": "..."}}"""


def stage4_prompt(kb: dict, shortlist: list[str], refs: list[dict],
                  diff_knowledge: str, policy: str) -> str:
    ref_lines = []
    for i, r in enumerate(refs, 1):
        ref_lines.append(f"  REFERENCE {i} — confirmed example of '{r['class']}'"
                         f" (role={r['role']})")
    ref_block = "\n".join(ref_lines) if ref_lines else "  (no reference images)"
    dk = f"\nSOURCE-GROUNDED DIFFERENTIATING KNOWLEDGE for the contrasting pair:\n{diff_knowledge}\n" \
        if (policy in ("contrasting", "counterfactual") and diff_knowledge) else ""
    return f"""You are making the FINAL DIAGNOSIS for the query image (shown first).
You are also given {len(refs)} labeled REFERENCE images from a curated dataset —
compare the query against them. The reference set was chosen to contrast the
most confusable candidates; decide which the query matches.

CANDIDATE DESCRIPTIONS (source-grounded):
{_kb_block(kb, shortlist)}
{dk}
REFERENCE IMAGES (in order after the query):
{ref_block}

Weigh supporting vs contradicting evidence and the differentiating cues, then
commit to ONE label from: {', '.join(shortlist)}.
JSON schema:
{{"prediction": "<one label>",
  "evidence_summary": "...",
  "decisive_cue": "...",
  "alternatives": ["..."],
  "confidence": 0.0}}"""


# ─────────────────────────────────────────────────────────────────────────────
#  Stages
# ─────────────────────────────────────────────────────────────────────────────
def run_stage1(vlm, kb, query_img: Path, k: int) -> dict:
    r = vlm.call(stage1_prompt(kb, k), [(str(query_img), "QUERY IMAGE")], tag="stage1")
    p = r["parsed"] or {}
    cands = [c.get("label") for c in p.get("candidates", []) if c.get("label") in kb]
    # keep order, dedup, cap to k
    seen, shortlist = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c); shortlist.append(c)
    return {"raw": r, "parsed": p, "shortlist": shortlist[:k]}


def run_stage2(vlm, kb, query_img: Path, shortlist: list[str]) -> dict:
    r = vlm.call(stage2_prompt(kb, shortlist),
                 [(str(query_img), "QUERY IMAGE")], tag="stage2")
    p = r["parsed"] or {}
    ranked = [c.get("label") for c in p.get("candidates", []) if c.get("label") in shortlist]
    ranked = ranked or shortlist
    return {"raw": r, "parsed": p, "ranked": ranked, "top1": p.get("top1", ranked[0])}


def route_references(cfg, graph, kb, ranked: list[str], budget: int, policy: str,
                     query_ids: set[str], seed: int) -> dict:
    """Stage 3 — deterministic routing. Returns selected reference specs + rationale.
    policy: general | similarity | contrasting | counterfactual | none."""
    if budget == 0 or policy == "none":
        return {"refs": [], "edge": None, "policy": policy, "rationale": "0-shot"}

    edge, diff_knowledge = None, ""
    refs: list[dict] = []
    if policy in ("contrasting", "counterfactual"):
        # induce look-alike subgraph over the shortlist; pick highest-priority edge
        sub = [e for e in graph
               if e["a"] in ranked and e["b"] in ranked]
        # priority: more validation evidence (n_pairs) first, then deterministic pair_id
        sub.sort(key=lambda e: (-e["n_pairs"], e["pair_id"]))
        if sub:
            edge = sub[0]
            diff_knowledge = edge["diagnostic_knowledge"]
            half = budget // 2
            for cls, role in ((edge["a"], "contrast-A"), (edge["b"], "contrast-B")):
                for p in sample_support(cfg, cls, half, query_ids, seed):
                    refs.append({"class": cls, "role": role, "path": str(p),
                                 "pair_id": edge["pair_id"]})
    if not refs:  # general fallback OR general/similarity policy
        policy_eff = "general_fallback" if policy in ("contrasting", "counterfactual") else policy
        per = max(1, budget // max(1, min(len(ranked), budget)))
        i = 0
        for cls in ranked:
            take = per if i + per <= budget else budget - i
            for p in sample_support(cfg, cls, take, query_ids, seed):
                refs.append({"class": cls, "role": policy_eff, "path": str(p), "pair_id": None})
                i += 1
            if i >= budget:
                break
        policy = policy_eff if policy in ("contrasting", "counterfactual") else policy
    refs = refs[:budget]
    return {"refs": refs, "edge": edge, "diff_knowledge": diff_knowledge,
            "policy": policy, "rationale":
            (f"look-alike edge {edge['a']}<->{edge['b']} (n={edge['n_pairs']}); "
             f"symmetric {len(refs)} refs" if edge else
             f"{policy}: {len(refs)} candidate-specific refs (no eligible edge)")}


def run_stage4(vlm, kb, query_img: Path, ranked, routing, policy) -> dict:
    refs = routing["refs"]
    specs = [(str(query_img), "QUERY IMAGE")]
    for i, r in enumerate(refs, 1):
        specs.append((r["path"], f"REFERENCE {i} (labeled {r['class']})"))
    prompt = stage4_prompt(kb, ranked, refs, routing.get("diff_knowledge", ""), policy)
    r = vlm.call(prompt, specs, tag=f"stage4_{policy}")
    p = r["parsed"] or {}
    return {"raw": r, "parsed": p, "prediction": p.get("prediction"),
            "decisive_cue": p.get("decisive_cue", ""), "n_refs": len(refs)}
