#!/usr/bin/env python3
"""
Structured contrasting-feature disambiguation for look-alike disease pairs.
===========================================================================
Improves the SAGE look-alike method (Dr. Sarkar's direction): rather than
merely *showing* reference images, the model reads the pair's source-grounded
distinguishing descriptions, turns them into a feature checklist, and fills an
explicit evidence table over the query and both reference groups before
reassessing the initial prediction.

Three matched conditions per test image (isolate the structured-comparison effect):
  A  cond_kb_only        image + both KB descriptions               (initial prediction)
  B  cond_refs_only      image + KB + N+N ordinary references       (references shown, free-form)
  C  cond_feature_compare image + KB + the SAME N+N references
                         + structured distinguishing-feature comparison   (proposed)

Design rules enforced (per the six requested improvements):
 1. The pair = leading prediction + one DOCUMENTED look-alike; skip if none.
 2. Equal references per class; same-organ / comparable presentation preferred;
    query ground-truth label never used in retrieval.
 3. Compare specific KB-documented features via a checklist; do not invent features.
 4. Explicit evidence table; a not-assessable feature is NOT evidence against a class.
 5. Crop inspection for too-small features (query + baseline), no enhancement/generation.
 6. Non-forced reassessment: keep the initial label unless the assessable evidence
    margin favors the alternative.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
from PIL import Image
import pipeline as P
import kb_loader as K

# margin (assessable supports) the ALTERNATIVE must exceed the initial by, to flip
FLIP_MARGIN = 1


# ─────────────────────────────────────────────────────────────────────────────
# 1. Relevant-pair selection
# ─────────────────────────────────────────────────────────────────────────────
def select_pair(graph, top1: str, shortlist=None):
    """Return the highest-priority DOCUMENTED look-alike edge incident to `top1`
    (optionally restricted to `shortlist`). None if no documented pair exists."""
    inc = [e for e in graph if top1 in (e["a"], e["b"])]
    if shortlist is not None:
        inc = [e for e in inc if (e["a"] in shortlist and e["b"] in shortlist)]
    inc = [e for e in inc if e.get("diagnostic_knowledge")]
    if not inc:
        return None
    inc.sort(key=lambda e: (-e["n_pairs"], e["pair_id"]))
    return inc[0]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Balanced held-out reference retrieval (query GT label never used)
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_pair_refs(cfg, edge, n_per_class: int, held_out: set[str], seed: int):
    """n references per class from the LOCAL CyAg support pool, disjoint from all
    test images. Soybean references are foliar (same organ as the leaf queries);
    within that, deterministic sampling illustrates documented features."""
    refs = {}
    for cls in (edge["a"], edge["b"]):
        refs[cls] = [str(p) for p in
                     P.sample_support(cfg, cls, n_per_class, held_out, seed)]
    return refs


# ─────────────────────────────────────────────────────────────────────────────
# 3. KB distinguishing-feature checklist (parsed once per pair, cached)
# ─────────────────────────────────────────────────────────────────────────────
def extract_checklist(vlm, edge) -> dict:
    a, b = edge["a"], edge["b"]
    prompt = f"""You are given the SOURCE-GROUNDED text that documents how to
distinguish two look-alike soybean diseases. Convert it into a SHORT checklist
of concrete, visually-checkable DISTINGUISHING features. Use ONLY features that
are stated or clearly implied by the text; do NOT invent botanical differences.

CLASS A = {a}
CLASS B = {b}

DOCUMENTED DISTINGUISHING TEXT:
{edge['diagnostic_knowledge']}

For each feature give its name and the signature expected for each class.
JSON schema:
{{"features": [{{"feature": "...", "class_a_signature": "...",
                 "class_b_signature": "...", "typical_size": "large|small"}}]}}
Return at most 6 features, ordered by how decisive they are."""
    r = vlm.call(prompt, [], tag=f"checklist_{edge['pair_id']}")
    feats = (r["parsed"] or {}).get("features", []) if r["parsed"] else []
    return {"raw": r, "features": feats}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Crop inspection helper (center crop + upscale; NO enhancement/generation)
# ─────────────────────────────────────────────────────────────────────────────
def make_crop(src: str, out_dir: Path, frac: float = 0.45) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()[:12]
    dst = out_dir / f"crop_{sha}.jpg"
    if not dst.exists():
        im = Image.open(src).convert("RGB")
        w, h = im.size
        cw, ch = int(w*frac), int(h*frac)
        left, top = (w-cw)//2, (h-ch)//2
        im.crop((left, top, left+cw, top+ch)).resize((min(w, 768), min(h, 768))).save(dst, quality=90)
    return str(dst)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt blocks
# ─────────────────────────────────────────────────────────────────────────────
def _pair_kb_block(kb, a, b):
    return (f"- {a}: {kb[a]['description']}\n- {b}: {kb[b]['description']}")


def _ref_specs(query, refsA, refsB, a, b):
    specs = [(query, "QUERY IMAGE (unlabeled)")]
    for i, p in enumerate(refsA, 1):
        specs.append((p, f"CLASS A reference {i} (labeled {a})"))
    for i, p in enumerate(refsB, 1):
        specs.append((p, f"CLASS B reference {i} (labeled {b})"))
    return specs


# ─────────────────────────────────────────────────────────────────────────────
# Condition A — image + KB (initial prediction)
# ─────────────────────────────────────────────────────────────────────────────
def cond_kb_only(vlm, kb, query: str, a: str, b: str) -> dict:
    prompt = f"""Diagnose the soybean disease in the query image. It is exactly one
of two candidates. Use ONLY the source-grounded descriptions below.

CANDIDATES:
{_pair_kb_block(kb, a, b)}

JSON: {{"prediction": "{a}|{b}", "evidence": "...", "confidence": 0.0}}"""
    r = vlm.call(prompt, [(query, "QUERY IMAGE (unlabeled)")], tag="condA")
    p = r["parsed"] or {}
    return {"raw": r, "prediction": p.get("prediction"), "evidence": p.get("evidence", "")}


# ─────────────────────────────────────────────────────────────────────────────
# Condition B — image + KB + ordinary references (free-form, no structured table)
# ─────────────────────────────────────────────────────────────────────────────
def cond_refs_only(vlm, kb, query: str, a: str, b: str, refsA, refsB) -> dict:
    specs = _ref_specs(query, refsA, refsB, a, b)
    prompt = f"""Diagnose the soybean disease in the query image (exactly one of two
candidates). You are given labeled reference images of each class. Use the
descriptions and the references as you see fit.

CANDIDATES:
{_pair_kb_block(kb, a, b)}

There are {len(refsA)} CLASS A references and {len(refsB)} CLASS B references.
JSON: {{"prediction": "{a}|{b}", "evidence": "...", "confidence": 0.0}}"""
    r = vlm.call(prompt, specs, tag="condB")
    p = r["parsed"] or {}
    return {"raw": r, "prediction": p.get("prediction"), "evidence": p.get("evidence", "")}


# ─────────────────────────────────────────────────────────────────────────────
# Condition C — structured distinguishing-feature comparison (proposed)
# ─────────────────────────────────────────────────────────────────────────────
def cond_feature_compare(vlm, kb, query: str, a: str, b: str, refsA, refsB,
                         checklist, edge, initial_pred: str,
                         crop_dir: Path | None = None) -> dict:
    specs = _ref_specs(query, refsA, refsB, a, b)
    feat_lines = "\n".join(
        f"  - {f.get('feature')}: A({a})={f.get('class_a_signature')} | "
        f"B({b})={f.get('class_b_signature')}" for f in checklist)
    prompt = f"""Disambiguate two look-alike soybean diseases by comparing SPECIFIC
distinguishing features (not overall resemblance). Work ONLY from the checklist
below (do not invent features). For EACH feature, examine the QUERY and BOTH
labeled reference groups, then fill the evidence table.

CANDIDATES:
{_pair_kb_block(kb, a, b)}

DISTINGUISHING-FEATURE CHECKLIST (source-grounded):
{feat_lines}

Rules:
- For the query, mark each feature visible / absent / not_assessable, with a location.
- A feature that is not_assessable is NOT evidence against either class.
- 'supports' is A, B, or neither, justified by what is actually visible.
- Do not change your view just because references were provided.

Initial prediction (from image + KB only): {initial_pred}

JSON schema:
{{"evidence_table": [
   {{"feature": "...", "class_a_refs": "...", "class_b_refs": "...",
     "query_observation": "...", "assessable": true, "location": "...",
     "supports": "A|B|neither"}}],
  "too_small_features": ["feature names that need a zoomed crop, if any"],
  "decision": "A|B|inconclusive",
  "final_label": "{a}|{b}",
  "explanation": "tie decision to observed features + their KB source"}}"""
    r = vlm.call(prompt, specs, tag="condC")
    p = r["parsed"] or {}
    table = p.get("evidence_table", []) or []

    # 5. Crop inspection for features the model flagged too small / not assessable
    zoom = None
    need = set(p.get("too_small_features", []) or [])
    need |= {t["feature"] for t in table
             if not t.get("assessable", True) or t.get("query_observation", "") == "not_assessable"}
    if need and crop_dir is not None:
        qcrop = make_crop(query, crop_dir)
        zoom_specs = [(qcrop, "QUERY IMAGE — CENTER CROP (zoom, not enhanced)")]
        # apply the same crop opportunity to one reference of each class (baseline)
        if refsA: zoom_specs.append((make_crop(refsA[0], crop_dir), f"CLASS A crop (labeled {a})"))
        if refsB: zoom_specs.append((make_crop(refsB[0], crop_dir), f"CLASS B crop (labeled {b})"))
        zprompt = f"""Re-examine ONLY these features that were too small to assess at
full-frame, using the zoomed center crops (crops are not enhanced or generated):
{sorted(need)}
For each, report query observation (visible/absent/not_assessable + location) and
which class it supports (A={a}, B={b}, or neither).
JSON: {{"updated": [{{"feature": "...", "query_observation": "...",
        "assessable": true, "supports": "A|B|neither"}}]}}"""
        rz = vlm.call(zprompt, zoom_specs, tag="condC_zoom")
        zoom = rz["parsed"] or {}
        upd = {u["feature"]: u for u in zoom.get("updated", [])}
        for t in table:
            if t["feature"] in upd:
                u = upd[t["feature"]]
                t["query_observation"] = u.get("query_observation", t["query_observation"])
                t["assessable"] = u.get("assessable", t.get("assessable"))
                t["supports"] = u.get("supports", t["supports"])
                t["zoomed"] = True

    # 4/6. Recompute the tally in Python (assessable features only) — do not trust model tally
    supA = sum(1 for t in table if t.get("assessable", True) and t.get("supports") == "A")
    supB = sum(1 for t in table if t.get("assessable", True) and t.get("supports") == "B")
    init_side = "A" if initial_pred == a else "B"
    other_side = "B" if init_side == "A" else "A"
    sup_init, sup_other = (supA, supB) if init_side == "A" else (supB, supA)
    if sup_other - sup_init >= FLIP_MARGIN:
        decision, final = other_side, (b if other_side == "B" else a)
    elif sup_init > sup_other:
        decision, final = init_side, initial_pred
    else:
        decision, final = "inconclusive", initial_pred     # non-forced: keep initial
    return {"raw": r, "zoom": zoom, "evidence_table": table,
            "supports_A": supA, "supports_B": supB,
            "decision": decision, "prediction": final,
            "changed_from_initial": final != initial_pred,
            "explanation": p.get("explanation", "")}
