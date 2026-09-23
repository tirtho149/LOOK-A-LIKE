#!/usr/bin/env python3
"""
Evidence adjudication for look-alike disambiguation (fixes the C regression).
=============================================================================
Separates VISUAL OBSERVATION (Stage 1) from DIAGNOSTIC INTERPRETATION (Stage 2)
and replaces naive row-count voting with rule-based adjudication.

Core rule that fixes the bug audited on Bacterial_Blight__Soybean_PNAS_5031:
    "A class-A feature was not observed" is NOT positive evidence for class B.
Absence of a class's positive feature can, at most, weakly CONTRADICT that class
(and only when the source deems its absence diagnostic and the region is visible);
it never becomes support for the other class. A flip requires an assessable,
source-supported *positive* distinguishing observation for the other class, or a
source-supported *exclusion* of the initial class.

Nothing here hard-codes an image id or a ground-truth label.
"""
from __future__ import annotations
import re

# evidence labels (ASCII identifiers, per spec)
SUPPORTS_A = "supports_blight"
SUPPORTS_B = "supports_pustule"
CONTRA_A = "contradicts_blight"
CONTRA_B = "contradicts_pustule"
NONDISC = "nondiscriminating"
INSUFF = "insufficient_evidence"

NON_VISUAL_HINTS = ("fingertip", "tactile", "touch", "texture on underside",
                    "weather", "temperature", "humid", "climate", "position on plant",
                    "canopy", "node", "lower leaves", "upper leaves", "whole plant")


# ─────────────────────────────────────────────────────────────────────────────
# Feature semantics (derived from the source-grounded checklist, not from labels)
# ─────────────────────────────────────────────────────────────────────────────
def classify_feature(feat: dict, class_a: str, class_b: str) -> dict:
    """Tag a checklist feature with is_visual, which class its PRESENCE supports,
    whether it discriminates by observed value, and whether absence is diagnostic.
    Purely from the feature's own signatures — general across pairs."""
    name = str(feat.get("feature", "")).lower()
    sa = str(feat.get("class_a_signature", "")).lower()
    sb = str(feat.get("class_b_signature", "")).lower()
    is_visual = not any(h in name or h in sa or h in sb for h in NON_VISUAL_HINTS)

    # A structural feature that one class shows and the other lacks -> presence supports that class.
    present_words = ("raised", "pustule", "erumpent", "halo", "water-soaked", "shot-hole",
                     "shot hole", "tear", "ragged", "bump", "blister", "concentric")
    a_struct = any(w in sa for w in present_words)
    b_struct = any(w in sb for w in present_words)
    # colour/appearance features discriminate by the OBSERVED VALUE, not presence/absence
    color_words = ("color", "colour", "brown", "black", "yellow", "pale", "green", "tan", "reddish")
    by_value = ("color" in name or "colour" in name or
                (sum(w in sa for w in color_words) >= 2 and sum(w in sb for w in color_words) >= 2))

    positive_for = None
    if not by_value:
        if a_struct and not b_struct:
            positive_for = "A"
        elif b_struct and not a_struct:
            positive_for = "B"
    # absence is diagnostic ONLY if the source frames the feature as (near-)always present
    # for its class; qualifiers 'may/sometimes/can/often' mean absence is NOT decisive.
    src = f"{sa} {sb} {feat.get('source','')}".lower()
    hedged = any(q in src for q in (" may ", "sometimes", " can ", "often", "variable",
                                    "stage", "early", "late"))
    absence_is_diagnostic = bool(positive_for) and not hedged and (
        "always" in src or "characteristic" in src or "defining" in src)
    return {"feature": feat.get("feature"), "is_visual": is_visual,
            "positive_for": positive_for, "by_value": by_value,
            "absence_is_diagnostic": absence_is_diagnostic,
            "class_a": class_a, "class_b": class_b}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: map a single visual observation -> evidence label (pure rules)
# ─────────────────────────────────────────────────────────────────────────────
def interpret(tags: dict, status: str, value_side: str | None = None) -> str:
    """status in {present, absent, not_assessable}. value_side (for by_value
    features) in {A, B, None}. Returns an evidence label. Never turns 'absent A'
    into 'supports B'."""
    if not tags["is_visual"]:
        return INSUFF                      # non-visual feature: no image evidence
    if status == "not_assessable":
        return INSUFF                      # unassessable contributes no evidence
    if tags["by_value"]:
        if status == "present" and value_side == "A":
            return SUPPORTS_A
        if status == "present" and value_side == "B":
            return SUPPORTS_B
        return INSUFF
    pf = tags["positive_for"]
    if pf is None:
        return NONDISC                     # feature shared / not class-distinctive
    if status == "present":
        return SUPPORTS_A if pf == "A" else SUPPORTS_B
    # status == absent : at most a weak contradiction of the feature's own class,
    # and only if the source deems its absence diagnostic. NEVER supports the other.
    if tags["absence_is_diagnostic"]:
        return CONTRA_A if pf == "A" else CONTRA_B
    return NONDISC


# ─────────────────────────────────────────────────────────────────────────────
# Adjudication: decide final label from interpreted evidence (replaces voting)
# ─────────────────────────────────────────────────────────────────────────────
def adjudicate(labels: list[str], initial_side: str) -> dict:
    """initial_side in {A, B} = condition B's prediction (predeclared).
    Flip only on a POSITIVE, assessable, source-supported distinguishing
    observation for the other class, or a source-supported exclusion of the
    initial class. Otherwise keep initial and mark inconclusive."""
    supA = labels.count(SUPPORTS_A)
    supB = labels.count(SUPPORTS_B)
    contraA = labels.count(CONTRA_A)
    contraB = labels.count(CONTRA_B)
    other_side = "B" if initial_side == "A" else "A"
    sup_other = supB if other_side == "B" else supA
    sup_init = supA if initial_side == "A" else supB
    contra_init = contraA if initial_side == "A" else contraB

    reason = None
    if sup_other >= 1 and sup_other > sup_init:
        decision, final_side = other_side, other_side          # positive distinguishing obs
        reason = f"positive {('supports_pustule' if other_side=='B' else 'supports_blight')} observation"
    elif contra_init >= 1 and sup_init == 0 and sup_other >= 1:
        decision, final_side = other_side, other_side           # source-supported exclusion + some positive
        reason = "source-supported exclusion of initial class with positive alternative evidence"
    elif sup_init >= 1 and sup_init >= sup_other:
        decision, final_side = initial_side, initial_side       # initial confirmed by positive evidence
        reason = "positive evidence confirms initial prediction"
    else:
        decision, final_side = "inconclusive", initial_side     # fallback: keep initial (NOT confirmation)
        reason = "no assessable source-supported distinguishing evidence"
    return {"decision": decision, "final_side": final_side,
            "supports_A": supA, "supports_B": supB,
            "contradicts_A": contraA, "contradicts_B": contraB,
            "reason": reason, "changed_from_initial": final_side != initial_side}


# ─────────────────────────────────────────────────────────────────────────────
# Re-adjudicate EXISTING cached observations (zero new inference) — isolates the
# decision-procedure change with references held fixed.
# ─────────────────────────────────────────────────────────────────────────────
def parse_status(query_obs: str, assessable) -> str:
    t = str(query_obs).strip().lower()
    if assessable is False or t.startswith("not_assessable") or "not assessable" in t[:40]:
        return "not_assessable"
    if t.startswith("absent") or re.match(r"^(no |none|without|intact|absent)\b", t):
        return "absent"
    return "present"


def value_side_from_text(query_obs: str, class_a_sig: str, class_b_sig: str) -> str | None:
    t = str(query_obs).lower()
    a_hit = sum(w in t for w in ("dark", "brown", "black", "water-soaked", "necrotic", "angular"))
    b_hit = sum(w in t for w in ("pale", "yellow", "light", "reddish", "tan", "speck"))
    if a_hit > b_hit:
        return "A"
    if b_hit > a_hit:
        return "B"
    return None


def reaudit_record(rec: dict, checklist: list[dict], class_a: str, class_b: str,
                   initial_pred: str) -> dict:
    """Apply Stage-2 interpretation + adjudication to the record's existing
    Stage-1 observations (rec['C_table']). initial_pred = condition B prediction."""
    tag_by_name = {f["feature"]: classify_feature(f, class_a, class_b) for f in checklist}
    interp_rows, labels = [], []
    for t in rec.get("C_table", []):
        name = t.get("feature")
        tags = tag_by_name.get(name) or classify_feature({"feature": name}, class_a, class_b)
        status = parse_status(t.get("query_observation", ""), t.get("assessable"))
        vside = value_side_from_text(t.get("query_observation", ""),
                                     "", "") if tags["by_value"] else None
        label = interpret(tags, status, vside)
        labels.append(label)
        interp_rows.append({"feature": name, "is_visual": tags["is_visual"],
                            "status": status, "positive_for": tags["positive_for"],
                            "by_value": tags["by_value"], "label": label,
                            "query_observation": t.get("query_observation", "")[:160]})
    init_side = "A" if initial_pred == class_a else "B"
    adj = adjudicate(labels, init_side)
    final = class_a if adj["final_side"] == "A" else class_b
    return {"interpretation": interp_rows, "adjudication": adj,
            "initial_pred": initial_pred, "final_pred": final,
            "decision": adj["decision"]}
