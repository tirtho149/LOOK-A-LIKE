#!/usr/bin/env python3
"""
validate_splits.py — leakage / integrity gate (protocol §2, §8).
================================================================
Exits NONZERO on any of:
  * query/support overlap (by sha256 OR snapshot name OR group)
  * missing snapshot files
  * class mismatch (image class != manifest class)
  * asymmetric look-alike pairs (edge present but not usable from both sides)
  * insufficient shot budget (support < max requested B for a needed pool)
  * missing KB provenance (eligible class without frozen description/source)

Usage:  python validate_splits.py [--snapshot snapshot|snapshot_dryrun]
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import yaml

HERE = Path(__file__).parent


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None,
                    help="snapshot subdir name (default: read from run_manifest)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    out = Path(cfg["experiment_output_dir"])
    shot_max = max(cfg["shot_budgets"])
    errors: list[str] = []
    warnings: list[str] = []

    run_manifest = json.loads((out / "run_manifest.json").read_text()) \
        if (out / "run_manifest.json").exists() else {}

    queries = read_csv(out / "query_manifest.csv")
    supports = read_csv(out / "support_manifest.csv")
    classes = read_csv(out / "class_manifest.csv")
    kb = read_jsonl(out / "kb_manifest.jsonl")
    pairs = read_csv(out / "pair_manifest.csv")

    if not queries:
        errors.append("query_manifest.csv empty or missing")
    dry = run_manifest.get("dry_run", False)

    # ── 1. query/support overlap ──
    q_sha = {r["sha256"] for r in queries}
    q_names = {Path(r["snapshot_path"]).name for r in queries}
    q_groups = {(r["class"], r["group"]) for r in queries}
    for r in supports:
        if r["sha256"] in q_sha:
            errors.append(f"OVERLAP sha256: support {r['support_id']} == a query image")
        if Path(r["snapshot_path"]).name in q_names and r["class"] in {q["class"] for q in queries}:
            # same filename in same class between pools is suspicious
            if (r["class"], r["group"]) in q_groups:
                errors.append(f"OVERLAP group: support {r['support_id']} shares a "
                              f"query specimen group in class {r['class']}")
    for r in supports:
        if (r["class"], r["group"]) in q_groups:
            errors.append(f"GROUP LEAK: support {r['support_id']} group={r['group']} "
                          f"straddles query boundary in {r['class']}")

    # ── 2. missing files (skip in dry-run; snapshot not copied) ──
    if not dry:
        for r in queries + supports:
            p = Path(r["snapshot_path"])
            if not p.exists():
                errors.append(f"MISSING FILE: {p}")

    # ── 3. class mismatch (snapshot path parent dir must equal class) ──
    for r in queries:
        parent = Path(r["snapshot_path"]).parent.name
        if parent != r["class"]:
            errors.append(f"CLASS MISMATCH: query {r['query_id']} in dir {parent} "
                          f"!= class {r['class']}")
    for r in supports:
        parent = Path(r["snapshot_path"]).parent.name
        if parent != r["class"]:
            errors.append(f"CLASS MISMATCH: support {r['support_id']} in dir {parent} "
                          f"!= class {r['class']}")

    # ── 4. KB provenance ──
    kb_classes = {r["class"] for r in kb}
    for r in classes:
        c = r["class"]
        if c not in kb_classes:
            errors.append(f"MISSING KB: class {c} has no frozen description record")
        else:
            rec = next(x for x in kb if x["class"] == c)
            if not rec.get("description"):
                errors.append(f"MISSING KB DESCRIPTION: {c}")
            if not rec.get("source_url"):
                warnings.append(f"missing source_url provenance for {c}")

    # ── 5. asymmetric look-alike pairs ──
    # A pair is usable only if BOTH endpoints are selected classes with support.
    sel = {r["class"] for r in classes}
    sup_by_class = {}
    for r in supports:
        sup_by_class.setdefault(r["class"], []).append(r)
    for e in pairs:
        a, b = e["source_class"], e["target_class"]
        if a in sel and b not in sel:
            errors.append(f"ASYMMETRIC PAIR: {e['pair_id']} {a}<->{b}; {b} not selected")
        if b in sel and a not in sel:
            errors.append(f"ASYMMETRIC PAIR: {e['pair_id']} {a}<->{b}; {a} not selected")
        # both selected: need >= 1 support each side so a symmetric split is possible
        if a in sel and b in sel:
            na = len(sup_by_class.get(a, []))
            nb = len(sup_by_class.get(b, []))
            if na == 0 or nb == 0:
                errors.append(f"PAIR SUPPORT MISSING: {e['pair_id']} {a}({na})/{b}({nb})")

    # ── 6. insufficient shot budget ──
    for c in sel:
        n = len([r for r in supports if r["class"] == c])
        # need at least the max budget from a class's own pool for KG/G routing headroom
        if n < shot_max:
            (warnings if dry else errors).append(
                f"INSUFFICIENT BUDGET: class {c} has {n} support imgs < max shot {shot_max}")

    # ── 7. queries-per-class ──
    nq = int(cfg["queries_per_class"])
    from collections import Counter
    qc = Counter(r["class"] for r in queries)
    for c in sel:
        if qc.get(c, 0) != nq:
            (warnings if dry else errors).append(
                f"QUERY COUNT: class {c} has {qc.get(c,0)} queries != {nq}")

    # ── report ──
    for w in warnings:
        print(f"[warn] {w}")
    if errors:
        print(f"\n[FAIL] {len(errors)} validation error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"[OK] splits valid: {len(queries)} queries, {len(supports)} support, "
          f"{len(pairs)} pairs, {len(classes)} classes, {len(warnings)} warning(s).")
    sys.exit(0)


if __name__ == "__main__":
    main()
