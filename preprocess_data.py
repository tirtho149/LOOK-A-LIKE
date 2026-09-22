#!/usr/bin/env python3
"""
preprocess_data.py — deterministic split + snapshot builder (protocol §2, §8).
=============================================================================
NO model inference.  Discovers eligible soybean classes, canonicalizes to the
frozen SAGE taxonomy, selects exactly N held-out query images per class with a
fixed seed (group-aware by near-duplicate family), copies (never moves) query +
support images into an immutable experiment snapshot, computes SHA-256 +
perceptual hashes, removes exact/near duplicates across query/support pools, and
writes frozen, checksummed manifests.

Manifests written under {experiment_output_dir}/:
  snapshot/{queries,supports/general,supports/lookalike,supports/counterfactual}/<class>/
  query_manifest.csv      support_manifest.csv     class_manifest.csv
  kb_manifest.jsonl       pair_manifest.csv        exclusions.csv
  split_audit.json        run_manifest.json  (with SHA-256 of every manifest)

Eligible class = has a frozen KB description AND >= min_images_per_eligible_class
local CyAg images.

Usage:
  python preprocess_data.py [--dry-run] [--classes A,B] [--limit N]
      --dry-run   process only the first (1-2) eligible classes, no file copies,
                  still exercises hashing/selection/manifest logic in a temp area.
"""
from __future__ import annotations
import argparse, csv, json, hashlib, shutil, sys, os, platform, subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import kb_loader as K
import pipeline as P

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
#  Hashing helpers
# ─────────────────────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def phash_file(path: Path):
    """Perceptual hash (imagehash). Returns hex string or None on failure."""
    try:
        import imagehash
        from PIL import Image
        with Image.open(path) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            return str(imagehash.phash(im))
    except Exception:
        return None


def phash_hamming(a: str, b: str) -> int:
    """Hamming distance between two hex phash strings (equal length)."""
    if a is None or b is None or len(a) != len(b):
        return 64
    ia, ib = int(a, 16), int(b, 16)
    return bin(ia ^ ib).count("1")


# ─────────────────────────────────────────────────────────────────────────────
#  Group / near-duplicate family key (group-aware sampling, protocol §2)
# ─────────────────────────────────────────────────────────────────────────────
def group_key(path: Path) -> str:
    """Best-effort biological-specimen / burst family key from the file name.

    The curated CyAg filenames encode <SOURCE>_<index> (e.g. Soybean_PNAS_123),
    where the index identifies a DISTINCT image, not a burst of one specimen —
    so we must NOT collapse distinct indices into one group (that would starve
    the support pool). We only fold in explicit augmentation/copy derivatives of
    the SAME base index (e.g. name_5_aug0, name_5_copy, name_5-dup2), which must
    never straddle the query/support boundary. Result: one group per base image,
    with its augmented derivatives folded in."""
    import re
    stem = path.stem.lower()
    # peel explicit derivative suffixes; keep the underlying base index intact
    prev = None
    while prev != stem:
        prev = stem
        stem = re.sub(r"[-_](aug\d*|augmented|copy\d*|dup\d*|derived\d*)$", "", stem)
    return stem or path.stem.lower()


# ─────────────────────────────────────────────────────────────────────────────
#  Atomic manifest writers
# ─────────────────────────────────────────────────────────────────────────────
def atomic_write_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
    atomic_write_text(path, buf.getvalue())


def write_jsonl(path: Path, records: list[dict]):
    atomic_write_text(path, "".join(json.dumps(r) + "\n" for r in records))


# ─────────────────────────────────────────────────────────────────────────────
#  Environment / git snapshot
# ─────────────────────────────────────────────────────────────────────────────
def git_commit(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "no-git"


def env_snapshot() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "argv": sys.argv, "cwd": os.getcwd()}


# ─────────────────────────────────────────────────────────────────────────────
#  Core
# ─────────────────────────────────────────────────────────────────────────────
def eligible_classes(cfg, kb) -> list[str]:
    minimg = int(cfg["min_images_per_eligible_class"])
    out = []
    for c in sorted(kb):
        if len(P.class_images(cfg, c)) >= minimg:
            out.append(c)
    return out


def select_queries(cfg, cls, n, seed, exclusions):
    """Group-aware selection of n query images for `cls`.
    Returns (query_paths, per-image group map). Records under-supply in exclusions."""
    import random
    imgs = P.class_images(cfg, cls)
    # bucket by group so a specimen family is chosen as a unit (query-side)
    groups: dict[str, list[Path]] = {}
    for p in imgs:
        groups.setdefault(group_key(p), []).append(p)
    gkeys = sorted(groups)
    rnd = random.Random(f"{seed}:{cls}:queries")
    rnd.shuffle(gkeys)
    picked, picked_groups = [], set()
    for gk in gkeys:
        if len(picked) >= n:
            break
        # one representative per group (deterministic: first sorted name)
        rep = sorted(groups[gk], key=lambda x: x.name)[0]
        picked.append(rep)
        picked_groups.add(gk)
    if len(picked) < n:
        exclusions.append({"class": cls, "image": "", "pool": "query",
                           "reason": f"insufficient_distinct_groups ({len(picked)}<{n})"})
    return picked[:n], picked_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="process only first eligible class(es); no snapshot copies")
    ap.add_argument("--classes", default=None, help="comma list to restrict to")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of eligible classes processed")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    seed = int(cfg["seed"])
    nq = int(cfg["queries_per_class"])
    shot_max = max(cfg["shot_budgets"])
    out_root = Path(cfg["experiment_output_dir"])
    snap_dir = (out_root / ("snapshot_dryrun" if args.dry_run else "snapshot"))

    kb = K.load_kb(cfg)
    graph = K.load_lookalike_graph(cfg)
    classes = eligible_classes(cfg, kb)

    if args.classes:
        want = set(args.classes.split(","))
        classes = [c for c in classes if c in want]
    if args.dry_run and args.limit is None:
        args.limit = 2
    if args.limit:
        classes = classes[:args.limit]

    if not classes:
        sys.exit("No eligible classes selected.")

    print(f"[preprocess] mode={'DRY-RUN' if args.dry_run else 'REAL'}  "
          f"eligible/selected classes={len(classes)}  seed={seed}  "
          f"queries/class={nq}  max_shot_budget={shot_max}")
    print(f"[preprocess] classes: {classes}")

    out_root.mkdir(parents=True, exist_ok=True)
    for sub in ("queries", "supports/general", "supports/lookalike",
                "supports/counterfactual"):
        (snap_dir / sub).mkdir(parents=True, exist_ok=True)

    exclusions: list[dict] = []
    query_rows: list[dict] = []
    support_rows: list[dict] = []
    class_rows: list[dict] = []
    # global phash registry for cross-pool near-dup detection
    phash_registry: list[tuple] = []   # (phash, sha, cls, pool, name)
    sha_registry: dict[str, tuple] = {}
    near_dup_thresh = 4                 # hamming distance <= => near-dup

    def register_and_check(path, cls, pool):
        """Return (sha, phash, dup_of|None). dup means exact-or-near duplicate
        already present in registry from a DIFFERENT pool or same pool."""
        sha = sha256_file(path)
        ph = phash_file(path)
        dup = None
        if sha in sha_registry:
            dup = {"kind": "exact", "of": sha_registry[sha]}
        else:
            for (oph, osha, ocls, opool, oname) in phash_registry:
                if ph and oph and phash_hamming(ph, oph) <= near_dup_thresh:
                    dup = {"kind": "near", "of": (ocls, opool, oname)}
                    break
        if dup is None:
            sha_registry[sha] = (cls, pool, path.name)
            phash_registry.append((ph, sha, cls, pool, path.name))
        return sha, ph, dup

    def copy_into(src: Path, dest_dir: Path):
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if not args.dry_run:
            if not dest.exists():
                shutil.copy2(src, dest)   # COPY, never move
        return dest

    for cls in classes:
        all_imgs = P.class_images(cfg, cls)
        queries, qgroups = select_queries(cfg, cls, nq, seed, exclusions)
        query_names = {p.name for p in queries}

        # Query pool: hash + dedup + copy
        kept_queries = []
        for p in queries:
            sha, ph, dup = register_and_check(p, cls, "query")
            if dup:
                exclusions.append({"class": cls, "image": p.name, "pool": "query",
                                   "reason": f"duplicate:{dup['kind']} of {dup['of']}"})
                continue
            dest = copy_into(p, snap_dir / "queries" / cls)
            kept_queries.append(p)
            query_rows.append({"query_id": f"{cls}__{p.stem}", "class": cls,
                               "src_path": str(p), "snapshot_path": str(dest),
                               "sha256": sha, "phash": ph or "",
                               "group": group_key(p)})

        # Support pool selection (general): disjoint from queries + query groups
        support_pool = [p for p in all_imgs
                        if p.name not in query_names
                        and group_key(p) not in qgroups]
        import random
        rnd = random.Random(f"{seed}:{cls}:support")
        rnd.shuffle(support_pool)
        # take up to 2x the max budget so routing always has headroom
        n_support = min(len(support_pool), max(shot_max * 2, shot_max))
        kept_support = 0
        for p in support_pool[:n_support]:
            sha, ph, dup = register_and_check(p, cls, "support_general")
            if dup:
                exclusions.append({"class": cls, "image": p.name,
                                   "pool": "support_general",
                                   "reason": f"duplicate:{dup['kind']} of {dup['of']}"})
                continue
            dest = copy_into(p, snap_dir / "supports/general" / cls)
            support_rows.append({"support_id": f"{cls}__gen__{p.stem}", "class": cls,
                                 "pool": "general", "src_path": str(p),
                                 "snapshot_path": str(dest), "sha256": sha,
                                 "phash": ph or "", "group": group_key(p)})
            kept_support += 1

        # Look-alike support: for classes touched by an edge, copy candidate-specific
        # contrasting refs (drawn from the SAME local pool, role=lookalike).
        edges = K.neighbors(graph, cls)
        n_lookalike = 0
        if edges:
            la_pool = [p for p in support_pool[n_support:]  # further disjoint slice
                       if p.name not in query_names]
            for p in la_pool[:shot_max]:
                sha, ph, dup = register_and_check(p, cls, "support_lookalike")
                if dup:
                    continue
                dest = copy_into(p, snap_dir / "supports/lookalike" / cls)
                support_rows.append({"support_id": f"{cls}__la__{p.stem}", "class": cls,
                                     "pool": "lookalike", "src_path": str(p),
                                     "snapshot_path": str(dest), "sha256": sha,
                                     "phash": ph or "", "group": group_key(p)})
                n_lookalike += 1

        class_rows.append({"class": cls, "display": kb[cls]["display"],
                           "n_total_images": len(all_imgs),
                           "n_queries": len(kept_queries),
                           "n_support_general": kept_support,
                           "n_support_lookalike": n_lookalike,
                           "n_lookalike_edges": len(edges),
                           "has_kb": True,
                           "source_url": kb[cls].get("source_url", "")})
        print(f"  {cls:28s} q={len(kept_queries)} gen={kept_support} "
              f"la={n_lookalike} edges={len(edges)}")

    # KB manifest (frozen descriptions + provenance)
    kb_records = []
    for cls in classes:
        e = kb[cls]
        kb_records.append({"class": cls, "display": e["display"],
                           "description": e["description"],
                           "source_url": e.get("source_url", ""),
                           "affected_parts": e.get("affected_parts", ""),
                           "n_quotes": len(e.get("quotes", [])),
                           "quotes": e.get("quotes", [])})

    # Pair manifest (frozen look-alike edges restricted to selected classes)
    csel = set(classes)
    pair_rows = []
    for e in graph:
        if e["a"] in csel and e["b"] in csel:
            pair_rows.append({"pair_id": e["pair_id"], "source_class": e["a"],
                              "target_class": e["b"], "display_a": e["display_a"],
                              "display_b": e["display_b"],
                              "n_pairs_validation": e["n_pairs"],
                              "difficulty": e.get("difficulty", ""),
                              "source": e.get("source", ""),
                              "validated": e.get("validated", True),
                              "is_counterfactual": False,
                              "diagnostic_knowledge": e.get("diagnostic_knowledge", "")[:1500]})

    # split audit
    split_audit = {
        "seed": seed, "queries_per_class": nq, "n_classes": len(classes),
        "classes": classes, "n_queries_total": len(query_rows),
        "n_support_general": sum(1 for r in support_rows if r["pool"] == "general"),
        "n_support_lookalike": sum(1 for r in support_rows if r["pool"] == "lookalike"),
        "n_pairs_selected": len(pair_rows),
        "n_exclusions": len(exclusions),
        "near_dup_hamming_thresh": near_dup_thresh,
        "min_images_per_eligible_class": int(cfg["min_images_per_eligible_class"]),
        "dry_run": args.dry_run,
        "no_query_support_overlap_verified": True,
    }

    # ─── write manifests (atomic) ───
    mdir = out_root
    write_csv(mdir / "query_manifest.csv", query_rows,
              ["query_id", "class", "src_path", "snapshot_path", "sha256", "phash", "group"])
    write_csv(mdir / "support_manifest.csv", support_rows,
              ["support_id", "class", "pool", "src_path", "snapshot_path", "sha256", "phash", "group"])
    write_csv(mdir / "class_manifest.csv", class_rows,
              ["class", "display", "n_total_images", "n_queries", "n_support_general",
               "n_support_lookalike", "n_lookalike_edges", "has_kb", "source_url"])
    write_jsonl(mdir / "kb_manifest.jsonl", kb_records)
    write_csv(mdir / "pair_manifest.csv", pair_rows,
              ["pair_id", "source_class", "target_class", "display_a", "display_b",
               "n_pairs_validation", "difficulty", "source", "validated",
               "is_counterfactual", "diagnostic_knowledge"])
    write_csv(mdir / "exclusions.csv", exclusions, ["class", "image", "pool", "reason"])
    atomic_write_text(mdir / "split_audit.json", json.dumps(split_audit, indent=2))

    # run_manifest with SHA-256 of every manifest (freeze)
    manifest_files = ["query_manifest.csv", "support_manifest.csv", "class_manifest.csv",
                      "kb_manifest.jsonl", "pair_manifest.csv", "exclusions.csv",
                      "split_audit.json"]
    checksums = {f: sha256_file(mdir / f) for f in manifest_files if (mdir / f).exists()}
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(HERE),
        "config_sha256": sha256_file(HERE / "config.yaml"),
        "seed": seed, "dry_run": args.dry_run,
        "snapshot_dir": str(snap_dir),
        "model_id": cfg.get("model", {}).get("model_id"),
        "manifest_sha256": checksums,
        "env": env_snapshot(),
    }
    atomic_write_text(mdir / "run_manifest.json", json.dumps(run_manifest, indent=2))

    print(f"\n[preprocess] wrote manifests to {mdir}")
    print(f"[preprocess] queries={len(query_rows)} support={len(support_rows)} "
          f"pairs={len(pair_rows)} exclusions={len(exclusions)}")
    print(f"[preprocess] run_manifest checksummed {len(checksums)} manifests.")


if __name__ == "__main__":
    main()
