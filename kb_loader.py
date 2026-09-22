#!/usr/bin/env python3
"""
Source-grounded SAGE KB + look-alike graph loaders (protocol §2, §4).
=====================================================================
Everything here is FROZEN and never derived from a query image:
  * load_kb()              -> canonical_class -> visual description + provenance
  * load_lookalike_graph() -> validated look-alike edges w/ reference images
                              and source-grounded differentiating knowledge
Class names from the KB spreadsheet, the look-alike dataset, and the image
directory tree are reconciled onto one canonical taxonomy (the image-dir names).
"""
from __future__ import annotations
import ast, json, re
from pathlib import Path
import pandas as pd


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


# Aliases: normalized variant -> canonical image-dir class name.
ALIASES = {
    "septoria leaf spot": "Septoria_Brown_Spot",
    "septoria brown spot": "Septoria_Brown_Spot",
    "brown spot": "Septoria_Brown_Spot",
    "frogeye leaf spot": "Frogeye_Leaf_Spot",
    "bacterial blight": "Bacterial_Blight",
    "bacterial pustule": "Bacterial_Pustule",
    "bacterial pustule of soybean disease": "Bacterial_Pustule",
    "soybean rust": "Soybean_Rust",
    "rust": "Soybean_Rust",
    "downy mildew": "Downy_Mildew",
    "cercospora blight": "Cercospora_Blight",
    "cercospora": "Cercospora",
    "charcoal rot": "Charcoal_Rot",
    "brown stem rot": "Brown_Stem_Rot",
    "sudden death syndrome": "Sudden_Death_Syndrome",
    "soybean vein necrosis virus": "Soybean_Vein_Necrosis_Virus",
    "soybean mosaic virus": "Soybean_Mosaic_Virus",
    "bean pod mottle virus": "Bean_Pod_Mottle_Virus",
    "alfalfa mosaic virus": "Alfalfa_Mosaic_Virus",
    "genus diaporthe": "Diaporthe",
    "diaporthe": "Diaporthe",
    "stem canker": "Stem_Canker",
    "southern blight": "Southern_Blight",
    "sclerotinia timber rot": "White_Mold",
    "white mold": "White_Mold",
    "root and stem rot": "Root_And_Stem_Rot",
    "rhizoctonia damping off blight and rot": "Rhizoctonia",
    "taproot decline on soybean": "Taproot_Decline_On_Soybean",
}


def canonicalize(name: str, image_dirs=None) -> str | None:
    n = normalize(name)
    if n in ALIASES:
        return ALIASES[n]
    if image_dirs:
        # exact / underscore match against real class dirs
        for d in image_dirs:
            if normalize(d) == n:
                return d
    return None


def list_image_dirs(cfg) -> list[str]:
    root = Path(cfg["sage_curated_dataset_dir"]) / cfg["sage_curated_images_subdir"]
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def load_kb(cfg) -> dict:
    """canonical_class -> {display, description, source_url, affected_parts, quotes[]}."""
    image_dirs = list_image_dirs(cfg)
    desc = pd.read_excel(cfg["kb_visual_descriptions_xlsx"])
    desc = desc[desc["Crop"].astype(str).str.contains("oybean", case=False, na=False)]
    kb = {}
    for _, r in desc.iterrows():
        canon = canonicalize(r["Disease Class"], image_dirs)
        d = r.get("Visual Description")
        if canon is None or pd.isna(d):
            continue
        kb[canon] = {"display": str(r["Disease Class"]).strip(),
                     "description": str(d).strip(),
                     "source_url": str(r.get("Source Link", "") or "").strip(),
                     "affected_parts": "", "quotes": []}
    # attach verbatim provenance from the merged sourced KB
    try:
        prov = pd.read_excel(cfg["kb_sourced_provenance_xlsx"])
        prov = prov[prov["Crop"].astype(str).str.contains("oybean", case=False, na=False)]
        for _, r in prov.iterrows():
            canon = canonicalize(r["Disease Class"], image_dirs)
            if canon in kb:
                q = str(r.get("Verbatim Source Quote", "") or "").strip()
                if q and q.lower() != "nan":
                    kb[canon]["quotes"].append({
                        "quote": q[:400],
                        "url": str(r.get("Source URL", "") or "").strip(),
                        "parts": str(r.get("Affected Parts", "") or "").strip()})
                if not kb[canon]["affected_parts"]:
                    kb[canon]["affected_parts"] = str(r.get("Affected Parts", "") or "").strip()
    except Exception:
        pass
    return kb


def _parse_labels(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def _parse_dk(v):
    try:
        dk = ast.literal_eval(v) if isinstance(v, str) else v
        return (dk.get("text", "") if isinstance(dk, dict) else str(dk)).strip()
    except Exception:
        # tolerate un-literal-able text; take the substring after 'text':
        m = re.search(r"text['\"]?\s*:\s*['\"](.+)", str(v), re.DOTALL)
        return (m.group(1)[:1500] if m else str(v)[:1500]).strip()


def load_lookalike_graph(cfg, max_edges: int | None = None) -> list[dict]:
    """Frozen, validated look-alike edges + source-grounded differentiating knowledge.

    Only the RELATIONSHIP and the differentiating KNOWLEDGE come from the
    look-alike dataset. Reference IMAGES are drawn from the local CyAg support
    pool by retrieve_references (protocol §2: support = SAGE curated dataset).
    Edge = {pair_id, a, b, display_a, display_b, diagnostic_knowledge,
            n_pairs, difficulty, source, validated}.
    """
    image_dirs = list_image_dirs(cfg)
    p = Path(cfg["lookalike_local_jsonl"])
    seen, edges = {}, []
    if not p.exists():
        return edges
    for line in p.open():
        try:
            d = json.loads(line)
        except Exception:
            continue
        pid = d.get("pair_id")
        if pid in seen:
            seen[pid]["n_pairs"] += 1
            continue
        try:
            labs = _parse_labels(d["candidate_labels"])
        except Exception:
            continue
        ca = canonicalize(labs[0], image_dirs)
        cb = canonicalize(labs[1], image_dirs)
        if ca is None or cb is None or ca == cb:
            continue
        edge = {"pair_id": pid, "a": ca, "b": cb,
                "display_a": str(labs[0]), "display_b": str(labs[1]),
                "diagnostic_knowledge": _parse_dk(d.get("diagnostic_knowledge", "")),
                "n_pairs": 1, "difficulty": d.get("difficulty", ""),
                "source": "plantseg-lookalike", "validated": True}
        seen[pid] = edge
        edges.append(edge)
    edges.sort(key=lambda e: -e["n_pairs"])
    return edges[:max_edges] if max_edges else edges


def neighbors(graph: list[dict], cls: str) -> list[dict]:
    return [e for e in graph if e["a"] == cls or e["b"] == cls]


if __name__ == "__main__":
    import yaml, sys
    cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
    kb = load_kb(cfg)
    g = load_lookalike_graph(cfg)
    print(f"KB classes: {len(kb)}  | look-alike edges: {len(g)}")
    for c in sorted(kb)[:5]:
        print(f"  {c}: {kb[c]['description'][:70]}...")
    print("edges w/ Frogeye_Leaf_Spot:")
    for e in neighbors(g, "Frogeye_Leaf_Spot"):
        print(f"  {e['a']} <-> {e['b']}  n={e['n_pairs']}  dk={e['diagnostic_knowledge'][:60]}...")
