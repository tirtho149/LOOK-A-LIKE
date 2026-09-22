#!/usr/bin/env python3
"""
Unit tests (protocol §8): shot-budget accounting, no overlap, class mapping,
pair symmetry, bootstrap pairing, Holm correction.
Run:  python -m pytest tests/ -q
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import kb_loader as K
import pipeline as P
import compute_statistics as S


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open(HERE / "config.yaml"))


@pytest.fixture(scope="module")
def kb(cfg):
    return K.load_kb(cfg)


@pytest.fixture(scope="module")
def graph(cfg):
    return K.load_lookalike_graph(cfg)


# ─────────────────────────────────────────────────────────────────────────────
#  1. Shot-budget accounting: B = TOTAL refs, symmetric contrasting split
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("budget", [2, 4, 8])
def test_shot_budget_total_not_per_class(cfg, kb, graph, budget):
    # pick a shortlist containing a real look-alike edge if possible
    edge = graph[0] if graph else None
    ranked = [edge["a"], edge["b"]] + [c for c in sorted(kb)][:3] if edge \
        else sorted(kb)[:5]
    ranked = list(dict.fromkeys(ranked))
    routing = P.route_references(cfg, graph, kb, ranked, budget,
                                 "contrasting", set(), cfg["seed"])
    assert len(routing["refs"]) <= budget, "B must be the TOTAL image count"


def test_contrasting_split_symmetric(cfg, kb, graph):
    if not graph:
        pytest.skip("no edges")
    edge = graph[0]
    ranked = [edge["a"], edge["b"]]
    routing = P.route_references(cfg, graph, kb, ranked, 8, "contrasting",
                                 set(), cfg["seed"])
    if routing.get("edge"):
        from collections import Counter
        cnt = Counter(r["class"] for r in routing["refs"])
        a, b = edge["a"], edge["b"]
        # symmetric allocation: both competing classes represented equally
        assert cnt[a] == cnt[b], f"asymmetric split {cnt}"
        assert cnt[a] + cnt[b] == len(routing["refs"])


def test_general_fallback_when_no_edge(cfg, kb, graph):
    # a shortlist with no induced edge should fall back to general refs, still <= B
    lonely = [c for c in sorted(kb)][:1]
    routing = P.route_references(cfg, graph, kb, lonely, 4, "contrasting",
                                 set(), cfg["seed"])
    assert routing["edge"] is None
    assert len(routing["refs"]) <= 4


# ─────────────────────────────────────────────────────────────────────────────
#  2. No query/support overlap (sample_support excludes query ids)
# ─────────────────────────────────────────────────────────────────────────────
def test_sample_support_excludes_queries(cfg, kb):
    cls = next(iter(sorted(kb)))
    imgs = P.class_images(cfg, cls)
    if len(imgs) < 3:
        pytest.skip("too few images")
    exclude = {imgs[0].name, imgs[1].name}
    sup = P.sample_support(cfg, cls, 5, exclude, cfg["seed"])
    names = {p.name for p in sup}
    assert names.isdisjoint(exclude), "support leaked a held-out query image"


def test_group_key_folds_augmentations_only():
    from preprocess_data import group_key
    # distinct indices must NOT collapse (else support pool starves)
    assert group_key(Path("Soybean_PNAS_1.png")) != group_key(Path("Soybean_PNAS_2.png"))
    # augmented derivatives of the SAME base fold together
    base = group_key(Path("Soybean_PNAS_5.png"))
    assert group_key(Path("Soybean_PNAS_5_aug0.png")) == base
    assert group_key(Path("Soybean_PNAS_5_copy.png")) == base


# ─────────────────────────────────────────────────────────────────────────────
#  3. Class-mapping canonicalization
# ─────────────────────────────────────────────────────────────────────────────
def test_canonicalization(cfg):
    dirs = K.list_image_dirs(cfg)
    assert K.canonicalize("Frogeye Leaf Spot", dirs) == "Frogeye_Leaf_Spot"
    assert K.canonicalize("soybean rust", dirs) == "Soybean_Rust"
    assert K.canonicalize("brown spot", dirs) == "Septoria_Brown_Spot"
    # unknown label maps to None (never silently mislabeled)
    assert K.canonicalize("totally unknown xyz", dirs) is None


# ─────────────────────────────────────────────────────────────────────────────
#  4. Look-alike pair symmetry
# ─────────────────────────────────────────────────────────────────────────────
def test_pair_symmetry(graph):
    # neighbors(cls) must find every edge from both endpoints
    for e in graph[:20]:
        na = K.neighbors(graph, e["a"])
        nb = K.neighbors(graph, e["b"])
        assert e in na and e in nb, "edge not reachable from both endpoints"


def test_no_self_loops(graph):
    for e in graph:
        assert e["a"] != e["b"], "self-loop edge is invalid"


# ─────────────────────────────────────────────────────────────────────────────
#  5. Bootstrap pairing preserved
# ─────────────────────────────────────────────────────────────────────────────
def test_bootstrap_pairing_preserved():
    # Build paired records where cond A is always correct, cond B never.
    # A class-stratified paired bootstrap of (accA - accB) must be exactly 1.0
    # in EVERY replicate (pairing preserved => delta invariant to resampling).
    recs_by_class = {
        "clsX": [{"A": {"correct": 1}, "B": {"correct": 0}, "truth": "clsX"} for _ in range(5)],
        "clsY": [{"A": {"correct": 1}, "B": {"correct": 0}, "truth": "clsY"} for _ in range(5)],
    }

    def delta(rs):
        accA = sum(r["A"]["correct"] for r in rs) / len(rs)
        accB = sum(r["B"]["correct"] for r in rs) / len(rs)
        return accA - accB

    res = S.class_stratified_bootstrap(recs_by_class, delta, 200, seed=42)
    assert abs(res["point"] - 1.0) < 1e-9
    assert abs(res["ci_lo"] - 1.0) < 1e-9 and abs(res["ci_hi"] - 1.0) < 1e-9
    assert res["prob_gt0"] == 1.0


def test_bootstrap_stratified_uses_all_classes():
    recs = {"a": [{"v": 1}], "b": [{"v": 0}]}
    res = S.class_stratified_bootstrap(recs, lambda rs: sum(r["v"] for r in rs) / len(rs),
                                       100, seed=1)
    # mean over the 2 stratified draws is always 0.5 (one from each class)
    assert abs(res["point"] - 0.5) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
#  6. Holm correction correctness
# ─────────────────────────────────────────────────────────────────────────────
def test_holm_correction():
    p = {"t1": 0.01, "t2": 0.02, "t3": 0.03, "t4": 0.04}
    adj = S.holm(p)
    # step-down: sorted p * (m-i); m=4
    assert abs(adj["t1"] - 0.04) < 1e-9   # 0.01*4
    assert abs(adj["t2"] - 0.06) < 1e-9   # 0.02*3
    assert abs(adj["t3"] - 0.06) < 1e-9   # 0.03*2 => 0.06, monotone-clamped
    assert abs(adj["t4"] - 0.06) < 1e-9   # 0.04*1 => 0.04, clamped up to 0.06
    # monotone non-decreasing in original p order
    assert adj["t1"] <= adj["t2"] <= adj["t3"] <= adj["t4"]


def test_holm_clamps_at_one():
    adj = S.holm({"a": 0.6, "b": 0.7})
    assert all(v <= 1.0 for v in adj.values())


# ─────────────────────────────────────────────────────────────────────────────
#  Bonus: exact McNemar + Cochran's Q sanity
# ─────────────────────────────────────────────────────────────────────────────
def test_exact_mcnemar_symmetric_is_one():
    assert abs(S.exact_mcnemar(5, 5)["p_value"] - 1.0) < 1e-9


def test_exact_mcnemar_extreme_small_p():
    r = S.exact_mcnemar(10, 0)
    assert r["p_value"] < 0.01


def test_cochran_q_no_variation():
    # all conditions identical => Q = 0, p = 1
    m = [[1, 1, 1], [0, 0, 0], [1, 1, 1]]
    r = S.cochran_q(m)
    assert abs(r["Q"]) < 1e-9
    assert r["p_value"] == pytest.approx(1.0, abs=1e-6)


def test_wilson_bounds():
    p, lo, hi = S.wilson(3, 5)
    assert 0 <= lo <= p <= hi <= 1
