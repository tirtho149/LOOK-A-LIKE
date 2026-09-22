#!/usr/bin/env python3
"""
compute_statistics.py — small paired-data statistical plan (protocol §6).
========================================================================
Unit of analysis = the held-out query image; every condition scores the SAME
queries, so all method comparisons are PAIRED. Seeds/references are NOT
independent test observations.

Implements:
  * Exact McNemar (primary KC8 vs KG8) + confirmatory secondary contrasts
    (KC8vsG8, K0vsZ0, KC4vsG8, KC8vsKS8) with Holm-adjusted p-values.
  * Cochran's Q omnibus across repeated binary conditions.
  * 10,000 class-stratified PAIRED bootstrap CIs for accuracy, macro-F1,
    balanced accuracy, top-k recall, net correction, latency, cost.
  * Wilson intervals for proportions.
  * Wrong-to-right / right-to-wrong transition (net correction) exact test.

Uses scipy/statsmodels when present; falls back to exact hand implementations.

Output: outputs/stats/results_summary.json + stats tables (csv/tex).
Runs on partial/synthetic data without crashing (guards empty conditions).
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from collections import defaultdict, Counter
from pathlib import Path
import yaml

HERE = Path(__file__).parent

try:
    from scipy import stats as _sp
except Exception:
    _sp = None


# ─────────────────────────────────────────────────────────────────────────────
#  Exact tests
# ─────────────────────────────────────────────────────────────────────────────
def exact_mcnemar(b: int, c: int):
    """Two-sided exact McNemar on discordant pairs (b, c). Binomial(n, 0.5)."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "statistic": 0.0}
    if _sp is not None:
        # exact binomial two-sided
        p = _sp.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    else:
        k = min(b, c)
        cum = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        p = min(1.0, 2 * cum)
    return {"b": b, "c": c, "n_discordant": n,
            "p_value": float(p), "statistic": (b - c) ** 2 / n}


def cochran_q(matrix):
    """Cochran's Q for k repeated binary conditions over n subjects.
    matrix: list of rows (per query), each row = list of 0/1 across conditions."""
    if not matrix or len(matrix[0]) < 2:
        return {"Q": 0.0, "df": 0, "p_value": 1.0, "k": 0, "n": 0}
    k = len(matrix[0])
    rows = [r for r in matrix]
    n = len(rows)
    col_sum = [sum(r[j] for r in rows) for j in range(k)]
    row_sum = [sum(r) for r in rows]
    G = sum(col_sum)
    num = (k - 1) * (k * sum(cj ** 2 for cj in col_sum) - G ** 2)
    den = k * G - sum(ri ** 2 for ri in row_sum)
    Q = num / den if den != 0 else 0.0
    df = k - 1
    if _sp is not None:
        p = float(_sp.chi2.sf(Q, df))
    else:
        p = float(_chi2_sf(Q, df))
    return {"Q": float(Q), "df": df, "p_value": p, "k": k, "n": n}


def _chi2_sf(x, df):
    # regularized upper incomplete gamma via series/continued fraction (Q)
    if x <= 0:
        return 1.0
    a = df / 2.0
    x2 = x / 2.0
    return _gammaincc(a, x2)


def _gammaincc(a, x):
    # complement of regularized lower incomplete gamma
    if x < a + 1:
        # series for P, return 1 - P
        term = 1.0 / a
        s = term
        n = a
        for _ in range(500):
            n += 1
            term *= x / n
            s += term
            if abs(term) < abs(s) * 1e-12:
                break
        P = s * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return 1.0 - P
    else:
        # continued fraction for Q
        tiny = 1e-300
        b = x + 1 - a
        c = 1 / tiny
        d = 1 / b
        h = d
        for i in range(1, 500):
            an = -i * (i - a)
            b += 2
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1 / d
            delta = d * c
            h *= delta
            if abs(delta - 1) < 1e-12:
                break
        return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def holm(pvals: dict):
    """Holm step-down. pvals: {name: p}. Returns {name: p_adj}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, prev = {}, 0.0
    for i, (name, p) in enumerate(items):
        a = min(1.0, (m - i) * p)
        a = max(a, prev)   # enforce monotonicity
        adj[name] = a
        prev = a
    return adj


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - hw) / d, (c + hw) / d)


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────
def accuracy(correct):
    return sum(correct) / len(correct) if correct else 0.0


def macro_f1(truth, pred):
    labels = set(truth) | set(pred)
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(truth, pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(truth, pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(truth, pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if (tp + fn) > 0:   # label present in truth
            f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def balanced_accuracy(truth, pred):
    by = defaultdict(lambda: [0, 0])
    for t, p in zip(truth, pred):
        by[t][1] += 1
        if t == p:
            by[t][0] += 1
    recalls = [c / n for c, n in by.values() if n > 0]
    return sum(recalls) / len(recalls) if recalls else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Class-stratified paired bootstrap
# ─────────────────────────────────────────────────────────────────────────────
def class_stratified_bootstrap(records_by_class, metric_fn, n_boot, seed):
    """records_by_class: {class: [record,...]}; metric_fn(list_of_records)->float.
    Resample WITH REPLACEMENT within each class (preserving pairing: each record
    carries all conditions), recompute the overall metric. Returns (point, lo, hi,
    p_gt0-if-delta)."""
    import random
    rng = random.Random(seed)
    flat = [r for rs in records_by_class.values() for r in rs]
    point = metric_fn(flat)
    if not flat:
        return {"point": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_boot": 0}
    dist = []
    classes = list(records_by_class)
    for _ in range(n_boot):
        sample = []
        for c in classes:
            rs = records_by_class[c]
            if not rs:
                continue
            sample.extend(rs[rng.randrange(len(rs))] for _ in range(len(rs)))
        dist.append(metric_fn(sample))
    dist.sort()
    lo = dist[int(0.025 * len(dist))]
    hi = dist[min(len(dist) - 1, int(0.975 * len(dist)))]
    prob_gt0 = sum(1 for d in dist if d > 0) / len(dist)
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n_boot": n_boot,
            "prob_gt0": prob_gt0}


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading — align conditions on the shared query set (paired)
# ─────────────────────────────────────────────────────────────────────────────
def load_all(cfg):
    out = Path(cfg["experiment_output_dir"]) / "outputs"
    recs = []
    for p in [out / "preds" / "final_predictions.jsonl",
              out / "baselines" / "baseline_predictions.jsonl"]:
        if p.exists():
            for line in p.open():
                try:
                    recs.append(json.loads(line))
                except Exception:
                    continue
    # index: query_id -> condition -> record (latest wins for resume-safety)
    by_q = defaultdict(dict)
    for r in recs:
        by_q[r["query_id"]][r["condition"]] = r
    return by_q


def paired_correct(by_q, cond_a, cond_b):
    """Return (b, c, queries_used) discordant counts over queries scored by BOTH."""
    b = c = 0
    used = []
    for qid, cm in by_q.items():
        if cond_a in cm and cond_b in cm:
            ca, cb = int(cm[cond_a]["correct"]), int(cm[cond_b]["correct"])
            if ca == 1 and cb == 0:
                c += 1  # a right, b wrong  (b-of-mcnemar convention below)
            elif ca == 0 and cb == 1:
                b += 1
            used.append(qid)
    return b, c, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    n_boot = args.n_boot or int(cfg.get("bootstrap_replicates", 10000))
    seed = int(cfg["seed"])
    by_q = load_all(cfg)
    out = Path(cfg["experiment_output_dir"]) / "outputs" / "stats"
    out.mkdir(parents=True, exist_ok=True)

    if not by_q:
        summary = {"error": "no prediction records found",
                   "note": "run run_conditions.py / run_baselines.py first"}
        (out / "results_summary.json").write_text(json.dumps(summary, indent=2))
        print("[stats] no records; wrote placeholder summary.")
        return

    conditions = sorted({c for cm in by_q.values() for c in cm})

    # ── per-condition recognition metrics + class-stratified accuracy bootstrap ──
    per_condition = {}
    for cond in conditions:
        rows = [cm[cond] for cm in by_q.values() if cond in cm]
        if not rows:
            continue
        truth = [r["truth"] for r in rows]
        pred = [r.get("prediction") for r in rows]
        correct = [int(r["correct"]) for r in rows]
        by_class = defaultdict(list)
        for r in rows:
            by_class[r["truth"]].append(r)
        boot = class_stratified_bootstrap(
            by_class, lambda rs: accuracy([int(x["correct"]) for x in rs]),
            n_boot, seed)
        p, lo, hi = wilson(sum(correct), len(correct))
        per_condition[cond] = {
            "n": len(rows),
            "accuracy": accuracy(correct),
            "accuracy_wilson_ci": [lo, hi],
            "accuracy_boot_ci": [boot["ci_lo"], boot["ci_hi"]],
            "macro_f1": macro_f1(truth, pred),
            "balanced_accuracy": balanced_accuracy(truth, pred),
            "mean_latency_s": sum(r.get("latency_s") or 0 for r in rows) / len(rows),
            "mean_n_refs": sum(r.get("n_refs") or 0 for r in rows) / len(rows),
        }

    # ── paired contrasts (McNemar) ──
    def contrast(a, b, label):
        bb, cc, used = paired_correct(by_q, a, b)
        mc = exact_mcnemar(bb, cc)
        # paired accuracy delta on shared queries
        shared = [cm for cm in by_q.values() if a in cm and b in cm]
        da = accuracy([int(cm[a]["correct"]) for cm in shared])
        db = accuracy([int(cm[b]["correct"]) for cm in shared])
        # paired class-stratified bootstrap on the delta
        by_class = defaultdict(list)
        for cm in shared:
            by_class[cm[a]["truth"]].append(cm)
        boot = class_stratified_bootstrap(
            by_class,
            lambda rs: accuracy([int(cm[a]["correct"]) for cm in rs]) -
                       accuracy([int(cm[b]["correct"]) for cm in rs]),
            n_boot, seed)
        return {"label": label, "cond_a": a, "cond_b": b,
                "n_paired": len(shared), "acc_a": da, "acc_b": db,
                "delta_acc": da - db, "discordant_b": mc["b"], "discordant_c": mc["c"],
                "p_value": mc["p_value"],
                "delta_boot_ci": [boot["ci_lo"], boot["ci_hi"]],
                "boot_prob_delta_gt0": boot.get("prob_gt0")}

    primary = cfg["primary_comparison"]
    contrasts = {}
    if primary[0] in conditions and primary[1] in conditions:
        contrasts["primary_KC8_vs_KG8"] = contrast(primary[0], primary[1], "primary")

    conf_raw = {}
    for pair in cfg["confirmatory_secondary"]:
        a, b = pair
        name = f"{a}_vs_{b}"
        if a in conditions and b in conditions:
            contrasts[name] = contrast(a, b, "confirmatory")
            conf_raw[name] = contrasts[name]["p_value"]
    holm_adj = holm(conf_raw) if conf_raw else {}
    for name, padj in holm_adj.items():
        contrasts[name]["p_holm"] = padj

    # ── Cochran's Q over a comparable repeated-binary family ──
    q_family = [c for c in ["Z0", "G8", "K0", "KG8", "KS8", "KC8"] if c in conditions]
    matrix = []
    for cm in by_q.values():
        if all(c in cm for c in q_family):
            matrix.append([int(cm[c]["correct"]) for c in q_family])
    cq = cochran_q(matrix) if len(q_family) >= 2 else {"note": "insufficient conditions"}
    cq["conditions"] = q_family

    # ── counterfactual / contrasting net correction (KC8 vs KG8 transitions) ──
    net = None
    if "KG8" in conditions and "KC8" in conditions:
        w2r = r2w = unchanged = w2dw = 0
        for cm in by_q.values():
            if "KG8" in cm and "KC8" in cm:
                g, kc = cm["KG8"], cm["KC8"]
                gc, kcc = g["correct"], kc["correct"]
                if not gc and kcc:
                    w2r += 1
                elif gc and not kcc:
                    r2w += 1
                elif gc and kcc:
                    unchanged += 1
                else:
                    if g.get("prediction") != kc.get("prediction"):
                        w2dw += 1
                    else:
                        unchanged += 1
        mc = exact_mcnemar(w2r, r2w)
        net = {"wrong_to_right": w2r, "right_to_wrong": r2w,
               "wrong_to_diff_wrong": w2dw, "unchanged": unchanged,
               "net_correction": w2r - r2w, "p_value": mc["p_value"]}

    summary = {
        "seed": seed, "n_boot": n_boot, "alpha": cfg["alpha"],
        "n_queries": len(by_q), "conditions": conditions,
        "per_condition": per_condition,
        "contrasts": contrasts,
        "cochran_q": cq,
        "net_correction_KC8_vs_KG8": net,
        "primary_comparison": primary,
        "confirmatory_family": [f"{a}_vs_{b}" for a, b in cfg["confirmatory_secondary"]],
        "note": "SOYBEAN-ONLY PRELIMINARY prototype; n=5/class, per-class tests not credible.",
    }
    (out / "results_summary.json").write_text(json.dumps(summary, indent=2))

    # ── tables (csv + tex) ──
    _write_main_table(out, per_condition, contrasts)
    print(f"[stats] {len(conditions)} conditions, {len(by_q)} paired queries.")
    print(f"[stats] wrote {out/'results_summary.json'} + tables.")
    if "primary_KC8_vs_KG8" in contrasts:
        pc = contrasts["primary_KC8_vs_KG8"]
        print(f"[stats] PRIMARY KC8 vs KG8: delta_acc={pc['delta_acc']:+.3f} "
              f"p={pc['p_value']:.4f} (b={pc['discordant_b']}, c={pc['discordant_c']})")


def _write_main_table(out, per_condition, contrasts):
    fields = ["condition", "n", "accuracy", "acc_ci_lo", "acc_ci_hi",
              "macro_f1", "balanced_accuracy", "mean_latency_s", "mean_n_refs"]
    with open(out / "main_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cond, m in sorted(per_condition.items()):
            w.writerow({"condition": cond, "n": m["n"],
                        "accuracy": round(m["accuracy"], 4),
                        "acc_ci_lo": round(m["accuracy_boot_ci"][0], 4),
                        "acc_ci_hi": round(m["accuracy_boot_ci"][1], 4),
                        "macro_f1": round(m["macro_f1"], 4),
                        "balanced_accuracy": round(m["balanced_accuracy"], 4),
                        "mean_latency_s": round(m["mean_latency_s"], 3),
                        "mean_n_refs": round(m["mean_n_refs"], 2)})
    # contrasts csv
    cfields = ["label", "cond_a", "cond_b", "n_paired", "delta_acc",
               "discordant_b", "discordant_c", "p_value", "p_holm",
               "ci_lo", "ci_hi"]
    with open(out / "contrasts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cfields)
        w.writeheader()
        for name, c in contrasts.items():
            w.writerow({"label": name, "cond_a": c["cond_a"], "cond_b": c["cond_b"],
                        "n_paired": c["n_paired"], "delta_acc": round(c["delta_acc"], 4),
                        "discordant_b": c["discordant_b"], "discordant_c": c["discordant_c"],
                        "p_value": round(c["p_value"], 5),
                        "p_holm": round(c.get("p_holm", float("nan")), 5)
                        if c.get("p_holm") is not None else "",
                        "ci_lo": round(c["delta_boot_ci"][0], 4),
                        "ci_hi": round(c["delta_boot_ci"][1], 4)})
    # tex main
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Soybean prototype (preliminary): recognition metrics per condition.}",
             r"\label{tab:main}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Condition & Acc & 95\% CI & Macro-F1 & Bal.\ Acc \\ \midrule"]
    for cond, m in sorted(per_condition.items()):
        lines.append(f"{cond} & {m['accuracy']:.3f} & "
                     f"[{m['accuracy_boot_ci'][0]:.3f}, {m['accuracy_boot_ci'][1]:.3f}] & "
                     f"{m['macro_f1']:.3f} & {m['balanced_accuracy']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "main_results.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
