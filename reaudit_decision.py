#!/usr/bin/env python3
"""
Isolate the decision-procedure change with references held fixed (no new inference).
Re-adjudicates the EXISTING condition-C observations under the new rules
(initial prediction = condition B). Writes to a NEW dir; never overwrites the
original experiment.
"""
import json, collections
from pathlib import Path
import yaml
import evidence_adjudication as EA

A, B = "Bacterial_Blight", "Bacterial_Pustule"
cfg = yaml.safe_load(open("config.yaml"))
EXP = Path(cfg["experiment_output_dir"]) / "pair_experiment" / f"{A}__vs__{B}"
OUT = EXP / "reaudit_decision"; OUT.mkdir(parents=True, exist_ok=True)

records = [json.loads(l) for l in (EXP / "records.jsonl").open()]
checklist = json.loads((EXP / "checklist.json").read_text())

rows, out = [], (OUT / "reaudit.jsonl").open("w")
for rec in records:
    initial = rec["B_pred"]                       # predeclared initial = condition B
    ra = EA.reaudit_record(rec, checklist, A, B, initial)
    d = {"query_id": rec["query_id"], "truth": rec["truth"],
         "B_pred": rec["B_pred"], "B_correct": rec["B_correct"],
         "oldC_pred": rec["C_pred"], "oldC_correct": rec["C_correct"],
         "newD_pred": ra["final_pred"], "newD_correct": ra["final_pred"] == rec["truth"],
         "newD_decision": ra["decision"], "newD_changed": ra["adjudication"]["changed_from_initial"],
         "newD_reason": ra["adjudication"]["reason"],
         "supports": {"A": ra["adjudication"]["supports_A"], "B": ra["adjudication"]["supports_B"],
                      "contradicts_A": ra["adjudication"]["contradicts_A"],
                      "contradicts_B": ra["adjudication"]["contradicts_B"]},
         "interpretation": ra["interpretation"]}
    rows.append(d); out.write(json.dumps(d) + "\n")
out.close()


def acc(key, cls=None):
    r = [x for x in rows if cls is None or x["truth"] == cls]
    return (sum(x[key] for x in r), len(r))


summary = {
    "n": len(rows),
    "accuracy": {
        "B_refs_only":   {"overall": acc("B_correct"), A: acc("B_correct", A), B: acc("B_correct", B)},
        "oldC_voting":   {"overall": acc("oldC_correct"), A: acc("oldC_correct", A), B: acc("oldC_correct", B)},
        "newD_adjudicated": {"overall": acc("newD_correct"), A: acc("newD_correct", A), B: acc("newD_correct", B)}},
    "D_vs_B": {
        "fixed": sum(1 for x in rows if not x["B_correct"] and x["newD_correct"]),
        "introduced": sum(1 for x in rows if x["B_correct"] and not x["newD_correct"])},
    "oldC_vs_B": {
        "fixed": sum(1 for x in rows if not x["B_correct"] and x["oldC_correct"]),
        "introduced": sum(1 for x in rows if x["B_correct"] and not x["oldC_correct"])},
    "flips_made_by_D": [x["query_id"] for x in rows if x["newD_changed"]],
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))

print("== Decision-procedure change (references fixed, no new inference) ==\n")
def fmt(t): return f"{t[0]}/{t[1]}"
print(f"{'condition':22s} overall  {A[:14]:14s} {B}")
for name, k in [("B (refs only)", "B_refs_only"), ("old C (row voting)", "oldC_voting"),
                ("new D (adjudicated)", "newD_adjudicated")]:
    a = summary["accuracy"][k]
    print(f"  {name:20s} {fmt(a['overall']):7s} {fmt(a[A]):14s} {fmt(a[B])}")
print(f"\n  D vs B: fixed={summary['D_vs_B']['fixed']} introduced={summary['D_vs_B']['introduced']}")
print(f"  oldC vs B: fixed={summary['oldC_vs_B']['fixed']} introduced={summary['oldC_vs_B']['introduced']}")
print(f"  flips made by D: {summary['flips_made_by_D']}")

print("\n== spot: Bacterial_Blight__Soybean_PNAS_5031 ==")
r = next(x for x in rows if x["query_id"].endswith("5031"))
print(f"  truth={r['truth']}  B={r['B_pred']}  oldC={r['oldC_pred']}({'ok' if r['oldC_correct'] else 'WRONG'})  "
      f"newD={r['newD_pred']}({'ok' if r['newD_correct'] else 'WRONG'})  decision={r['newD_decision']}")
print(f"  reason: {r['newD_reason']}")
for it in r["interpretation"]:
    print(f"    - {it['feature'][:42]:42s} status={it['status']:14s} -> {it['label']}")
