# Leveraging Contrasting Visual References for Knowledge-Guided Few-Shot Crop Disease Diagnosis
### Soybean prototype — source-grounded KB + contrasting visual references

> **PRELIMINARY / SOYBEAN-ONLY.** Five held-out queries per eligible class. Per-class
> p-values are not credible (n=5); all formal tests pool paired query outcomes while
> preserving class structure (protocol §6). A larger evaluation is required for
> agricultural claims.

Implements the experimental protocol
`Soybean_KB_Contrasting_Visual_References_Experimental_Protocol.pdf`.

---

## Data rule (non-negotiable)
ALL images — queries **and** references — come from the **local CyAg dataset only**:
`{sage_curated_dataset_dir}/Images/Soybean/<class>/*.jpg`.
The Hugging Face `plantseg-lookalike` dataset supplies **only** the frozen look-alike
relationships + source-grounded differentiating knowledge (loaded by
`kb_loader.load_lookalike_graph`). Source images are never moved or altered (copy only).

An **eligible class** has a frozen KB description AND ≥ `min_images_per_eligible_class`
local images. In this snapshot: **15 eligible soybean classes**.

---

## Pipeline (protocol §4)
```
Stage 1  Broad KB screening   query + full taxonomy + KB descriptions -> ranked top-k
Stage 2  Focused re-read      query + shortlist descriptions           -> re-ranked + evidence
Stage 3  Reference routing    look-alike subgraph + budget B           -> selected refs
Stage 4  Final diagnosis      query + candidate desc + exactly B refs  -> final label
```
Stage 1/2 are **cached per query** and shared across KB conditions, so cross-policy
comparisons isolate the reference effect only.

The frozen VLM is driven through the `claude -p` headless CLI (`vlm_backend.VLMBackend`),
which reads each image via the Read tool. No API key is stored in the repo.

---

## Condition registry (protocol §3)
`B` = **total** reference images in the prompt (never per class).

| Code | KB | Two-stage | Reference policy | Shots |
|---|---|---|---|---|
| Z0  | no  | no  | none | 0 |
| K0  | yes | yes | none | 0 |
| G2/G4/G8 | no | no | general (AgEval-style) | 2/4/8 |
| KG2/KG4/KG8 | yes | yes | general candidate-specific | 2/4/8 |
| KS2/KS4/KS8 | yes | yes | similarity retrieval | 2/4/8 |
| KC2/KC4/KC8 | yes | yes | **contrasting pairs + general fallback** (proposed) | 2/4/8 |
| KCF2/KCF4/KCF8 | yes | yes | validated counterfactuals only (sensitivity) | 2/4/8 |

**Primary comparison:** KC8 vs KG8 (two-sided exact McNemar, α=0.05).
**Confirmatory secondary (Holm-corrected):** KC8 vs G8, K0 vs Z0, KC4 vs G8, KC8 vs KS8.

Baselines Z0 and G2/G4/G8 reuse AgEval's verbatim `universal_prompt`
(`third_party_AgEval/inference.py`) with a closed candidate list = full soybean taxonomy.

---

## Exact commands

Configure paths in `config.yaml` first, then activate the venv:
```bash
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
```

**1. Preprocess (no model calls).** Copies query/support images into an immutable
snapshot, computes SHA-256 + perceptual hashes, removes exact/near duplicates,
writes frozen checksummed manifests.
```bash
python preprocess_data.py                 # full
python preprocess_data.py --dry-run       # 2 classes, no snapshot copies
python validate_splits.py                 # exits nonzero on any leakage/integrity error
python build_lookalike_graph.py           # freeze graph.json (validation-only weights)
```

**2. Inference.** Proposed conditions + AgEval baselines over the shared query set.
```bash
python run_conditions.py                  # all conditions × queries (claude -p)
python run_conditions.py --dry-run        # deterministic mock VLM, no model/GPU calls
python run_baselines.py                   # Z0/G2/G4/G8 (AgEval-style)
python run_baselines.py --dry-run
# flags: --conditions Z0,KC8  --classes Frogeye_Leaf_Spot  --limit 10
```

**3. Statistics, figures, report.**
```bash
python compute_statistics.py              # McNemar, Cochran's Q, Holm, bootstrap CIs
python make_plots.py                      # 9 figures -> outputs/figures/ (PDF+PNG)
python make_report.py                     # outputs/report/REPORT.md + results_summary.json
```

**SLURM (submit only — never run compute on a login node):**
```bash
sbatch slurm/preprocess.sbatch
sbatch slurm/inference_array.sbatch       # sharded by class (array 0-14)
sbatch slurm/stats_and_plots.sbatch
```
All sbatch scripts self-activate `/work/mech-ai-scratch/tirtho/.venv` and use
`--account=mech-ai --qos=normal --partition=nova`.

---

## Outputs layout
```
outputs/
  snapshot/{queries,supports/general,supports/lookalike,supports/counterfactual}/<class>/
  query_manifest.csv  support_manifest.csv  class_manifest.csv
  kb_manifest.jsonl   pair_manifest.csv     exclusions.csv
  split_audit.json    run_manifest.json     graph.json
  outputs/
    stage1.jsonl  stage2.jsonl  retrieval.jsonl
    preds/final_predictions.jsonl
    baselines/baseline_predictions.jsonl  baseline_results.csv
    stats/results_summary.json  main_results.{csv,tex}  contrasts.csv
    figures/fig1..fig9.{pdf,png}
    report/REPORT.md  results_summary.json
```

---

## Reproducibility contract (protocol §8)
- Configuration-driven paths; no hard-coded credentials or HF tokens.
- Resume-safe JSONL, one record per (query, condition, seed), atomic writes.
- Deterministic manifests; unavailable examples are recorded in `exclusions.csv`,
  never silently replaced.
- `run_manifest.json` records git commit, config SHA-256, model id, seed, and the
  SHA-256 of every frozen manifest.
- Look-alike graph weights use **validation-only** `n_pairs`; test labels are never
  used to build/tune the graph, choose k, or retrieve references.
- A dry-run mode runs preprocessing and the full stage composition with **no** model/GPU
  calls (`--dry-run` uses a deterministic mock VLM).

## Tests
```bash
python -m pytest tests/ -q
```
Covers shot-budget accounting (B = total refs, symmetric split), no query/support
overlap, class-mapping canonicalization, look-alike pair symmetry, bootstrap pairing
preservation, and Holm correction correctness.

---
_Findings from this repository are preliminary and soybean-only._

# LOOK-A-LIKE
Source-grounded, KB-guided contrasting-visual-reference few-shot crop-disease diagnosis (soybean prototype).
Built on AgEval; images from local CyAg curated dataset; look-alike graph + differentiating knowledge from plantseg-lookalike.
See protocol and `run_gold_example.py` for the gold-standard worked example.

Note: vendored baseline is git-ignored; clone it with:
```
git clone https://github.com/arbab-ml/AgEval third_party_AgEval
```
