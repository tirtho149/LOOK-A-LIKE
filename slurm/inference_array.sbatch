#!/bin/bash
#SBATCH --job-name=soy_kb_infer
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=slurm/logs/infer_%A_%a.out
#SBATCH --error=slurm/logs/infer_%A_%a.err
# ---------------------------------------------------------------------------
# Full inference (proposed conditions + AgEval baselines) sharded by CLASS.
# Each array task processes one eligible soybean class through every condition.
# Drives `claude -p` headless; resume-safe (JSONL skips completed records).
# Submit with `sbatch`; never run on a login node.
# ---------------------------------------------------------------------------
set -euo pipefail
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
cd /work/mech-ai-scratch/tirtho/SoybeanContrastKB
mkdir -p slurm/logs

# Map SLURM array id -> class name from the frozen class manifest.
CLASS=$(python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys, csv, yaml
from pathlib import Path
cfg = yaml.safe_load(open("config.yaml"))
rows = list(csv.DictReader(open(Path(cfg["experiment_output_dir"])/"class_manifest.csv")))
i = int(sys.argv[1])
print(rows[i]["class"] if i < len(rows) else "")
PY
)
if [ -z "$CLASS" ]; then echo "no class for array id $SLURM_ARRAY_TASK_ID"; exit 0; fi
echo "[infer] class = $CLASS"

python run_conditions.py --classes "$CLASS"
python run_baselines.py  --classes "$CLASS"
echo "[inference_array.sbatch] done class=$CLASS"
