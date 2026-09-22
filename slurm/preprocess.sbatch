#!/bin/bash
#SBATCH --job-name=soy_kb_preprocess
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/preprocess_%j.out
#SBATCH --error=slurm/logs/preprocess_%j.err
# ---------------------------------------------------------------------------
# Preprocessing only — NO model / GPU calls. Submit with `sbatch`, never run
# on a login node. Self-activates the venv.
# ---------------------------------------------------------------------------
set -euo pipefail
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
cd /work/mech-ai-scratch/tirtho/SoybeanContrastKB
mkdir -p slurm/logs

python preprocess_data.py
python validate_splits.py
python build_lookalike_graph.py
echo "[preprocess.sbatch] done."
