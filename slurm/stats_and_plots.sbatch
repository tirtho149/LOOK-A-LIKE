#!/bin/bash
#SBATCH --job-name=soy_kb_stats
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/stats_%j.out
#SBATCH --error=slurm/logs/stats_%j.err
# ---------------------------------------------------------------------------
# Statistics + figures + report. NO model / GPU calls. Submit with `sbatch`.
# ---------------------------------------------------------------------------
set -euo pipefail
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
cd /work/mech-ai-scratch/tirtho/SoybeanContrastKB
mkdir -p slurm/logs

python compute_statistics.py
python make_plots.py
python make_report.py
echo "[stats_and_plots.sbatch] done."
