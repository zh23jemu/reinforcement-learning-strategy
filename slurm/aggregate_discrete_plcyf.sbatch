#!/bin/bash
#SBATCH --job-name=ops-demo-agg
#SBATCH --account=mfk-gsr2
#SBATCH --partition=plcyf-com
#SBATCH --qos=normal
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# 汇总 sweep 结果，生成 runs/discrete_sweep_summary.csv。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

.venv/bin/python scripts/aggregate_discrete_sweep.py \
  --runs-root runs \
  --output runs/discrete_sweep_summary.csv

