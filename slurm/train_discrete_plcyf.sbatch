#!/bin/bash
#SBATCH --job-name=ops-demo-train
#SBATCH --account=mfk-gsr2
#SBATCH --partition=plcyf-com
#SBATCH --qos=normal
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# 离散 OPS-DeMo 论文级 PPO 长训练脚本。
# 该脚本只训练一次 opponent/response/baseline 三组 PPO 策略库。
# 训练完成后，再用 sweep 脚本批量评估不同 alpha/threshold，无需重复训练模型。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm artifacts/discrete runs

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m rl_strategy.cli train --config configs/discrete_paper_like.yaml

