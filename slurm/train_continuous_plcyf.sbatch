#!/bin/bash
#SBATCH --job-name=cont-train
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

# 连续二维拦截捕猎场景的 PPO 训练脚本。
# 该脚本会训练三类入侵策略对应的响应策略，以及一个面对非平稳入侵者的 baseline。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm artifacts/continuous runs

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m rl_strategy.cli train --config configs/continuous_paper_like.yaml

