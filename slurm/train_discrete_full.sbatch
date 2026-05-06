#!/bin/bash
#SBATCH --job-name=ops-demo-discrete
#SBATCH --account=mfk-gsr2
#SBATCH --partition=plcyf-com
#SBATCH --qos=normal
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# 该脚本用于运行离散 OPS-DeMo 的论文级 PPO 训练。
# 已按 plcyf-com 集群的 mfk-gsr2 account 和 normal QOS 调整。

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/slurm

.venv/bin/python -m rl_strategy.cli train --config configs/discrete_paper_like.yaml
