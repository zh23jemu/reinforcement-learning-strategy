#!/bin/bash
#SBATCH --job-name=ops-demo-discrete
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

# 该脚本用于运行离散 OPS-DeMo 的论文级 PPO 训练。
# 如集群没有 GPU，可删除 --gres 行；Stable-Baselines3 PPO 也可以在 CPU 上运行。

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/slurm

.venv/bin/python -m rl_strategy.cli train --config configs/discrete_paper_like.yaml

