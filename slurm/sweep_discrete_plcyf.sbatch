#!/bin/bash
#SBATCH --job-name=ops-demo-sweep
#SBATCH --account=mfk-gsr2
#SBATCH --partition=plcyf-com
#SBATCH --qos=normal
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-23

# 离散 OPS-DeMo 参数 sweep 脚本。
# 默认组合：alpha 4 个值 x threshold 3 个值 x seed 2 个值 = 24 个任务。
# 提交前请先运行 train_discrete_plcyf.sbatch，确保 artifacts/discrete 下已有 PPO 模型。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm runs

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -e ".[dev]"
fi

ALPHAS=(0.80 0.90 0.95 0.99)
THRESHOLDS=(4.0 6.0 8.0)
SEEDS=(42 43)

N_ALPHA=${#ALPHAS[@]}
N_THRESHOLD=${#THRESHOLDS[@]}
N_SEED=${#SEEDS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED_INDEX=$(( TASK_ID % N_SEED ))
THRESHOLD_INDEX=$(( (TASK_ID / N_SEED) % N_THRESHOLD ))
ALPHA_INDEX=$(( TASK_ID / (N_SEED * N_THRESHOLD) ))

ALPHA=${ALPHAS[$ALPHA_INDEX]}
THRESHOLD=${THRESHOLDS[$THRESHOLD_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "运行 sweep: alpha=${ALPHA}, threshold=${THRESHOLD}, seed=${SEED}"

.venv/bin/python scripts/run_discrete_sweep.py \
  --base-config configs/discrete_paper_like.yaml \
  --alpha "${ALPHA}" \
  --threshold "${THRESHOLD}" \
  --seed "${SEED}"

