#!/bin/bash
#SBATCH --job-name=ops-demo-sweep
#SBATCH --account=gpo-ifv7xx
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-23

# 离散 OPS-DeMo 参数 sweep 脚本。
# 默认组合：alpha 4 个值 x threshold 3 个值 x seed 2 个值 = 24 个任务。
# 提交前请先运行 train_discrete_plcyf.sbatch，确保 artifacts/discrete 下已有 PPO 模型。
# 默认使用本地 CPU 分区 defq，优先落到本地 com 节点；aws-com 可作为备用分区手动切换。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs runs

module purge
module load python/3.11.7

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "错误：找不到可执行 Python：${PYTHON_BIN}"
  echo "请先在服务器创建 .venv，或用 --export=ALL,PYTHON_BIN=/path/to/python 指定解释器。"
  exit 1
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
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}, Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Python: ${PYTHON_BIN}"

"${PYTHON_BIN}" scripts/run_discrete_sweep.py \
  --base-config configs/discrete_paper_like.yaml \
  --alpha "${ALPHA}" \
  --threshold "${THRESHOLD}" \
  --seed "${SEED}"
