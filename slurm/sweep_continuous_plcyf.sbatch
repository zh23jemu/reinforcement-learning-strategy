#!/bin/bash
#SBATCH --job-name=cont-sweep
#SBATCH --account=gpo-ifv7xx
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --array=0-11

# 连续二维拦截捕猎参数 sweep 脚本。
# 默认扫描 12 组中训参数：拦截者速度 3 档 x 入侵者速度 2 档 x 碰撞半径 2 档。
# 每个 array task 会训练独立模型目录、执行评估，并生成 analysis 诊断结果。
# 默认使用本地 CPU 分区 defq，优先落到本地 com 节点；aws-com 可作为备用分区手动切换。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs runs artifacts/continuous_sweep

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

# 第一轮先用 300k timesteps 找趋势；后续再把最优 2-3 组提高到 1M+ 确认。
INTERCEPTOR_SPEEDS=(0.022 0.026 0.030)
INTRUDER_SPEEDS=(0.016 0.018)
COLLISION_RADII=(0.08 0.10)
SEEDS=(42)
TIMESTEPS="${TIMESTEPS:-300000}"
EPISODES="${EPISODES:-300}"

N_INTERCEPTOR=${#INTERCEPTOR_SPEEDS[@]}
N_INTRUDER=${#INTRUDER_SPEEDS[@]}
N_COLLISION=${#COLLISION_RADII[@]}
N_SEED=${#SEEDS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED_INDEX=$(( TASK_ID % N_SEED ))
COLLISION_INDEX=$(( (TASK_ID / N_SEED) % N_COLLISION ))
INTRUDER_INDEX=$(( (TASK_ID / (N_SEED * N_COLLISION)) % N_INTRUDER ))
INTERCEPTOR_INDEX=$(( TASK_ID / (N_SEED * N_COLLISION * N_INTRUDER) ))

INTERCEPTOR_SPEED=${INTERCEPTOR_SPEEDS[$INTERCEPTOR_INDEX]}
INTRUDER_SPEED=${INTRUDER_SPEEDS[$INTRUDER_INDEX]}
COLLISION_RADIUS=${COLLISION_RADII[$COLLISION_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "============================================================"
echo "连续 OPS-DeMo sweep"
echo "Host:              $(hostname)"
echo "Job ID:            ${SLURM_JOB_ID:-local}"
echo "Array ID:          ${SLURM_ARRAY_TASK_ID}"
echo "Python:            ${PYTHON_BIN}"
echo "Interceptor speed: ${INTERCEPTOR_SPEED}"
echo "Intruder speed:    ${INTRUDER_SPEED}"
echo "Collision radius:  ${COLLISION_RADIUS}"
echo "Timesteps:         ${TIMESTEPS}"
echo "Episodes:          ${EPISODES}"
echo "Seed:              ${SEED}"
echo "Start time:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

"${PYTHON_BIN}" -m pip install -e ".[dev]"
"${PYTHON_BIN}" scripts/run_continuous_sweep.py \
  --base-config configs/continuous_paper_like.yaml \
  --interceptor-speed "${INTERCEPTOR_SPEED}" \
  --intruder-speed "${INTRUDER_SPEED}" \
  --collision-radius "${COLLISION_RADIUS}" \
  --timesteps "${TIMESTEPS}" \
  --episodes "${EPISODES}" \
  --seed "${SEED}"

echo "Finished time: $(date '+%Y-%m-%d %H:%M:%S')"
