#!/bin/bash
#SBATCH --job-name=cont-oracle
#SBATCH --account=gpo-ifv7xx
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-2

# 连续 response policy oracle 对照脚本。
# 目标：不再调 SAM 检测器，而是假设检测完全正确，直接按真实入侵策略选择
# response_direct / response_detour / response_attack。这样可以判断 seed 43 的
# direct/attack 短板到底来自响应策略库本身，还是来自 SAM 检测/切换时序。
# 默认使用 defq 分区；提交时仍可用 `sbatch --partition=目标分区 ...` 覆盖。

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
INSTALL_DEPS="${INSTALL_DEPS:-0}"
TIMESTEPS="${TIMESTEPS:-1500000}"
EPISODES="${EPISODES:-800}"
NAME_PREFIX="${NAME_PREFIX:-continuous_oracle_compare}"
INTERCEPTOR_SPEED="${INTERCEPTOR_SPEED:-0.030}"
INTRUDER_SPEED="${INTRUDER_SPEED:-0.016}"
COLLISION_RADIUS="${COLLISION_RADIUS:-0.08}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "错误：找不到可执行 Python：${PYTHON_BIN}"
  echo "请先在服务器创建 .venv 并安装依赖，或用 --export=ALL,PYTHON_BIN=/path/to/python 指定解释器。"
  exit 1
fi

SEEDS=(42 43 44)
TASK_ID=${SLURM_ARRAY_TASK_ID}

if [ "${TASK_ID}" -ge "${#SEEDS[@]}" ]; then
  echo "错误：SLURM_ARRAY_TASK_ID=${TASK_ID} 超出 seeds 范围。"
  exit 1
fi

SEED=${SEEDS[$TASK_ID]}

echo "============================================================"
echo "连续 response policy oracle 对照"
echo "Host:              $(hostname)"
echo "Job ID:            ${SLURM_JOB_ID:-local}"
echo "Array ID:          ${SLURM_ARRAY_TASK_ID}"
echo "Python:            ${PYTHON_BIN}"
echo "Install deps:      ${INSTALL_DEPS}"
echo "Environment:       ${INTERCEPTOR_SPEED} / ${INTRUDER_SPEED} / ${COLLISION_RADIUS}"
echo "Timesteps:         ${TIMESTEPS}"
echo "Episodes:          ${EPISODES}"
echo "Seed:              ${SEED}"
echo "Name prefix:       ${NAME_PREFIX}"
echo "Start time:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

if [ "${INSTALL_DEPS}" = "1" ]; then
  # 默认关闭安装步骤，防止 Slurm array 并发写同一个 editable 安装元数据。
  "${PYTHON_BIN}" -m pip install -e ".[dev]"
fi

"${PYTHON_BIN}" scripts/run_continuous_oracle_compare.py \
  --base-config configs/continuous_paper_like.yaml \
  --name-prefix "${NAME_PREFIX}" \
  --interceptor-speed "${INTERCEPTOR_SPEED}" \
  --intruder-speed "${INTRUDER_SPEED}" \
  --collision-radius "${COLLISION_RADIUS}" \
  --timesteps "${TIMESTEPS}" \
  --episodes "${EPISODES}" \
  --seed "${SEED}"

echo "Finished time: $(date '+%Y-%m-%d %H:%M:%S')"
