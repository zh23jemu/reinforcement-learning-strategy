#!/bin/bash
#SBATCH --job-name=cont-resp-focus
#SBATCH --account=gpo-ifv7xx
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-5

# 连续 response policy 专项补强脚本。
# 背景：oracle 对照整体通过验收，但 seed 43 的 direct/attack 分策略仍弱于 baseline。
# 本脚本保持 detour 和 baseline 为基础训练步数，只加训 direct/attack response policy，
# 并继续用 oracle 对照评估，判断“更长训练”能否修复 direct/attack 短板。
# 默认使用 defq 分区；提交时可用 `sbatch --partition=目标分区 ...` 覆盖。

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
BASE_TIMESTEPS="${BASE_TIMESTEPS:-1500000}"
EPISODES="${EPISODES:-800}"
NAME_PREFIX="${NAME_PREFIX:-continuous_response_focus}"
INTERCEPTOR_SPEED="${INTERCEPTOR_SPEED:-0.030}"
INTRUDER_SPEED="${INTRUDER_SPEED:-0.016}"
COLLISION_RADIUS="${COLLISION_RADIUS:-0.08}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "错误：找不到可执行 Python：${PYTHON_BIN}"
  echo "请先在服务器创建 .venv 并安装依赖，或用 --export=ALL,PYTHON_BIN=/path/to/python 指定解释器。"
  exit 1
fi

SEEDS=(42 43 44)
# 两档加训强度：3M 用于验证训练不足假设，5M 用于确认加训是否继续改善。
DIRECT_ATTACK_TIMESTEPS=(3000000 5000000)

N_SEED=${#SEEDS[@]}
N_COMBO=${#DIRECT_ATTACK_TIMESTEPS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED_INDEX=$(( TASK_ID % N_SEED ))
COMBO_INDEX=$(( TASK_ID / N_SEED ))

if [ "${COMBO_INDEX}" -ge "${N_COMBO}" ]; then
  echo "错误：SLURM_ARRAY_TASK_ID=${TASK_ID} 超出 direct/attack 加训组合范围。"
  exit 1
fi

SEED=${SEEDS[$SEED_INDEX]}
FOCUS_TIMESTEPS=${DIRECT_ATTACK_TIMESTEPS[$COMBO_INDEX]}
RUN_PREFIX="${NAME_PREFIX}_da${FOCUS_TIMESTEPS}"

echo "============================================================"
echo "连续 direct/attack response policy 专项补强"
echo "Host:                  $(hostname)"
echo "Job ID:                ${SLURM_JOB_ID:-local}"
echo "Array ID:              ${SLURM_ARRAY_TASK_ID}"
echo "Combo index:           ${COMBO_INDEX}"
echo "Python:                ${PYTHON_BIN}"
echo "Install deps:          ${INSTALL_DEPS}"
echo "Environment:           ${INTERCEPTOR_SPEED} / ${INTRUDER_SPEED} / ${COLLISION_RADIUS}"
echo "Base timesteps:        ${BASE_TIMESTEPS}"
echo "Direct/attack steps:   ${FOCUS_TIMESTEPS}"
echo "Episodes:              ${EPISODES}"
echo "Seed:                  ${SEED}"
echo "Name prefix:           ${RUN_PREFIX}"
echo "Start time:            $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

if [ "${INSTALL_DEPS}" = "1" ]; then
  # 默认关闭安装步骤，防止 Slurm array 并发写同一个 editable 安装元数据。
  "${PYTHON_BIN}" -m pip install -e ".[dev]"
fi

"${PYTHON_BIN}" scripts/run_continuous_oracle_compare.py \
  --base-config configs/continuous_paper_like.yaml \
  --name-prefix "${RUN_PREFIX}" \
  --interceptor-speed "${INTERCEPTOR_SPEED}" \
  --intruder-speed "${INTRUDER_SPEED}" \
  --collision-radius "${COLLISION_RADIUS}" \
  --timesteps "${BASE_TIMESTEPS}" \
  --response-direct-timesteps "${FOCUS_TIMESTEPS}" \
  --response-attack-timesteps "${FOCUS_TIMESTEPS}" \
  --baseline-timesteps "${BASE_TIMESTEPS}" \
  --episodes "${EPISODES}" \
  --seed "${SEED}"

echo "Finished time: $(date '+%Y-%m-%d %H:%M:%S')"
