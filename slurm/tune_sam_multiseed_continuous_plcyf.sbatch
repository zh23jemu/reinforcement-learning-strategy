#!/bin/bash
#SBATCH --job-name=cont-sam-ms
#SBATCH --account=gpo-ifv7xx
#SBATCH --partition=ISP
#SBATCH --qos=normal
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --array=0-17%2

# 连续场景 SAM 检测器多 seed 小规模调参脚本。
# 上一轮单 seed tune 在 seed 42 上表现好，但 1.5M confirm 中 seed 43/44
# 经常 0 次切换，说明检测器偏保守。本脚本固定环境参数 0.030 / 0.016 / 0.08，
# 用 6 组更激进的切换参数 x 3 个 seed 做 300k 快速筛选。
# 目标不是马上追最高单次回报，而是找出每个 seed 都能合理触发切换的组合。

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

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "错误：找不到可执行 Python：${PYTHON_BIN}"
  echo "请先在服务器创建 .venv 并安装依赖，或用 --export=ALL,PYTHON_BIN=/path/to/python 指定解释器。"
  exit 1
fi

# 固定上一轮相对最优的连续环境，只搜索 SAM 检测器参数。
INTERCEPTOR_SPEED="${INTERCEPTOR_SPEED:-0.030}"
INTRUDER_SPEED="${INTRUDER_SPEED:-0.016}"
COLLISION_RADIUS="${COLLISION_RADIUS:-0.08}"
TIMESTEPS="${TIMESTEPS:-300000}"
EPISODES="${EPISODES:-300}"
SAM_FEATURE_MODE="${SAM_FEATURE_MODE:-geometry}"
SEEDS=(42 43 44)

# 6 组更激进的检测器组合，重点降低 threshold/cooldown/margin，避免 0 次切换。
# 组合 0-2 偏“保守但会切”，组合 3-5 偏“更容易切”，用于观察误切换与收益的平衡。
THRESHOLDS=(8 10 10 12 12 14)
DECAYS=(0.02 0.02 0.01 0.02 0.01 0.02)
WARMUPS=(20 20 30 30 40 40)
COOLDOWNS=(40 60 80 80 100 120)
MARGINS=(0.20 0.25 0.35 0.50 0.50 0.75)
NOISES=(0.0005 0.0005 0.001 0.0005 0.001 0.0005)
MAX_ERRORS=(8 8 8 8 8 8)

N_SEED=${#SEEDS[@]}
N_COMBO=${#THRESHOLDS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED_INDEX=$(( TASK_ID % N_SEED ))
COMBO_INDEX=$(( TASK_ID / N_SEED ))

if [ "${COMBO_INDEX}" -ge "${N_COMBO}" ]; then
  echo "错误：SLURM_ARRAY_TASK_ID=${TASK_ID} 超出 SAM 调参组合范围。"
  exit 1
fi

SEED=${SEEDS[$SEED_INDEX]}
THRESHOLD=${THRESHOLDS[$COMBO_INDEX]}
DECAY=${DECAYS[$COMBO_INDEX]}
WARMUP=${WARMUPS[$COMBO_INDEX]}
COOLDOWN=${COOLDOWNS[$COMBO_INDEX]}
MARGIN=${MARGINS[$COMBO_INDEX]}
NOISE=${NOISES[$COMBO_INDEX]}
MAX_ERROR=${MAX_ERRORS[$COMBO_INDEX]}

echo "============================================================"
echo "连续 SAM 多 seed 检测器调参"
echo "Host:                 $(hostname)"
echo "Job ID:               ${SLURM_JOB_ID:-local}"
echo "Array ID:             ${SLURM_ARRAY_TASK_ID}"
echo "Combo index:          ${COMBO_INDEX}"
echo "Python:               ${PYTHON_BIN}"
echo "Install deps:         ${INSTALL_DEPS}"
echo "Environment:          ${INTERCEPTOR_SPEED} / ${INTRUDER_SPEED} / ${COLLISION_RADIUS}"
echo "Timesteps:            ${TIMESTEPS}"
echo "Episodes:             ${EPISODES}"
echo "Seed:                 ${SEED}"
echo "Threshold:            ${THRESHOLD}"
echo "Decay:                ${DECAY}"
echo "Warmup steps:         ${WARMUP}"
echo "Cooldown steps:       ${COOLDOWN}"
echo "Switch margin:        ${MARGIN}"
echo "Noise variance:       ${NOISE}"
echo "Max normalized error: ${MAX_ERROR}"
echo "SAM feature mode:     ${SAM_FEATURE_MODE}"
echo "Start time:           $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

if [ "${INSTALL_DEPS}" = "1" ]; then
  # 默认关闭安装步骤，防止 Slurm array 并发写同一个 editable 安装元数据。
  "${PYTHON_BIN}" -m pip install -e ".[dev]"
fi

"${PYTHON_BIN}" scripts/run_continuous_sweep.py \
  --base-config configs/continuous_paper_like.yaml \
  --name-prefix continuous_sam_multiseed_tune \
  --interceptor-speed "${INTERCEPTOR_SPEED}" \
  --intruder-speed "${INTRUDER_SPEED}" \
  --collision-radius "${COLLISION_RADIUS}" \
  --timesteps "${TIMESTEPS}" \
  --episodes "${EPISODES}" \
  --seed "${SEED}" \
  --detector-threshold "${THRESHOLD}" \
  --detector-decay "${DECAY}" \
  --sam-warmup-steps "${WARMUP}" \
  --sam-cooldown-steps "${COOLDOWN}" \
  --sam-switch-margin "${MARGIN}" \
  --sam-noise-variance "${NOISE}" \
  --sam-max-normalized-error "${MAX_ERROR}" \
  --sam-feature-mode "${SAM_FEATURE_MODE}" \
  --sam-online-updates false

echo "Finished time: $(date '+%Y-%m-%d %H:%M:%S')"
