#!/bin/bash
#SBATCH --job-name=cont-sam-tune
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

# 连续场景 SAM 检测器小规模调参脚本。
# 该脚本固定使用当前表现相对最好的环境参数 `0.030 / 0.016 / 0.08`，
# 只扫描 SAM 检测相关参数，目的是先降低误切换、提升 response_policy_accuracy。
# 默认使用 300k timesteps / 300 episodes；找到候选后再用 confirm 脚本做 1.5M 长训。

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

# 固定环境参数，减少本轮 sweep 的搜索维度。
INTERCEPTOR_SPEED="${INTERCEPTOR_SPEED:-0.030}"
INTRUDER_SPEED="${INTRUDER_SPEED:-0.016}"
COLLISION_RADIUS="${COLLISION_RADIUS:-0.08}"
SEED="${SEED:-42}"
TIMESTEPS="${TIMESTEPS:-300000}"
EPISODES="${EPISODES:-300}"
SAM_FEATURE_MODE="${SAM_FEATURE_MODE:-geometry}"

# 12 组检测器组合：
# - 提高 threshold、加入 cooldown，抑制一轮长训中出现数百次误切换。
# - 调整 margin 与 noise_variance，筛选更稳定的 normalized running error 尺度。
# - 默认关闭 online_updates，避免当前假设模型被错误策略样本持续污染。
THRESHOLDS=(10 12 12 14 14 16 16 18 18 20 20 24)
DECAYS=(0.02 0.02 0.01 0.02 0.01 0.02 0.01 0.02 0.01 0.02 0.01 0.01)
WARMUPS=(40 40 40 40 40 40 40 40 40 40 40 40)
COOLDOWNS=(80 80 120 120 160 120 160 160 120 160 200 200)
MARGINS=(0.25 0.50 0.50 0.75 0.75 1.00 1.00 1.25 1.25 1.50 1.50 2.00)
NOISES=(0.0005 0.0005 0.001 0.0005 0.001 0.0005 0.001 0.0005 0.001 0.0005 0.001 0.001)
MAX_ERRORS=(8 8 8 8 8 8 8 8 8 8 8 8)

TASK_ID=${SLURM_ARRAY_TASK_ID}
THRESHOLD=${THRESHOLDS[$TASK_ID]}
DECAY=${DECAYS[$TASK_ID]}
WARMUP=${WARMUPS[$TASK_ID]}
COOLDOWN=${COOLDOWNS[$TASK_ID]}
MARGIN=${MARGINS[$TASK_ID]}
NOISE=${NOISES[$TASK_ID]}
MAX_ERROR=${MAX_ERRORS[$TASK_ID]}

echo "============================================================"
echo "连续 SAM 检测器调参"
echo "Host:                 $(hostname)"
echo "Job ID:               ${SLURM_JOB_ID:-local}"
echo "Array ID:             ${SLURM_ARRAY_TASK_ID}"
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
  --name-prefix continuous_sam_tune \
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
