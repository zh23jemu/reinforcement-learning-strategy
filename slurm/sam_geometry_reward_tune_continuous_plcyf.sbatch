#!/bin/bash
#SBATCH --job-name=cont-sam-reward-tune
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
#SBATCH --array=0-17

# 连续场景 geometry SAM + 按策略 reward profile 的检测参数细扫脚本。
# 当前 response 库配置固定为 direct=guard、attack=attack_guard_safe；该组合在
# oracle 和真实 SAM confirm 中都提升了整体回报，但真实 SAM 的响应准确率仍偏低。
# 因此本脚本只扫描更保守的阈值、冷却和切换 margin，目标是减少无效切换并提高
# direct/attack 的 response policy accuracy，而不再继续改变 reward profile。

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

# 固定当前最有希望的连续环境和 response reward profile。
INTERCEPTOR_SPEED="${INTERCEPTOR_SPEED:-0.030}"
INTRUDER_SPEED="${INTRUDER_SPEED:-0.016}"
COLLISION_RADIUS="${COLLISION_RADIUS:-0.08}"
TIMESTEPS="${TIMESTEPS:-300000}"
EPISODES="${EPISODES:-300}"
SAM_FEATURE_MODE="${SAM_FEATURE_MODE:-geometry}"
DIRECT_REWARD_PROFILE="${DIRECT_REWARD_PROFILE:-guard}"
ATTACK_REWARD_PROFILE="${ATTACK_REWARD_PROFILE:-attack_guard_safe}"
NAME_PREFIX="${NAME_PREFIX:-continuous_sam_geometry_reward_tune}"
SEEDS=(42 43 44)

# 6 组更保守的检测组合：
# - 组合 0 是 reward confirm 使用的锚点，便于和 1.5M 结果对齐。
# - 组合 1-3 逐步增加 cooldown / margin，压制短时间内反复切换。
# - 组合 4-5 提高 threshold 和 warmup，测试更保守触发能否提升准确率。
THRESHOLDS=(10 10 11 12 12 14)
DECAYS=(0.02 0.02 0.02 0.02 0.02 0.02)
WARMUPS=(20 20 20 20 30 30)
COOLDOWNS=(60 100 100 120 120 160)
MARGINS=(0.25 0.35 0.40 0.50 0.60 0.75)
NOISES=(0.0005 0.0005 0.0005 0.0005 0.0005 0.0005)
MAX_ERRORS=(8 8 8 8 8 8)

N_SEED=${#SEEDS[@]}
N_COMBO=${#THRESHOLDS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED_INDEX=$(( TASK_ID % N_SEED ))
COMBO_INDEX=$(( TASK_ID / N_SEED ))

if [ "${COMBO_INDEX}" -ge "${N_COMBO}" ]; then
  echo "错误：SLURM_ARRAY_TASK_ID=${TASK_ID} 超出 SAM reward 细扫组合范围。"
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
echo "连续 SAM geometry + reward profile 检测细扫"
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
echo "Direct reward:        ${DIRECT_REWARD_PROFILE}"
echo "Attack reward:        ${ATTACK_REWARD_PROFILE}"
echo "Name prefix:          ${NAME_PREFIX}"
echo "Start time:           $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

if [ "${INSTALL_DEPS}" = "1" ]; then
  # 默认关闭安装步骤，防止 Slurm array 并发写同一个 editable 安装元数据。
  "${PYTHON_BIN}" -m pip install -e ".[dev]"
fi

"${PYTHON_BIN}" scripts/run_continuous_sweep.py \
  --base-config configs/continuous_paper_like.yaml \
  --name-prefix "${NAME_PREFIX}" \
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
  --sam-online-updates false \
  --direct-reward-profile "${DIRECT_REWARD_PROFILE}" \
  --attack-reward-profile "${ATTACK_REWARD_PROFILE}"

echo "Finished time: $(date '+%Y-%m-%d %H:%M:%S')"
