#!/bin/bash
#SBATCH --job-name=cont-confirm
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
#SBATCH --array=0-8

# 连续二维拦截捕猎 1M+ 候选参数确认脚本。
# 该脚本只覆盖 300k sweep 中排名靠前的 3 组参数，并用 3 个 seed 做稳定性确认。
# 每个 array task 会训练独立模型目录、执行评估，并生成 analysis 诊断结果。
# 注意：array 任务默认不执行 pip install，避免多个任务同时改写同一个 .venv。
# 提交前请先在登录节点或单独作业中完成依赖安装；确需在任务内安装时再显式设置 INSTALL_DEPS=1。

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
TIMESTEPS="${TIMESTEPS:-1000000}"
EPISODES="${EPISODES:-500}"
NAME_PREFIX="${NAME_PREFIX:-continuous_confirm}"
SAM_THRESHOLD="${SAM_THRESHOLD:-}"
SAM_DECAY="${SAM_DECAY:-}"
SAM_WARMUP_STEPS="${SAM_WARMUP_STEPS:-}"
SAM_COOLDOWN_STEPS="${SAM_COOLDOWN_STEPS:-}"
SAM_SWITCH_MARGIN="${SAM_SWITCH_MARGIN:-}"
SAM_NOISE_VARIANCE="${SAM_NOISE_VARIANCE:-}"
SAM_MAX_NORMALIZED_ERROR="${SAM_MAX_NORMALIZED_ERROR:-}"
SAM_FEATURE_MODE="${SAM_FEATURE_MODE:-}"
SAM_ONLINE_UPDATES="${SAM_ONLINE_UPDATES:-}"
DIRECT_REWARD_PROFILE="${DIRECT_REWARD_PROFILE:-none}"
ATTACK_REWARD_PROFILE="${ATTACK_REWARD_PROFILE:-none}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "错误：找不到可执行 Python：${PYTHON_BIN}"
  echo "请先在服务器创建 .venv 并安装依赖，或用 --export=ALL,PYTHON_BIN=/path/to/python 指定解释器。"
  exit 1
fi

# 候选参数来自 2026-05-13 的 300k sweep：
# 1. 0.030 / 0.016 / 0.08：综合胜率提升和回报提升最强。
# 2. 0.030 / 0.016 / 0.10：绝对拦截胜率最高。
# 3. 0.026 / 0.018 / 0.10：覆盖更快入侵者设置，仍有较强提升。
INTERCEPTOR_SPEEDS=(0.030 0.030 0.026)
INTRUDER_SPEEDS=(0.016 0.016 0.018)
COLLISION_RADII=(0.08 0.10 0.10)
SEEDS=(42 43 44)

N_CANDIDATE=${#INTERCEPTOR_SPEEDS[@]}
N_SEED=${#SEEDS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID}

SEED_INDEX=$(( TASK_ID % N_SEED ))
CANDIDATE_INDEX=$(( TASK_ID / N_SEED ))

if [ "${CANDIDATE_INDEX}" -ge "${N_CANDIDATE}" ]; then
  echo "错误：SLURM_ARRAY_TASK_ID=${TASK_ID} 超出候选参数范围。"
  exit 1
fi

INTERCEPTOR_SPEED=${INTERCEPTOR_SPEEDS[$CANDIDATE_INDEX]}
INTRUDER_SPEED=${INTRUDER_SPEEDS[$CANDIDATE_INDEX]}
COLLISION_RADIUS=${COLLISION_RADII[$CANDIDATE_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "============================================================"
echo "连续 OPS-DeMo 1M+ 确认长训"
echo "Host:              $(hostname)"
echo "Job ID:            ${SLURM_JOB_ID:-local}"
echo "Array ID:          ${SLURM_ARRAY_TASK_ID}"
echo "Candidate index:   ${CANDIDATE_INDEX}"
echo "Python:            ${PYTHON_BIN}"
echo "Install deps:      ${INSTALL_DEPS}"
echo "Interceptor speed: ${INTERCEPTOR_SPEED}"
echo "Intruder speed:    ${INTRUDER_SPEED}"
echo "Collision radius:  ${COLLISION_RADIUS}"
echo "Timesteps:         ${TIMESTEPS}"
echo "Episodes:          ${EPISODES}"
echo "Seed:              ${SEED}"
echo "Name prefix:       ${NAME_PREFIX}"
echo "SAM threshold:     ${SAM_THRESHOLD:-<config>}"
echo "SAM decay:         ${SAM_DECAY:-<config>}"
echo "SAM warmup:        ${SAM_WARMUP_STEPS:-<config>}"
echo "SAM cooldown:      ${SAM_COOLDOWN_STEPS:-<config>}"
echo "SAM margin:        ${SAM_SWITCH_MARGIN:-<config>}"
echo "SAM noise:         ${SAM_NOISE_VARIANCE:-<config>}"
echo "SAM feature mode:  ${SAM_FEATURE_MODE:-<config>}"
echo "Direct reward:     ${DIRECT_REWARD_PROFILE}"
echo "Attack reward:     ${ATTACK_REWARD_PROFILE}"
echo "Start time:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

if [ "${INSTALL_DEPS}" = "1" ]; then
  # 默认关闭安装步骤，防止 Slurm array 并发写同一个 editable 安装元数据。
  # 如果确实需要任务内安装，建议只在单任务或互斥环境中设置 INSTALL_DEPS=1。
  "${PYTHON_BIN}" -m pip install -e ".[dev]"
fi

SAM_ARGS=()
if [ -n "${SAM_THRESHOLD}" ]; then
  SAM_ARGS+=(--detector-threshold "${SAM_THRESHOLD}")
fi
if [ -n "${SAM_DECAY}" ]; then
  SAM_ARGS+=(--detector-decay "${SAM_DECAY}")
fi
if [ -n "${SAM_WARMUP_STEPS}" ]; then
  SAM_ARGS+=(--sam-warmup-steps "${SAM_WARMUP_STEPS}")
fi
if [ -n "${SAM_COOLDOWN_STEPS}" ]; then
  SAM_ARGS+=(--sam-cooldown-steps "${SAM_COOLDOWN_STEPS}")
fi
if [ -n "${SAM_SWITCH_MARGIN}" ]; then
  SAM_ARGS+=(--sam-switch-margin "${SAM_SWITCH_MARGIN}")
fi
if [ -n "${SAM_NOISE_VARIANCE}" ]; then
  SAM_ARGS+=(--sam-noise-variance "${SAM_NOISE_VARIANCE}")
fi
if [ -n "${SAM_MAX_NORMALIZED_ERROR}" ]; then
  SAM_ARGS+=(--sam-max-normalized-error "${SAM_MAX_NORMALIZED_ERROR}")
fi
if [ -n "${SAM_FEATURE_MODE}" ]; then
  SAM_ARGS+=(--sam-feature-mode "${SAM_FEATURE_MODE}")
fi
if [ -n "${SAM_ONLINE_UPDATES}" ]; then
  SAM_ARGS+=(--sam-online-updates "${SAM_ONLINE_UPDATES}")
fi
if [ "${DIRECT_REWARD_PROFILE}" != "none" ]; then
  SAM_ARGS+=(--direct-reward-profile "${DIRECT_REWARD_PROFILE}")
fi
if [ "${ATTACK_REWARD_PROFILE}" != "none" ]; then
  SAM_ARGS+=(--attack-reward-profile "${ATTACK_REWARD_PROFILE}")
fi

"${PYTHON_BIN}" scripts/run_continuous_sweep.py \
  --base-config configs/continuous_paper_like.yaml \
  --interceptor-speed "${INTERCEPTOR_SPEED}" \
  --intruder-speed "${INTRUDER_SPEED}" \
  --collision-radius "${COLLISION_RADIUS}" \
  --timesteps "${TIMESTEPS}" \
  --episodes "${EPISODES}" \
  --seed "${SEED}" \
  --name-prefix "${NAME_PREFIX}" \
  "${SAM_ARGS[@]}"

echo "Finished time: $(date '+%Y-%m-%d %H:%M:%S')"
