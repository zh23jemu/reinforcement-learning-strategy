#!/bin/bash
#SBATCH --job-name=cont-attack-reward
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
#SBATCH --array=0-11

# Attack response policy 窄范围奖励扫描脚本。
# 背景：按策略 reward profile 已经把 direct 的 seed 43 reward gap 从负值拉到正值；
# attack 仍是主要残留短板。这里固定 direct=guard，只细扫 attack 奖励 profile，
# 保持 detour 原始奖励，避免继续扩大无关组合。
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
TIMESTEPS="${TIMESTEPS:-1500000}"
EPISODES="${EPISODES:-800}"
NAME_PREFIX="${NAME_PREFIX:-continuous_response_attack_reward}"
INTERCEPTOR_SPEED="${INTERCEPTOR_SPEED:-0.030}"
INTRUDER_SPEED="${INTRUDER_SPEED:-0.016}"
COLLISION_RADIUS="${COLLISION_RADIUS:-0.08}"
DIRECT_REWARD_PROFILE="${DIRECT_REWARD_PROFILE:-guard}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "错误：找不到可执行 Python：${PYTHON_BIN}"
  echo "请先在服务器创建 .venv 并安装依赖，或用 --export=ALL,PYTHON_BIN=/path/to/python 指定解释器。"
  exit 1
fi

SEEDS=(42 43 44)

# 组合格式：tag|attack_reward_profile
# light/guard/balanced 主要围绕 dguard_achase 和 dstrong_asafe 的有效信号收窄：
# 避免 attacksafe_strong 那种过强主动碰撞惩罚造成大负 gap。
ATTACK_PROFILES=(
  "achase_light|attack_chase_light"
  "aguard|attack_guard"
  "aguard_safe|attack_guard_safe"
  "abalanced|attack_balanced"
)

N_SEED=${#SEEDS[@]}
N_COMBO=${#ATTACK_PROFILES[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED_INDEX=$(( TASK_ID % N_SEED ))
COMBO_INDEX=$(( TASK_ID / N_SEED ))

if [ "${COMBO_INDEX}" -ge "${N_COMBO}" ]; then
  echo "错误：SLURM_ARRAY_TASK_ID=${TASK_ID} 超出 attack profile 组合范围。"
  exit 1
fi

SEED=${SEEDS[$SEED_INDEX]}
IFS='|' read -r ATTACK_TAG ATTACK_REWARD_PROFILE <<< "${ATTACK_PROFILES[$COMBO_INDEX]}"
RUN_PREFIX="${NAME_PREFIX}_d${DIRECT_REWARD_PROFILE}_${ATTACK_TAG}"

echo "============================================================"
echo "连续 attack response policy 窄范围奖励扫描"
echo "Host:                  $(hostname)"
echo "Job ID:                ${SLURM_JOB_ID:-local}"
echo "Array ID:              ${SLURM_ARRAY_TASK_ID}"
echo "Combo index:           ${COMBO_INDEX}"
echo "Python:                ${PYTHON_BIN}"
echo "Install deps:          ${INSTALL_DEPS}"
echo "Environment:           ${INTERCEPTOR_SPEED} / ${INTRUDER_SPEED} / ${COLLISION_RADIUS}"
echo "Timesteps:             ${TIMESTEPS}"
echo "Episodes:              ${EPISODES}"
echo "Seed:                  ${SEED}"
echo "Direct reward profile: ${DIRECT_REWARD_PROFILE}"
echo "Attack tag:            ${ATTACK_TAG}"
echo "Attack reward profile: ${ATTACK_REWARD_PROFILE}"
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
  --timesteps "${TIMESTEPS}" \
  --baseline-timesteps "${TIMESTEPS}" \
  --episodes "${EPISODES}" \
  --seed "${SEED}" \
  --direct-reward-profile "${DIRECT_REWARD_PROFILE}" \
  --attack-reward-profile "${ATTACK_REWARD_PROFILE}"

echo "Finished time: $(date '+%Y-%m-%d %H:%M:%S')"
