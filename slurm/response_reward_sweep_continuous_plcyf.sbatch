#!/bin/bash
#SBATCH --job-name=cont-resp-reward
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

# 连续 response policy 奖励塑形扫描脚本。
# 背景：direct/attack 只加训到 3M/5M 后，整体 oracle 指标仍通过，
# 但 seed 43 的 direct/attack 分策略仍弱于 baseline，说明“训练步数不足”
# 不是唯一瓶颈。本脚本保持 oracle 对照口径，扫描几组更强的终局/过程奖励信号，
# 用来判断响应策略短板是否来自奖励塑形不够明确。
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
NAME_PREFIX="${NAME_PREFIX:-continuous_response_reward}"
INTERCEPTOR_SPEED="${INTERCEPTOR_SPEED:-0.030}"
INTRUDER_SPEED="${INTRUDER_SPEED:-0.016}"
COLLISION_RADIUS="${COLLISION_RADIUS:-0.08}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "错误：找不到可执行 Python：${PYTHON_BIN}"
  echo "请先在服务器创建 .venv 并安装依赖，或用 --export=ALL,PYTHON_BIN=/path/to/python 指定解释器。"
  exit 1
fi

SEEDS=(42 43 44)

# 奖励组合格式：
#   tag|win_reward|loss_reward|step_penalty|agent_distance_weight|intruder_distance_weight|active_collision_loss_reward
# baseline 复刻原始奖励，用于和新实现等价性做同批对照。
# chase 强化靠近入侵者的过程塑形；guard 强化把入侵者挡在目标外侧；
# attacksafe 对主动碰撞失败给更大惩罚，鼓励 attack response policy 学会避让/诱导。
REWARD_COMBOS=(
  "base|100|-100|-0.01|-0.20|0.05|-100"
  "chase|120|-120|-0.01|-0.35|0.05|-120"
  "guard|120|-120|-0.01|-0.25|0.12|-120"
  "attacksafe|120|-150|-0.01|-0.30|0.08|-220"
)

N_SEED=${#SEEDS[@]}
N_COMBO=${#REWARD_COMBOS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED_INDEX=$(( TASK_ID % N_SEED ))
COMBO_INDEX=$(( TASK_ID / N_SEED ))

if [ "${COMBO_INDEX}" -ge "${N_COMBO}" ]; then
  echo "错误：SLURM_ARRAY_TASK_ID=${TASK_ID} 超出奖励组合范围。"
  exit 1
fi

SEED=${SEEDS[$SEED_INDEX]}
IFS='|' read -r REWARD_TAG WIN_REWARD LOSS_REWARD STEP_PENALTY AGENT_DISTANCE_WEIGHT INTRUDER_DISTANCE_WEIGHT ACTIVE_COLLISION_LOSS_REWARD <<< "${REWARD_COMBOS[$COMBO_INDEX]}"
RUN_PREFIX="${NAME_PREFIX}_${REWARD_TAG}"

echo "============================================================"
echo "连续 response policy 奖励塑形扫描"
echo "Host:                         $(hostname)"
echo "Job ID:                       ${SLURM_JOB_ID:-local}"
echo "Array ID:                     ${SLURM_ARRAY_TASK_ID}"
echo "Combo index:                  ${COMBO_INDEX}"
echo "Python:                       ${PYTHON_BIN}"
echo "Install deps:                 ${INSTALL_DEPS}"
echo "Environment:                  ${INTERCEPTOR_SPEED} / ${INTRUDER_SPEED} / ${COLLISION_RADIUS}"
echo "Timesteps:                    ${TIMESTEPS}"
echo "Episodes:                     ${EPISODES}"
echo "Seed:                         ${SEED}"
echo "Reward tag:                   ${REWARD_TAG}"
echo "Win/loss reward:              ${WIN_REWARD} / ${LOSS_REWARD}"
echo "Step penalty:                 ${STEP_PENALTY}"
echo "Agent distance weight:        ${AGENT_DISTANCE_WEIGHT}"
echo "Intruder distance weight:     ${INTRUDER_DISTANCE_WEIGHT}"
echo "Active collision loss reward: ${ACTIVE_COLLISION_LOSS_REWARD}"
echo "Name prefix:                  ${RUN_PREFIX}"
echo "Start time:                   $(date '+%Y-%m-%d %H:%M:%S')"
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
  --win-reward "${WIN_REWARD}" \
  --loss-reward "${LOSS_REWARD}" \
  --step-penalty "${STEP_PENALTY}" \
  --agent-distance-weight "${AGENT_DISTANCE_WEIGHT}" \
  --intruder-distance-weight "${INTRUDER_DISTANCE_WEIGHT}" \
  --active-collision-loss-reward "${ACTIVE_COLLISION_LOSS_REWARD}"

echo "Finished time: $(date '+%Y-%m-%d %H:%M:%S')"
