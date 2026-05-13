#!/bin/bash
#SBATCH --job-name=cont-train
#SBATCH --account=gpo-ifv7xx
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# 连续二维拦截捕猎场景的 PPO 训练脚本。
# 该脚本会训练三类入侵策略对应的响应策略，以及一个面对非平稳入侵者的 baseline。
# 默认使用本地 CPU 分区 defq，优先落到本地 com 节点；aws-com 可作为备用分区手动切换。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs artifacts/continuous runs

module purge
module load python/3.11.7

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

echo "============================================================"
echo "连续二维拦截捕猎 PPO 长训练"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Python:     ${PYTHON_BIN}"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "错误：找不到可执行 Python：${PYTHON_BIN}"
  echo "请先在服务器创建 .venv，或用 --export=ALL,PYTHON_BIN=/path/to/python 指定解释器。"
  exit 1
fi

"${PYTHON_BIN}" -m pip install -e ".[dev]"
"${PYTHON_BIN}" -m rl_strategy.cli train --config configs/continuous_paper_like.yaml

echo "Finished time: $(date '+%Y-%m-%d %H:%M:%S')"
