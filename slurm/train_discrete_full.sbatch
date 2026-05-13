#!/bin/bash
#SBATCH --job-name=ops-demo-discrete
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

# 该脚本用于运行离散 OPS-DeMo 的论文级 PPO 训练。
# 已按新集群默认账号 gpo-ifv7xx、本地 CPU 分区 defq、QOS normal 调整。
# 本项目训练主要消耗 CPU，因此不照搬参考脚本中的 GPU 分区和 --gres=gpu:1。
# defq 优先使用本地 com 节点，aws-com 仅建议在本地 CPU 队列不可用时手动切换。

set -euo pipefail

# 切到 sbatch 提交时所在的项目根目录，确保 configs、runs、artifacts 等相对路径一致。
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs artifacts/discrete runs

# 新集群参考脚本使用 python/3.11.7；实际训练仍默认走项目本地 .venv。
module purge
module load python/3.11.7

# 限制数值库线程，避免 Slurm 分配 CPU 数与 BLAS/OpenMP 自动开线程叠加导致过度抢占。
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 默认使用项目本地虚拟环境；如服务器路径不同，可通过 sbatch --export 覆盖。
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

echo "============================================================"
echo "离散 OPS-DeMo PPO full 训练"
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
"${PYTHON_BIN}" -m rl_strategy.cli train --config configs/discrete_paper_like.yaml

echo "Finished time: $(date '+%Y-%m-%d %H:%M:%S')"
