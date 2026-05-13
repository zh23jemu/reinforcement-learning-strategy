#!/bin/bash
#SBATCH --job-name=ops-demo-agg
#SBATCH --account=gpo-ifv7xx
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# 汇总 sweep 结果，生成 runs/discrete_sweep_summary.csv。
# 默认使用本地 CPU 分区 defq，避免优先占用 aws-com 云节点。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs

module purge
module load python/3.11.7

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "错误：找不到可执行 Python：${PYTHON_BIN}"
  echo "请先在服务器创建 .venv，或用 --export=ALL,PYTHON_BIN=/path/to/python 指定解释器。"
  exit 1
fi

# 汇总阶段也固定走项目本地虚拟环境，避免服务器环境差异影响结果读取。
"${PYTHON_BIN}" -m pip install -e ".[dev]"
"${PYTHON_BIN}" scripts/aggregate_discrete_sweep.py \
  --runs-root runs \
  --output runs/discrete_sweep_summary.csv
