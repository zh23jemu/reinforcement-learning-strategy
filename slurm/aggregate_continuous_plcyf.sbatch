#!/bin/bash
#SBATCH --job-name=cont-agg
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

# 汇总连续 sweep 结果，生成 runs/continuous_sweep_summary.csv。
# 该脚本只读取 analysis JSON，不重新训练或评估模型。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs runs

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

"${PYTHON_BIN}" -m pip install -e ".[dev]"
"${PYTHON_BIN}" scripts/aggregate_continuous_sweep.py \
  --runs-root runs \
  --output runs/continuous_sweep_summary.csv
