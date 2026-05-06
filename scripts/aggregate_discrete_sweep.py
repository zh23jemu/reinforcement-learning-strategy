"""汇总离散 OPS-DeMo sweep 结果。

读取 `runs/discrete_sweep_*/*/analysis/analysis_summary.json`，生成一个方便比较
不同 `alpha/threshold/seed` 的 CSV。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SWEEP_PATTERN = re.compile(r"_a(?P<alpha>[^_]+)_t(?P<threshold>[^_]+)_s(?P<seed>[^_]+)$")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="汇总离散 OPS-DeMo sweep 结果")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"), help="runs 根目录")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/discrete_sweep_summary.csv"),
        help="输出 CSV 路径",
    )
    return parser


def main() -> None:
    """读取所有 analysis summary 并写出总表。"""

    args = build_parser().parse_args()
    rows: list[dict[str, Any]] = []
    for summary_path in args.runs_root.glob("discrete_sweep_*/*/analysis/analysis_summary.json"):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        experiment_name = summary_path.parents[2].name
        match = SWEEP_PATTERN.search(experiment_name)
        if match:
            data["alpha"] = float(match.group("alpha"))
            data["threshold"] = float(match.group("threshold"))
            data["seed"] = int(match.group("seed"))
        data["experiment_name"] = experiment_name
        rows.append(data)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["alpha", "threshold", "seed"]).to_csv(output, index=False)
    print(f"已汇总 {len(rows)} 条 sweep 结果到: {output}")


if __name__ == "__main__":
    main()

