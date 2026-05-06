"""离散 OPS-DeMo 参数批量评估脚本。

该脚本用于 Slurm array：每个任务只覆盖一组 `alpha/threshold/seed`，
复用已经训练好的 PPO opponent/response/baseline 模型，然后自动执行评估和分析。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_strategy.config import load_config
from rl_strategy.discrete.analysis import analyze_discrete_run
from rl_strategy.discrete.experiment import evaluate_discrete


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="离散 OPS-DeMo 参数批量评估")
    parser.add_argument("--base-config", type=Path, required=True, help="基础 YAML 配置文件")
    parser.add_argument("--alpha", type=float, required=True, help="OPS-DeMo strictness factor")
    parser.add_argument("--threshold", type=float, required=True, help="running error 切换阈值")
    parser.add_argument("--seed", type=int, required=True, help="评估随机种子")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="覆盖 evaluation.episodes；为空时使用配置文件值",
    )
    parser.add_argument(
        "--name-prefix",
        default="discrete_sweep",
        help="写入 runs/ 下的实验名前缀",
    )
    return parser


def main() -> None:
    """加载配置、覆盖参数、运行评估并生成分析图。"""

    args = build_parser().parse_args()
    config = load_config(args.base_config)

    # 为每组 sweep 参数设置独立实验名，避免结果目录混在一起。
    config["experiment"]["name"] = (
        f"{args.name_prefix}_a{args.alpha:g}_t{args.threshold:g}_s{args.seed}"
    )
    config["experiment"]["seed"] = args.seed
    config["detector"]["alpha"] = args.alpha
    config["detector"]["threshold"] = args.threshold
    if args.episodes is not None:
        config["evaluation"]["episodes"] = args.episodes

    run_dir = evaluate_discrete(config)
    analyze_discrete_run(run_dir)


if __name__ == "__main__":
    main()

