"""命令行入口。

当前优先实现离散动作空间 OPS-DeMo 复现流程：

1. `train`：训练 Predator B 的候选策略，以及 Predator A 的响应策略。
2. `evaluate`：加载策略库，运行策略切换检测并保存过程数据。

所有命令都通过配置文件控制，便于在 smoke 配置和论文级配置之间切换。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_strategy.config import load_config
from rl_strategy.continuous.experiment import evaluate_continuous, train_continuous
from rl_strategy.discrete.analysis import analyze_discrete_run
from rl_strategy.discrete.experiment import evaluate_discrete, train_discrete


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="策略变化检测复现项目 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("train", "evaluate"):
        sub = subparsers.add_parser(command)
        sub.add_argument(
            "--config",
            type=Path,
            required=True,
            help="YAML 配置文件路径，例如 configs/discrete_smoke.yaml",
        )

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="evaluate 生成的运行目录，例如 runs/discrete_smoke/20260506_064331",
    )
    analyze.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="分析结果输出目录；默认写入 run-dir/analysis",
    )

    return parser


def main() -> None:
    """根据子命令启动训练或评估流程。"""

    args = build_parser().parse_args()
    if args.command == "train":
        config = load_config(args.config)
        if _is_continuous_config(config):
            train_continuous(config)
        else:
            train_discrete(config)
    elif args.command == "evaluate":
        config = load_config(args.config)
        if _is_continuous_config(config):
            evaluate_continuous(config)
        else:
            evaluate_discrete(config)
    elif args.command == "analyze":
        summary = analyze_discrete_run(args.run_dir, args.output_dir)
        print(f"分析完成，生成 {len(summary['figures'])} 张图。")
    else:  # pragma: no cover - argparse 已保证不会进入该分支
        raise ValueError(f"未知命令: {args.command}")


def _is_continuous_config(config: dict) -> bool:
    """判断配置是否指向连续场景模块。

    旧的离散配置没有 `experiment.module` 字段，因此默认仍走离散复现流程，
    这样可以保持已有命令完全兼容。
    """

    return str(config.get("experiment", {}).get("module", "discrete")) == "continuous"


if __name__ == "__main__":
    main()
