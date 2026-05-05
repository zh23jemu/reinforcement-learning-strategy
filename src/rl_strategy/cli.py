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

    return parser


def main() -> None:
    """根据子命令启动训练或评估流程。"""

    args = build_parser().parse_args()
    config = load_config(args.config)

    if args.command == "train":
        train_discrete(config)
    elif args.command == "evaluate":
        evaluate_discrete(config)
    else:  # pragma: no cover - argparse 已保证不会进入该分支
        raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()

