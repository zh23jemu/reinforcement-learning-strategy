"""连续二维拦截捕猎参数 sweep 单任务脚本。

该脚本面向 Slurm array 的单个任务：从基础 YAML 配置读取连续实验参数，
覆盖一组环境/训练参数，训练独立的响应策略库与 baseline，随后执行评估
和分析。每个参数组都会写入独立的 `experiment.name` 与 `artifact_dir`，
避免 array 任务之间互相覆盖模型文件。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from rl_strategy.config import load_config
from rl_strategy.continuous.experiment import evaluate_continuous, train_continuous

from analyze_continuous_run import analyze_continuous_run


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="连续 OPS-DeMo 参数 sweep 单任务")
    parser.add_argument("--base-config", type=Path, required=True, help="基础连续 YAML 配置")
    parser.add_argument("--interceptor-speed", type=float, required=True, help="拦截者最大速度")
    parser.add_argument("--intruder-speed", type=float, required=True, help="入侵者最大速度")
    parser.add_argument("--collision-radius", type=float, required=True, help="主动碰撞半径")
    parser.add_argument("--timesteps", type=int, required=True, help="每个 PPO 模型训练步数")
    parser.add_argument("--seed", type=int, required=True, help="随机种子")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="覆盖 evaluation.episodes；为空时使用基础配置值",
    )
    parser.add_argument(
        "--name-prefix",
        default="continuous_sweep",
        help="写入 runs/ 和 artifacts/ 下的实验名前缀",
    )
    return parser


def main() -> None:
    """加载基础配置，覆盖当前 sweep 参数，并执行训练、评估、分析。"""

    args = build_parser().parse_args()
    config = load_config(args.base_config)
    experiment_name = _build_experiment_name(
        prefix=args.name_prefix,
        interceptor_speed=args.interceptor_speed,
        intruder_speed=args.intruder_speed,
        collision_radius=args.collision_radius,
        timesteps=args.timesteps,
        seed=args.seed,
    )

    # 每个 sweep 任务都写入独立目录，避免 Slurm array 并发时覆盖模型和结果。
    config["experiment"]["name"] = experiment_name
    config["experiment"]["seed"] = args.seed
    config["experiment"]["artifact_dir"] = str(Path("artifacts") / "continuous_sweep" / experiment_name)

    config["environment"]["interceptor_max_speed"] = args.interceptor_speed
    config["environment"]["intruder_max_speed"] = args.intruder_speed
    config["environment"]["collision_radius"] = args.collision_radius
    config["ppo"]["total_timesteps"] = args.timesteps
    if args.episodes is not None:
        config["evaluation"]["episodes"] = args.episodes

    print("============================================================")
    print("连续 OPS-DeMo sweep 单任务")
    print(json.dumps(_sweep_metadata(args, experiment_name), ensure_ascii=False, indent=2))
    print("============================================================")

    train_continuous(config)
    run_dir = evaluate_continuous(config)
    analysis_summary = analyze_continuous_run(run_dir)
    print("连续 sweep 分析摘要:")
    print(json.dumps(analysis_summary, ensure_ascii=False, indent=2))


def _build_experiment_name(
    *,
    prefix: str,
    interceptor_speed: float,
    intruder_speed: float,
    collision_radius: float,
    timesteps: int,
    seed: int,
) -> str:
    """构造稳定、可排序且适合作为目录名的实验名。"""

    return (
        f"{prefix}"
        f"_is{_slug_float(interceptor_speed)}"
        f"_us{_slug_float(intruder_speed)}"
        f"_cr{_slug_float(collision_radius)}"
        f"_ts{timesteps}"
        f"_s{seed}"
    )


def _slug_float(value: float) -> str:
    """把小数转成目录友好的短字符串，例如 0.026 -> 0p026。"""

    return re.sub(r"[^0-9a-zA-Z]+", "p", f"{value:g}")


def _sweep_metadata(args: argparse.Namespace, experiment_name: str) -> dict[str, Any]:
    """整理当前任务元信息，便于日志中快速定位参数组。"""

    return {
        "experiment_name": experiment_name,
        "interceptor_max_speed": args.interceptor_speed,
        "intruder_max_speed": args.intruder_speed,
        "collision_radius": args.collision_radius,
        "total_timesteps": args.timesteps,
        "seed": args.seed,
        "episodes": args.episodes,
    }


if __name__ == "__main__":
    main()
