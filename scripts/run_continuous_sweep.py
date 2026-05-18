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
    parser.add_argument("--detector-threshold", type=float, default=None, help="覆盖 detector.threshold")
    parser.add_argument("--detector-decay", type=float, default=None, help="覆盖 detector.decay")
    parser.add_argument("--sam-warmup-steps", type=int, default=None, help="覆盖 sam.warmup_steps")
    parser.add_argument("--sam-cooldown-steps", type=int, default=None, help="覆盖 sam.cooldown_steps")
    parser.add_argument("--sam-switch-margin", type=float, default=None, help="覆盖 sam.switch_margin")
    parser.add_argument("--sam-max-normalized-error", type=float, default=None, help="覆盖 sam.max_normalized_error")
    parser.add_argument("--sam-noise-variance", type=float, default=None, help="覆盖 sam.noise_variance")
    parser.add_argument("--sam-mc-passes", type=int, default=None, help="覆盖 sam.mc_passes")
    parser.add_argument("--sam-sample-steps", type=int, default=None, help="覆盖 sam.sample_steps")
    parser.add_argument("--sam-epochs", type=int, default=None, help="覆盖 sam.epochs")
    parser.add_argument(
        "--direct-reward-profile",
        choices=("none", "chase", "guard", "guard_strong"),
        default="none",
        help="只覆盖 direct 响应策略训练时的奖励塑形 profile",
    )
    parser.add_argument(
        "--attack-reward-profile",
        choices=(
            "none",
            "chase",
            "attacksafe",
            "attacksafe_strong",
            "attack_chase_light",
            "attack_guard",
            "attack_guard_safe",
            "attack_balanced",
        ),
        default="none",
        help="只覆盖 attack 响应策略训练时的奖励塑形 profile",
    )
    parser.add_argument(
        "--sam-feature-mode",
        choices=("raw", "geometry"),
        default=None,
        help="覆盖 sam.feature_mode；geometry 会为 opponent model 追加连续相对几何特征",
    )
    parser.add_argument(
        "--sam-online-updates",
        choices=("true", "false"),
        default=None,
        help="覆盖 sam.online_updates，调参时通常设为 false 以避免错误样本污染模型",
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
    ) + _build_sam_suffix(args)

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
    _apply_policy_reward_overrides(config, args)
    _apply_sam_overrides(config, args)

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


def _apply_sam_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """应用 SAM 检测调参覆盖。

    Slurm array 会用同一个基础 YAML 扫不同检测器组合；这里集中处理覆盖项，避免
    复制多份配置文件。未显式传入的参数保持基础配置原值。
    """

    detector = config.setdefault("detector", {})
    sam = config.setdefault("sam", {})
    detector_overrides = {
        "threshold": args.detector_threshold,
        "decay": args.detector_decay,
    }
    sam_overrides = {
        "warmup_steps": args.sam_warmup_steps,
        "cooldown_steps": args.sam_cooldown_steps,
        "switch_margin": args.sam_switch_margin,
        "max_normalized_error": args.sam_max_normalized_error,
        "noise_variance": args.sam_noise_variance,
        "mc_passes": args.sam_mc_passes,
        "sample_steps": args.sam_sample_steps,
        "epochs": args.sam_epochs,
        "feature_mode": args.sam_feature_mode,
    }
    for key, value in detector_overrides.items():
        if value is not None:
            detector[key] = value
    for key, value in sam_overrides.items():
        if value is not None:
            sam[key] = value
    if args.sam_online_updates is not None:
        sam["online_updates"] = args.sam_online_updates == "true"


def _apply_policy_reward_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """按入侵策略覆盖 response policy 的训练奖励。

    该入口复用 oracle 对照阶段筛出的奖励 profile，但不改变检测器逻辑；真实 SAM
    confirm 仍然通过 MC dropout normalized error 做策略识别，只是在训练 response
    policy 库时给 direct/attack 使用更合适的奖励塑形。
    """

    profiles = _build_policy_reward_overrides(args)
    if profiles:
        config["environment"]["reward_overrides_by_policy"] = profiles


def _build_policy_reward_overrides(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """根据命令行 profile 构造按策略奖励覆盖表。"""

    profiles: dict[str, dict[str, float]] = {}
    if args.direct_reward_profile == "chase":
        profiles["direct"] = {
            "win_reward": 120.0,
            "loss_reward": -120.0,
            "agent_distance_weight": -0.35,
            "intruder_distance_weight": 0.05,
        }
    elif args.direct_reward_profile == "guard":
        profiles["direct"] = {
            "win_reward": 120.0,
            "loss_reward": -120.0,
            "agent_distance_weight": -0.25,
            "intruder_distance_weight": 0.12,
        }
    elif args.direct_reward_profile == "guard_strong":
        profiles["direct"] = {
            "win_reward": 140.0,
            "loss_reward": -140.0,
            "agent_distance_weight": -0.20,
            "intruder_distance_weight": 0.18,
        }

    if args.attack_reward_profile == "chase":
        profiles["attack"] = {
            "win_reward": 120.0,
            "loss_reward": -120.0,
            "agent_distance_weight": -0.35,
            "intruder_distance_weight": 0.05,
        }
    elif args.attack_reward_profile == "attacksafe":
        profiles["attack"] = {
            "win_reward": 120.0,
            "loss_reward": -150.0,
            "agent_distance_weight": -0.30,
            "intruder_distance_weight": 0.08,
            "active_collision_loss_reward": -220.0,
        }
    elif args.attack_reward_profile == "attacksafe_strong":
        profiles["attack"] = {
            "win_reward": 140.0,
            "loss_reward": -180.0,
            "agent_distance_weight": -0.25,
            "intruder_distance_weight": 0.12,
            "active_collision_loss_reward": -300.0,
        }
    elif args.attack_reward_profile == "attack_chase_light":
        profiles["attack"] = {
            "win_reward": 120.0,
            "loss_reward": -120.0,
            "agent_distance_weight": -0.28,
            "intruder_distance_weight": 0.04,
            "active_collision_loss_reward": -140.0,
        }
    elif args.attack_reward_profile == "attack_guard":
        profiles["attack"] = {
            "win_reward": 120.0,
            "loss_reward": -120.0,
            "agent_distance_weight": -0.20,
            "intruder_distance_weight": 0.14,
            "active_collision_loss_reward": -160.0,
        }
    elif args.attack_reward_profile == "attack_guard_safe":
        profiles["attack"] = {
            "win_reward": 120.0,
            "loss_reward": -140.0,
            "agent_distance_weight": -0.22,
            "intruder_distance_weight": 0.12,
            "active_collision_loss_reward": -220.0,
        }
    elif args.attack_reward_profile == "attack_balanced":
        profiles["attack"] = {
            "win_reward": 120.0,
            "loss_reward": -130.0,
            "agent_distance_weight": -0.26,
            "intruder_distance_weight": 0.08,
            "active_collision_loss_reward": -180.0,
        }
    return profiles


def _slug_float(value: float) -> str:
    """把小数转成目录友好的短字符串，例如 0.026 -> 0p026。"""

    return re.sub(r"[^0-9a-zA-Z]+", "p", f"{value:g}")


def _build_sam_suffix(args: argparse.Namespace) -> str:
    """为 SAM 调参任务追加目录后缀，避免不同检测器组合覆盖同一目录。"""

    parts: list[str] = []
    if args.detector_threshold is not None:
        parts.append(f"th{_slug_float(args.detector_threshold)}")
    if args.detector_decay is not None:
        parts.append(f"dc{_slug_float(args.detector_decay)}")
    if args.sam_warmup_steps is not None:
        parts.append(f"wu{args.sam_warmup_steps}")
    if args.sam_cooldown_steps is not None:
        parts.append(f"cd{args.sam_cooldown_steps}")
    if args.sam_switch_margin is not None:
        parts.append(f"mg{_slug_float(args.sam_switch_margin)}")
    if args.sam_noise_variance is not None:
        parts.append(f"nv{_slug_float(args.sam_noise_variance)}")
    if args.sam_max_normalized_error is not None:
        parts.append(f"mx{_slug_float(args.sam_max_normalized_error)}")
    if args.sam_feature_mode is not None:
        parts.append(f"fm{args.sam_feature_mode}")
    if args.sam_online_updates is not None:
        parts.append(f"ou{args.sam_online_updates}")
    if args.direct_reward_profile != "none":
        parts.append(f"dr{args.direct_reward_profile}")
    if args.attack_reward_profile != "none":
        parts.append(f"ar{args.attack_reward_profile}")
    return "" if not parts else "_sam" + "_".join(parts)


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
        "detector_threshold": args.detector_threshold,
        "detector_decay": args.detector_decay,
        "sam_warmup_steps": args.sam_warmup_steps,
        "sam_cooldown_steps": args.sam_cooldown_steps,
        "sam_switch_margin": args.sam_switch_margin,
        "sam_max_normalized_error": args.sam_max_normalized_error,
        "sam_noise_variance": args.sam_noise_variance,
        "sam_mc_passes": args.sam_mc_passes,
        "sam_sample_steps": args.sam_sample_steps,
        "sam_epochs": args.sam_epochs,
        "sam_feature_mode": args.sam_feature_mode,
        "sam_online_updates": args.sam_online_updates,
        "direct_reward_profile": args.direct_reward_profile,
        "attack_reward_profile": args.attack_reward_profile,
    }


if __name__ == "__main__":
    main()
