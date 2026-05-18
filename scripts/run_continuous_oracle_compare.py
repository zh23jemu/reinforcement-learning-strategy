"""连续场景 response policy oracle 对照单任务脚本。

该脚本用于回答 seed 43 诊断后的关键问题：如果检测器完全知道真实入侵策略，
直接选择对应的 `response_<policy>.zip`，`direct/attack` 响应策略是否仍然弱于
普通 PPO baseline。它复用现有 `train_continuous` / `evaluate_continuous` /
`analyze_continuous_run` / `diagnose_continuous_run` 流程，只把 detector.method
显式改为 `oracle`，因此不会影响 SAM 主链路。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from rl_strategy.config import load_config
from rl_strategy.continuous.experiment import evaluate_continuous, train_continuous

# 该脚本既会被 `python scripts/run_continuous_oracle_compare.py` 直接执行，
# 也会在测试中作为模块导入。显式加入脚本目录，保证同目录分析脚本在两种入口下
# 都能稳定解析，而不依赖调用者当前的 PYTHONPATH 细节。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_continuous_run import analyze_continuous_run
from diagnose_continuous_run import diagnose_continuous_run


def build_parser() -> argparse.ArgumentParser:
    """构建 oracle 对照任务的命令行参数。"""

    parser = argparse.ArgumentParser(description="连续 response policy oracle 对照单任务")
    parser.add_argument("--base-config", type=Path, required=True, help="基础连续 YAML 配置")
    parser.add_argument("--interceptor-speed", type=float, default=0.030, help="拦截者最大速度")
    parser.add_argument("--intruder-speed", type=float, default=0.016, help="入侵者最大速度")
    parser.add_argument("--collision-radius", type=float, default=0.08, help="主动碰撞半径")
    parser.add_argument("--win-reward", type=float, default=None, help="覆盖拦截者获胜终局奖励")
    parser.add_argument("--loss-reward", type=float, default=None, help="覆盖入侵者获胜默认终局惩罚")
    parser.add_argument("--step-penalty", type=float, default=None, help="覆盖未终止 step 的基础惩罚")
    parser.add_argument("--agent-distance-weight", type=float, default=None, help="覆盖双方距离塑形权重")
    parser.add_argument("--intruder-distance-weight", type=float, default=None, help="覆盖入侵者到目标距离塑形权重")
    parser.add_argument(
        "--active-collision-loss-reward",
        type=float,
        default=None,
        help="覆盖 attack 主动碰撞失败的专用终局惩罚",
    )
    parser.add_argument(
        "--direct-reward-profile",
        choices=("none", "chase", "guard", "guard_strong"),
        default="none",
        help="只覆盖 direct 入侵策略下的奖励塑形 profile",
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
        help="只覆盖 attack 入侵策略下的奖励塑形 profile",
    )
    parser.add_argument("--timesteps", type=int, required=True, help="每个 PPO 模型训练步数")
    parser.add_argument(
        "--response-direct-timesteps",
        type=int,
        default=None,
        help="只覆盖 response_direct 训练步数；为空时使用 --timesteps",
    )
    parser.add_argument(
        "--response-detour-timesteps",
        type=int,
        default=None,
        help="只覆盖 response_detour 训练步数；为空时使用 --timesteps",
    )
    parser.add_argument(
        "--response-attack-timesteps",
        type=int,
        default=None,
        help="只覆盖 response_attack 训练步数；为空时使用 --timesteps",
    )
    parser.add_argument(
        "--baseline-timesteps",
        type=int,
        default=None,
        help="只覆盖 baseline 训练步数；为空时使用 --timesteps",
    )
    parser.add_argument("--seed", type=int, required=True, help="随机种子")
    parser.add_argument("--episodes", type=int, default=None, help="覆盖 evaluation.episodes")
    parser.add_argument(
        "--name-prefix",
        default="continuous_oracle_compare",
        help="写入 runs/ 和 artifacts/ 下的实验名前缀",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help=(
            "跳过训练，直接加载 artifact-dir 中已有模型评估；"
            "仅当确认对应 response/baseline 模型已存在时使用。"
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="显式指定模型目录；默认使用 artifacts/continuous_sweep/<实验名>",
    )
    parser.add_argument(
        "--run-diagnosis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="评估和常规分析后是否继续生成 episode 级诊断表，默认开启。",
    )
    return parser


def main() -> None:
    """加载基础配置，改成 oracle detector 后执行训练、评估和诊断。"""

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
    _apply_overrides(config, args, experiment_name)

    print("============================================================")
    print("连续 response policy oracle 对照单任务")
    print(json.dumps(_metadata(args, experiment_name), ensure_ascii=False, indent=2))
    print("============================================================")

    if not args.skip_train:
        train_continuous(config)
    else:
        _require_oracle_artifacts(Path(config["experiment"]["artifact_dir"]))

    run_dir = evaluate_continuous(config)
    analysis_summary = analyze_continuous_run(run_dir)
    print("oracle 对照常规分析摘要:")
    print(json.dumps(analysis_summary, ensure_ascii=False, indent=2))

    if args.run_diagnosis:
        diagnosis_summary = diagnose_continuous_run(run_dir=run_dir)
        print("oracle 对照 episode 级诊断摘要:")
        print(json.dumps(diagnosis_summary, ensure_ascii=False, indent=2))


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace, experiment_name: str) -> None:
    """把命令行参数写入配置，并强制启用 oracle 响应策略选择。

    这里不修改基础 YAML 文件，而是在单任务运行时覆盖配置，保证 Slurm array 中每个
    seed 都写入独立模型和结果目录，不会互相覆盖。
    """

    config["experiment"]["name"] = experiment_name
    config["experiment"]["seed"] = args.seed
    config["experiment"]["artifact_dir"] = str(
        args.artifact_dir or Path("artifacts") / "continuous_sweep" / experiment_name
    )
    config["environment"]["interceptor_max_speed"] = args.interceptor_speed
    config["environment"]["intruder_max_speed"] = args.intruder_speed
    config["environment"]["collision_radius"] = args.collision_radius
    reward_overrides = {
        "win_reward": getattr(args, "win_reward", None),
        "loss_reward": getattr(args, "loss_reward", None),
        "step_penalty": getattr(args, "step_penalty", None),
        "agent_distance_weight": getattr(args, "agent_distance_weight", None),
        "intruder_distance_weight": getattr(args, "intruder_distance_weight", None),
        "active_collision_loss_reward": getattr(args, "active_collision_loss_reward", None),
    }
    for key, value in reward_overrides.items():
        if value is not None:
            config["environment"][key] = float(value)
    policy_reward_overrides = _build_policy_reward_overrides(args)
    if policy_reward_overrides:
        config["environment"]["reward_overrides_by_policy"] = policy_reward_overrides
    config["ppo"]["total_timesteps"] = args.timesteps
    response_timesteps = {
        "direct": getattr(args, "response_direct_timesteps", None),
        "detour": getattr(args, "response_detour_timesteps", None),
        "attack": getattr(args, "response_attack_timesteps", None),
    }
    response_timesteps = {
        name: int(value)
        for name, value in response_timesteps.items()
        if value is not None
    }
    if response_timesteps:
        config["ppo"]["response_timesteps_by_policy"] = response_timesteps
    baseline_timesteps = getattr(args, "baseline_timesteps", None)
    if baseline_timesteps is not None:
        config["ppo"]["baseline_total_timesteps"] = int(baseline_timesteps)
    if args.episodes is not None:
        config["evaluation"]["episodes"] = args.episodes

    # evaluate_continuous 中非 sam detector 会按真实入侵策略直接选择 response policy。
    # 这相当于“检测完全正确”的 oracle 上限，用来隔离响应策略库本身的质量问题。
    detector = config.setdefault("detector", {})
    detector["method"] = "oracle"
    detector.setdefault("initial_policy", "direct")


def _build_policy_reward_overrides(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """根据命令行 profile 生成按策略奖励覆盖。

    全局 reward sweep 已经证明“统一改所有策略奖励”只能缓解 seed 43。这里把 direct
    和 attack 的奖励塑形拆开，让 detour 保持原始奖励，便于判断短板是否来自某类
    response policy 的专用训练信号不足。
    """

    profiles: dict[str, dict[str, float]] = {}
    direct_profile = getattr(args, "direct_reward_profile", "none")
    attack_profile = getattr(args, "attack_reward_profile", "none")
    if direct_profile == "chase":
        profiles["direct"] = {
            "win_reward": 120.0,
            "loss_reward": -120.0,
            "agent_distance_weight": -0.35,
            "intruder_distance_weight": 0.05,
        }
    elif direct_profile == "guard":
        profiles["direct"] = {
            "win_reward": 120.0,
            "loss_reward": -120.0,
            "agent_distance_weight": -0.25,
            "intruder_distance_weight": 0.12,
        }
    elif direct_profile == "guard_strong":
        profiles["direct"] = {
            "win_reward": 140.0,
            "loss_reward": -140.0,
            "agent_distance_weight": -0.20,
            "intruder_distance_weight": 0.18,
        }

    if attack_profile == "chase":
        profiles["attack"] = {
            "win_reward": 120.0,
            "loss_reward": -120.0,
            "agent_distance_weight": -0.35,
            "intruder_distance_weight": 0.05,
        }
    elif attack_profile == "attacksafe":
        profiles["attack"] = {
            "win_reward": 120.0,
            "loss_reward": -150.0,
            "agent_distance_weight": -0.30,
            "intruder_distance_weight": 0.08,
            "active_collision_loss_reward": -220.0,
        }
    elif attack_profile == "attacksafe_strong":
        profiles["attack"] = {
            "win_reward": 140.0,
            "loss_reward": -180.0,
            "agent_distance_weight": -0.25,
            "intruder_distance_weight": 0.12,
            "active_collision_loss_reward": -300.0,
        }
    elif attack_profile == "attack_chase_light":
        profiles["attack"] = {
            "win_reward": 120.0,
            "loss_reward": -120.0,
            "agent_distance_weight": -0.28,
            "intruder_distance_weight": 0.04,
            "active_collision_loss_reward": -140.0,
        }
    elif attack_profile == "attack_guard":
        profiles["attack"] = {
            "win_reward": 120.0,
            "loss_reward": -120.0,
            "agent_distance_weight": -0.20,
            "intruder_distance_weight": 0.14,
            "active_collision_loss_reward": -160.0,
        }
    elif attack_profile == "attack_guard_safe":
        profiles["attack"] = {
            "win_reward": 120.0,
            "loss_reward": -140.0,
            "agent_distance_weight": -0.22,
            "intruder_distance_weight": 0.12,
            "active_collision_loss_reward": -220.0,
        }
    elif attack_profile == "attack_balanced":
        profiles["attack"] = {
            "win_reward": 120.0,
            "loss_reward": -130.0,
            "agent_distance_weight": -0.26,
            "intruder_distance_weight": 0.08,
            "active_collision_loss_reward": -180.0,
        }
    return profiles


def _require_oracle_artifacts(artifact_dir: Path) -> None:
    """跳过训练时检查必需模型是否齐全，避免评估中途才失败。"""

    required = [
        artifact_dir / "response_direct.zip",
        artifact_dir / "response_detour.zip",
        artifact_dir / "response_attack.zip",
        artifact_dir / "baseline_switching_ppo.zip",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"跳过训练但缺少 oracle 对照所需模型:\n{joined}")


def _build_experiment_name(
    *,
    prefix: str,
    interceptor_speed: float,
    intruder_speed: float,
    collision_radius: float,
    timesteps: int,
    seed: int,
) -> str:
    """构造和 sweep/confirm 一致的实验目录名，方便聚合脚本解析参数。"""

    return (
        f"{prefix}"
        f"_is{_slug_float(interceptor_speed)}"
        f"_us{_slug_float(intruder_speed)}"
        f"_cr{_slug_float(collision_radius)}"
        f"_ts{timesteps}"
        f"_s{seed}"
    )


def _slug_float(value: float) -> str:
    """把小数转成目录友好的短字符串，例如 0.030 -> 0p03。"""

    return re.sub(r"[^0-9a-zA-Z]+", "p", f"{value:g}")


def _metadata(args: argparse.Namespace, experiment_name: str) -> dict[str, Any]:
    """整理日志元信息，便于 Slurm 输出中快速定位参数组。"""

    return {
        "experiment_name": experiment_name,
        "interceptor_max_speed": args.interceptor_speed,
        "intruder_max_speed": args.intruder_speed,
        "collision_radius": args.collision_radius,
        "total_timesteps": args.timesteps,
        "response_direct_timesteps": args.response_direct_timesteps,
        "response_detour_timesteps": args.response_detour_timesteps,
        "response_attack_timesteps": args.response_attack_timesteps,
        "baseline_timesteps": args.baseline_timesteps,
        "win_reward": getattr(args, "win_reward", None),
        "loss_reward": getattr(args, "loss_reward", None),
        "step_penalty": getattr(args, "step_penalty", None),
        "agent_distance_weight": getattr(args, "agent_distance_weight", None),
        "intruder_distance_weight": getattr(args, "intruder_distance_weight", None),
        "active_collision_loss_reward": getattr(args, "active_collision_loss_reward", None),
        "direct_reward_profile": getattr(args, "direct_reward_profile", "none"),
        "attack_reward_profile": getattr(args, "attack_reward_profile", "none"),
        "seed": args.seed,
        "episodes": args.episodes,
        "detector_method": "oracle",
        "skip_train": args.skip_train,
        "artifact_dir": None if args.artifact_dir is None else str(args.artifact_dir),
        "run_diagnosis": args.run_diagnosis,
    }


if __name__ == "__main__":
    main()
