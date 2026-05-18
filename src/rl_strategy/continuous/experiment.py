"""连续拦截捕猎实验训练与评估流程。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from rl_strategy.continuous.env import ContinuousInterceptEnv, ContinuousPolicyConditionedEnv
from rl_strategy.continuous.intruder import POLICY_NAMES, IntruderPolicyName
from rl_strategy.continuous.sam_detector import (
    SamSwitchboard,
    load_opponent_model,
    save_opponent_model,
    train_dropout_opponent_model,
)


def train_continuous(config: dict[str, Any]) -> None:
    """训练连续场景中的 PPO 响应策略和 baseline。

    当前阶段入侵者使用三类脚本策略，拦截者使用 Stable-Baselines3 PPO。训练会
    产出三类响应策略 `response_<policy>.zip`，以及一个面对周期切换入侵者的
    `baseline_switching_ppo.zip`。
    """

    artifact_dir = Path(config["experiment"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    response_timesteps_by_policy = config.get("ppo", {}).get("response_timesteps_by_policy", {})
    if not isinstance(response_timesteps_by_policy, dict):
        response_timesteps_by_policy = {}
    default_timesteps = int(config["ppo"]["total_timesteps"])

    for index, policy_name in enumerate(POLICY_NAMES):
        env = ContinuousPolicyConditionedEnv(
            _make_env(config, intruder_policy=policy_name, switch_interval=None),
            policy_index=index,
            policy_count=len(POLICY_NAMES),
        )
        model = _make_ppo(config, env)
        # 默认三类响应策略使用同一训练步数；专项补强实验可以只加训 direct/attack，
        # 用于确认 seed 43 的短板是否来自某类 response policy 训练不足。
        policy_timesteps = int(response_timesteps_by_policy.get(policy_name, default_timesteps))
        model.learn(total_timesteps=policy_timesteps)
        model.save(artifact_dir / f"response_{policy_name}.zip")
        _train_sam_opponent_model(config, policy_name, artifact_dir)

    baseline_env = _make_env(
        config,
        intruder_policy=str(config["detector"]["initial_policy"]),
        switch_interval=int(config["environment"]["switch_interval"]),
    )
    baseline_model = _make_ppo(config, baseline_env)
    baseline_timesteps = int(config["ppo"].get("baseline_total_timesteps", default_timesteps))
    baseline_model.learn(total_timesteps=baseline_timesteps)
    baseline_model.save(artifact_dir / "baseline_switching_ppo.zip")


def evaluate_continuous(config: dict[str, Any]) -> Path:
    """评估连续场景的策略库切换方案，并保存逐步数据。"""

    run_dir = _create_run_dir(config)
    _save_json(run_dir / "config.json", config)

    response_policies = _load_response_policies(config)
    detector_method = str(config.get("detector", {}).get("method", "sam"))
    switchboard = _make_sam_switchboard(config) if detector_method == "sam" else None
    env = _make_env(
        config,
        intruder_policy=str(config["detector"]["initial_policy"]),
        switch_interval=int(config["environment"]["switch_interval"]),
    )

    rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    episode_rewards: list[float] = []
    interceptor_wins = 0
    total_steps = 0
    correct_response_policy_steps = 0

    for episode in range(int(config["evaluation"]["episodes"])):
        observation, _ = env.reset()
        done = False
        episode_reward = 0.0
        final_winner: str | None = None

        while not done:
            observation_before_action = np.asarray(observation, dtype=np.float32).copy()
            true_policy_before_step = env.intruder_policy
            assumed_policy = (
                switchboard.assumed_policy
                if switchboard is not None
                else str(true_policy_before_step)
            )
            model = response_policies.get(assumed_policy)
            if model is None:
                action = _heuristic_interceptor_action(env)
            else:
                conditioned_observation = _condition_observation(observation, assumed_policy)
                action, _ = model.predict(
                    conditioned_observation,
                    deterministic=bool(config["evaluation"]["deterministic"]),
                )

            observation, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            true_policy_after_step = str(info["intruder_policy"])
            sam_result = None
            if switchboard is not None:
                observed_intruder_action = np.asarray(env.state.intruder_velocity, dtype=np.float32)
                sam_result = switchboard.update(observation_before_action, observed_intruder_action)
                if sam_result.switched:
                    switch_rows.append(
                        {
                            "episode": episode,
                            "global_step": info["global_step"],
                            "from_policy": assumed_policy,
                            "to_policy": sam_result.assumed_policy,
                            "true_policy": true_policy_after_step,
                            "normalized_error": sam_result.normalized_error,
                            "running_error": sam_result.running_error,
                            "best_candidate": sam_result.best_candidate,
                            "best_candidate_error": sam_result.best_candidate_error,
                            "current_candidate_error": sam_result.current_candidate_error,
                            "switch_margin": sam_result.switch_margin,
                        }
                    )
            done = bool(terminated or truncated)
            episode_reward += float(reward)
            total_steps += 1
            correct_response_policy_steps += int(assumed_policy == true_policy_after_step)
            final_winner = info["winner"] or final_winner

            row = {
                "episode": episode,
                "global_step": info["global_step"],
                "episode_step": info["episode_step"],
                "reward": reward,
                "episode_reward_so_far": episode_reward,
                "intruder_policy": true_policy_after_step,
                "intruder_policy_before_step": true_policy_before_step,
                "response_policy": assumed_policy,
                "response_policy_correct": assumed_policy == true_policy_after_step,
                "winner": info["winner"],
                "reason": info["reason"],
                "collision": info["collision"],
                "intruder_distance": info["intruder_distance"],
                "agent_distance": info["agent_distance"],
                "interceptor_action_x": float(np.asarray(action).reshape(2)[0]),
                "interceptor_action_y": float(np.asarray(action).reshape(2)[1]),
                "intruder_action_x": float(env.state.intruder_velocity[0]),
                "intruder_action_y": float(env.state.intruder_velocity[1]),
            }
            if sam_result is not None:
                row.update(
                    {
                        "sam_pred_action_x": float(sam_result.prediction.mean[0]),
                        "sam_pred_action_y": float(sam_result.prediction.mean[1]),
                        "sam_uncertainty_x": float(sam_result.prediction.uncertainty[0]),
                        "sam_uncertainty_y": float(sam_result.prediction.uncertainty[1]),
                        "sam_uncertainty_mean": float(np.mean(sam_result.prediction.uncertainty)),
                        "sam_normalized_error": sam_result.normalized_error,
                        "sam_running_error": sam_result.running_error,
                        "sam_switched": sam_result.switched,
                        "sam_switch_ready": sam_result.switch_ready,
                        "sam_best_candidate": sam_result.best_candidate,
                        "sam_best_candidate_error": sam_result.best_candidate_error,
                        "sam_current_candidate_error": sam_result.current_candidate_error,
                        "sam_switch_margin": sam_result.switch_margin,
                        "sam_steps_since_switch": sam_result.steps_since_switch,
                    }
                )
                for name, value in sam_result.candidate_errors.items():
                    row[f"sam_candidate_error_{name}"] = value
            rows.append(row)

        episode_rewards.append(episode_reward)
        interceptor_wins += int(final_winner == "interceptor")

    pd.DataFrame(rows).to_csv(run_dir / "step_trace.csv", index=False)
    pd.DataFrame(switch_rows).to_csv(run_dir / "switch_events.csv", index=False)
    baseline_rewards = _evaluate_baseline(config, run_dir)
    summary = {
        "episodes": int(config["evaluation"]["episodes"]),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "interceptor_win_rate": float(interceptor_wins / max(int(config["evaluation"]["episodes"]), 1)),
        "response_policy_accuracy": float(correct_response_policy_steps / max(total_steps, 1)),
        "detector_method": detector_method,
        "switch_count": len(switch_rows),
        "baseline_mean_episode_reward": baseline_rewards["mean"],
        "baseline_std_episode_reward": baseline_rewards["std"],
        "baseline_interceptor_win_rate": baseline_rewards["win_rate"],
    }
    _save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"连续场景过程数据已保存到: {run_dir}")
    return run_dir


def _evaluate_baseline(config: dict[str, Any], run_dir: Path) -> dict[str, float | None]:
    """评估不切换响应策略的普通 PPO baseline。"""

    artifact_dir = Path(config["experiment"]["artifact_dir"])
    baseline_path = artifact_dir / "baseline_switching_ppo.zip"
    if not baseline_path.exists():
        return {"mean": None, "std": None, "win_rate": None}

    model = PPO.load(baseline_path)
    env = _make_env(
        config,
        intruder_policy=str(config["detector"]["initial_policy"]),
        switch_interval=int(config["environment"]["switch_interval"]),
    )
    rows: list[dict[str, Any]] = []
    rewards: list[float] = []
    wins = 0

    for episode in range(int(config["evaluation"]["episodes"])):
        observation, _ = env.reset()
        done = False
        episode_reward = 0.0
        final_winner: str | None = None
        while not done:
            action, _ = model.predict(
                observation,
                deterministic=bool(config["evaluation"]["deterministic"]),
            )
            observation, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            done = bool(terminated or truncated)
            episode_reward += float(reward)
            final_winner = info["winner"] or final_winner
            rows.append(
                {
                    "episode": episode,
                    "global_step": info["global_step"],
                    "episode_step": info["episode_step"],
                    "reward": reward,
                    "intruder_policy": info["intruder_policy"],
                    "winner": info["winner"],
                    "reason": info["reason"],
                }
            )
        rewards.append(episode_reward)
        wins += int(final_winner == "interceptor")

    pd.DataFrame(rows).to_csv(run_dir / "baseline_step_trace.csv", index=False)
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "win_rate": float(wins / max(int(config["evaluation"]["episodes"]), 1)),
    }


def _make_env(
    config: dict[str, Any],
    *,
    intruder_policy: str,
    switch_interval: int | None,
) -> ContinuousInterceptEnv:
    """根据 YAML 配置创建连续拦截环境。"""

    environment = config["environment"]
    return ContinuousInterceptEnv(
        big_radius=float(environment["big_radius"]),
        small_radius=float(environment["small_radius"]),
        collision_radius=float(environment["collision_radius"]),
        max_steps=int(environment["max_steps"]),
        intruder_policy=intruder_policy,  # type: ignore[arg-type]
        switch_interval=switch_interval,
        intruder_max_speed=float(environment["intruder_max_speed"]),
        interceptor_max_speed=float(environment["interceptor_max_speed"]),
        world_radius=float(environment["world_radius"]),
        detour_safe_distance=float(environment["detour_safe_distance"]),
        win_reward=float(environment.get("win_reward", 100.0)),
        loss_reward=float(environment.get("loss_reward", -100.0)),
        step_penalty=float(environment.get("step_penalty", -0.01)),
        agent_distance_weight=float(environment.get("agent_distance_weight", -0.2)),
        intruder_distance_weight=float(environment.get("intruder_distance_weight", 0.05)),
        active_collision_loss_reward=_optional_float(environment.get("active_collision_loss_reward")),
        reward_overrides_by_policy=_reward_overrides_by_policy(environment.get("reward_overrides_by_policy")),
        seed=int(config["experiment"]["seed"]),
    )


def _optional_float(value: Any) -> float | None:
    """把可选配置值转换为浮点数，缺省时保留 None。

    Slurm sweep 会通过命令行把奖励权重写入配置；其中 attack 主动碰撞惩罚是可选项。
    这里集中处理空值，避免环境构造处混入字符串判断。
    """

    if value is None:
        return None
    return float(value)


def _reward_overrides_by_policy(value: Any) -> dict[str, dict[str, float | None]]:
    """解析按入侵策略覆盖的奖励配置。

    配置文件或命令行包装脚本可以只给 `direct`、`attack` 中某几个字段传覆盖值。
    这里统一转成环境可直接消费的浮点数字典，并保留 None 表示“该字段不覆盖”。
    """

    if not isinstance(value, dict):
        return {}
    result: dict[str, dict[str, float | None]] = {}
    for policy_name, overrides in value.items():
        if not isinstance(overrides, dict):
            continue
        result[str(policy_name)] = {
            str(key): _optional_float(override_value)
            for key, override_value in overrides.items()
        }
    return result


def _train_sam_opponent_model(config: dict[str, Any], policy_name: str, artifact_dir: Path) -> None:
    """为指定入侵策略训练 SAM dropout opponent model。

    训练数据来自固定入侵策略环境中的短 rollout。标签是环境真实产生的入侵者
    二维速度，符合 SAM 原文用状态-动作轨迹学习 opponent policy 的设定。
    """

    sam_config = config.get("sam", {})
    if not bool(sam_config.get("enabled", True)):
        return

    observations, actions = _collect_opponent_model_data(config, policy_name)
    model = train_dropout_opponent_model(
        observations,
        actions,
        hidden_dim=int(sam_config.get("hidden_dim", 64)),
        dropout=float(sam_config.get("dropout", 0.1)),
        epochs=int(sam_config.get("epochs", 8)),
        batch_size=int(sam_config.get("batch_size", 128)),
        learning_rate=float(sam_config.get("learning_rate", 0.001)),
        seed=int(config["experiment"]["seed"]),
        feature_mode=str(sam_config.get("feature_mode", "geometry")),
    )
    save_opponent_model(
        artifact_dir / f"opponent_model_{policy_name}.zip",
        model,
        {
            "input_dim": int(model.sam_input_dim),
            "raw_input_dim": int(model.sam_raw_input_dim),
            "output_dim": int(actions.shape[1]),
            "hidden_dim": int(sam_config.get("hidden_dim", 64)),
            "dropout": float(sam_config.get("dropout", 0.1)),
            "feature_mode": str(model.sam_feature_mode),
            "policy_name": policy_name,
        },
    )


def _collect_opponent_model_data(config: dict[str, Any], policy_name: str) -> tuple[np.ndarray, np.ndarray]:
    """采集 opponent model 的监督训练样本。"""

    sam_config = config.get("sam", {})
    sample_steps = int(sam_config.get("sample_steps", 2000))
    env = _make_env(config, intruder_policy=policy_name, switch_interval=None)
    rng = np.random.default_rng(int(config["experiment"]["seed"]))
    observation, _ = env.reset()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for _ in range(sample_steps):
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        # 用随机拦截者动作覆盖不同相对位置，使 opponent model 不只记住单一路径。
        interceptor_action = rng.uniform(
            low=-env.interceptor_max_speed,
            high=env.interceptor_max_speed,
            size=2,
        ).astype(np.float32)
        observation, _, terminated, truncated, _ = env.step(interceptor_action)
        actions.append(np.asarray(env.state.intruder_velocity, dtype=np.float32).copy())
        if terminated or truncated:
            observation, _ = env.reset()
    return np.stack(observations, axis=0), np.stack(actions, axis=0)


def _make_sam_switchboard(config: dict[str, Any]) -> SamSwitchboard:
    """加载 opponent models 并创建 SAM switchboard。"""

    artifact_dir = Path(config["experiment"]["artifact_dir"])
    opponent_models = {
        name: load_opponent_model(artifact_dir / f"opponent_model_{name}.zip")
        for name in POLICY_NAMES
    }
    detector = config.get("detector", {})
    sam_config = config.get("sam", {})
    return SamSwitchboard(
        policy_names=list(POLICY_NAMES),
        opponent_models=opponent_models,
        initial_policy=str(detector["initial_policy"]),
        threshold=float(detector.get("threshold", 8.0)),
        decay=float(detector.get("decay", 0.05)),
        mc_passes=int(sam_config.get("mc_passes", 20)),
        noise_variance=float(sam_config.get("noise_variance", 1e-4)),
        online_learning_rate=float(sam_config.get("online_learning_rate", 0.0003)),
        warmup_steps=int(sam_config.get("warmup_steps", detector.get("warmup_steps", 0))),
        cooldown_steps=int(sam_config.get("cooldown_steps", detector.get("cooldown_steps", 0))),
        switch_margin=float(sam_config.get("switch_margin", detector.get("switch_margin", 0.0))),
        max_normalized_error=(
            None
            if sam_config.get("max_normalized_error", detector.get("max_normalized_error")) is None
            else float(sam_config.get("max_normalized_error", detector.get("max_normalized_error")))
        ),
        online_updates=bool(sam_config.get("online_updates", True)),
    )


def _make_ppo(config: dict[str, Any], env: gym.Env) -> PPO:
    """统一创建 Stable-Baselines3 PPO，避免连续模块手写 PPO。"""

    return PPO(
        "MlpPolicy",
        env,
        seed=int(config["experiment"]["seed"]),
        n_steps=int(config["ppo"]["n_steps"]),
        batch_size=int(config["ppo"]["batch_size"]),
        learning_rate=float(config["ppo"]["learning_rate"]),
        gamma=float(config["ppo"]["gamma"]),
        verbose=1,
    )


def _load_response_policies(config: dict[str, Any]) -> dict[str, PPO | None]:
    """加载三类入侵策略对应的响应 PPO 模型。"""

    artifact_dir = Path(config["experiment"]["artifact_dir"])
    policies: dict[str, PPO | None] = {}
    for name in POLICY_NAMES:
        path = artifact_dir / f"response_{name}.zip"
        policies[name] = PPO.load(path) if path.exists() else None
    return policies


def _condition_observation(observation: np.ndarray, policy_name: str) -> np.ndarray:
    """给固定响应策略模型补上训练时使用的策略编号。"""

    index = POLICY_NAMES.index(policy_name)  # type: ignore[arg-type]
    denom = max(len(POLICY_NAMES) - 1, 1)
    return np.concatenate([observation, np.array([index / denom], dtype=np.float32)]).astype(np.float32)


def _heuristic_interceptor_action(env: ContinuousInterceptEnv) -> np.ndarray:
    """未训练 PPO 时使用的兜底拦截动作，直接朝入侵者移动。"""

    direction = env.state.intruder_position - env.state.interceptor_position
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return np.zeros(2, dtype=np.float32)
    return (direction / norm * env.interceptor_max_speed).astype(np.float32)


def _create_run_dir(config: dict[str, Any]) -> Path:
    """创建本次连续评估输出目录。"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["experiment"]["run_dir"]) / config["experiment"]["name"] / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_json(path: Path, data: dict[str, Any]) -> None:
    """保存 JSON 文件，确保中文内容不被转义。"""

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
