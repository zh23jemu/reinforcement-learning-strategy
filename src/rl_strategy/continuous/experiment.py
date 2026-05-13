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


def train_continuous(config: dict[str, Any]) -> None:
    """训练连续场景中的 PPO 响应策略和 baseline。

    当前阶段入侵者使用三类脚本策略，拦截者使用 Stable-Baselines3 PPO。训练会
    产出三类响应策略 `response_<policy>.zip`，以及一个面对周期切换入侵者的
    `baseline_switching_ppo.zip`。
    """

    artifact_dir = Path(config["experiment"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for index, policy_name in enumerate(POLICY_NAMES):
        env = ContinuousPolicyConditionedEnv(
            _make_env(config, intruder_policy=policy_name, switch_interval=None),
            policy_index=index,
            policy_count=len(POLICY_NAMES),
        )
        model = _make_ppo(config, env)
        model.learn(total_timesteps=int(config["ppo"]["total_timesteps"]))
        model.save(artifact_dir / f"response_{policy_name}.zip")

    baseline_env = _make_env(
        config,
        intruder_policy=str(config["detector"]["initial_policy"]),
        switch_interval=int(config["environment"]["switch_interval"]),
    )
    baseline_model = _make_ppo(config, baseline_env)
    baseline_model.learn(total_timesteps=int(config["ppo"]["total_timesteps"]))
    baseline_model.save(artifact_dir / "baseline_switching_ppo.zip")


def evaluate_continuous(config: dict[str, Any]) -> Path:
    """评估连续场景的策略库切换方案，并保存逐步数据。"""

    run_dir = _create_run_dir(config)
    _save_json(run_dir / "config.json", config)

    response_policies = _load_response_policies(config)
    env = _make_env(
        config,
        intruder_policy=str(config["detector"]["initial_policy"]),
        switch_interval=int(config["environment"]["switch_interval"]),
    )

    rows: list[dict[str, Any]] = []
    episode_rewards: list[float] = []
    interceptor_wins = 0
    total_steps = 0
    oracle_policy_steps = 0

    for episode in range(int(config["evaluation"]["episodes"])):
        observation, _ = env.reset()
        done = False
        episode_reward = 0.0
        final_winner: str | None = None

        while not done:
            true_policy = env.intruder_policy
            model = response_policies.get(true_policy)
            if model is None:
                action = _heuristic_interceptor_action(env)
            else:
                conditioned_observation = _condition_observation(observation, true_policy)
                action, _ = model.predict(
                    conditioned_observation,
                    deterministic=bool(config["evaluation"]["deterministic"]),
                )

            observation, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            done = bool(terminated or truncated)
            episode_reward += float(reward)
            total_steps += 1
            oracle_policy_steps += int(true_policy == info["intruder_policy"])
            final_winner = info["winner"] or final_winner

            rows.append(
                {
                    "episode": episode,
                    "global_step": info["global_step"],
                    "episode_step": info["episode_step"],
                    "reward": reward,
                    "episode_reward_so_far": episode_reward,
                    "intruder_policy": info["intruder_policy"],
                    "response_policy": true_policy,
                    "response_policy_correct": true_policy == info["intruder_policy"],
                    "winner": info["winner"],
                    "reason": info["reason"],
                    "collision": info["collision"],
                    "intruder_distance": info["intruder_distance"],
                    "agent_distance": info["agent_distance"],
                    "interceptor_action_x": float(np.asarray(action).reshape(2)[0]),
                    "interceptor_action_y": float(np.asarray(action).reshape(2)[1]),
                }
            )

        episode_rewards.append(episode_reward)
        interceptor_wins += int(final_winner == "interceptor")

    pd.DataFrame(rows).to_csv(run_dir / "step_trace.csv", index=False)
    baseline_rewards = _evaluate_baseline(config, run_dir)
    summary = {
        "episodes": int(config["evaluation"]["episodes"]),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "interceptor_win_rate": float(interceptor_wins / max(int(config["evaluation"]["episodes"]), 1)),
        "response_policy_accuracy": float(oracle_policy_steps / max(total_steps, 1)),
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
        seed=int(config["experiment"]["seed"]),
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
