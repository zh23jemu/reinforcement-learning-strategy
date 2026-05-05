"""离散 OPS-DeMo 训练与评估流程。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from rl_strategy.discrete.detector import DiscreteOpsDemoDetector
from rl_strategy.discrete.env import PredatorPreyEnv, greedy_action_towards

POLICY_NAMES = ["chase_x", "chase_y"]


def train_discrete(config: dict[str, Any]) -> None:
    """训练离散环境中的 PPO 响应策略。

    当前版本优先复现 OPS-DeMo 主框架。Predator B 的候选策略使用可解释启发式
    策略，Predator A 的两个响应策略用 PPO 分别在固定 Predator B 策略下训练。
    """

    artifact_dir = Path(config["experiment"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for policy_name in POLICY_NAMES:
        env = _make_env(config, predator_b_target=policy_name, switch_interval=None)
        model = PPO(
            "MlpPolicy",
            env,
            seed=int(config["experiment"]["seed"]),
            n_steps=int(config["ppo"]["n_steps"]),
            batch_size=int(config["ppo"]["batch_size"]),
            learning_rate=float(config["ppo"]["learning_rate"]),
            gamma=float(config["ppo"]["gamma"]),
            verbose=1,
        )
        model.learn(total_timesteps=int(config["ppo"]["total_timesteps"]))
        model.save(artifact_dir / f"response_{policy_name}.zip")


def evaluate_discrete(config: dict[str, Any]) -> None:
    """评估 OPS-DeMo + PPO，并保存每一步过程数据。"""

    run_dir = _create_run_dir(config)
    _save_json(run_dir / "config.json", config)

    response_policies = _load_or_create_response_policies(config)
    env = _make_env(
        config,
        predator_b_target=str(config["detector"]["initial_policy"]),
        switch_interval=int(config["environment"]["switch_interval"]),
    )
    detector = DiscreteOpsDemoDetector(
        POLICY_NAMES,
        alpha=float(config["detector"]["alpha"]),
        threshold=float(config["detector"]["threshold"]),
        initial_policy=str(config["detector"]["initial_policy"]),
    )

    step_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    episode_rewards: list[float] = []
    correct_assumptions = 0
    total_steps = 0

    for episode in range(int(config["evaluation"]["episodes"])):
        observation, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            assumed = detector.assumed_policy
            model = response_policies.get(assumed)
            if model is None:
                action = _heuristic_response_action(env, assumed)
            else:
                action, _ = model.predict(
                    observation,
                    deterministic=bool(config["evaluation"]["deterministic"]),
                )
                action = int(action)

            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_reward += float(reward)

            predator_b_action = int(info["predator_b_action"])
            probabilities = {
                name: env.predator_b_action_probabilities(name) for name in POLICY_NAMES
            }
            result = detector.update(probabilities, predator_b_action)

            total_steps += 1
            is_correct = result.assumed_policy == info["predator_b_policy"]
            correct_assumptions += int(is_correct)

            if result.switched:
                switch_rows.append(
                    {
                        "episode": episode,
                        "global_step": info["global_step"],
                        "episode_step": info["episode_step"],
                        "true_policy": info["predator_b_policy"],
                        "new_assumed_policy": result.assumed_policy,
                    }
                )

            row = {
                "episode": episode,
                "global_step": info["global_step"],
                "episode_step": info["episode_step"],
                "reward": reward,
                "episode_reward_so_far": episode_reward,
                "true_policy": info["predator_b_policy"],
                "assumed_policy": result.assumed_policy,
                "assumption_correct": is_correct,
                "predator_a_action": action,
                "predator_b_action": predator_b_action,
                "switched": result.switched,
            }
            for name, value in result.running_errors.items():
                row[f"running_error_{name}"] = value
            step_rows.append(row)

        episode_rewards.append(episode_reward)

    pd.DataFrame(step_rows).to_csv(run_dir / "step_trace.csv", index=False)
    pd.DataFrame(switch_rows).to_csv(run_dir / "switch_events.csv", index=False)
    summary = {
        "episodes": int(config["evaluation"]["episodes"]),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "aop_accuracy": float(correct_assumptions / max(total_steps, 1)),
        "switch_count": len(switch_rows),
    }
    _save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"过程数据已保存到: {run_dir}")


def _make_env(
    config: dict[str, Any],
    *,
    predator_b_target: str,
    switch_interval: int | None,
) -> PredatorPreyEnv:
    """根据配置创建 Predator-Prey 环境。"""

    return PredatorPreyEnv(
        grid_size=int(config["environment"]["grid_size"]),
        max_steps=int(config["environment"]["max_steps"]),
        predator_b_target=predator_b_target,
        switch_interval=switch_interval,
        prey_random_prob=float(config["environment"]["prey_random_prob"]),
        seed=int(config["experiment"]["seed"]),
    )


def _load_or_create_response_policies(config: dict[str, Any]) -> dict[str, PPO | None]:
    """加载已训练响应策略；缺失时退回启发式响应。

    这样 smoke 评估可以在尚未训练 PPO 的情况下直接跑通检测器和日志链路。
    论文级实验应先执行 `train` 生成 PPO 模型。
    """

    artifact_dir = Path(config["experiment"]["artifact_dir"])
    policies: dict[str, PPO | None] = {}
    for name in POLICY_NAMES:
        path = artifact_dir / f"response_{name}.zip"
        policies[name] = PPO.load(path) if path.exists() else None
    return policies


def _heuristic_response_action(env: PredatorPreyEnv, assumed_policy: str) -> int:
    """未训练 PPO 时使用的响应策略。

    如果认为 Predator B 正在追某个猎物，Predator A 就追另一个猎物，形成协作捕获。
    该策略只用于快速验证流程，不替代论文级 PPO 训练结果。
    """

    target = env.state.prey_y if assumed_policy == "chase_x" else env.state.prey_x
    return greedy_action_towards(env.state.predator_a, target)


def _create_run_dir(config: dict[str, Any]) -> Path:
    """创建本次评估的输出目录。"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["experiment"]["run_dir"]) / config["experiment"]["name"] / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_json(path: Path, data: dict[str, Any]) -> None:
    """保存 JSON 文件，确保中文内容不被转义。"""

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

