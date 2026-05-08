"""离散 OPS-DeMo 训练与评估流程。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO

from rl_strategy.discrete.detector import DiscreteOpsDemoDetector
from rl_strategy.discrete.env import PredatorBTrainingEnv, PredatorPreyEnv, greedy_action_towards

POLICY_NAMES = ["chase_x", "chase_y"]


def train_discrete(config: dict[str, Any]) -> None:
    """训练离散环境中的 PPO 响应策略。

    训练分成两组策略库：

    1. `opponent_*.zip`：Predator B 的候选策略库，用于产生真实对手动作和概率。
    2. `response_*.zip`：Predator A 的响应策略库，用于在检测到不同对手策略后切换。
    """

    artifact_dir = Path(config["experiment"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for policy_name in POLICY_NAMES:
        opponent_env = _make_b_training_env(config, target_policy=policy_name)
        opponent_model = PPO(
            "MlpPolicy",
            opponent_env,
            seed=int(config["experiment"]["seed"]),
            n_steps=int(config["ppo"]["n_steps"]),
            batch_size=int(config["ppo"]["batch_size"]),
            learning_rate=float(config["ppo"]["learning_rate"]),
            gamma=float(config["ppo"]["gamma"]),
            verbose=1,
        )
        opponent_model.learn(total_timesteps=int(config["ppo"]["total_timesteps"]))
        opponent_model.save(artifact_dir / f"opponent_{policy_name}.zip")

        response_env = _make_env(config, predator_b_target=policy_name, switch_interval=None)
        # 响应策略训练阶段优先使用刚训练好的 PPO 对手策略作为固定对手；
        # 这样 response policy 与真实评估时的 opponent policy bank 更一致。
        response_env = FixedOpponentPolicyEnv(response_env, policy_name, opponent_model)
        response_model = PPO(
            "MlpPolicy",
            response_env,
            seed=int(config["experiment"]["seed"]),
            n_steps=int(config["ppo"]["n_steps"]),
            batch_size=int(config["ppo"]["batch_size"]),
            learning_rate=float(config["ppo"]["learning_rate"]),
            gamma=float(config["ppo"]["gamma"]),
            verbose=1,
        )
        response_model.learn(total_timesteps=int(config["ppo"]["total_timesteps"]))
        response_model.save(artifact_dir / f"response_{policy_name}.zip")

    opponent_bank = _load_policy_bank(config, prefix="opponent")
    baseline_env = SwitchingOpponentPolicyEnv(
        _make_env(
            config,
            predator_b_target=str(config["detector"]["initial_policy"]),
            switch_interval=int(config["environment"]["switch_interval"]),
        ),
        opponent_bank,
    )
    baseline_model = PPO(
        "MlpPolicy",
        baseline_env,
        seed=int(config["experiment"]["seed"]),
        n_steps=int(config["ppo"]["n_steps"]),
        batch_size=int(config["ppo"]["batch_size"]),
        learning_rate=float(config["ppo"]["learning_rate"]),
        gamma=float(config["ppo"]["gamma"]),
        verbose=1,
    )
    baseline_model.learn(total_timesteps=int(config["ppo"]["total_timesteps"]))
    baseline_model.save(artifact_dir / "baseline_switching_ppo.zip")


def evaluate_discrete(config: dict[str, Any]) -> Path:
    """评估 OPS-DeMo + PPO，并保存每一步过程数据。"""

    run_dir = _create_run_dir(config)
    _save_json(run_dir / "config.json", config)

    opponent_policies = _load_policy_bank(config, prefix="opponent")
    response_policies = _load_policy_bank(config, prefix="response")
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
    correct_response_policies = 0
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

            env.prepare_predator_b_policy_for_next_step()
            probabilities = {
                name: _predator_b_action_probabilities(env, name, opponent_policies)
                for name in POLICY_NAMES
            }
            predator_b_action = _predator_b_action(env, opponent_policies)
            observation, reward, terminated, truncated, info = env.step_with_predator_b_action(
                action,
                predator_b_action,
                policy_already_advanced=True,
            )
            done = bool(terminated or truncated)
            episode_reward += float(reward)

            result = detector.update(probabilities, predator_b_action)

            total_steps += 1
            is_correct = result.assumed_policy == info["predator_b_policy"]
            response_policy_correct = assumed == info["predator_b_policy"]
            correct_assumptions += int(is_correct)
            correct_response_policies += int(response_policy_correct)

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
                "response_policy": assumed,
                "response_policy_correct": response_policy_correct,
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
    baseline_rewards = _evaluate_baseline(config, run_dir, opponent_policies)
    summary = {
        "episodes": int(config["evaluation"]["episodes"]),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "aop_accuracy": float(correct_assumptions / max(total_steps, 1)),
        "response_policy_accuracy": float(correct_response_policies / max(total_steps, 1)),
        "switch_count": len(switch_rows),
        "baseline_mean_episode_reward": baseline_rewards["mean"],
        "baseline_std_episode_reward": baseline_rewards["std"],
    }
    _save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"过程数据已保存到: {run_dir}")
    return run_dir


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
        predator_b_policy_mode="heuristic",
        switch_interval=switch_interval,
        prey_random_prob=float(config["environment"]["prey_random_prob"]),
        seed=int(config["experiment"]["seed"]),
    )


def _make_b_training_env(config: dict[str, Any], *, target_policy: str) -> PredatorBTrainingEnv:
    """创建用于训练 Predator B 候选策略的环境。"""

    return PredatorBTrainingEnv(
        target_policy=target_policy,
        grid_size=int(config["environment"]["grid_size"]),
        max_steps=int(config["environment"]["max_steps"]),
        prey_random_prob=float(config["environment"]["prey_random_prob"]),
        seed=int(config["experiment"]["seed"]),
    )


def _load_policy_bank(config: dict[str, Any], *, prefix: str) -> dict[str, PPO | None]:
    """加载 PPO 策略库。

    参数:
        config: 实验配置。
        prefix: 模型文件前缀，取值通常为 `opponent` 或 `response`。
    """

    artifact_dir = Path(config["experiment"]["artifact_dir"])
    policies: dict[str, PPO | None] = {}
    for name in POLICY_NAMES:
        path = artifact_dir / f"{prefix}_{name}.zip"
        policies[name] = PPO.load(path) if path.exists() else None
    return policies


def _predator_b_action(
    env: PredatorPreyEnv,
    opponent_policies: dict[str, PPO | None],
) -> int:
    """根据 Predator B 的真实隐藏策略获得动作。

    评估阶段 Predator A 不知道这个真实策略；这里只在环境外部驱动对手动作。
    """

    true_policy = env.predator_b_target
    model = opponent_policies.get(true_policy)
    if model is None:
        return env.predator_b_policy_action()
    action, _ = model.predict(
        env.observation_for_predator_b(true_policy),
        deterministic=False,
    )
    return int(action)


def _predator_b_action_probabilities(
    env: PredatorPreyEnv,
    policy_name: str,
    opponent_policies: dict[str, PPO | None],
) -> np.ndarray:
    """返回用于 OPS-DeMo 检测的对手策略动作概率。

    对于 PPO 模型，我们用其动作分布近似；若模型尚未训练，则回退到可解释的
    epsilon-greedy 分布，确保 smoke 评估阶段流程不中断。
    """

    model = opponent_policies.get(policy_name)
    if model is None:
        return env.predator_b_action_probabilities(policy_name)

    obs = env.observation_for_predator_b(policy_name)
    dist = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0])
    probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)
    if probs.shape[0] != env.action_space.n:
        return env.predator_b_action_probabilities(policy_name)
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / max(float(probs.sum()), 1e-12)
    return probs


def _evaluate_baseline(
    config: dict[str, Any],
    run_dir: Path,
    opponent_policies: dict[str, PPO | None],
) -> dict[str, float]:
    """评估普通 PPO baseline。

    baseline 不使用 OPS-DeMo 检测器，也不切换响应策略；它只依赖当前可见状态，
    用于复现论文中 standalone PPO 的对照组。
    """

    artifact_dir = Path(config["experiment"]["artifact_dir"])
    baseline_path = artifact_dir / "baseline_switching_ppo.zip"
    if not baseline_path.exists():
        return {"mean": float("nan"), "std": float("nan")}

    model = PPO.load(baseline_path)
    env = _make_env(
        config,
        predator_b_target=str(config["detector"]["initial_policy"]),
        switch_interval=int(config["environment"]["switch_interval"]),
    )
    rows: list[dict[str, Any]] = []
    rewards: list[float] = []

    for episode in range(int(config["evaluation"]["episodes"])):
        observation, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            env.prepare_predator_b_policy_for_next_step()
            action, _ = model.predict(
                observation,
                deterministic=bool(config["evaluation"]["deterministic"]),
            )
            predator_b_action = _predator_b_action(env, opponent_policies)
            observation, reward, terminated, truncated, info = env.step_with_predator_b_action(
                int(action),
                predator_b_action,
                policy_already_advanced=True,
            )
            done = bool(terminated or truncated)
            episode_reward += float(reward)
            rows.append(
                {
                    "episode": episode,
                    "global_step": info["global_step"],
                    "episode_step": info["episode_step"],
                    "reward": reward,
                    "episode_reward_so_far": episode_reward,
                    "true_policy": info["predator_b_policy"],
                    "predator_a_action": int(action),
                    "predator_b_action": predator_b_action,
                }
            )
        rewards.append(episode_reward)

    pd.DataFrame(rows).to_csv(run_dir / "baseline_step_trace.csv", index=False)
    return {"mean": float(np.mean(rewards)), "std": float(np.std(rewards))}


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


class FixedOpponentPolicyEnv(gym.Env):
    """给 Predator A 响应策略训练使用的固定 PPO 对手环境包装器。

    Stable-Baselines3 只要求对象具备 Gymnasium 环境接口。该包装器将 Predator A
    的动作传给基础环境，同时用固定的 Predator B PPO 模型生成对手动作。
    """

    metadata = {"render_modes": []}

    def __init__(self, env: PredatorPreyEnv, opponent_policy_name: str, opponent_model: PPO) -> None:
        super().__init__()
        self.env = env
        self.opponent_policy_name = opponent_policy_name
        self.opponent_model = opponent_model
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """重置底层环境。"""

        return self.env.reset(seed=seed, options=options)

    def step(self, action: int):
        """用固定 PPO 对手动作推进一步。"""

        opponent_action, _ = self.opponent_model.predict(
            self.env.observation_for_predator_b(self.opponent_policy_name),
            deterministic=False,
        )
        return self.env.step_with_predator_b_action(int(action), int(opponent_action))

    def render(self):  # pragma: no cover - 当前环境不需要可视化渲染
        """兼容 Gymnasium 渲染接口。"""

        return None

    def close(self) -> None:
        """关闭底层环境。"""

        self.env.close()


class SwitchingOpponentPolicyEnv(gym.Env):
    """普通 PPO baseline 训练环境。

    该环境让 Predator B 按隐藏策略周期性切换，但 Predator A 的观测仍只有 8 维
    可见状态。baseline 因此必须学习一个单一 PPO 策略来应对混合对手行为。
    """

    metadata = {"render_modes": []}

    def __init__(self, env: PredatorPreyEnv, opponent_policies: dict[str, PPO | None]) -> None:
        super().__init__()
        self.env = env
        self.opponent_policies = opponent_policies
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """重置底层环境。"""

        return self.env.reset(seed=seed, options=options)

    def step(self, action: int):
        """用当前真实隐藏策略对应的对手模型推进一步。"""

        predator_b_action = _predator_b_action(self.env, self.opponent_policies)
        return self.env.step_with_predator_b_action(int(action), predator_b_action)

    def render(self):  # pragma: no cover - 当前环境不需要可视化渲染
        """兼容 Gymnasium 渲染接口。"""

        return None

    def close(self) -> None:
        """关闭底层环境。"""

        self.env.close()
