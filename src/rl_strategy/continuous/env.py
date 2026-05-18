"""连续二维拦截捕猎环境。

该环境根据新增需求文档实现：中心保护目标点、大小两个同心圆、一个入侵者和
一个拦截者。动作空间为连续二维速度，当前由 Stable-Baselines3 PPO 控制拦截者，
入侵者使用 `intruder.py` 中的三类脚本策略。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_strategy.continuous.intruder import (
    IntruderPolicyContext,
    IntruderPolicyName,
    intruder_velocity,
)


@dataclass(frozen=True)
class ContinuousInterceptState:
    """连续拦截环境内部状态。"""

    intruder_position: np.ndarray
    interceptor_position: np.ndarray
    intruder_velocity: np.ndarray
    interceptor_velocity: np.ndarray


class ContinuousInterceptEnv(gym.Env):
    """OpenAI Multi-Particle Environment 风格的连续二维拦截环境。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        big_radius: float = 1.0,
        small_radius: float = 0.3,
        collision_radius: float = 0.08,
        max_steps: int = 500,
        intruder_policy: IntruderPolicyName = "direct",
        switch_interval: int | None = None,
        intruder_max_speed: float = 0.018,
        interceptor_max_speed: float = 0.022,
        world_radius: float = 1.5,
        detour_safe_distance: float = 0.35,
        win_reward: float = 100.0,
        loss_reward: float = -100.0,
        step_penalty: float = -0.01,
        agent_distance_weight: float = -0.2,
        intruder_distance_weight: float = 0.05,
        active_collision_loss_reward: float | None = None,
        reward_overrides_by_policy: Mapping[str, Mapping[str, float | None]] | None = None,
        seed: int | None = None,
    ) -> None:
        """初始化连续拦截环境。

        参数:
            big_radius: 大保护圆半径。
            small_radius: 小保护圆半径。
            collision_radius: 两智能体中心距离小于该值时视为碰撞。
            max_steps: 每个 episode 最大步数。
            intruder_policy: 当前真实入侵者策略。
            switch_interval: 若不为空，按全局步数在三类策略间循环切换。
            intruder_max_speed: 入侵者最大速度。
            interceptor_max_speed: 拦截者最大速度，也是动作向量裁剪上限。
            world_radius: 观测归一化和初始采样使用的世界半径。
            detour_safe_distance: 迂回策略尝试维持的安全距离。
            win_reward: 拦截者获胜时的终局奖励。
            loss_reward: 入侵者获胜时的默认终局惩罚。
            step_penalty: 每个未终止 step 的基础惩罚。
            agent_distance_weight: 未终止时，双方距离的塑形权重；负值鼓励靠近入侵者。
            intruder_distance_weight: 未终止时，入侵者到目标距离的塑形权重；正值鼓励把入侵者挡在外侧。
            active_collision_loss_reward: attack 策略主动碰撞造成失败时的专用惩罚；
                为空时沿用 loss_reward。该参数用于专项检查 attack response policy 是否需要
                更强的避让/诱导塑形，不改变默认实验口径。
            reward_overrides_by_policy: 按入侵策略覆盖的奖励权重。键为 `direct`、`detour`
                或 `attack`，值只需要包含要覆盖的奖励字段。该入口用于下一轮只改
                direct/attack 训练奖励、保留 detour 原始奖励的诊断实验。
            seed: 随机种子。
        """

        super().__init__()
        if not 0 < small_radius < big_radius:
            raise ValueError("small_radius 必须大于 0 且小于 big_radius")
        if collision_radius <= 0:
            raise ValueError("collision_radius 必须为正数")

        self.big_radius = float(big_radius)
        self.small_radius = float(small_radius)
        self.collision_radius = float(collision_radius)
        self.max_steps = int(max_steps)
        self.intruder_policy = intruder_policy
        self.switch_interval = switch_interval
        self.intruder_max_speed = float(intruder_max_speed)
        self.interceptor_max_speed = float(interceptor_max_speed)
        self.world_radius = float(world_radius)
        self.detour_safe_distance = float(detour_safe_distance)
        self.win_reward = float(win_reward)
        self.loss_reward = float(loss_reward)
        self.step_penalty = float(step_penalty)
        self.agent_distance_weight = float(agent_distance_weight)
        self.intruder_distance_weight = float(intruder_distance_weight)
        self.active_collision_loss_reward = (
            None if active_collision_loss_reward is None else float(active_collision_loss_reward)
        )
        self.reward_overrides_by_policy = {
            str(policy_name): {
                str(key): None if value is None else float(value)
                for key, value in overrides.items()
            }
            for policy_name, overrides in (reward_overrides_by_policy or {}).items()
        }
        self.rng = np.random.default_rng(seed)
        self.global_step = 0
        self.episode_step = 0
        self.target_position = np.zeros(2, dtype=np.float32)
        self.state = ContinuousInterceptState(
            intruder_position=np.zeros(2, dtype=np.float32),
            interceptor_position=np.zeros(2, dtype=np.float32),
            intruder_velocity=np.zeros(2, dtype=np.float32),
            interceptor_velocity=np.zeros(2, dtype=np.float32),
        )

        self.action_space = spaces.Box(
            low=-self.interceptor_max_speed,
            high=self.interceptor_max_speed,
            shape=(2,),
            dtype=np.float32,
        )
        # 观测包含双方位置、速度，以及到目标距离、双方距离和当前策略编号占位。
        # 评估 baseline 时不暴露真实策略；固定响应策略训练时策略编号由 wrapper 追加。
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(10,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """重置 episode，采样入侵者在大圆外、拦截者在小圆内的初始位置。"""

        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        intruder_position = self._sample_outside_big_circle()
        interceptor_position = self._sample_inside_small_circle()
        self.state = ContinuousInterceptState(
            intruder_position=intruder_position,
            interceptor_position=interceptor_position,
            intruder_velocity=np.zeros(2, dtype=np.float32),
            interceptor_velocity=np.zeros(2, dtype=np.float32),
        )
        self.episode_step = 0
        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        """用拦截者连续速度动作推进环境一步。"""

        self.episode_step += 1
        self.global_step += 1
        self._maybe_switch_intruder_policy(self.global_step)

        interceptor_velocity = self._clip_velocity(action, self.interceptor_max_speed)
        context = IntruderPolicyContext(
            intruder_position=self.state.intruder_position,
            interceptor_position=self.state.interceptor_position,
            target_position=self.target_position,
            max_speed=self.intruder_max_speed,
            safe_distance=self.detour_safe_distance,
        )
        intruder_step_velocity = intruder_velocity(self.intruder_policy, context)

        next_intruder = self._clip_position(self.state.intruder_position + intruder_step_velocity)
        next_interceptor = self._clip_position(self.state.interceptor_position + interceptor_velocity)
        self.state = ContinuousInterceptState(
            intruder_position=next_intruder,
            interceptor_position=next_interceptor,
            intruder_velocity=intruder_step_velocity,
            interceptor_velocity=interceptor_velocity,
        )

        outcome = self._outcome()
        reward = self._reward(outcome)
        terminated = outcome["winner"] is not None
        truncated = self.episode_step >= self.max_steps
        if truncated and outcome["winner"] is None:
            outcome = self._timeout_outcome()
            reward = self._reward(outcome)
            terminated = outcome["winner"] is not None

        return self._observation(), float(reward), bool(terminated), bool(truncated), self._info(outcome)

    def _outcome(self) -> dict[str, Any]:
        """根据三类入侵策略的规则判断当前步胜负。"""

        intruder_distance = self._distance_to_target(self.state.intruder_position)
        collision = self._agent_distance() <= self.collision_radius
        passive_intercept = collision and not self._active_intruder_collision()

        if self.intruder_policy == "direct":
            if collision and intruder_distance < self.big_radius:
                return {"winner": "interceptor", "reason": "intercepted_in_big_circle", "collision": collision}
            if intruder_distance < self.big_radius:
                return {"winner": "intruder", "reason": "entered_big_circle", "collision": collision}
        elif self.intruder_policy == "detour":
            if collision and intruder_distance < self.big_radius:
                return {"winner": "interceptor", "reason": "intercepted_before_small_circle", "collision": collision}
            if intruder_distance < self.small_radius:
                return {"winner": "intruder", "reason": "entered_small_circle", "collision": collision}
        elif self.intruder_policy == "attack":
            if collision and self._active_intruder_collision():
                return {"winner": "intruder", "reason": "active_collision", "collision": collision}
            if passive_intercept:
                return {"winner": "interceptor", "reason": "passive_intercept", "collision": collision}
            if intruder_distance < self.small_radius:
                return {"winner": "intruder", "reason": "entered_small_circle", "collision": collision}

        return {"winner": None, "reason": "running", "collision": collision}

    def _timeout_outcome(self) -> dict[str, Any]:
        """按需求文档定义超时胜负。"""

        return {"winner": "interceptor", "reason": "timeout", "collision": self._agent_distance() <= self.collision_radius}

    def _reward(self, outcome: dict[str, Any]) -> float:
        """拦截者视角奖励函数，兼顾胜负终局和过程塑形。"""

        reward_config = self._reward_config_for_current_policy()
        if outcome["winner"] == "interceptor":
            return float(reward_config["win_reward"])
        if outcome["winner"] == "intruder":
            active_collision_loss_reward = reward_config["active_collision_loss_reward"]
            if outcome.get("reason") == "active_collision" and active_collision_loss_reward is not None:
                return float(active_collision_loss_reward)
            return float(reward_config["loss_reward"])
        intruder_distance = self._distance_to_target(self.state.intruder_position)
        agent_distance = self._agent_distance()
        # 过程奖励默认保持原有口径；专项实验可通过配置调大靠近入侵者或挡在外圈的信号，
        # 用来判断 direct/attack 响应策略短板是否来自奖励塑形不足。
        return (
            float(reward_config["step_penalty"])
            + float(reward_config["agent_distance_weight"]) * agent_distance
            + float(reward_config["intruder_distance_weight"]) * intruder_distance
        )

    def _reward_config_for_current_policy(self) -> dict[str, float | None]:
        """合并默认奖励权重和当前入侵策略的专用覆盖。

        训练固定 response policy 时，环境中的 `intruder_policy` 不会切换；评估 oracle
        或 SAM 时则会随 episode 进程切换。按当前真实策略动态取配置，可以让下一轮
        实验只强化 direct/attack，而不牺牲 detour 已经很强的原始奖励设定。
        """

        base: dict[str, float | None] = {
            "win_reward": self.win_reward,
            "loss_reward": self.loss_reward,
            "step_penalty": self.step_penalty,
            "agent_distance_weight": self.agent_distance_weight,
            "intruder_distance_weight": self.intruder_distance_weight,
            "active_collision_loss_reward": self.active_collision_loss_reward,
        }
        overrides = self.reward_overrides_by_policy.get(self.intruder_policy, {})
        for key, value in overrides.items():
            if key in base:
                base[key] = value
        return base

    def _active_intruder_collision(self) -> bool:
        """判断攻击型碰撞是否由入侵者主动朝拦截者冲撞造成。"""

        direction = self.state.interceptor_position - self.state.intruder_position
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            return False
        unit = direction / norm
        projection = float(np.dot(self.state.intruder_velocity, unit))
        return projection > 0.0

    def _observation(self) -> np.ndarray:
        """生成拦截者观测，不暴露真实入侵策略。"""

        scale = max(self.world_radius, 1e-6)
        intruder_distance = self._distance_to_target(self.state.intruder_position) / scale
        agent_distance = self._agent_distance() / scale
        return np.array(
            [
                *(self.state.intruder_position / scale),
                *(self.state.interceptor_position / scale),
                *(self.state.intruder_velocity / max(self.intruder_max_speed, 1e-6)),
                *(self.state.interceptor_velocity / max(self.interceptor_max_speed, 1e-6)),
                intruder_distance,
                agent_distance,
            ],
            dtype=np.float32,
        )

    def _info(self, outcome: dict[str, Any] | None = None) -> dict[str, Any]:
        """返回评估和日志需要的状态信息。"""

        outcome = outcome or {"winner": None, "reason": "reset", "collision": False}
        return {
            "intruder_policy": self.intruder_policy,
            "winner": outcome["winner"],
            "reason": outcome["reason"],
            "collision": outcome["collision"],
            "episode_step": self.episode_step,
            "global_step": self.global_step,
            "intruder_distance": self._distance_to_target(self.state.intruder_position),
            "agent_distance": self._agent_distance(),
        }

    def _maybe_switch_intruder_policy(self, step_number: int) -> None:
        """按全局步数在三类入侵策略间循环切换。"""

        if not self.switch_interval:
            return
        if step_number > 0 and step_number % self.switch_interval == 0:
            order: list[IntruderPolicyName] = ["direct", "detour", "attack"]
            current = order.index(self.intruder_policy)
            self.intruder_policy = order[(current + 1) % len(order)]

    def _sample_outside_big_circle(self) -> np.ndarray:
        """采样大圆外的入侵者初始位置。"""

        angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
        radius = float(self.rng.uniform(self.big_radius * 1.05, self.world_radius))
        return np.array([np.cos(angle) * radius, np.sin(angle) * radius], dtype=np.float32)

    def _sample_inside_small_circle(self) -> np.ndarray:
        """采样小圆内的拦截者初始位置。"""

        angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
        radius = float(self.rng.uniform(0.0, self.small_radius * 0.8))
        return np.array([np.cos(angle) * radius, np.sin(angle) * radius], dtype=np.float32)

    def _clip_velocity(self, action: np.ndarray, max_speed: float) -> np.ndarray:
        """将连续动作裁剪为最大速度约束内的速度向量。"""

        velocity = np.asarray(action, dtype=np.float32).reshape(2)
        norm = float(np.linalg.norm(velocity))
        if norm > max_speed:
            velocity = velocity / norm * max_speed
        return velocity.astype(np.float32)

    def _clip_position(self, position: np.ndarray) -> np.ndarray:
        """将位置裁剪在仿真世界边界内，避免 PPO 探索导致数值漂移。"""

        return np.clip(position, -self.world_radius, self.world_radius).astype(np.float32)

    def _distance_to_target(self, position: np.ndarray) -> float:
        """计算某个实体到中心目标的距离。"""

        return float(np.linalg.norm(position - self.target_position))

    def _agent_distance(self) -> float:
        """计算入侵者与拦截者之间的中心距离。"""

        return float(np.linalg.norm(self.state.intruder_position - self.state.interceptor_position))


class ContinuousPolicyConditionedEnv(gym.Env):
    """给响应策略训练使用的策略条件包装器。

    固定响应策略训练时，在观测末尾追加入侵策略编号，使 PPO 可以学习与特定
    入侵策略绑定的响应行为；评估时仍通过策略库切换选择相应模型。
    """

    metadata = {"render_modes": []}

    def __init__(self, env: ContinuousInterceptEnv, policy_index: int, policy_count: int) -> None:
        super().__init__()
        self.env = env
        self.policy_index = int(policy_index)
        self.policy_count = int(policy_count)
        self.action_space = env.action_space
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(11,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """重置底层环境并追加策略编号。"""

        observation, info = self.env.reset(seed=seed, options=options)
        return self._augment(observation), info

    def step(self, action: np.ndarray):
        """推进底层环境并追加策略编号。"""

        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(observation), reward, terminated, truncated, info

    def _augment(self, observation: np.ndarray) -> np.ndarray:
        """在观测末尾追加归一化策略编号。"""

        denom = max(self.policy_count - 1, 1)
        marker = np.array([self.policy_index / denom], dtype=np.float32)
        return np.concatenate([observation, marker]).astype(np.float32)

    def render(self):  # pragma: no cover - 当前环境不需要图形渲染
        """兼容 Gymnasium 渲染接口。"""

        return None

    def close(self) -> None:
        """关闭底层环境。"""

        self.env.close()
