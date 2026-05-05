"""Predator-Prey 离散网格环境。

该环境用于复现 OPS-DeMo 论文中的离散动作空间设置：

- 网格大小默认 10x10。
- 两个捕食者：Predator A 是待训练/评估主体，Predator B 是会切换策略的对手。
- 两个猎物：Prey X 与 Prey Y 随机移动。
- Predator B 可在“追 Prey X”和“追 Prey Y”两种策略间切换。

环境实现尽量保持简单、可控，重点服务策略切换检测，而不是追求复杂生态模拟。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

MoveAction = Literal[0, 1, 2, 3, 4]

STAY: MoveAction = 0
UP: MoveAction = 1
DOWN: MoveAction = 2
LEFT: MoveAction = 3
RIGHT: MoveAction = 4

ACTION_DELTAS: dict[int, tuple[int, int]] = {
    STAY: (0, 0),
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}


@dataclass(frozen=True)
class PredatorPreyState:
    """环境内部状态，统一保存四个智能体的位置。"""

    predator_a: tuple[int, int]
    predator_b: tuple[int, int]
    prey_x: tuple[int, int]
    prey_y: tuple[int, int]


class PredatorPreyEnv(gym.Env):
    """用于训练 Predator A 响应策略的 Gymnasium 环境。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        grid_size: int = 10,
        max_steps: int = 40,
        predator_b_target: str = "chase_x",
        switch_interval: int | None = None,
        prey_random_prob: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """初始化环境。

        参数:
            grid_size: 正方形网格边长。
            max_steps: 每个 episode 的最大步数。
            predator_b_target: Predator B 的固定目标策略，或初始策略。
            switch_interval: 如果不为空，Predator B 每隔该全局步数切换目标。
            prey_random_prob: 猎物随机移动概率，保留配置接口以便调节场景难度。
            seed: 随机种子。
        """

        super().__init__()
        if grid_size < 4:
            raise ValueError("grid_size 至少为 4，避免初始位置过度拥挤")
        if predator_b_target not in {"chase_x", "chase_y"}:
            raise ValueError("predator_b_target 必须是 chase_x 或 chase_y")

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.predator_b_target = predator_b_target
        self.switch_interval = switch_interval
        self.prey_random_prob = prey_random_prob
        self.rng = np.random.default_rng(seed)
        self.global_step = 0
        self.episode_step = 0
        self.state = PredatorPreyState((0, 0), (0, 0), (0, 0), (0, 0))

        # Predator A 使用 5 个离散动作：停留、上、下、左、右。
        self.action_space = spaces.Discrete(5)

        # 观测包含四个实体的归一化坐标和 Predator B 当前目标编号。
        # 训练响应策略时暴露目标编号，评估普通 PPO baseline 时可以通过 wrapper 隐藏。
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """重置 episode，并返回初始观测。"""

        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        positions = self._sample_unique_positions(4)
        self.state = PredatorPreyState(
            predator_a=positions[0],
            predator_b=positions[1],
            prey_x=positions[2],
            prey_y=positions[3],
        )
        self.episode_step = 0
        return self._observation(), self._info()

    def step(self, action: int):
        """推进环境一步。

        参数:
            action: Predator A 的动作。

        返回:
            Gymnasium 标准五元组。
        """

        self.episode_step += 1
        self.global_step += 1
        self._maybe_switch_predator_b()

        predator_b_action = self.predator_b_policy_action()
        prey_x_action = self._random_action() if self.rng.random() < self.prey_random_prob else STAY
        prey_y_action = self._random_action() if self.rng.random() < self.prey_random_prob else STAY

        self.state = PredatorPreyState(
            predator_a=self._move(self.state.predator_a, int(action)),
            predator_b=self._move(self.state.predator_b, predator_b_action),
            prey_x=self._move(self.state.prey_x, prey_x_action),
            prey_y=self._move(self.state.prey_y, prey_y_action),
        )

        caught_x = self._is_caught(self.state.prey_x)
        caught_y = self._is_caught(self.state.prey_y)
        collision = self.state.predator_a == self.state.predator_b

        reward = -1.0
        if collision:
            reward -= 1.0
        if caught_x and caught_y:
            reward += 100.0

        terminated = bool(caught_x and caught_y)
        truncated = self.episode_step >= self.max_steps
        return self._observation(), reward, terminated, truncated, self._info(predator_b_action)

    def predator_b_policy_action(self) -> int:
        """根据当前 Predator B 策略选择动作。

        这里实现为明确的启发式追逐策略，便于训练稳定并可直接获得动作概率。
        后续若要加载 PPO 版 Predator B，只需替换策略库实现，不影响检测器接口。
        """

        target = self.state.prey_x if self.predator_b_target == "chase_x" else self.state.prey_y
        return greedy_action_towards(self.state.predator_b, target)

    def predator_b_action_probabilities(self, policy_name: str) -> np.ndarray:
        """返回候选 Predator B 策略在当前状态下的离散动作概率。

        为了让 OPS-DeMo 的随机策略误差有意义，这里使用 epsilon-greedy 分布：
        目标方向动作获得高概率，其余动作共享探索概率。
        """

        target = self.state.prey_x if policy_name == "chase_x" else self.state.prey_y
        greedy = greedy_action_towards(self.state.predator_b, target)
        probs = np.full(5, 0.05 / 4.0, dtype=np.float64)
        probs[greedy] = 0.95
        return probs

    def _maybe_switch_predator_b(self) -> None:
        """按全局步数周期性切换 Predator B 的真实策略。"""

        if not self.switch_interval:
            return
        if self.global_step > 0 and self.global_step % self.switch_interval == 0:
            self.predator_b_target = "chase_y" if self.predator_b_target == "chase_x" else "chase_x"

    def _sample_unique_positions(self, count: int) -> list[tuple[int, int]]:
        """采样互不重叠的初始位置。"""

        all_positions = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        indexes = self.rng.choice(len(all_positions), size=count, replace=False)
        return [all_positions[int(i)] for i in indexes]

    def _move(self, position: tuple[int, int], action: int) -> tuple[int, int]:
        """根据动作移动，并将位置裁剪到网格范围内。"""

        dr, dc = ACTION_DELTAS[int(action)]
        row = int(np.clip(position[0] + dr, 0, self.grid_size - 1))
        col = int(np.clip(position[1] + dc, 0, self.grid_size - 1))
        return row, col

    def _random_action(self) -> int:
        """采样一个随机离散动作。"""

        return int(self.rng.integers(0, 5))

    def _is_caught(self, prey: tuple[int, int]) -> bool:
        """判断某个猎物是否与任一捕食者相邻或重合。"""

        return (
            manhattan(self.state.predator_a, prey) <= 1
            or manhattan(self.state.predator_b, prey) <= 1
        )

    def _observation(self) -> np.ndarray:
        """生成归一化观测向量。"""

        scale = float(self.grid_size - 1)
        coords = [
            *self.state.predator_a,
            *self.state.predator_b,
            *self.state.prey_x,
            *self.state.prey_y,
        ]
        target_id = 0.0 if self.predator_b_target == "chase_x" else 1.0
        return np.array([v / scale for v in coords] + [target_id], dtype=np.float32)

    def _info(self, predator_b_action: int | None = None) -> dict:
        """返回便于日志记录和检测器使用的过程信息。"""

        return {
            "predator_b_policy": self.predator_b_target,
            "predator_b_action": predator_b_action,
            "state": self.state,
            "episode_step": self.episode_step,
            "global_step": self.global_step,
        }


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    """计算两个网格坐标的曼哈顿距离。"""

    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def greedy_action_towards(source: tuple[int, int], target: tuple[int, int]) -> int:
    """返回从 source 朝 target 移动的贪心动作。

    若行方向距离更大，优先上下移动；否则左右移动。距离为 0 时停留。
    """

    row_delta = target[0] - source[0]
    col_delta = target[1] - source[1]
    if abs(row_delta) >= abs(col_delta) and row_delta != 0:
        return DOWN if row_delta > 0 else UP
    if col_delta != 0:
        return RIGHT if col_delta > 0 else LEFT
    return STAY

