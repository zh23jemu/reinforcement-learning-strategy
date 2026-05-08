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
        predator_b_policy_mode: str = "heuristic",
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
        if predator_b_policy_mode not in {"heuristic", "ppo"}:
            raise ValueError("predator_b_policy_mode 必须是 heuristic 或 ppo")

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.predator_b_target = predator_b_target
        self.predator_b_policy_mode = predator_b_policy_mode
        self.switch_interval = switch_interval
        self.prey_random_prob = prey_random_prob
        self.rng = np.random.default_rng(seed)
        self.global_step = 0
        self.episode_step = 0
        self.state = PredatorPreyState((0, 0), (0, 0), (0, 0), (0, 0))

        # Predator A 使用 5 个离散动作：停留、上、下、左、右。
        self.action_space = spaces.Discrete(5)

        # Predator A 的可见观测只包含四个实体的归一化坐标，不暴露真实对手策略。
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

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

        return self._advance(int(action), predator_b_action=None)

    def step_with_predator_b_action(
        self,
        action: int,
        predator_b_action: int,
        *,
        policy_already_advanced: bool = False,
    ):
        """使用外部给定的 Predator B 动作推进环境。

        评估 PPO 版对手策略时，需要由外部策略模型决定 Predator B 动作；该方法
        复用环境奖励、猎物移动、终止条件和日志信息，避免复制 step 逻辑。

        参数:
            action: Predator A 的动作。
            predator_b_action: 外部策略在当前状态下采样出的 Predator B 动作。
            policy_already_advanced: 外部调用方是否已经按下一步编号处理过隐藏
                策略切换。评估 PPO 对手时需要先切换再选动作，避免动作来自旧策略
                但日志记录为新策略。
        """

        return self._advance(
            int(action),
            predator_b_action=int(predator_b_action),
            policy_already_advanced=policy_already_advanced,
        )

    def prepare_predator_b_policy_for_next_step(self) -> None:
        """在外部策略选动作前，按下一步编号推进 Predator B 隐藏策略。

        环境内部 `step` 会自己处理切换；但评估 PPO 对手时，Predator B 的动作由
        环境外部模型先生成。如果不在生成动作前处理切换，就会出现“动作来自旧
        真实策略，info 却记录新真实策略”的一拍错位。
        """

        self._maybe_switch_predator_b(self.global_step + 1)

    def _advance(
        self,
        action: int,
        predator_b_action: int | None,
        *,
        policy_already_advanced: bool = False,
    ):
        """环境推进的内部实现。

        参数:
            action: Predator A 动作。
            predator_b_action: 外部指定的 Predator B 动作；为空时使用环境内置策略。
            policy_already_advanced: 是否已经在动作选择前处理过隐藏策略切换。
        """

        self.episode_step += 1
        self.global_step += 1
        if not policy_already_advanced:
            self._maybe_switch_predator_b(self.global_step)

        if predator_b_action is None:
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

        if self.predator_b_policy_mode == "ppo":
            # PPO 版本的 Predator B 在当前实现中仍以目标驱动的状态编码作为输入，
            # 真正的 PPO 策略会在训练和评估流程中被装载到对应 wrapper 上。
            # 这里保留函数接口，避免环境逻辑与策略加载逻辑耦合。
            target = self.state.prey_x if self.predator_b_target == "chase_x" else self.state.prey_y
            return greedy_action_towards(self.state.predator_b, target)

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

    def observation_for_predator_b(self, policy_name: str) -> np.ndarray:
        """为 Predator B 候选策略构造输入观测。

        为了让 PPO 能学习不同目标策略，这里在原始观测基础上显式加入候选目标编号。
        这样两个策略的输入分布可区分，训练也更稳定。
        """

        scale = float(self.grid_size - 1)
        coords = [
            *self.state.predator_a,
            *self.state.predator_b,
            *self.state.prey_x,
            *self.state.prey_y,
        ]
        target_id = 0.0 if policy_name == "chase_x" else 1.0
        return np.array([v / scale for v in coords] + [target_id], dtype=np.float32)

    def _maybe_switch_predator_b(self, step_number: int) -> None:
        """按全局步数周期性切换 Predator B 的真实策略。

        参数:
            step_number: 即将执行或正在执行的全局步数。外部 PPO 对手会在动作选择
                前用下一步编号调用该方法，环境内置策略则在 `_advance` 增加步数后
                调用，两种路径都保持同一个切换时刻。
        """

        if not self.switch_interval:
            return
        if step_number > 0 and step_number % self.switch_interval == 0:
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
        """生成 Predator A 可见的归一化观测向量。

        注意：这里故意不包含 Predator B 的真实策略编号，避免信息泄漏。
        """

        scale = float(self.grid_size - 1)
        coords = [
            *self.state.predator_a,
            *self.state.predator_b,
            *self.state.prey_x,
            *self.state.prey_y,
        ]
        return np.array([v / scale for v in coords], dtype=np.float32)

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


class PredatorBTrainingEnv(PredatorPreyEnv):
    """专门用于训练 Predator B 候选策略的环境。

    基类环境的动作控制 Predator A；该子类将动作解释为 Predator B 的动作，并用
    目标猎物的曼哈顿距离构造密集奖励。这样可以用 Stable-Baselines3 PPO 分别训练
    “追 Prey X”和“追 Prey Y”两个候选对手策略。
    """

    def __init__(self, *, target_policy: str, **kwargs) -> None:
        super().__init__(predator_b_target=target_policy, switch_interval=None, **kwargs)
        self.target_policy = target_policy
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """重置后返回 Predator B 专用的 9 维观测。"""

        super().reset(seed=seed, options=options)
        return self.observation_for_predator_b(self.target_policy), self._info()

    def step(self, action: int):
        """推进 Predator B 训练环境一步。"""

        self.episode_step += 1
        self.global_step += 1

        # 训练 B 时，Predator A 使用简单协作启发式：追另一个猎物。
        predator_a_target = self.state.prey_y if self.target_policy == "chase_x" else self.state.prey_x
        predator_a_action = greedy_action_towards(self.state.predator_a, predator_a_target)
        prey_x_action = self._random_action() if self.rng.random() < self.prey_random_prob else STAY
        prey_y_action = self._random_action() if self.rng.random() < self.prey_random_prob else STAY

        self.state = PredatorPreyState(
            predator_a=self._move(self.state.predator_a, predator_a_action),
            predator_b=self._move(self.state.predator_b, int(action)),
            prey_x=self._move(self.state.prey_x, prey_x_action),
            prey_y=self._move(self.state.prey_y, prey_y_action),
        )

        target = self.state.prey_x if self.target_policy == "chase_x" else self.state.prey_y
        distance = manhattan(self.state.predator_b, target)
        caught_target = distance <= 1
        caught_both = self._is_caught(self.state.prey_x) and self._is_caught(self.state.prey_y)

        # 密集奖励让短训也能学到朝目标移动；命中目标和完成双捕获额外奖励。
        reward = -float(distance) / max(float(self.grid_size), 1.0)
        if caught_target:
            reward += 10.0
        if caught_both:
            reward += 20.0

        terminated = bool(caught_both)
        truncated = self.episode_step >= self.max_steps
        return self.observation_for_predator_b(self.target_policy), reward, terminated, truncated, self._info(int(action))
