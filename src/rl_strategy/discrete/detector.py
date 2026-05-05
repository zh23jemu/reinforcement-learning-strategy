"""OPS-DeMo 离散动作空间策略变化检测器。

论文中的核心思想是：对每个候选对手策略维护一个 running error。
如果某个策略在当前状态下选择观测动作的概率越低，那么该策略解释当前动作的
误差越大。为了避免随机策略的自然采样误差无限累积，running error 每一步会减去
一个由 strictness factor `alpha` 控制的动态衰减项。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    """一次检测更新后的结果。

    属性:
        assumed_policy: 当前认为对手正在执行的策略名称。
        switched: 本步是否检测到策略切换。
        running_errors: 每个候选策略当前的运行误差快照。
    """

    assumed_policy: str
    switched: bool
    running_errors: dict[str, float]


class DiscreteOpsDemoDetector:
    """面向离散动作空间随机策略的 OPS-DeMo 检测器。"""

    def __init__(
        self,
        policy_names: list[str],
        *,
        alpha: float,
        threshold: float,
        initial_policy: str | None = None,
    ) -> None:
        """初始化检测器。

        参数:
            policy_names: 候选对手策略名称。
            alpha: 严格系数，越接近 1，切换检测越敏感。
            threshold: running error 的切换阈值和上限。
            initial_policy: 初始假设策略；如果为空则使用第一个候选策略。
        """

        if not policy_names:
            raise ValueError("policy_names 不能为空")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha 必须位于 [0, 1]")
        if threshold <= 0:
            raise ValueError("threshold 必须为正数")

        self.policy_names = list(policy_names)
        self.alpha = alpha
        self.threshold = threshold
        self.assumed_policy = initial_policy or self.policy_names[0]
        if self.assumed_policy not in self.policy_names:
            raise ValueError("initial_policy 必须属于 policy_names")

        self.running_errors = {name: 0.0 for name in self.policy_names}

    @staticmethod
    def observed_error(action_probabilities: np.ndarray, observed_action: int) -> float:
        """计算论文 Lemma 1 中的观测误差 `1 - P(a|s)`。

        参数:
            action_probabilities: 候选策略在当前状态下的离散动作概率分布。
            observed_action: 真实观测到的动作编号。

        返回:
            当前动作相对该候选策略的观测误差。
        """

        probs = np.asarray(action_probabilities, dtype=np.float64)
        if observed_action < 0 or observed_action >= probs.shape[0]:
            raise ValueError("observed_action 超出动作概率范围")
        return float(1.0 - probs[observed_action])

    @staticmethod
    def expected_following_error(action_probabilities: np.ndarray) -> float:
        """计算遵循候选策略时的自然期望误差。

        对离散随机策略 `pi`，论文给出的期望误差为
        `sum_j p_j * (1 - p_j)`。
        """

        probs = np.asarray(action_probabilities, dtype=np.float64)
        return float(np.sum(probs * (1.0 - probs)))

    @staticmethod
    def expected_not_following_error(action_count: int) -> float:
        """计算不遵循候选策略时的近似期望误差。

        论文在未知替代策略时采用均匀动作假设，因此为 `(n - 1) / n`。
        """

        if action_count <= 0:
            raise ValueError("action_count 必须为正数")
        return float((action_count - 1) / action_count)

    def update(
        self,
        policy_action_probabilities: Mapping[str, np.ndarray],
        observed_action: int,
    ) -> DetectionResult:
        """根据一个观测动作更新所有候选策略的 running error。

        参数:
            policy_action_probabilities: 每个候选策略在当前状态下的动作概率。
            observed_action: 对手真实执行的动作。

        返回:
            检测结果，包含当前假设策略和所有 running error。
        """

        for name in self.policy_names:
            probs = np.asarray(policy_action_probabilities[name], dtype=np.float64)
            observed = self.observed_error(probs, observed_action)
            expected_follow = self.expected_following_error(probs)
            expected_not_follow = self.expected_not_following_error(probs.shape[0])
            decay = self.alpha * expected_follow + (1.0 - self.alpha) * expected_not_follow

            next_error = self.running_errors[name] + observed - decay
            self.running_errors[name] = float(np.clip(next_error, 0.0, self.threshold))

        switched = False
        if self.running_errors[self.assumed_policy] >= self.threshold:
            switched = True
            self.assumed_policy = min(self.running_errors, key=self.running_errors.get)
            self.running_errors[self.assumed_policy] *= 0.5

        return DetectionResult(
            assumed_policy=self.assumed_policy,
            switched=switched,
            running_errors=dict(self.running_errors),
        )

