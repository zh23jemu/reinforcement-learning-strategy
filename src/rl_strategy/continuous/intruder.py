"""连续场景中的脚本化入侵者策略。

该模块先按需求文档实现三类可解释入侵策略，用于构建连续动作空间的工程
复现闭环。拦截者仍由 Stable-Baselines3 PPO 训练；入侵者策略在当前阶段
作为非平稳对手库使用，后续如果论文需要学习型入侵者，可在这里替换接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

IntruderPolicyName = Literal["direct", "detour", "attack"]


POLICY_NAMES: list[IntruderPolicyName] = ["direct", "detour", "attack"]


@dataclass(frozen=True)
class IntruderPolicyContext:
    """入侵者策略决策所需的连续状态。

    属性:
        intruder_position: 入侵者二维坐标。
        interceptor_position: 拦截者二维坐标。
        target_position: 被保护目标点坐标，默认是原点。
        max_speed: 入侵者允许的最大速度。
        safe_distance: 迂回型策略尝试与拦截者保持的安全距离。
    """

    intruder_position: np.ndarray
    interceptor_position: np.ndarray
    target_position: np.ndarray
    max_speed: float
    safe_distance: float


def intruder_velocity(policy_name: IntruderPolicyName, context: IntruderPolicyContext) -> np.ndarray:
    """根据策略名称返回入侵者速度向量。

    参数:
        policy_name: 三类入侵策略之一。
        context: 当前环境状态和速度约束。

    返回:
        长度为 2 的速度向量，模长不超过 `context.max_speed`。
    """

    if policy_name == "direct":
        return _direct_velocity(context)
    if policy_name == "detour":
        return _detour_velocity(context)
    if policy_name == "attack":
        return _attack_velocity(context)
    raise ValueError(f"未知入侵者策略: {policy_name}")


def _direct_velocity(context: IntruderPolicyContext) -> np.ndarray:
    """直入型：无视拦截者，沿当前位置到目标点的方向全速前进。"""

    return _unit(context.target_position - context.intruder_position) * context.max_speed


def _attack_velocity(context: IntruderPolicyContext) -> np.ndarray:
    """攻击型：主动朝拦截者方向全速冲撞。"""

    return _unit(context.interceptor_position - context.intruder_position) * context.max_speed


def _detour_velocity(context: IntruderPolicyContext) -> np.ndarray:
    """迂回型：在靠近目标的同时绕开拦截者。

    该策略由三部分合成：朝目标推进、围绕拦截者的切向绕行、距离过近时的
    排斥项。它不是最优控制器，但能稳定产生“侧翼绕行”的非平稳对手行为。
    """

    to_target = _unit(context.target_position - context.intruder_position)
    from_interceptor = context.intruder_position - context.interceptor_position
    distance = float(np.linalg.norm(from_interceptor))
    away = _unit(from_interceptor)
    tangent = np.array([-away[1], away[0]], dtype=np.float32)

    avoid_strength = max(0.0, (context.safe_distance - distance) / max(context.safe_distance, 1e-6))
    direction = 0.65 * to_target + 0.65 * tangent + 1.25 * avoid_strength * away
    return _unit(direction) * context.max_speed


def _unit(vector: np.ndarray) -> np.ndarray:
    """返回单位向量；零向量时返回零向量，避免数值异常。"""

    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return np.zeros_like(vector, dtype=np.float32)
    return vector / norm

