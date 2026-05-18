import numpy as np

from rl_strategy.continuous.env import ContinuousInterceptEnv, ContinuousPolicyConditionedEnv
from rl_strategy.continuous.intruder import IntruderPolicyContext, intruder_velocity


def test_continuous_env_reset_and_step_shapes():
    env = ContinuousInterceptEnv(seed=1)
    observation, info = env.reset()

    next_observation, reward, terminated, truncated, next_info = env.step(np.zeros(2, dtype=np.float32))

    assert observation.shape == (10,)
    assert next_observation.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["intruder_policy"] == "direct"
    assert "intruder_distance" in next_info


def test_continuous_initial_positions_follow_requirement():
    env = ContinuousInterceptEnv(seed=2, big_radius=1.0, small_radius=0.3)
    env.reset()

    intruder_distance = np.linalg.norm(env.state.intruder_position)
    interceptor_distance = np.linalg.norm(env.state.interceptor_position)

    assert intruder_distance > env.big_radius
    assert interceptor_distance < env.small_radius


def test_intruder_attack_velocity_points_to_interceptor():
    context = IntruderPolicyContext(
        intruder_position=np.array([1.0, 0.0], dtype=np.float32),
        interceptor_position=np.array([0.0, 0.0], dtype=np.float32),
        target_position=np.array([0.0, 0.0], dtype=np.float32),
        max_speed=0.2,
        safe_distance=0.3,
    )

    velocity = intruder_velocity("attack", context)

    assert velocity[0] < 0
    assert abs(float(np.linalg.norm(velocity)) - 0.2) < 1e-6


def test_policy_conditioned_env_appends_policy_marker():
    base_env = ContinuousInterceptEnv(seed=3)
    env = ContinuousPolicyConditionedEnv(base_env, policy_index=2, policy_count=3)

    observation, _ = env.reset()

    assert observation.shape == (11,)
    assert observation[-1] == 1.0


def test_continuous_policy_switch_cycles_three_intruder_modes():
    env = ContinuousInterceptEnv(seed=4, switch_interval=1)
    env.reset()

    _, _, _, _, first = env.step(np.zeros(2, dtype=np.float32))
    _, _, _, _, second = env.step(np.zeros(2, dtype=np.float32))
    _, _, _, _, third = env.step(np.zeros(2, dtype=np.float32))

    assert first["intruder_policy"] == "detour"
    assert second["intruder_policy"] == "attack"
    assert third["intruder_policy"] == "direct"


def test_reward_overrides_can_target_single_intruder_policy():
    """按策略奖励覆盖应只影响指定入侵策略，避免把 detour 原始奖励一起改掉。"""

    env = ContinuousInterceptEnv(
        seed=5,
        intruder_policy="direct",
        reward_overrides_by_policy={
            "direct": {"win_reward": 140.0, "agent_distance_weight": -0.4},
            "attack": {"active_collision_loss_reward": -250.0},
        },
    )

    direct_config = env._reward_config_for_current_policy()
    env.intruder_policy = "detour"
    detour_config = env._reward_config_for_current_policy()
    env.intruder_policy = "attack"
    attack_config = env._reward_config_for_current_policy()

    assert direct_config["win_reward"] == 140.0
    assert direct_config["agent_distance_weight"] == -0.4
    assert detour_config["win_reward"] == 100.0
    assert detour_config["agent_distance_weight"] == -0.2
    assert attack_config["active_collision_loss_reward"] == -250.0
