from rl_strategy.discrete.env import PredatorPreyEnv


def test_predator_prey_env_step_shape():
    env = PredatorPreyEnv(seed=1)
    observation, info = env.reset()

    next_observation, reward, terminated, truncated, next_info = env.step(0)

    assert observation.shape == (8,)
    assert next_observation.shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "predator_b_policy" in info
    assert "predator_b_action" in next_info


def test_predator_b_training_observation_keeps_target_signal_separate():
    env = PredatorPreyEnv(seed=1)
    env.reset()

    observation = env.observation_for_predator_b("chase_x")

    assert observation.shape == (9,)


def test_predator_b_probability_distribution_is_valid():
    env = PredatorPreyEnv(seed=1)
    env.reset()

    probs = env.predator_b_action_probabilities("chase_x")

    assert probs.shape == (5,)
    assert abs(float(probs.sum()) - 1.0) < 1e-9


def test_external_predator_b_policy_switches_before_action_selection():
    env = PredatorPreyEnv(seed=1, switch_interval=2)
    env.reset()

    env.prepare_predator_b_policy_for_next_step()
    assert env.predator_b_target == "chase_x"
    _, _, _, _, first_info = env.step_with_predator_b_action(
        0,
        0,
        policy_already_advanced=True,
    )

    env.prepare_predator_b_policy_for_next_step()
    assert env.predator_b_target == "chase_y"
    _, _, _, _, second_info = env.step_with_predator_b_action(
        0,
        0,
        policy_already_advanced=True,
    )

    assert first_info["global_step"] == 1
    assert first_info["predator_b_policy"] == "chase_x"
    assert second_info["global_step"] == 2
    assert second_info["predator_b_policy"] == "chase_y"


def test_internal_step_keeps_same_switch_timing_as_external_policy_path():
    env = PredatorPreyEnv(seed=1, switch_interval=2)
    env.reset()

    _, _, _, _, first_info = env.step(0)
    _, _, _, _, second_info = env.step(0)

    assert first_info["global_step"] == 1
    assert first_info["predator_b_policy"] == "chase_x"
    assert second_info["global_step"] == 2
    assert second_info["predator_b_policy"] == "chase_y"
