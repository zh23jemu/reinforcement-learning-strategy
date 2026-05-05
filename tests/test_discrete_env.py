from rl_strategy.discrete.env import PredatorPreyEnv


def test_predator_prey_env_step_shape():
    env = PredatorPreyEnv(seed=1)
    observation, info = env.reset()

    next_observation, reward, terminated, truncated, next_info = env.step(0)

    assert observation.shape == (9,)
    assert next_observation.shape == (9,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "predator_b_policy" in info
    assert "predator_b_action" in next_info


def test_predator_b_probability_distribution_is_valid():
    env = PredatorPreyEnv(seed=1)
    env.reset()

    probs = env.predator_b_action_probabilities("chase_x")

    assert probs.shape == (5,)
    assert abs(float(probs.sum()) - 1.0) < 1e-9

