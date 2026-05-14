import numpy as np

from rl_strategy.continuous.sam_detector import SamSwitchboard, train_dropout_opponent_model


def test_sam_switchboard_uses_mc_dropout_uncertainty():
    observations = np.random.default_rng(1).normal(size=(64, 10)).astype(np.float32)
    actions = np.repeat(np.array([[0.1, -0.05]], dtype=np.float32), repeats=64, axis=0)
    model = train_dropout_opponent_model(
        observations,
        actions,
        hidden_dim=16,
        dropout=0.2,
        epochs=2,
        batch_size=16,
        learning_rate=0.01,
        seed=1,
    )
    switchboard = SamSwitchboard(
        policy_names=["direct"],
        opponent_models={"direct": model},
        initial_policy="direct",
        threshold=100.0,
        decay=0.0,
        mc_passes=5,
        noise_variance=1e-4,
        online_learning_rate=0.001,
    )

    result = switchboard.update(observations[0], actions[0])

    assert result.assumed_policy == "direct"
    assert result.normalized_error >= 0.0
    assert result.prediction.mean.shape == (2,)
    assert result.prediction.uncertainty.shape == (2,)
    assert np.all(result.prediction.uncertainty > 0.0)
