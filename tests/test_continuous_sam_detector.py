import numpy as np
import torch

from rl_strategy.continuous.sam_detector import SamSwitchboard, train_dropout_opponent_model


class ConstantOpponentModel(torch.nn.Module):
    """测试用常量 opponent model，用于稳定构造候选误差差异。"""

    def __init__(self, value: tuple[float, float]) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """忽略观测并输出固定动作，便于测试 switchboard 的切换门控。"""

        return self.value.reshape(1, -1).expand(observation.shape[0], -1)


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


def test_sam_switchboard_respects_warmup_cooldown_and_margin():
    observation = np.zeros(10, dtype=np.float32)
    observed_attack_action = np.array([1.0, 1.0], dtype=np.float32)
    switchboard = SamSwitchboard(
        policy_names=["direct", "attack"],
        opponent_models={
            "direct": ConstantOpponentModel((0.0, 0.0)),
            "attack": ConstantOpponentModel((1.0, 1.0)),
        },
        initial_policy="direct",
        threshold=1.0,
        decay=0.0,
        mc_passes=3,
        noise_variance=0.01,
        online_learning_rate=0.0,
        warmup_steps=2,
        cooldown_steps=2,
        switch_margin=0.5,
        online_updates=False,
    )

    first = switchboard.update(observation, observed_attack_action)
    second = switchboard.update(observation, observed_attack_action)
    third = switchboard.update(observation, observed_attack_action)

    assert first.switched is False
    assert second.switched is False
    assert third.switched is True
    assert third.switch_ready is True
    assert third.assumed_policy == "attack"
    assert third.best_candidate == "attack"
    assert third.switch_margin >= 0.5
