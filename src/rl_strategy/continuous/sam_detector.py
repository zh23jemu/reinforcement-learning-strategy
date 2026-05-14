"""连续场景 SAM opponent model 与 switchboard。

该模块实现 Everett SAM 论文中的核心检测逻辑：

1. 为每个候选对手行为训练一个带 dropout 的 opponent model。
2. 预测时保持 dropout 开启，执行多次前向传播，得到动作预测均值和方差。
3. 用观测动作误差除以预测不确定性，形成归一化误差并累积为 running error。
4. running error 超过阈值后切换到更匹配当前观测的 opponent model / response policy。

当前连续环境中的“对手动作”是入侵者二维速度向量；观测为拦截者可见的连续环境
观测。这样可以保留本项目自定义连续拦截环境，同时让检测方法回到 SAM 原文的
MC dropout 不确定性归一化误差。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SamPrediction:
    """一次 MC dropout 动作预测结果。"""

    mean: np.ndarray
    uncertainty: np.ndarray


@dataclass(frozen=True)
class SamSwitchResult:
    """switchboard 更新后的检测结果。"""

    assumed_policy: str
    switched: bool
    switch_ready: bool
    normalized_error: float
    running_error: float
    best_candidate: str
    best_candidate_error: float
    current_candidate_error: float
    switch_margin: float
    steps_since_switch: int
    prediction: SamPrediction
    candidate_errors: dict[str, float]


class DropoutOpponentModel(nn.Module):
    """带 dropout 的连续动作 opponent model。

    模型输入为连续环境观测，输出为入侵者下一步二维速度。Dropout 层既用于训练，
    也在预测阶段保持开启，以便通过多次采样近似 Bayesian predictive uncertainty。
    """

    def __init__(self, input_dim: int, output_dim: int = 2, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """预测入侵者动作。"""

        return self.net(observation)


def train_dropout_opponent_model(
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> DropoutOpponentModel:
    """用监督学习训练一个 SAM opponent model。

    参数:
        observations: 形状为 ``[N, obs_dim]`` 的环境观测。
        actions: 形状为 ``[N, 2]`` 的真实入侵者速度。
        hidden_dim: MLP 隐层宽度。
        dropout: dropout 概率，对应论文 Predict Action Algorithm 中的 ``p``。
        epochs: 训练轮数。
        batch_size: mini-batch 大小。
        learning_rate: Adam 学习率。
        seed: 随机种子，保证 smoke 测试可复现。

    返回:
        训练完成的 opponent model。
    """

    torch.manual_seed(seed)
    model = DropoutOpponentModel(
        input_dim=int(observations.shape[1]),
        output_dim=int(actions.shape[1]),
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    x = torch.as_tensor(observations, dtype=torch.float32)
    y = torch.as_tensor(actions, dtype=torch.float32)
    n_samples = int(len(x))
    generator = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(int(epochs)):
        indices = torch.randperm(n_samples, generator=generator)
        for start in range(0, n_samples, int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            prediction = model(x[batch_idx])
            loss = loss_fn(prediction, y[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def save_opponent_model(path: Path, model: DropoutOpponentModel, metadata: dict[str, Any]) -> None:
    """保存 opponent model 权重和可复原元数据。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_opponent_model(path: Path) -> DropoutOpponentModel:
    """从磁盘加载 dropout opponent model。"""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload["metadata"])
    model = DropoutOpponentModel(
        input_dim=int(metadata["input_dim"]),
        output_dim=int(metadata.get("output_dim", 2)),
        hidden_dim=int(metadata["hidden_dim"]),
        dropout=float(metadata["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


class SamSwitchboard:
    """SAM switchboard，使用不确定性归一化误差检测对手策略变化。"""

    def __init__(
        self,
        *,
        policy_names: list[str],
        opponent_models: dict[str, DropoutOpponentModel],
        initial_policy: str,
        threshold: float,
        decay: float,
        mc_passes: int,
        noise_variance: float,
        online_learning_rate: float,
        warmup_steps: int = 0,
        cooldown_steps: int = 0,
        switch_margin: float = 0.0,
        max_normalized_error: float | None = None,
        online_updates: bool = True,
    ) -> None:
        """创建 SAM switchboard。

        参数中的 warmup/cooldown/margin 不改变 SAM 的核心误差定义，只限制“何时
        允许切换”。这样可以降低 MC dropout 早期不稳定和连续环境短时噪声造成的
        误切换，同时仍保留原文的 normalized running error 检测机制。
        """

        self.policy_names = list(policy_names)
        self.opponent_models = opponent_models
        self.assumed_policy = initial_policy
        self.threshold = float(threshold)
        self.decay = float(decay)
        self.mc_passes = int(mc_passes)
        self.noise_variance = float(noise_variance)
        self.warmup_steps = max(0, int(warmup_steps))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.switch_margin = max(0.0, float(switch_margin))
        self.max_normalized_error = (
            None if max_normalized_error is None else max(0.0, float(max_normalized_error))
        )
        self.online_updates = bool(online_updates) and float(online_learning_rate) > 0.0
        self.running_error = 0.0
        self.step_count = 0
        self.steps_since_switch = self.cooldown_steps
        self._optimizers = {
            name: torch.optim.Adam(model.parameters(), lr=float(online_learning_rate))
            for name, model in opponent_models.items()
        }
        self._loss_fn = nn.MSELoss()

    def predict(self, policy_name: str, observation: np.ndarray) -> SamPrediction:
        """按论文 Algorithm 2 执行 MC dropout 预测。"""

        model = self.opponent_models[policy_name]
        model.train()  # 预测时保持 dropout 开启，近似 Bayesian 后验采样。
        x = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        samples: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(self.mc_passes):
                samples.append(model(x).detach().cpu().numpy().reshape(-1))
        stacked = np.stack(samples, axis=0)
        mean = stacked.mean(axis=0)
        variance = stacked.var(axis=0)
        uncertainty = np.sqrt(variance + self.noise_variance)
        return SamPrediction(mean=mean.astype(np.float32), uncertainty=uncertainty.astype(np.float32))

    def update(self, observation: np.ndarray, observed_action: np.ndarray) -> SamSwitchResult:
        """用当前观测动作更新 running error，并在必要时切换策略。

        这里的 ``observation`` 是动作发生前的观测，``observed_action`` 是下一步
        环境返回的真实入侵者速度。误差计算对应论文公式 ``r += |a-a_hat|/eta``。
        """

        self.step_count += 1
        self.steps_since_switch += 1
        observed = np.asarray(observed_action, dtype=np.float32).reshape(-1)
        prediction = self.predict(self.assumed_policy, observation)
        normalized_error = self._normalized_error(observed, prediction)
        self.running_error += normalized_error

        candidate_errors = self._candidate_errors(observation, observed)
        best_candidate = min(candidate_errors, key=candidate_errors.get)
        best_candidate_error = float(candidate_errors[best_candidate])
        current_candidate_error = float(candidate_errors[self.assumed_policy])
        candidate_margin = current_candidate_error - best_candidate_error
        switch_ready = self._switch_ready(best_candidate, candidate_margin)
        switched = False
        if switch_ready:
            switched = True
            self.assumed_policy = best_candidate
            self.running_error = 0.0
            self.steps_since_switch = 0
        else:
            self.running_error = max(0.0, self.running_error - self.decay)

        # SAM 原文会持续更新当前 opponent model。这里用最新观测对当前假设模型做
        # 一步在线微调，使模型能适配当前连续环境中的轻微分布漂移。
        if self.online_updates:
            self._train_online(self.assumed_policy, observation, observed)
        return SamSwitchResult(
            assumed_policy=self.assumed_policy,
            switched=switched,
            switch_ready=switch_ready,
            normalized_error=float(normalized_error),
            running_error=float(self.running_error),
            best_candidate=best_candidate,
            best_candidate_error=best_candidate_error,
            current_candidate_error=current_candidate_error,
            switch_margin=float(candidate_margin),
            steps_since_switch=int(self.steps_since_switch),
            prediction=prediction,
            candidate_errors=candidate_errors,
        )

    def _switch_ready(self, best_candidate: str, candidate_margin: float) -> bool:
        """判断本步是否允许切换到候选模型。

        SAM 的核心触发条件仍是 running error 超过阈值；额外条件用于过滤连续
        环境中的瞬时噪声：warmup 让 MC dropout 初期预测先稳定，cooldown 防止
        连续来回切换，margin 要求新候选确实比当前模型更匹配。
        """

        if self.running_error < self.threshold:
            return False
        if self.step_count <= self.warmup_steps:
            return False
        if self.steps_since_switch < self.cooldown_steps:
            return False
        if best_candidate == self.assumed_policy:
            return False
        return candidate_margin >= self.switch_margin

    def _candidate_errors(self, observation: np.ndarray, observed_action: np.ndarray) -> dict[str, float]:
        """计算每个候选模型对当前动作的归一化误差，用于切换时选最匹配模型。"""

        return {
            name: self._normalized_error(observed_action, self.predict(name, observation))
            for name in self.policy_names
        }

    def _normalized_error(self, observed_action: np.ndarray, prediction: SamPrediction) -> float:
        """计算 ``|a-a_hat|/eta``，对二维动作取均值保持量纲稳定。"""

        denominator = np.maximum(prediction.uncertainty, 1e-6)
        error = float(np.mean(np.abs(observed_action - prediction.mean) / denominator))
        if self.max_normalized_error is not None:
            return min(error, self.max_normalized_error)
        return error

    def _train_online(self, policy_name: str, observation: np.ndarray, observed_action: np.ndarray) -> None:
        """用单个观测样本在线更新 opponent model。"""

        model = self.opponent_models[policy_name]
        optimizer = self._optimizers[policy_name]
        model.train()
        x = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        y = torch.as_tensor(observed_action, dtype=torch.float32).reshape(1, -1)
        prediction = model(x)
        loss = self._loss_fn(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
