# 强化学习策略变化检测复现项目

本项目当前优先复现离散动作空间论文 `2406.06500v1.pdf` 中的 OPS-DeMo 框架：在 Predator-Prey 网格环境中，使用 PPO 训练对手策略库与响应策略库，并用运行误差检测对手策略切换。

连续动作空间 SAM 复现会作为后续模块加入，当前不会影响离散环境代码。

## 快速开始

```powershell
.venv\Scripts\python.exe -m pip install -e .[dev]
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m rl_strategy.cli evaluate --config configs/discrete_smoke.yaml
```

如果需要重新训练 PPO 策略：

```powershell
.venv\Scripts\python.exe -m rl_strategy.cli train --config configs/discrete_smoke.yaml
```

## 输出数据

评估会在 `runs/` 下保留过程数据：

- `step_trace.csv`：每一步真实对手策略、假设策略、运行误差、动作、奖励。
- `switch_events.csv`：检测到的策略切换事件。
- `summary.json`：平均回报、策略识别准确率等汇总指标。

