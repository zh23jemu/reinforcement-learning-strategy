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
.venv\Scripts\python.exe -m rl_strategy.cli evaluate --config configs/discrete_smoke.yaml
.venv\Scripts\python.exe -m rl_strategy.cli analyze --run-dir runs\discrete_smoke\<运行目录>
```

`discrete_smoke.yaml` 只用于验证工程链路，训练步数很短，指标不代表论文效果。
接近论文设置时使用：

```powershell
.venv\Scripts\python.exe -m rl_strategy.cli train --config configs/discrete_paper_like.yaml
.venv\Scripts\python.exe -m rl_strategy.cli evaluate --config configs/discrete_paper_like.yaml
```

## 离散模块模型

训练后会在 `artifacts/discrete/` 下生成：

- `opponent_chase_x.zip` / `opponent_chase_y.zip`：Predator B 的两个 PPO 候选对手策略。
- `response_chase_x.zip` / `response_chase_y.zip`：Predator A 针对不同对手策略训练的 PPO 响应策略。
- `baseline_switching_ppo.zip`：不使用 OPS-DeMo 检测器的普通 PPO 对照组。

## 输出数据

评估会在 `runs/` 下保留过程数据：

- `step_trace.csv`：每一步真实对手策略、假设策略、运行误差、动作、奖励。
- `switch_events.csv`：检测到的策略切换事件。
- `baseline_step_trace.csv`：普通 PPO baseline 的逐步过程数据。
- `summary.json`：平均回报、策略识别准确率等汇总指标。

## 分析图表

`analyze` 命令会在运行目录下创建 `analysis/`，包含：

- `running_errors.png`：两个候选对手策略的 running error 曲线。
- `aop_accuracy.png`：按 episode 汇总的 AOP 识别准确率。
- `reward_comparison.png`：OPS-DeMo + PPO 与 standalone PPO 的回报对比。
- `policy_timeline.png`：真实对手策略与检测器假设策略时间线。
- `episode_metrics.csv` / `analysis_summary.json`：可继续处理的结构化汇总。

## Slurm 长训与参数扫描

已按当前集群账号 `mfk-gsr2`、分区 `plcyf-com`、QOS `normal` 提供脚本：

```bash
# 1. 先训练 PPO 策略库，只需跑一次
sbatch slurm/train_discrete_plcyf.sbatch

# 2. 训练完成后，批量扫描 OPS-DeMo 参数
sbatch slurm/sweep_discrete_plcyf.sbatch

# 3. 所有 array 任务完成后汇总结果
sbatch slurm/aggregate_discrete_plcyf.sbatch
```

默认 sweep 组合为 `alpha={0.80,0.90,0.95,0.99}`、`threshold={4.0,6.0,8.0}`、`seed={42,43}`，共 24 个 array 任务。每个任务会自动执行 `evaluate` 和 `analyze`，最终汇总到 `runs/discrete_sweep_summary.csv`。
