# 强化学习策略变化检测复现项目

本项目当前优先复现离散动作空间论文 `2406.06500v1.pdf` 中的 OPS-DeMo 框架：在 Predator-Prey 网格环境中，使用 PPO 训练对手策略库与响应策略库，并用运行误差检测对手策略切换。

连续动作空间场景已加入 SAM 风格工程链路：使用 Stable-Baselines3 PPO 训练三类入侵策略对应的响应策略，并使用 MC dropout opponent model 的不确定性归一化误差进行策略切换检测，评估相对普通 PPO baseline 的收益。

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

连续场景 smoke 与 paper-like 配置：

```powershell
.venv\Scripts\python.exe -m rl_strategy.cli train --config configs/continuous_smoke.yaml
.venv\Scripts\python.exe -m rl_strategy.cli evaluate --config configs/continuous_smoke.yaml
.venv\Scripts\python.exe scripts\analyze_continuous_run.py --run-dir runs\continuous_smoke\<运行目录>
```

接近论文设置时使用：

```powershell
.venv\Scripts\python.exe -m rl_strategy.cli train --config configs/continuous_paper_like.yaml
.venv\Scripts\python.exe -m rl_strategy.cli evaluate --config configs/continuous_paper_like.yaml
.venv\Scripts\python.exe scripts\analyze_continuous_run.py --run-dir runs\continuous_paper_like\<运行目录>
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

## 连续模块验收与分析

连续环境验收标准见 `docs/continuous_acceptance_checklist.md`。当前分为三档：

- 流程过关：训练、评估、日志、模型产物、过程数据完整。
- 工程复现过关：响应策略准确率高于 0.95，平均回报和拦截胜率均优于 baseline。
- 接近论文效果：需要多 seed 稳定优于 baseline，且绝对拦截胜率达到明确目标。

1.5M timesteps 确认长训结果见 `docs/continuous_confirm_results.md`。当前同时保留两批结果：早期 oracle/工程化响应策略选择版本作为理想策略选择上限参考；SAM 原文检测版本使用 MC dropout opponent model 预测入侵者动作，并用观测误差除以预测不确定性得到归一化 running error 后触发策略切换。SAM 检测版本已完成长训确认，但当前平均收益尚未稳定优于 baseline，需要继续优化检测阈值、opponent model 和切换策略。

连续评估会在 `runs/continuous_*` 下生成：

- `summary.json`：平均回报、拦截胜率、响应策略准确率、baseline 对照指标。
- `step_trace.csv`：SAM switchboard、MC dropout 预测、不确定性、归一化误差和响应策略库逐步过程数据。
- `baseline_step_trace.csv`：普通 PPO baseline 逐步过程数据。

可使用以下脚本生成连续诊断结果：

```powershell
.venv\Scripts\python.exe scripts\analyze_continuous_run.py --run-dir runs\continuous_paper_like\<运行目录>
```

脚本会写出：

- `analysis/continuous_analysis_summary.json`：核心指标、胜负分布、终止原因、验收判断。
- `analysis/episode_metrics.csv`：episode 级 OPS-DeMo 与 baseline 对比表。
- `analysis/sam_uncertainty_analysis.png`：MC dropout 动作预测不确定性、归一化误差和 running error 可视化。

## Slurm 长训与参数扫描

已按当前集群账号 `gpo-ifv7xx`、本地 CPU 分区 `defq`、QOS `normal` 提供脚本。`defq` 优先使用本地 com 节点，`aws-com` 仅建议在本地 CPU 队列不可用时手动切换：

```bash
# 1. 先训练 PPO 策略库，只需跑一次
sbatch slurm/train_discrete_plcyf.sbatch

# 2. 训练完成后，批量扫描 OPS-DeMo 参数
sbatch slurm/sweep_discrete_plcyf.sbatch

# 3. 所有 array 任务完成后汇总结果
sbatch slurm/aggregate_discrete_plcyf.sbatch
```

默认 sweep 组合为 `alpha={0.80,0.90,0.95,0.99}`、`threshold={4.0,6.0,8.0}`、`seed={42,43}`，共 24 个 array 任务。每个任务会自动执行 `evaluate` 和 `analyze`，最终汇总到 `runs/discrete_sweep_summary.csv`。

连续长训：

```bash
sbatch slurm/train_continuous_plcyf.sbatch
```

训练完成后，在服务器或拉回本地执行连续评估和分析：

```bash
.venv/bin/python -m rl_strategy.cli evaluate --config configs/continuous_paper_like.yaml
.venv/bin/python scripts/analyze_continuous_run.py --run-dir runs/continuous_paper_like/<运行目录>
```

连续参数 sweep：

```bash
# 1. 默认跑 12 组中训参数，每组独立训练、评估并分析
sbatch slurm/sweep_continuous_plcyf.sbatch

# 2. 所有 array 任务完成后汇总结果
sbatch slurm/aggregate_continuous_plcyf.sbatch
```

默认连续 sweep 组合为 `interceptor_max_speed={0.022,0.026,0.030}`、`intruder_max_speed={0.016,0.018}`、`collision_radius={0.08,0.10}`、`seed={42}`，共 12 个 array 任务。每个任务默认训练 `300000` timesteps、评估 `300` episodes；如需临时覆盖，可在提交时使用：

```bash
sbatch --export=ALL,TIMESTEPS=500000,EPISODES=500 slurm/sweep_continuous_plcyf.sbatch
```

提交连续 sweep 前，应先在服务器项目 `.venv` 中完成依赖安装。`sweep_continuous_plcyf.sbatch` 默认不在每个 array task 内执行 `pip install`，避免多个任务并发改写同一个虚拟环境；确需任务内安装时可显式传入 `INSTALL_DEPS=1`，但不建议用于并发 array。

聚合结果会写入 `runs/continuous_sweep_summary.csv`，建议优先查看 `engineering_pass`、`reward_improvement`、`win_rate_improvement` 排名靠前的参数组，再把最优 2-3 组提升到 `1M+` timesteps 做确认。

SAM 检测器调参 sweep：

```bash
# 固定当前相对最优环境参数 0.030 / 0.016 / 0.08，扫描 threshold、cooldown、margin、noise 等检测参数
sbatch slurm/tune_sam_continuous_plcyf.sbatch

# 调参任务完成后同样用连续聚合脚本汇总，默认会扫描 continuous_sam_tune_* 运行目录
sbatch slurm/aggregate_continuous_plcyf.sbatch
```

`tune_sam_continuous_plcyf.sbatch` 默认训练 `300000` timesteps、评估 `300` episodes，目标是先提升 SAM 检测版的 `response_policy_accuracy`、降低误切换，再把最优组合提升到确认长训。

当前调参结果显示，`threshold=14`、`decay=0.01`、`warmup=40`、`cooldown=160`、`switch_margin=0.75`、`noise_variance=0.001` 的组合在 300k / seed 42 下表现最好，可作为下一轮 1.5M 多 seed confirm 的优先候选。

可用下面命令把该组合提升到 1.5M confirm：

```bash
sbatch --array=0-2 --export=ALL,NAME_PREFIX=continuous_sam_confirm,TIMESTEPS=1500000,EPISODES=800,SAM_THRESHOLD=14,SAM_DECAY=0.01,SAM_WARMUP_STEPS=40,SAM_COOLDOWN_STEPS=160,SAM_SWITCH_MARGIN=0.75,SAM_NOISE_VARIANCE=0.001,SAM_MAX_NORMALIZED_ERROR=8,SAM_ONLINE_UPDATES=false slurm/confirm_continuous_plcyf.sbatch
```

这里用 `--array=0-2` 只跑 `0.030 / 0.016 / 0.08` 这一组的 3 个 seed；如需同时确认原 3 组环境参数，可去掉 `--array=0-2`。

该组合的 1.5M / 800 episodes / 3 seed confirm 已完成：平均回报提升约 `6.29`，平均胜率提升约 `2.00` 个百分点，但 3 个 seed 都未达到 `engineering_pass`。主要问题是 seed 43 和 seed 44 没有触发切换，检测器仍偏保守。下一轮应先做多 seed 小规模调参，降低切换门槛后再 confirm。

多 seed 小规模调参：

```bash
# 6 组更容易触发切换的 SAM 参数 x 3 个 seed，共 18 个任务；ISP 当前空闲节点少时默认每次并发 2 个。
sbatch slurm/tune_sam_multiseed_continuous_plcyf.sbatch

# 完成后聚合，默认会收录 continuous_sam_multiseed_tune_* 结果
sbatch slurm/aggregate_continuous_plcyf.sbatch
```

筛选时优先看 `switch_count` 是否在每个 seed 都大于 0，再看 `reward_improvement`、`win_rate_improvement` 和 seed 间最小值；不要只按单个 seed 的最高回报选参数。

当前 300k sweep 中推荐进入 1M+ 确认的候选参数为：

- `interceptor_max_speed=0.030`、`intruder_max_speed=0.016`、`collision_radius=0.08`
- `interceptor_max_speed=0.030`、`intruder_max_speed=0.016`、`collision_radius=0.10`
- `interceptor_max_speed=0.026`、`intruder_max_speed=0.018`、`collision_radius=0.10`

资源充足时，可直接提交 3 组候选 x 3 个 seed 的确认长训：

```bash
sbatch slurm/confirm_continuous_plcyf.sbatch
```

默认确认脚本使用 `TIMESTEPS=1000000`、`EPISODES=500`、`SEEDS={42,43,44}`，并写入 `continuous_confirm_*` 运行目录。需要调整训练步数或评估 episode 时可覆盖：

```bash
sbatch --export=ALL,TIMESTEPS=1500000,EPISODES=800 slurm/confirm_continuous_plcyf.sbatch
```

确认长训完成后仍使用同一个聚合脚本收录结果；默认会同时扫描 `continuous_sweep_*` 和 `continuous_confirm_*` 运行目录，并写入 `runs/continuous_sweep_summary.csv`：

```bash
sbatch slurm/aggregate_continuous_plcyf.sbatch
```
