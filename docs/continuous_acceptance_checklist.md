# 连续环境复现验收清单

本文档用于验收连续二维拦截捕猎场景的复现进度。连续场景目前采用 Stable-Baselines3 PPO，不手写 PPO；默认目标是“工程可复现”，即训练、评估、诊断链路稳定可重复，`full/paper_like` 配置再逐步逼近论文效果。

## 1. 验收分级

### A. 流程过关

满足以下条件即可认为连续场景链路已经跑通：

- `configs/continuous_smoke.yaml` 可以在本地完成 `train` 和 `evaluate`。
- `configs/continuous_paper_like.yaml` 可以在 Slurm 上完成训练，`slurm/logs/*.err` 无 Python traceback 或 Slurm 异常。
- `artifacts/continuous/` 下生成 4 个模型文件：`response_direct.zip`、`response_detour.zip`、`response_attack.zip`、`baseline_switching_ppo.zip`。
- `runs/continuous_paper_like/<timestamp>/` 下生成 `summary.json`、`config.json`、`step_trace.csv`、`baseline_step_trace.csv`。
- `summary.json` 中 `episodes` 与配置一致，`response_policy_accuracy`、`mean_episode_reward`、`baseline_mean_episode_reward` 等核心字段存在且为有效数值。

### B. 工程复现过关

满足流程过关后，再满足以下指标，可认为连续场景达到工程复现粒度：

- `response_policy_accuracy >= 0.95`，说明入侵策略识别与响应策略选择机制稳定。
- `mean_episode_reward > baseline_mean_episode_reward`，说明 OPS-DeMo 响应策略库相对单一 PPO baseline 有正向收益。
- `interceptor_win_rate > baseline_interceptor_win_rate`，说明拦截成功率相对 baseline 有提升。
- `step_trace.csv` 中三类入侵策略 `direct/detour/attack` 都出现，且各策略响应准确率无明显单类崩溃。
- 训练日志显示 4 组 PPO 都完成目标步数，不能只依赖残缺或旧模型完成评估。

### C. 接近论文效果

连续场景目前尚未达到此档。后续若要声明接近论文效果，建议至少满足：

- 多 seed 评估下 OPS-DeMo 平均回报稳定优于 baseline，且提升幅度不依赖单次随机结果。
- 拦截胜率达到一个明确、可复验的绝对水平，例如 `interceptor_win_rate >= 0.20`，或达到论文/需求文档中给出的目标范围。
- 三类入侵策略下均有可解释的拦截收益，不只是某一类策略拉高整体均值。
- 参数 sweep 证明结果对 `interceptor_max_speed`、`intruder_max_speed`、`collision_radius`、奖励尺度等关键环境参数不过度敏感。

## 2. 当前连续长训结果定位

当前已拉回的长训结果位于：

- 训练日志：`slurm/logs/cont-train-32945557.out`、`slurm/logs/cont-train-32945557.err`
- 模型产物：`artifacts/continuous/`
- 评估数据：`runs/continuous_paper_like/20260513_032143/`

当前评估指标：

| 指标 | OPS-DeMo | baseline | 判断 |
|---|---:|---:|---|
| 平均 episode 回报 | -98.87 | -106.04 | OPS-DeMo 提升约 7.17 |
| 拦截胜率 | 3.4% | 0.5% | OPS-DeMo 相对提升，但绝对值偏低 |
| 响应策略准确率 | 99.38% | 不适用 | 策略识别/选择机制正常 |

因此当前结论是：连续场景达到“流程过关”，并且核心机制已满足“工程复现”的主要方向；但由于绝对拦截胜率仍低，暂不应声明达到论文效果级。

## 3. 推荐下一步

- 使用 `scripts/analyze_continuous_run.py` 固化每次连续评估的诊断指标。
- 做连续参数 sweep，优先扫描 `interceptor_max_speed`、`intruder_max_speed`、`collision_radius`、终止奖励/惩罚、`total_timesteps`。
- 保留当前结果作为 baseline checkpoint，后续所有优化都和 `runs/continuous_paper_like/20260513_032143/` 对比。
