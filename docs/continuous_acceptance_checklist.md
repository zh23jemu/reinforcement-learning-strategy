# 连续环境复现验收清单

本文档用于验收连续二维拦截捕猎场景的复现进度。连续场景目前采用 Stable-Baselines3 PPO，不手写 PPO；默认目标是“工程可复现”，即训练、评估、诊断链路稳定可重复，`full/paper_like` 配置再逐步逼近论文效果。

## 1. 验收分级

### A. 流程过关

满足以下条件即可认为连续场景链路已经跑通：

- `configs/continuous_smoke.yaml` 可以在本地完成 `train` 和 `evaluate`。
- `configs/continuous_paper_like.yaml` 可以在 Slurm 上完成训练，`slurm/logs/*.err` 无 Python traceback 或 Slurm 异常。
- `artifacts/continuous/` 下生成响应策略、baseline 和 SAM opponent model：`response_direct.zip`、`response_detour.zip`、`response_attack.zip`、`baseline_switching_ppo.zip`、`opponent_model_*.zip`。
- `runs/continuous_paper_like/<timestamp>/` 下生成 `summary.json`、`config.json`、`step_trace.csv`、`baseline_step_trace.csv`。
- `summary.json` 中 `episodes` 与配置一致，`response_policy_accuracy`、`mean_episode_reward`、`baseline_mean_episode_reward` 等核心字段存在且为有效数值。

### B. 工程复现过关

满足流程过关后，再满足以下指标，可认为连续场景达到工程复现粒度：

- `response_policy_accuracy >= 0.95`，说明入侵策略识别与响应策略选择机制稳定。
- `mean_episode_reward > baseline_mean_episode_reward`，说明 OPS-DeMo 响应策略库相对单一 PPO baseline 有正向收益。
- `interceptor_win_rate > baseline_interceptor_win_rate`，说明拦截成功率相对 baseline 有提升。
- `step_trace.csv` 中三类入侵策略 `direct/detour/attack` 都出现，且包含 SAM 的 MC dropout 预测、不确定性、归一化误差、running error 和策略切换记录。
- 训练日志显示 4 组 PPO 都完成目标步数，不能只依赖残缺或旧模型完成评估。

### C. 接近论文效果

早期 oracle/工程化响应策略选择版本已在 1.5M timesteps、800 episodes、3 个 seed 的确认实验中满足本项目定义的接近论文效果候选标准。当前代码已接入 SAM 原文检测方法，接入后的长训结果需要重新生成；若要声明完整论文效果级，仍建议至少满足：

- 多 seed 评估下 OPS-DeMo 平均回报稳定优于 baseline，且提升幅度不依赖单次随机结果。
- 拦截胜率达到一个明确、可复验的绝对水平，例如 `interceptor_win_rate >= 0.20`，或达到论文/需求文档中给出的目标范围。
- 三类入侵策略下均有可解释的拦截收益，不只是某一类策略拉高整体均值。
- 参数 sweep 证明结果对 `interceptor_max_speed`、`intruder_max_speed`、`collision_radius`、奖励尺度等关键环境参数不过度敏感。

## 2. 当前连续长训结果定位

早期 paper-like 长训结果位于：

- 训练日志：`slurm/logs/cont-train-32945557.out`、`slurm/logs/cont-train-32945557.err`
- 模型产物：`artifacts/continuous/`
- 评估数据：`runs/continuous_paper_like/20260513_032143/`

早期评估指标：

| 指标 | OPS-DeMo | baseline | 判断 |
|---|---:|---:|---|
| 平均 episode 回报 | -98.87 | -106.04 | OPS-DeMo 提升约 7.17 |
| 拦截胜率 | 3.4% | 0.5% | OPS-DeMo 相对提升，但绝对值偏低 |
| 响应策略准确率 | 99.38% | 不适用 | 策略识别/选择机制正常 |

早期 1.5M 确认长训结果见 `docs/continuous_confirm_results.md`。该批结果对应接入 SAM 检测前的工程化响应策略选择版本，结论是：

- 3 组候选参数 x 3 个 seed 共 9 条确认结果均 `engineering_pass=True` 且 `paper_like_pass=True`。
- 推荐主结果为 `interceptor_max_speed=0.030`、`intruder_max_speed=0.016`、`collision_radius=0.08`，平均回报提升约 `50.94`，平均胜率提升约 `23.96` 个百分点。
- 稳定性备选为 `0.026 / 0.018 / 0.10`，平均回报提升约 `40.67`，平均胜率提升约 `19.75` 个百分点。
- 当前 SAM 检测代码已完成 smoke 验证，但需要重新跑长训后，才能把上述指标更新为 SAM 原文检测方法下的最终结论。

## 3. 推荐下一步

- 将 `docs/continuous_confirm_results.md` 中的结果表整理进最终报告或论文复现说明。
- 如果继续优化连续场景，优先重新跑接入 SAM 检测后的长训确认结果，再处理绝对拦截胜率、baseline 方差和奖励/终止设计。
- 如需做新的参数实验，继续使用 `scripts/analyze_continuous_run.py` 和 `scripts/aggregate_continuous_sweep.py` 固化诊断指标。
