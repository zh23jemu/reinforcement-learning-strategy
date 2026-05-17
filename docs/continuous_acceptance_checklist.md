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

早期 oracle/工程化响应策略选择版本已在 1.5M timesteps、800 episodes、3 个 seed 的确认实验中满足本项目定义的接近论文效果候选标准。SAM 原文检测版本也已完成同规格长训确认，但当前指标尚未达到工程/论文效果标准；若要声明完整论文效果级，仍建议至少满足：

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

1.5M 确认长训结果见 `docs/continuous_confirm_results.md`。当前同时保留早期 oracle 版本和 SAM 原文检测版本，便于区分“理想策略选择上限”和“原文检测方法实测效果”。

早期 oracle/工程化响应策略选择版本结论是：

- 3 组候选参数 x 3 个 seed 共 9 条确认结果均 `engineering_pass=True` 且 `paper_like_pass=True`。
- 推荐主结果为 `interceptor_max_speed=0.030`、`intruder_max_speed=0.016`、`collision_radius=0.08`，平均回报提升约 `50.94`，平均胜率提升约 `23.96` 个百分点。
- 稳定性备选为 `0.026 / 0.018 / 0.10`，平均回报提升约 `40.67`，平均胜率提升约 `19.75` 个百分点。
- 该批结果不应作为 SAM 原文检测方法的最终效果，只适合作为上限参考。

SAM 原文检测版本结论是：

- 3 组候选参数 x 3 个 seed 共 9 条确认结果均已完成训练、评估、分析和聚合，`process_pass=True`。
- MC dropout opponent model、预测不确定性归一化误差、running error、switchboard 切换和 `sam_uncertainty_analysis.png` 可视化均已落地。
- 9 条结果中 `engineering_pass=0`、`paper_like_pass=0`；整体平均回报提升约 `-5.18`，平均胜率提升约 `-2.47` 个百分点，平均切换/响应准确率约 `35.35%`。
- 当前相对最好的参数组是 `0.030 / 0.016 / 0.08`，平均回报提升约 `13.12`，平均胜率提升约 `5.54` 个百分点，但 seed 间仍不稳定。
- 最新 SAM 检测器 300k 调参中，`threshold=14`、`decay=0.01`、`warmup=40`、`cooldown=160`、`switch_margin=0.75`、`noise_variance=0.001` 的组合表现最好：回报提升约 `50.12`，胜率提升约 `22.33` 个百分点，拦截胜率 `23.67%`，切换次数 `1`。该结果仍是单 seed / 300k 筛选结果，需要 1.5M 多 seed 确认。
- 该最优组合已完成 1.5M / 800 episodes / 3 seed 确认，平均回报提升约 `6.29`，平均胜率提升约 `2.00` 个百分点，但 `engineering_pass=0/3`、`paper_like_pass=0/3`。seed 43 和 seed 44 的切换次数为 0，说明检测器仍偏保守。
- 多 seed 小规模调参进一步筛出更稳定组合：`threshold=10`、`decay=0.01`、`warmup=30`、`cooldown=80`、`switch_margin=0.35`、`noise_variance=0.001`。该组合已完成 1.5M / 800 episodes / 3 seed 确认，三 seed 全部正收益且全部触发切换，平均回报提升约 `28.74`，平均胜率提升约 `13.00` 个百分点，平均 SAM 拦截胜率 `21.00%`。但当前验收脚本仍给出 `engineering_pass=0/3`、`paper_like_pass=0/3`，主要因为 SAM 检测版响应准确率约 `32%~33%`，未达到 oracle 风格的 `response_policy_accuracy >= 0.95` 阈值。

## 3. 推荐下一步

- 将 `docs/continuous_confirm_results.md` 中的两批结果表整理进最终报告或论文复现说明，并明确区分 oracle 上限参考与 SAM 检测实测结果。
- 如果继续优化连续场景，优先检查 opponent model 特征、监督样本覆盖和 online update 策略，提高 MC dropout normalized error 对策略变化的识别质量；验收口径上应区分“论文方法链路复现并产生稳定收益”和“接近 oracle 的策略识别准确率”。
- 当前代码已经支持 `sam.feature_mode=geometry`，下一轮实验优先用该模式对比 raw 输入，再决定是否继续扩大 confirm。
- geometry 1.5M confirm 显示 aggressive 组平均响应准确率约 `72.18%`、平均胜率提升约 `17.25` 个百分点，但切换次数 `183~269` 偏高且 seed 43 略负；balanced 组切换较少但准确率回落到约 `35.80%`。
- geometry 窄范围 300k 调参中 6 组参数的 3 个 seed 全部正收益，`th=10 / cd=60 / mg=0.25` 与 `th=10 / cd=80 / mg=0.25` 表现最好；其中 `cd80m25` 已完成 1.5M confirm，平均回报提升约 `32.13`、平均胜率提升约 `14.75` 个百分点、平均响应准确率约 `65.42%`，但 seed 43 仍为负收益。
- seed 43 的 episode 级诊断显示，`detour` 响应策略显著优于 baseline，但 `direct` 和 `attack` 响应策略系统性弱于 baseline。下一阶段应从继续微调 SAM threshold/cooldown/margin，转向补强 continuous response policy 的 `direct/attack` 控制质量，或先做 response policy 与 baseline 的 oracle 对照来确认策略库上限。
- `continuous_oracle_compare_*` 已完成 1.5M / 800 episodes / 3 seed 对照，三 seed 全部 `engineering_pass=True` 且 `paper_like_pass=True`，说明 response policy 库整体上限足够；但 seed 43 下 `direct` 与 `attack` 在 oracle 条件下仍弱于 baseline，下一轮应优先改这两类 response policy 的训练和奖励设计，而不是继续调 SAM 检测参数。
- 已新增 `slurm/response_focus_continuous_plcyf.sbatch`，用于只加训 `direct/attack` response policy 并继续用 oracle 对照评估；默认比较 `3M` 与 `5M` 两档 direct/attack 训练步数，`detour` 和 baseline 保持基础步数。
- 如需做新的参数实验，继续使用 `scripts/analyze_continuous_run.py`、`scripts/diagnose_continuous_run.py` 和 `scripts/aggregate_continuous_sweep.py` 固化诊断指标。
