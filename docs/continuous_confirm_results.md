# 连续场景 1.5M 确认长训结果

本文档汇总连续二维拦截捕猎场景在 1.5M timesteps、800 episodes、3 个 seed 下的确认实验结果。结果来自 `runs/continuous_sweep_summary.csv` 中的 `continuous_confirm_*` 行。

注意：该结果对应早期 oracle/工程化响应策略选择版本。当前代码已补入 SAM 原文方法中的 MC dropout opponent model、预测不确定性归一化误差和 switchboard 检测；接入 SAM 检测后的长训结果需要重新提交 Slurm 作业生成。

## 实验设置

- 训练脚本：`slurm/confirm_continuous_plcyf.sbatch`
- 聚合脚本：`slurm/aggregate_continuous_plcyf.sbatch`
- 总训练步数：`1500000`
- 评估 episodes：`800`
- seeds：`42`、`43`、`44`
- 候选参数：
  - `interceptor_max_speed=0.030`、`intruder_max_speed=0.016`、`collision_radius=0.08`
  - `interceptor_max_speed=0.030`、`intruder_max_speed=0.016`、`collision_radius=0.10`
  - `interceptor_max_speed=0.026`、`intruder_max_speed=0.018`、`collision_radius=0.10`

## 聚合结果

三组候选参数在 3 个 seed 上均达到 `engineering_pass=True` 和 `paper_like_pass=True`。

| 参数组 | 平均回报提升 | 最小回报提升 | 平均胜率提升 | 最小胜率提升 | OPS-DeMo 平均拦截胜率 | baseline 平均拦截胜率 | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `0.030 / 0.016 / 0.08` | 50.94 | 13.53 | 23.96 pp | 7.63 pp | 31.96% | 8.00% | 推荐主结果 |
| `0.026 / 0.018 / 0.10` | 40.67 | 18.71 | 19.75 pp | 9.50 pp | 30.00% | 10.25% | 稳定性备选 |
| `0.030 / 0.016 / 0.10` | 23.92 | 14.82 | 12.50 pp | 7.88 pp | 32.58% | 20.08% | 绝对胜率最高，但相对提升较弱 |

说明：

- `0.030 / 0.016 / 0.08` 的平均回报提升和平均胜率提升最高，适合作为连续场景当前主结果。
- `0.026 / 0.018 / 0.10` 的最小回报提升和最小胜率提升更稳，且覆盖更快入侵者设置，可作为稳定性备选。
- `0.030 / 0.016 / 0.10` 的 OPS-DeMo 绝对拦截胜率最高，但 baseline 也明显更强，因此相对提升弱于另外两组。

## 单次运行明细

| 实验名 | seed | 回报提升 | 胜率提升 | OPS-DeMo 拦截胜率 | baseline 拦截胜率 |
|---|---:|---:|---:|---:|---:|
| `continuous_confirm_is0p03_us0p016_cr0p08_ts1500000_s42` | 42 | 69.68 | 32.50 pp | 32.50% | 0.00% |
| `continuous_confirm_is0p03_us0p016_cr0p08_ts1500000_s43` | 43 | 13.53 | 7.63 pp | 31.13% | 23.50% |
| `continuous_confirm_is0p03_us0p016_cr0p08_ts1500000_s44` | 44 | 69.62 | 31.75 pp | 32.25% | 0.50% |
| `continuous_confirm_is0p03_us0p016_cr0p1_ts1500000_s42` | 42 | 23.18 | 12.50 pp | 33.25% | 20.75% |
| `continuous_confirm_is0p03_us0p016_cr0p1_ts1500000_s43` | 43 | 33.76 | 17.13 pp | 33.00% | 15.88% |
| `continuous_confirm_is0p03_us0p016_cr0p1_ts1500000_s44` | 44 | 14.82 | 7.88 pp | 31.50% | 23.63% |
| `continuous_confirm_is0p026_us0p018_cr0p1_ts1500000_s42` | 42 | 56.35 | 26.75 pp | 30.38% | 3.63% |
| `continuous_confirm_is0p026_us0p018_cr0p1_ts1500000_s43` | 43 | 18.71 | 9.50 pp | 31.00% | 21.50% |
| `continuous_confirm_is0p026_us0p018_cr0p1_ts1500000_s44` | 44 | 46.94 | 23.00 pp | 28.63% | 5.63% |

## 结论边界

该批早期结果可以表述为：在本项目二维连续拦截环境中，工程化响应策略库在 1.5M timesteps、多 seed 确认下稳定优于单一 PPO baseline。

仍需谨慎的是：绝对拦截胜率约为 30%-33%，且该批结果尚未使用当前 SAM 原文检测方法，还不适合直接宣称完整达到原论文效果级。更稳妥的表述是“早期工程化响应策略选择版本达到连续场景工程复现，并满足当时定义的接近论文效果候选标准”。
