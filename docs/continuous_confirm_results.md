# 连续场景 1.5M 确认长训结果

本文档汇总连续二维拦截捕猎场景在 `1.5M` timesteps、`800` episodes、3 个 seed 下的确认实验结果。结果来自 `runs/continuous_sweep_summary.csv` 中的 `continuous_confirm_*` 行。

当前 CSV 同时保留两批结果：

- 早期 oracle/工程化响应策略选择版本：运行时间为 `20260513_09*` / `20260513_10*`，用于记录接入 SAM 原文检测前的上限参考。
- SAM 原文检测版本：运行时间为 `20260513_234*`，使用 MC dropout opponent model 预测动作，并用观测误差除以预测不确定性形成 normalized running error，再由 switchboard 触发响应策略切换。

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

## SAM 检测版本结果

SAM 检测版本的 9 条确认结果均已完成训练、评估、分析和聚合，`process_pass=True`。但该批结果未达到当前定义的 `engineering_pass` / `paper_like_pass`，主要因为 MC dropout 检测后的响应策略切换准确率约为 35%，明显低于早期 oracle 版本，导致整体收益不稳定。

| 参数组 | 平均回报提升 | 最小回报提升 | 平均胜率提升 | 最小胜率提升 | SAM 平均拦截胜率 | baseline 平均拦截胜率 | 平均切换/响应准确率 | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `0.030 / 0.016 / 0.08` | 13.12 | -27.30 | 5.54 pp | -12.37 pp | 13.54% | 8.00% | 36.05% | 相对最优，但 seed 间不稳定 |
| `0.026 / 0.018 / 0.10` | -0.86 | -33.36 | -0.67 pp | -15.88 pp | 9.58% | 10.25% | 35.82% | 基本持平 baseline |
| `0.030 / 0.016 / 0.10` | -27.79 | -34.19 | -12.29 pp | -15.00 pp | 7.79% | 20.08% | 34.19% | 弱于 baseline |

总体均值：

- 平均 episode 回报：SAM `-84.49`，baseline `-79.31`，平均回报提升 `-5.18`。
- 平均拦截胜率：SAM `10.31%`，baseline `12.78%`，平均胜率提升 `-2.47` 个百分点。
- 平均切换/响应准确率：`35.35%`。
- 9 条结果中 `engineering_pass=0`、`paper_like_pass=0`。

## SAM 检测版本单次运行明细

| 实验名 | seed | 回报提升 | 胜率提升 | SAM 拦截胜率 | baseline 拦截胜率 | 响应准确率 |
|---|---:|---:|---:|---:|---:|---:|
| `continuous_confirm_is0p03_us0p016_cr0p08_ts1500000_s42` | 42 | 31.09 | 13.88 pp | 13.88% | 0.00% | 32.99% |
| `continuous_confirm_is0p03_us0p016_cr0p08_ts1500000_s43` | 43 | -27.30 | -12.37 pp | 11.12% | 23.50% | 36.17% |
| `continuous_confirm_is0p03_us0p016_cr0p08_ts1500000_s44` | 44 | 35.58 | 15.12 pp | 15.62% | 0.50% | 38.99% |
| `continuous_confirm_is0p03_us0p016_cr0p1_ts1500000_s42` | 42 | -34.19 | -15.00 pp | 5.75% | 20.75% | 33.25% |
| `continuous_confirm_is0p03_us0p016_cr0p1_ts1500000_s43` | 43 | -20.24 | -8.75 pp | 7.12% | 15.88% | 33.88% |
| `continuous_confirm_is0p03_us0p016_cr0p1_ts1500000_s44` | 44 | -28.94 | -13.12 pp | 10.50% | 23.62% | 35.44% |
| `continuous_confirm_is0p026_us0p018_cr0p1_ts1500000_s42` | 42 | 20.87 | 9.25 pp | 12.88% | 3.62% | 34.26% |
| `continuous_confirm_is0p026_us0p018_cr0p1_ts1500000_s43` | 43 | -33.36 | -15.88 pp | 5.62% | 21.50% | 35.72% |
| `continuous_confirm_is0p026_us0p018_cr0p1_ts1500000_s44` | 44 | 9.89 | 4.62 pp | 10.25% | 5.62% | 37.49% |

## 早期 oracle 版本参考

早期 oracle/工程化响应策略选择版本在同样的 `1.5M` timesteps、`800` episodes、3 个 seed 设置下，9 条确认结果均达到 `engineering_pass=True` 和 `paper_like_pass=True`。该批结果可作为“理想策略选择/上限参考”，但不应作为 SAM 原文检测方法的最终复现结果。

| 参数组 | 平均回报提升 | 最小回报提升 | 平均胜率提升 | 最小胜率提升 | OPS-DeMo 平均拦截胜率 | baseline 平均拦截胜率 | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `0.030 / 0.016 / 0.08` | 50.94 | 13.53 | 23.96 pp | 7.63 pp | 31.96% | 8.00% | 早期主结果 |
| `0.026 / 0.018 / 0.10` | 40.67 | 18.71 | 19.75 pp | 9.50 pp | 30.00% | 10.25% | 早期稳定性备选 |
| `0.030 / 0.016 / 0.10` | 23.92 | 14.82 | 12.50 pp | 7.88 pp | 32.58% | 20.08% | 早期绝对胜率最高 |

## 结论边界

当前可以如实表述为：连续环境、SAM 原文检测机制、MC dropout 不确定性归一化误差、switchboard 切换、长训评估和不确定性可视化都已经实现并完成确认实验；但 SAM 检测版本的实测收益尚未达到早期 oracle 上限，也未达到当前工程/论文效果验收标准。

## SAM 检测器调参结果

2026-05-14 已完成一轮 `continuous_sam_tune_*` 小规模调参。该轮固定当前相对最优环境参数 `0.030 / 0.016 / 0.08`，训练 `300000` timesteps、评估 `300` episodes、seed `42`，扫描 12 组 threshold、decay、warmup、cooldown、margin 和 noise 参数。

调参结论：

- 12 条调参结果均完成训练和评估，重新分析后全部写入 `runs/continuous_sweep_summary.csv`。
- 平均回报提升约 `20.93`，平均胜率提升约 `9.03` 个百分点，平均响应准确率约 `34.37%`。
- 低阈值组合切换较多，响应准确率略高但收益不最高；高阈值组合 0 次切换，效果退化到弱于 baseline。
- 当前最优组合为 `threshold=14`、`decay=0.01`、`warmup=40`、`cooldown=160`、`switch_margin=0.75`、`noise_variance=0.001`、`max_normalized_error=8`、`online_updates=false`，300k 下回报提升约 `50.12`，胜率提升约 `22.33` 个百分点，拦截胜率 `23.67%`，切换次数 `1`。
- 等价表现的备选组合为 `threshold=16`、`decay=0.02`、`warmup=40`、`cooldown=120`、`switch_margin=1.0`、`noise_variance=0.0005`，同样达到回报提升约 `50.12`、胜率提升约 `22.33` 个百分点。

下一步若要把连续场景推进到可对外宣称“效果复现”，建议优先把上述 1-2 个组合提升到 1.5M / 800 episodes / 多 seed 确认长训。

## SAM 调参最优组合确认结果

2026-05-14 已将调参最优组合提升到 `1.5M` timesteps、`800` episodes、3 个 seed 确认长训，实验名前缀为 `continuous_sam_confirm_*`。确认参数为：

- 环境参数：`interceptor_max_speed=0.030`、`intruder_max_speed=0.016`、`collision_radius=0.08`
- SAM 参数：`threshold=14`、`decay=0.01`、`warmup=40`、`cooldown=160`、`switch_margin=0.75`、`noise_variance=0.001`、`max_normalized_error=8`、`online_updates=false`

确认结果：

| seed | 回报提升 | 胜率提升 | SAM 拦截胜率 | baseline 拦截胜率 | 响应准确率 | 切换次数 | 判断 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 61.50 | 27.88 pp | 27.88% | 0.00% | 33.46% | 1 | 单 seed 效果好 |
| 43 | -48.82 | -22.50 pp | 1.00% | 23.50% | 33.46% | 0 | 未触发切换，弱于 baseline |
| 44 | 6.19 | 0.62 pp | 1.12% | 0.50% | 33.36% | 0 | 小幅优于 baseline |

三 seed 平均：

- 平均回报提升：`6.29`
- 平均胜率提升：`2.00` 个百分点
- 平均响应准确率：`33.43%`
- `engineering_pass=0/3`，`paper_like_pass=0/3`

结论：该参数组合在 300k 单 seed 调参中表现突出，但 1.5M 多 seed 确认后不稳定。主要失败模式是检测器过于保守，seed 43 和 seed 44 都没有触发策略切换，导致响应策略长期停留在初始假设。后续不宜直接宣称该组合完成效果复现，应继续降低切换门槛或引入多 seed 小规模调参。

## 下一轮多 seed 小规模调参

已新增 `slurm/tune_sam_multiseed_continuous_plcyf.sbatch`，用于验证更容易触发切换的 SAM 参数。该脚本固定环境参数 `0.030 / 0.016 / 0.08`，扫描 6 组更低 threshold/cooldown/margin 的组合，并覆盖 seeds `42/43/44`，共 18 个任务。默认仍使用 `300000` timesteps、`300` episodes。

建议筛选标准：

- 每个 seed 的 `switch_count > 0`，避免 confirm 阶段再次出现完全不切换。
- 平均 `reward_improvement` 和 `win_rate_improvement` 为正。
- 最差 seed 不出现大幅负收益，优先选择稳定性而不是单 seed 最高收益。

服务器提交命令：

```bash
sbatch slurm/tune_sam_multiseed_continuous_plcyf.sbatch
sbatch slurm/aggregate_continuous_plcyf.sbatch
```
