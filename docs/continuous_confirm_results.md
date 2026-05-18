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

## SAM 多 seed 小规模调参结果

2026-05-15 已 pull 回 `continuous_sam_multiseed_tune_*` 结果，并重新聚合到 `runs/continuous_sweep_summary.csv`。该轮 18 个任务均完成，`cont-sam-ms-33102641_*.err` 均为空。

表现最好的两组：

| 参数组 | seeds 切换次数 | 平均回报提升 | 最小回报提升 | 平均胜率提升 | 最小胜率提升 | 平均响应准确率 | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `threshold=10 / decay=0.01 / warmup=30 / cooldown=80 / margin=0.35 / noise=0.001` | `20 / 1 / 5` | 26.79 | 9.83 | 11.89 pp | 3.33 pp | 34.68% | 当前首选 |
| `threshold=12 / decay=0.02 / warmup=30 / cooldown=80 / margin=0.50 / noise=0.0005` | `11 / 1 / 3` | 23.69 | 9.78 | 10.44 pp | 3.33 pp | 34.16% | 稳定备选 |

首选组合分 seed 明细：

| seed | 回报提升 | 胜率提升 | SAM 拦截胜率 | baseline 拦截胜率 | 响应准确率 | 切换次数 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 40.32 | 18.33 pp | 19.67% | 1.33% | 37.29% | 20 |
| 43 | 30.23 | 14.00 pp | 21.00% | 7.00% | 32.90% | 1 |
| 44 | 9.83 | 3.33 pp | 3.33% | 0.00% | 33.84% | 5 |

结论：多 seed 小规模调参已经解决上一轮 confirm 的“0 次切换”问题。当前首选组合不是单 seed 最高收益，但三 seed 全部正收益、全部触发切换，适合进入下一轮 1.5M / 800 episodes / 3 seed confirm。

推荐 confirm 命令：

```bash
sbatch --array=0-2 --export=ALL,NAME_PREFIX=continuous_sam_confirm_msbest,TIMESTEPS=1500000,EPISODES=800,SAM_THRESHOLD=10,SAM_DECAY=0.01,SAM_WARMUP_STEPS=30,SAM_COOLDOWN_STEPS=80,SAM_SWITCH_MARGIN=0.35,SAM_NOISE_VARIANCE=0.001,SAM_MAX_NORMALIZED_ERROR=8,SAM_ONLINE_UPDATES=false slurm/confirm_continuous_plcyf.sbatch
```

## SAM msbest 1.5M 确认长训结果

2026-05-15 已 pull 回 `continuous_sam_confirm_msbest_*` 结果，并重新聚合到 `runs/continuous_sweep_summary.csv`。该轮使用多 seed 小规模调参筛出的首选参数：`threshold=10`、`decay=0.01`、`warmup=30`、`cooldown=80`、`switch_margin=0.35`、`noise_variance=0.001`、`max_normalized_error=8`、`online_updates=false`。

| seed | 回报提升 | 胜率提升 | SAM 拦截胜率 | baseline 拦截胜率 | 响应准确率 | 切换次数 | 验收 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 30.37 | 13.88 pp | 13.88% | 0.00% | 33.34% | 43 | `process_pass=True`，`engineering_pass=False` |
| 43 | 4.81 | 2.50 pp | 26.00% | 23.50% | 33.36% | 1 | `process_pass=True`，`engineering_pass=False` |
| 44 | 51.06 | 22.63 pp | 23.13% | 0.50% | 32.01% | 28 | `process_pass=True`，`engineering_pass=False` |

平均回报提升约 `28.74`，平均胜率提升约 `13.00` 个百分点，平均 SAM 拦截胜率 `21.00%`，平均 baseline 拦截胜率 `8.00%`。三个 seed 全部正收益、全部触发切换，说明 SAM 原文检测链路在连续拦截环境下已经跑通，并且相对 baseline 有稳定收益。

当前仍未标记为 `engineering_pass` / `paper_like_pass`，主要原因是分析脚本沿用了 oracle 风格的严格门槛：`response_policy_accuracy >= 0.95`。SAM 原文式 MC dropout 检测在该连续动作环境中的响应准确率约 `32%~33%`，明显低于直接使用真实策略标签的 oracle 版本。因此这批结果适合表述为“原文方法链路已复现，连续环境下有稳定收益”，但还不适合表述为“完全达到论文理想检测效果”。

后续优化方向：优先提高 opponent model 对连续动作的建模质量，扩展监督样本覆盖，检查归一化误差中的不确定性尺度，并继续围绕 threshold、cooldown、switch margin 与 online update 做稳定性实验。

已新增 `sam.feature_mode=geometry` 作为下一轮优化变量。该模式不改变 SAM 的 MC dropout 和 normalized error 检测公式，只是在 opponent model 输入中追加连续拦截场景的相对几何特征，便于模型区分 direct、detour、attack 三类动作模式。建议先运行 `continuous_sam_geometry_tune` 小规模多 seed 对比，再决定是否进入 1.5M confirm。

## SAM geometry 1.5M 确认长训结果

2026-05-16 已 pull 回 `continuous_sam_geometry_confirm_*` 结果，并修复聚合脚本默认前缀，使 `continuous_sam_geometry_confirm_*` 自动收录到 `runs/continuous_sweep_summary.csv`。本轮共确认两组：

| 参数组 | seed | 回报提升 | 胜率提升 | SAM 拦截胜率 | baseline 拦截胜率 | 响应准确率 | 切换次数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| aggressive `th=8/dc=0.02/wu=20/cd=40/mg=0.2/noise=0.0005` | 42 | 59.60 | 27.50 pp | 27.50% | 0.00% | 68.38% | 243 |
| aggressive `th=8/dc=0.02/wu=20/cd=40/mg=0.2/noise=0.0005` | 43 | -2.28 | -0.25 pp | 23.25% | 23.50% | 73.23% | 269 |
| aggressive `th=8/dc=0.02/wu=20/cd=40/mg=0.2/noise=0.0005` | 44 | 55.22 | 24.50 pp | 25.00% | 0.50% | 74.92% | 183 |
| balanced `th=12/dc=0.02/wu=30/cd=80/mg=0.5/noise=0.0005` | 42 | 23.26 | 10.62 pp | 10.62% | 0.00% | 38.85% | 61 |
| balanced `th=12/dc=0.02/wu=30/cd=80/mg=0.5/noise=0.0005` | 43 | -4.29 | -1.50 pp | 22.00% | 23.50% | 33.50% | 29 |
| balanced `th=12/dc=0.02/wu=30/cd=80/mg=0.5/noise=0.0005` | 44 | 41.51 | 18.12 pp | 18.62% | 0.50% | 35.05% | 47 |

aggressive 组平均回报提升约 `37.52`，平均胜率提升约 `17.25` 个百分点，平均响应准确率 `72.18%`，明显优于 raw / msbest 的约 `33%`，但切换次数 `183~269` 偏高，且 seed 43 略低于 baseline。balanced 组切换次数较合理，但准确率回落到 `35.80%`，收益也弱于 aggressive。下一轮建议搜索两者之间的中间参数，例如 `threshold=9~10`、`cooldown=60~80`、`switch_margin=0.25~0.35`，目标是在保持 geometry 高准确率的同时降低过度切换。

## SAM geometry 窄范围调参与 cd80m25 确认结果

2026-05-17 已 pull 回 `continuous_sam_geometry_narrow_tune_*` 与 `continuous_sam_geometry_confirm_cd80m25_*` 结果，并重新聚合到 `runs/continuous_sweep_summary.csv`。窄范围调参固定环境参数 `0.030 / 0.016 / 0.08`，使用 `feature_mode=geometry`、`decay=0.02`、`warmup=20`、`noise_variance=0.0005`、`max_normalized_error=8`、`online_updates=false`，主要扫描 `cooldown`、`switch_margin` 和 `threshold`。

窄范围 300k 调参中，6 组参数在 3 个 seed 下全部保持正收益，但收益随 `cooldown` 与 `switch_margin` 增大而下降。前两组结果如下：

| 参数组 | 平均回报提升 | 最小回报提升 | 平均胜率提升 | 平均响应准确率 | 平均切换次数 | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `th=10 / cd=60 / mg=0.25` | 33.43 | 14.36 | 15.11 pp | 60.95% | 81.3 | 300k 最优，等价于 midbest 短训复核 |
| `th=10 / cd=80 / mg=0.25` | 32.09 | 12.38 | 14.44 pp | 58.09% | 74.0 | 切换略少，提升到 confirm 验证 |

`cd80m25` 1.5M / 800 episodes / 3 seed 确认参数为 `threshold=10`、`decay=0.02`、`warmup=20`、`cooldown=80`、`switch_margin=0.25`、`noise_variance=0.0005`、`feature_mode=geometry`、`online_updates=false`。

| seed | 回报提升 | 胜率提升 | SAM 拦截胜率 | baseline 拦截胜率 | 响应准确率 | 切换次数 | 判断 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 53.14 | 24.62 pp | 24.62% | 0.00% | 63.79% | 175 | 明显优于 baseline |
| 43 | -13.04 | -5.37 pp | 18.12% | 23.50% | 59.41% | 197 | 仍弱于 baseline |
| 44 | 56.30 | 25.00 pp | 25.50% | 0.50% | 73.06% | 159 | 明显优于 baseline |

三 seed 平均回报提升约 `32.13`，平均胜率提升约 `14.75` 个百分点，平均响应准确率约 `65.42%`，平均切换次数 `177.0`。相比 midbest，`cd80m25` 降低了平均切换次数，但没有解决 seed 43 负收益；继续只调 SAM threshold/cooldown/margin 的边际收益已经变小。

## seed 43 诊断结论

已新增 `scripts/diagnose_continuous_run.py`，用于在已有 `analysis/episode_metrics.csv` 与逐步轨迹基础上，输出 episode 级胜负组合、最终响应策略拆分、终止原因差异和最大 reward gap。诊断产物写入各 run 的 `analysis/` 目录，包括 `continuous_run_diagnosis.json`、`episode_outcome_breakdown.csv`、`policy_outcome_breakdown.csv`、`reason_delta.csv` 和 `largest_reward_gaps.csv`。

对 aggressive、balanced、midbest、cd80m25 四组 seed 43 confirm 的诊断显示，geometry 特征确实提高了 SAM 检测/选择准确率，但 seed 43 的主要损失不是继续缺少切换，而是响应策略在 `direct` 和 `attack` 场景的控制质量弱于 baseline。以 `cd80m25` seed 43 为例：

- episode 胜负组合为 `baseline_only_win=142`、`ops_only_win=99`、`both_win=46`、`both_lose=513`；baseline 胜场 `188/800`，OPS/SAM 胜场 `145/800`。
- 按最终响应策略拆分，`detour` 表现很强：OPS/SAM 胜率约 `95.97%`，baseline 胜率约 `32.21%`，平均 reward gap 约 `+126.28`。
- `direct` 与 `attack` 是主要短板：`direct` 下 OPS/SAM 胜率约 `0.17%`、baseline 胜率约 `21.18%`，平均 reward gap 约 `-43.96`；`attack` 下 OPS/SAM 胜率约 `1.80%`、baseline 胜率约 `25.00%`，平均 reward gap 约 `-55.10`。

因此下一阶段不建议继续盲目扩大 SAM 参数扫描。更合理的路线是先补强 continuous response policy 的 `direct/attack` 控制质量，或做 response policy 与 baseline 的 oracle 对照，确认当前响应策略库本身的上限；只有当 `direct/attack` 不再系统性弱于 baseline 后，再继续做 SAM 检测参数微调或更大规模 confirm。

## response policy oracle 对照结果

2026-05-17 已 pull 回 `continuous_oracle_compare_*` 结果。本轮使用 `scripts/run_continuous_oracle_compare.py`，将 `detector.method` 设为 `oracle`，即评估时直接按真实入侵策略选择 `response_direct`、`response_detour` 或 `response_attack`，用于隔离“响应策略库上限”和“SAM 检测/切换误差”。

整体结果：

| seed | 回报提升 | 胜率提升 | oracle 拦截胜率 | baseline 拦截胜率 | 响应准确率 | 验收 |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 69.68 | 32.50 pp | 32.50% | 0.00% | 99.38% | `engineering_pass=True`, `paper_like_pass=True` |
| 43 | 13.53 | 7.63 pp | 31.12% | 23.50% | 99.38% | `engineering_pass=True`, `paper_like_pass=True` |
| 44 | 69.62 | 31.75 pp | 32.25% | 0.50% | 99.38% | `engineering_pass=True`, `paper_like_pass=True` |

结论：oracle 条件下三 seed 全部正收益并通过当前工程/论文候选验收，说明响应策略库的整体上限足够，SAM geometry 结果里的 seed 43 负收益不能简单归因于“策略库完全不可用”。但分策略诊断仍显示 seed 43 的短板集中在 `direct/attack`：

| seed | 策略 | episodes | oracle 胜率 | baseline 胜率 | 平均 reward gap | 判断 |
|---:|---|---:|---:|---:|---:|---|
| 43 | `attack` | 13 | 0.00% | 53.85% | -107.70 | 明显弱于 baseline |
| 43 | `direct` | 539 | 0.19% | 20.96% | -43.97 | 明显弱于 baseline |
| 43 | `detour` | 248 | 100.00% | 27.42% | 144.84 | 显著优于 baseline |

因此下一轮应做 `direct/attack` 响应策略专项补强，而不是继续扩大 SAM 参数扫描。可优先设计只训练/只评估 `direct` 与 `attack` response policy 的实验，比较更长训练步数、奖励项或按策略单独验收的变化；`detour` 当前已经足够强，不应成为主要优化对象。

## direct/attack 加训对照结果

2026-05-18 已 pull 回 `continuous_response_focus_*` 结果。本轮保持 oracle 对照口径，`detour` 与 baseline 仍使用 `1.5M`，只把 `response_direct` 和 `response_attack` 分别加训到 `3M` 与 `5M`，每档覆盖 seeds `42/43/44`。

整体结果：

| direct/attack 训练步数 | 平均回报提升 | 平均胜率提升 | seed 43 回报提升 | seed 43 胜率提升 | 验收 |
|---:|---:|---:|---:|---:|---|
| 3M | 51.50 | 24.25 pp | 14.74 | 8.25 pp | 3/3 `engineering_pass=True`, 3/3 `paper_like_pass=True` |
| 5M | 51.11 | 24.04 pp | 14.67 | 8.12 pp | 3/3 `engineering_pass=True`, 3/3 `paper_like_pass=True` |

结论：单纯增加 `direct/attack` 的 PPO 训练步数可以维持 oracle 整体通过，但没有根治 seed 43 的分策略短板，且 `5M` 没有优于 `3M`。seed 43 的分策略诊断仍显示：

| 步数 | 策略 | episodes | oracle 胜率 | baseline 胜率 | 平均 reward gap | 判断 |
|---:|---|---:|---:|---:|---:|---|
| 3M | `attack` | 13 | 7.69% | 30.77% | -48.55 | 仍弱于 baseline |
| 3M | `direct` | 535 | 0.19% | 21.87% | -45.78 | 仍弱于 baseline |
| 3M | `detour` | 252 | 100.00% | 26.59% | 146.49 | 显著优于 baseline |
| 5M | `attack` | 13 | 0.00% | 30.77% | -65.65 | 仍弱于 baseline |
| 5M | `direct` | 537 | 0.56% | 18.25% | -37.49 | 仍弱于 baseline |
| 5M | `detour` | 250 | 100.00% | 34.40% | 130.90 | 显著优于 baseline |

因此下一步不建议继续只堆训练步数。已新增奖励塑形配置入口和 `slurm/response_reward_sweep_continuous_plcyf.sbatch`，用于扫描更强的终局奖励、靠近入侵者过程塑形、目标外侧防守塑形以及 attack 主动碰撞惩罚，继续用 oracle 对照判断 `direct/attack` 短板是否来自奖励设计。

## response reward 塑形扫描结果

2026-05-18 已 pull 回 `continuous_response_reward_*` 结果。本轮比较 `base/chase/guard/attacksafe` 四组全局奖励塑形，每组 seeds `42/43/44`，仍使用 oracle 响应策略选择。

整体结果：

| 奖励组 | 平均回报提升 | 平均胜率提升 | seed 43 回报提升 | seed 43 胜率提升 | 验收 |
|---|---:|---:|---:|---:|---|
| `base` | 50.94 | 23.96 pp | 13.53 | 7.63 pp | 3/3 `engineering_pass=True` |
| `chase` | 27.69 | 12.00 pp | 39.27 | 16.50 pp | 3/3 `engineering_pass=True` |
| `guard` | 64.02 | 24.67 pp | 38.89 | 16.75 pp | 3/3 `engineering_pass=True` |
| `attacksafe` | 50.88 | 18.00 pp | 20.30 | 7.63 pp | 3/3 `engineering_pass=True` |

结论：`guard` 是当前全局奖励塑形的最佳折中，平均回报提升最高，且 seed 43 从 base 的 `13.53 / 7.63 pp` 提升到 `38.89 / 16.75 pp`。`chase` 对 seed 43 也有明显帮助，但三 seed 平均弱于 `guard`。`attacksafe` 对 seed 42 很强，但 seed 43 改善有限。

seed 43 分策略仍未完全修复：

| 奖励组 | 策略 | episodes | oracle 胜率 | baseline 胜率 | 平均 reward gap | 判断 |
|---|---|---:|---:|---:|---:|---|
| `base` | `attack` | 13 | 0.00% | 53.85% | -107.70 | 明显弱于 baseline |
| `base` | `direct` | 539 | 0.19% | 20.96% | -43.97 | 明显弱于 baseline |
| `chase` | `attack` | 41 | 7.32% | 31.71% | -62.21 | 缩小但仍弱 |
| `chase` | `direct` | 505 | 0.40% | 13.47% | -31.73 | 缩小但仍弱 |
| `guard` | `attack` | 14 | 0.00% | 35.71% | -86.84 | 仍弱 |
| `guard` | `direct` | 517 | 0.19% | 11.61% | -28.98 | 缩小但仍弱 |
| `attacksafe` | `attack` | 20 | 0.00% | 10.00% | -65.68 | 缩小但仍弱 |
| `attacksafe` | `direct` | 534 | 0.00% | 17.42% | -47.09 | 仍弱 |

下一步应避免继续用同一套全局奖励同时影响 `direct/detour/attack`。已新增按入侵策略覆盖奖励的入口和 `slurm/response_policy_reward_continuous_plcyf.sbatch`，下一轮只给 `direct` 和 `attack` 使用不同 profile，`detour` 保持原始奖励，以验证定向塑形是否能进一步缩小 seed 43 的分策略 gap。

## response policy 按策略奖励扫描结果

2026-05-18 已 pull 回 `continuous_response_policy_reward_*` 结果。本轮固定 detour 原始奖励，只给 direct 和 attack 使用不同 profile。整体上按策略塑形明显优于全局奖励塑形：

| profile 组合 | 平均回报提升 | 平均胜率提升 | 最小回报提升 | seed 43 回报提升 | seed 43 胜率提升 | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `dstrong_asafe` | 82.52 | 31.79 pp | 78.25 | 78.25 | 30.25 pp | 平均回报最高 |
| `dstrong_astrong` | 79.55 | 31.12 pp | 76.08 | 76.08 | 30.12 pp | 稳定但 attack gap 偏负 |
| `dguard_asafe` | 77.42 | 32.50 pp | 74.09 | 74.09 | 30.75 pp | 胜率最高且稳定 |
| `dguard_astrong` | 74.89 | 32.75 pp | 72.47 | 75.80 | 33.00 pp | seed 43 胜率最高 |
| `dchase_asafe` | 61.02 | 26.08 pp | 29.74 | 76.16 | 31.75 pp | seed 44 较弱 |
| `dguard_achase` | 46.62 | 20.00 pp | 17.95 | 74.58 | 31.00 pp | attack gap 最接近修复 |

seed 43 上，direct 已经基本从短板修复为正 gap：

| profile 组合 | `direct` reward gap | `attack` reward gap | `detour` reward gap | 结论 |
|---|---:|---:|---:|---|
| `dchase_asafe` | 7.23 | -51.40 | 227.34 | direct 已正，attack 仍负 |
| `dguard_achase` | 5.01 | 0.81 | 224.90 | seed 43 attack 也略正 |
| `dguard_asafe` | 4.71 | -68.42 | 230.20 | direct 已正，attack 仍负 |
| `dguard_astrong` | 9.26 | -148.58 | 230.01 | direct 强，attack 变差 |
| `dstrong_asafe` | 6.41 | -37.12 | 246.95 | 平均回报最高，attack 仍负 |
| `dstrong_astrong` | 9.87 | -125.53 | 246.63 | direct 强，attack 变差 |

跨 seed 的 attack 拆分显示，`dguard_achase` 是唯一让 attack 平均 gap 接近 0 的组合：seed 42 attack gap `+2.34`、seed 43 `+0.81`、seed 44 `-44.65`，三 seed 平均约 `-13.83`。其它组合的 attack 平均 gap 多数仍明显为负。因此下一轮应固定 `direct=guard`，只做更窄的 attack reward profile 扫描，围绕 chase/guard/balanced 与较温和主动碰撞惩罚寻找 seed 44 也不崩的设置。
