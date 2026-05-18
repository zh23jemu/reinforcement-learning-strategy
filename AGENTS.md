# 项目级代理记忆

## 项目目标

- 复现并工程化验证强化学习策略变化检测方法，主线是离散 Predator-Prey 场景中的 OPS-DeMo 框架。
- 扩展主线是连续二维拦截捕猎场景：训练三类入侵策略对应的响应策略库，用 SAM 风格 MC dropout opponent model 与 normalized running error 做策略切换检测，并对比普通 PPO baseline。

## 技术栈

- Python `>=3.10,<3.13`，依赖见 `pyproject.toml`。
- 强化学习训练使用 Stable-Baselines3 PPO、Gymnasium、PyTorch、NumPy、Pandas、Matplotlib。
- 本地必须使用项目 `.venv`：Windows 使用 `.venv\Scripts\python.exe`，Linux/Slurm 使用 `.venv/bin/python` 或脚本中的 `PYTHON_BIN` 覆盖。
- 长训练和参数扫描优先使用 Slurm 脚本；用户当前服务器实验默认放到 `defq` 分区，并可用 `sbatch --partition=defq ...` 覆盖脚本默认分区。

## 当前架构

- `src/rl_strategy/`：离散与连续环境、训练入口、评估入口、检测器和 CLI。
- `configs/`：离散、连续 smoke/paper-like 配置。
- `scripts/`：连续 sweep、聚合、分析和诊断脚本。
- `slurm/`：训练、确认长训、SAM 调参和聚合的 Slurm 提交脚本。
- `docs/continuous_confirm_results.md`：连续场景确认实验结果和结论边界。
- `docs/continuous_acceptance_checklist.md`：连续场景验收分级和当前定位。
- `.recallloom/`：长期项目上下文与跨会话进展记录。

## 开发规范

- 优先最小修改，不做无关重构。
- 新增或修改代码时保持较详细中文注释，重点解释用途、关键逻辑、重要分支和不明显实现细节。
- 编辑文件前先读取目标文件；不要直接覆盖用户已有内容。
- Python 命令不要使用系统 Python，也不要依赖激活虚拟环境。
- `.gitignore` 需要随项目真实内容维护；不要因为已有实验运行、模型产物或日志单文件小于 100MB 就整体通配忽略 `runs/`、`artifacts/` 或 `logs/`。

## 当前进度

- 离散 OPS-DeMo 工程链路已实现，包括 PPO 对手策略库、响应策略库、running error 策略切换检测、baseline、分析、sweep 和 Slurm。
- 连续 SAM 原文检测链路已实现：MC dropout opponent model 预测入侵者动作，观测误差除以预测不确定性形成 normalized running error，再由 switchboard 触发响应策略切换。
- `sam.feature_mode=geometry` 已加入连续 opponent model 输入，追加相对位置、单位方向、相对速度和速度投影等几何特征。
- geometry aggressive 1.5M confirm 平均回报提升约 `37.52`，平均胜率提升约 `17.25 pp`，平均响应准确率约 `72.18%`，但切换次数偏高且 seed 43 略负。
- geometry midbest 1.5M confirm 平均回报提升约 `32.87`，平均胜率提升约 `15.17 pp`，平均响应准确率约 `66.62%`，但 seed 43 回报提升 `-12.45`。
- geometry `cd80m25` 1.5M confirm 平均回报提升约 `32.13`，平均胜率提升约 `14.75 pp`，平均响应准确率约 `65.42%`，平均切换次数 `177.0`，但 seed 43 回报提升 `-13.04`。
- 已新增 `scripts/diagnose_continuous_run.py`，可输出 episode 胜负组合、最终响应策略拆分、终止原因差异和最大 reward gap。
- 已新增 `scripts/run_continuous_oracle_compare.py` 与 `slurm/oracle_compare_continuous_plcyf.sbatch`，可在 detector 完全正确的 oracle 条件下比较 response policy 库与 baseline，用于隔离 `direct/attack` 短板是否来自响应策略本身。
- `continuous_oracle_compare_*` 已完成 1.5M / 800 episodes / 3 seed 对照：seed 42/43/44 全部 `engineering_pass=True` 且 `paper_like_pass=True`，但 seed 43 的分策略诊断仍显示 `direct` 和 `attack` 弱于 baseline，`detour` 显著强于 baseline。
- 已新增 `slurm/response_focus_continuous_plcyf.sbatch`，默认比较 direct/attack response policy 加训到 `3M` 与 `5M` 两档，`detour` 和 baseline 保持基础步数，并继续用 oracle 对照评估。
- `continuous_response_focus_*` 已完成：`3M` 与 `5M` 两档整体仍通过验收，但 seed 43 的 `direct/attack` 分策略短板没有根治，`5M` 未优于 `3M`。
- 已给连续环境增加可配置奖励权重，并新增 `slurm/response_reward_sweep_continuous_plcyf.sbatch`，用于扫描 `base/chase/guard/attacksafe` 四组奖励塑形。
- `continuous_response_reward_*` 已完成：`guard` 全局奖励塑形三 seed 平均最好，seed 43 也明显改善，但 `direct/attack` 分策略仍未反超 baseline。
- 已新增按入侵策略覆盖奖励的入口和 `slurm/response_policy_reward_continuous_plcyf.sbatch`，用于只给 `direct/attack` 使用定向 profile，`detour` 保持原始奖励。
- `continuous_response_policy_reward_*` 已完成：按策略奖励 profile 明显优于全局塑形，`direct` 基本已从负 gap 修复为正 gap；残留短板集中在 `attack`。
- 已新增 `slurm/response_attack_reward_continuous_plcyf.sbatch` 和 `continuous_response_attack_reward` 聚合前缀，用于固定 `direct=guard` 后只细扫 attack profile。
- `continuous_response_attack_reward_*` 已完成：整体最强为 `attack_guard_safe`，三 seed 平均回报提升约 `77.81`、平均胜率提升约 `31.92 pp`，但 attack 分策略本身仍未修复；attack-only gap 最好的 `attack_chase_light` 平均仍约 `-25.46`，seed 44 仍约 `-39.61`。
- `scripts/run_continuous_sweep.py` 和 `slurm/confirm_continuous_plcyf.sbatch` 已支持在真实 SAM confirm 中传入 `--direct-reward-profile` / `--attack-reward-profile`，可验证 oracle 里表现最强的 response reward 库落到 SAM 检测链路后的收益。
- `continuous_sam_geometry_reward_confirm_*` 已完成：`direct=guard, attack=attack_guard_safe` 在真实 SAM 检测链路下三 seed 平均回报提升约 `48.03`，平均胜率提升约 `21.25 pp`，较前一轮 geometry confirm 整体明显更强；但平均响应准确率仍约 `58.02%`，仍未达到工程验收阈值。
- 已新增 `slurm/sam_geometry_reward_tune_continuous_plcyf.sbatch`，用于在固定 `direct=guard, attack=attack_guard_safe` 的 response reward profile 下细扫更保守的 SAM 检测参数。
- `scripts/aggregate_continuous_sweep.py` 已纳入 `continuous_sam_geometry_reward_confirm` 与 `continuous_sam_geometry_reward_tune` 前缀，后续聚合总表会自动收录 reward confirm / tune 结果。
- `continuous_sam_geometry_reward_tune_*` 已完成：更保守参数整体不如锚点，`th10/wu20/cd60/mg0.25` 仍是 300k 平均回报、胜率和响应准确率最高组合；保守参数能减少切换，但 seed 44 收益和 accuracy 明显下降。
- 已新增 `slurm/sam_reward_fast_tune_continuous_plcyf.sbatch`，围绕锚点做更灵敏的低阈值、短 cooldown、小 margin 搜索，并使用较短 `continuous_sam_reward_fast_tune` 前缀降低 Windows 长路径风险。

## TODO

- 优先诊断并补强连续 response policy 的 `direct` 和 `attack` 控制质量。
- 下一轮优先运行 `sam_reward_fast_tune_continuous_plcyf.sbatch`，在 `direct=guard, attack=attack_guard_safe` response 库基础上细扫更灵敏的 SAM 检测参数，目标是改善 seed 44 和整体 response accuracy；同时单独诊断 attack response policy 为什么在 attack-only 分组里持续弱于 baseline。
- 若 `direct/attack` 响应策略补强后不再系统性弱于 baseline，再继续做 SAM 参数微调或更大规模 confirm。
- 后续结果继续写入 `docs/continuous_confirm_results.md`、`docs/continuous_acceptance_checklist.md` 和 `.recallloom/`。

## 风险问题

- 当前 SAM 检测版 confirm 仍未达到 `response_policy_accuracy >= 0.95` 的 oracle 风格工程阈值，验收脚本仍会给出 `engineering_pass=False`。
- seed 43 的主要问题已经从“是否触发切换”转为“`direct/attack` 响应策略弱于 baseline”，继续只调 `threshold/cooldown/margin` 的收益有限。
- 对外说明必须区分 oracle 上限参考、raw SAM 检测结果和 geometry SAM 检测结果；不能把 oracle 结果当作 SAM 原文检测最终效果。

## Current Status

- 本地最新进展已包含 reward-profile 版 SAM 保守细扫结果：保守化会降低切换但损伤 accuracy 与 seed 44 收益，因此下一步改为更灵敏参数细扫。

## Recent Changes

- 补充连续确认结果文档中的 geometry 窄范围调参、`cd80m25` confirm 和 seed 43 分策略诊断结论。
- 补充连续验收清单和 README 中的诊断脚本说明。
- 创建项目级 `AGENTS.md`，记录长期代理上下文和当前下一步。
- pull 回并分析 `continuous_oracle_compare_*` 三 seed 结果，确认 oracle 整体通过验收但 seed 43 `direct/attack` 仍需补强。
- 准备提交 `continuous_oracle_smoke_*` 与 `continuous_response_focus_smoke_*` 的本地 smoke 验证产物，保留脚本可运行证据。
- pull 回并分析 `continuous_response_focus_*`，确认 `3M/5M` 加训整体通过但分策略短板仍在。
- 新增连续奖励权重配置入口、oracle runner 奖励覆盖参数、`continuous_response_reward` 聚合前缀和 reward sweep Slurm 脚本。
- pull 回并分析 `continuous_response_reward_*`，确认 `guard` 全局塑形最佳但仍未根治 `direct/attack`。
- 新增 `reward_overrides_by_policy`、`--direct-reward-profile` / `--attack-reward-profile` 和 `continuous_response_policy_reward` Slurm/聚合入口。
- pull 回并分析 `continuous_response_policy_reward_*`，确认 `dstrong_asafe` 平均回报最高、`dguard_asafe` 胜率稳定、`dguard_achase` 的 attack gap 最接近修复。
- 新增 attack-only profile：`attack_chase_light`、`attack_guard`、`attack_guard_safe`、`attack_balanced`，并新增 attack reward 窄范围 Slurm 脚本。
- pull 回并分析 `continuous_response_attack_reward_*`：`attack_guard_safe` 整体收益最高且 `direct/detour` 稳定，但 attack-only 仍为负；`attack_chase_light` 的 attack-only gap 最小但会拖累 `direct`。
- 给 `run_continuous_sweep.py` 和 `confirm_continuous_plcyf.sbatch` 增加按策略 reward profile 参数透传；本地 `.venv\Scripts\python.exe -m pytest` 通过，`27 passed`。
- pull 回并分析 `continuous_sam_geometry_reward_confirm_*`：seed 42/43/44 回报提升分别约 `45.10/54.48/44.52`，胜率提升约 `20.00/23.00/20.75 pp`，响应准确率约 `53.38/62.49/58.19%`。
- Windows 本地 pull 首次因长路径失败，已在仓库设置 `core.longpaths=true`，并把首次失败留下的半成品文件非破坏性移动到 `C:\Coding\reinforcement-learning-strategy_pull_backup_20260518_174728` 后完成 pull。
- 新增 reward-profile 版 SAM 检测细扫 Slurm 脚本，并更新聚合前缀；本地 `.venv\Scripts\python.exe -m pytest` 通过，`27 passed`；聚合 smoke 汇总 `188` 条连续结果。
- pull 回并分析 `continuous_sam_geometry_reward_tune_*`：锚点 `th10/wu20/cd60/mg0.25` 三 seed 平均回报提升约 `42.09`、胜率提升约 `17.78 pp`、响应准确率约 `62.49%`、平均切换约 `79.33`；其余更保守组合收益和 accuracy 均下降。
- 新增更灵敏 `continuous_sam_reward_fast_tune` Slurm 脚本，并把该短前缀加入聚合；本地 `.venv\Scripts\python.exe -m pytest` 通过，`27 passed`；聚合 smoke 汇总 `206` 条连续结果。

## Next TODO

- 服务器下一步运行 `sbatch --partition=defq --array=0-17 --export=ALL,TIMESTEPS=300000,EPISODES=300 slurm/sam_reward_fast_tune_continuous_plcyf.sbatch`，完成后聚合并分析 `continuous_sam_reward_fast_tune_*`。
- 同时准备 attack policy 专项诊断，检查 attack 任务是否存在奖励定义、起始状态分布、策略动作约束或 baseline 对照偏置问题；不建议继续盲目只扫当前四类 attack reward 权重。
- 长训继续沿用现有 Slurm 风格并提交到 `defq` 分区。

## Open Issues

- `direct` response policy 的 reward gap 已明显改善；`attack` 在 attack-only 窄扫后仍未稳定修复，说明问题可能不是单纯奖励权重强弱，而是 attack 训练任务定义或对照口径本身。
- 真实 SAM reward confirm 虽然整体收益变强，但 `response_policy_accuracy` 仍明显不足，下一步不能只看 reward improvement，需要继续提升检测选择质量。
- 更保守检测参数降低 switch count 的同时显著降低 response accuracy，说明“少切换”不是当前唯一目标，过强阻尼会让策略选择滞后。
- 当前验收阈值仍混用了 oracle 策略识别口径和 SAM 检测口径，最终报告需要明确解释。

## Architecture Decisions

- 连续检测主线保留 SAM 原文式 MC dropout normalized error，不回退到真实策略标签 oracle。
- geometry 特征作为当前连续 SAM opponent model 的主要改进方向保留。
- 下一阶段优化入口从 SAM 检测参数转向 response policy 控制质量，尤其是 `direct` 和 `attack`。
- 按策略 reward profile 现在既服务 oracle response policy 诊断，也可进入真实 SAM confirm；oracle 结果仍只作为 response 库上限参考。
