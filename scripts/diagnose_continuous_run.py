"""诊断连续场景单个 run 中 OPS-DeMo/SAM 与 baseline 的差异。

该脚本面向已经完成 `scripts/analyze_continuous_run.py` 分析的 run 目录，
进一步展开 episode 级对照，回答类似“某个 seed 为什么 baseline 更强”
的问题。输出包括：

- `continuous_run_diagnosis.json`：结构化诊断摘要；
- `episode_outcome_breakdown.csv`：按胜负组合拆分的 episode 统计；
- `policy_outcome_breakdown.csv`：按最终入侵策略拆分的 episode 统计；
- `reason_delta.csv`：OPS 与 baseline 的终止原因计数差异；
- `largest_reward_gaps.csv`：OPS 相对 baseline 最差/最好的 episode。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="诊断连续场景单个 run 的 seed 级差异")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="连续 evaluate 生成的 run 目录，需包含 analysis/episode_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="诊断输出目录；默认写入 run-dir/analysis",
    )
    parser.add_argument(
        "--gap-limit",
        type=int,
        default=20,
        help="largest_reward_gaps.csv 中每侧保留的极端 episode 数量",
    )
    return parser


def main() -> None:
    """执行单个 run 的诊断并打印摘要。"""

    args = build_parser().parse_args()
    summary = diagnose_continuous_run(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        gap_limit=args.gap_limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"连续场景诊断结果已保存到: {summary['output_dir']}")


def diagnose_continuous_run(
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    gap_limit: int = 20,
) -> dict[str, Any]:
    """诊断一个连续 run，并写出用于排查 seed 差异的表格。

    参数:
        run_dir: 单个连续评估输出目录。
        output_dir: 诊断输出目录；为空时复用 `run_dir / "analysis"`。
        gap_limit: 最差/最好 reward gap 各保留多少个 episode。

    返回:
        可直接写入 JSON 的诊断摘要。
    """

    run_dir = run_dir.resolve()
    output_dir = (output_dir or run_dir / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_metrics = _load_episode_metrics(run_dir)
    step_trace = pd.read_csv(run_dir / "step_trace.csv")
    switch_events = _read_csv_optional(run_dir / "switch_events.csv")

    enriched = _enrich_episode_metrics(episode_metrics)
    outcome_breakdown = _build_outcome_breakdown(enriched)
    policy_breakdown = _build_policy_breakdown(enriched)
    reason_delta = _build_reason_delta(enriched)
    reward_gaps = _build_largest_reward_gaps(enriched, gap_limit)
    switch_summary = _build_switch_summary(switch_events)

    outcome_breakdown.to_csv(output_dir / "episode_outcome_breakdown.csv", index=False)
    policy_breakdown.to_csv(output_dir / "policy_outcome_breakdown.csv", index=False)
    reason_delta.to_csv(output_dir / "reason_delta.csv", index=False)
    reward_gaps.to_csv(output_dir / "largest_reward_gaps.csv", index=False)

    summary = {
        "output_dir": _portable_path(output_dir),
        "episodes": int(len(enriched)),
        "ops_win_count": int(enriched["ops_interceptor_win"].sum()),
        "baseline_win_count": int(enriched["baseline_interceptor_win"].sum()),
        "baseline_only_win_count": int(enriched["baseline_only_win"].sum()),
        "ops_only_win_count": int(enriched["ops_only_win"].sum()),
        "both_win_count": int(enriched["both_win"].sum()),
        "both_lose_count": int(enriched["both_lose"].sum()),
        "mean_reward_gap": _safe_float(enriched["reward_gap"].mean()),
        "median_reward_gap": _safe_float(enriched["reward_gap"].median()),
        "baseline_only_mean_reward_gap": _safe_float(
            enriched.loc[enriched["baseline_only_win"], "reward_gap"].mean()
        ),
        "ops_only_mean_reward_gap": _safe_float(
            enriched.loc[enriched["ops_only_win"], "reward_gap"].mean()
        ),
        "policy_with_worst_mean_gap": _first_record(policy_breakdown, "mean_reward_gap"),
        "largest_baseline_advantage_episode": _episode_record(
            enriched.sort_values("reward_gap").head(1)
        ),
        "largest_ops_advantage_episode": _episode_record(
            enriched.sort_values("reward_gap", ascending=False).head(1)
        ),
        "switch_summary": switch_summary,
        "outputs": {
            "episode_outcome_breakdown": _portable_path(output_dir / "episode_outcome_breakdown.csv"),
            "policy_outcome_breakdown": _portable_path(output_dir / "policy_outcome_breakdown.csv"),
            "reason_delta": _portable_path(output_dir / "reason_delta.csv"),
            "largest_reward_gaps": _portable_path(output_dir / "largest_reward_gaps.csv"),
        },
    }
    (output_dir / "continuous_run_diagnosis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _load_episode_metrics(run_dir: Path) -> pd.DataFrame:
    """读取 episode_metrics；若不存在，则从 step trace 重新构造。"""

    metrics_path = run_dir / "analysis" / "episode_metrics.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)

    step_trace = pd.read_csv(run_dir / "step_trace.csv")
    baseline_trace = pd.read_csv(run_dir / "baseline_step_trace.csv")
    ops_last = step_trace.sort_values(["episode", "episode_step"]).groupby("episode").tail(1)
    baseline_last = (
        baseline_trace.sort_values(["episode", "episode_step"]).groupby("episode").tail(1)
    )
    baseline_rewards = baseline_trace.groupby("episode")["reward"].sum().rename(
        "baseline_episode_reward"
    )
    ops_metrics = ops_last[
        ["episode", "episode_step", "episode_reward_so_far", "winner", "reason", "intruder_policy"]
    ].rename(
        columns={
            "episode_step": "ops_episode_len",
            "episode_reward_so_far": "ops_episode_reward",
            "winner": "ops_winner",
            "reason": "ops_reason",
            "intruder_policy": "ops_final_intruder_policy",
        }
    )
    baseline_metrics = baseline_last[["episode", "episode_step", "winner", "reason"]].rename(
        columns={
            "episode_step": "baseline_episode_len",
            "winner": "baseline_winner",
            "reason": "baseline_reason",
        }
    )
    merged = ops_metrics.merge(baseline_metrics, on="episode", how="outer")
    merged = merged.merge(baseline_rewards, on="episode", how="left")
    merged["ops_interceptor_win"] = merged["ops_winner"].eq("interceptor")
    merged["baseline_interceptor_win"] = merged["baseline_winner"].eq("interceptor")
    return merged.sort_values("episode")


def _enrich_episode_metrics(episode_metrics: pd.DataFrame) -> pd.DataFrame:
    """补充 reward gap 与胜负组合列，方便后续聚合。"""

    enriched = episode_metrics.copy()
    enriched["reward_gap"] = (
        enriched["ops_episode_reward"] - enriched["baseline_episode_reward"]
    )
    enriched["both_win"] = enriched["ops_interceptor_win"] & enriched["baseline_interceptor_win"]
    enriched["ops_only_win"] = enriched["ops_interceptor_win"] & ~enriched["baseline_interceptor_win"]
    enriched["baseline_only_win"] = ~enriched["ops_interceptor_win"] & enriched[
        "baseline_interceptor_win"
    ]
    enriched["both_lose"] = ~enriched["ops_interceptor_win"] & ~enriched[
        "baseline_interceptor_win"
    ]
    enriched["outcome_pair"] = enriched.apply(_outcome_pair_label, axis=1)
    return enriched


def _outcome_pair_label(row: pd.Series) -> str:
    """把一行 episode 胜负组合转成稳定标签。"""

    if bool(row["both_win"]):
        return "both_win"
    if bool(row["ops_only_win"]):
        return "ops_only_win"
    if bool(row["baseline_only_win"]):
        return "baseline_only_win"
    return "both_lose"


def _build_outcome_breakdown(enriched: pd.DataFrame) -> pd.DataFrame:
    """按胜负组合统计回报差、长度差和样本数。"""

    grouped = (
        enriched.groupby("outcome_pair")
        .agg(
            episodes=("episode", "count"),
            mean_reward_gap=("reward_gap", "mean"),
            median_reward_gap=("reward_gap", "median"),
            mean_ops_reward=("ops_episode_reward", "mean"),
            mean_baseline_reward=("baseline_episode_reward", "mean"),
            mean_ops_len=("ops_episode_len", "mean"),
            mean_baseline_len=("baseline_episode_len", "mean"),
        )
        .reset_index()
    )
    return _round_numeric(grouped)


def _build_policy_breakdown(enriched: pd.DataFrame) -> pd.DataFrame:
    """按 OPS 最终入侵策略统计，定位是否某类策略拖累。"""

    grouped = (
        enriched.groupby("ops_final_intruder_policy")
        .agg(
            episodes=("episode", "count"),
            ops_win_rate=("ops_interceptor_win", "mean"),
            baseline_win_rate=("baseline_interceptor_win", "mean"),
            mean_reward_gap=("reward_gap", "mean"),
            baseline_only_wins=("baseline_only_win", "sum"),
            ops_only_wins=("ops_only_win", "sum"),
        )
        .reset_index()
        .sort_values("mean_reward_gap")
    )
    return _round_numeric(grouped)


def _build_reason_delta(enriched: pd.DataFrame) -> pd.DataFrame:
    """比较 OPS 与 baseline 终止原因计数差异。"""

    ops_counts = enriched["ops_reason"].value_counts(dropna=False)
    baseline_counts = enriched["baseline_reason"].value_counts(dropna=False)
    reasons = sorted(set(ops_counts.index).union(set(baseline_counts.index)), key=str)
    rows = []
    for reason in reasons:
        ops_count = int(ops_counts.get(reason, 0))
        baseline_count = int(baseline_counts.get(reason, 0))
        rows.append(
            {
                "reason": reason,
                "ops_count": ops_count,
                "baseline_count": baseline_count,
                "ops_minus_baseline": ops_count - baseline_count,
            }
        )
    return pd.DataFrame(rows).sort_values("ops_minus_baseline")


def _build_largest_reward_gaps(enriched: pd.DataFrame, gap_limit: int) -> pd.DataFrame:
    """输出 reward gap 两端的典型 episode，便于进一步画轨迹或查日志。"""

    columns = [
        "episode",
        "reward_gap",
        "ops_episode_reward",
        "baseline_episode_reward",
        "ops_interceptor_win",
        "baseline_interceptor_win",
        "ops_reason",
        "baseline_reason",
        "ops_final_intruder_policy",
        "ops_episode_len",
        "baseline_episode_len",
        "outcome_pair",
    ]
    worst = enriched.nsmallest(gap_limit, "reward_gap")[columns].assign(rank_type="baseline_advantage")
    best = enriched.nlargest(gap_limit, "reward_gap")[columns].assign(rank_type="ops_advantage")
    return _round_numeric(pd.concat([worst, best], ignore_index=True))


def _build_switch_summary(switch_events: pd.DataFrame) -> dict[str, Any]:
    """统计切换事件是否切向真实策略，辅助判断误切换风险。"""

    if switch_events.empty:
        return {
            "switch_count": 0,
            "switch_to_true_policy_rate": None,
            "switches_by_target": {},
            "switches_by_true_policy": {},
        }
    return {
        "switch_count": int(len(switch_events)),
        "switch_to_true_policy_rate": _safe_float(
            switch_events["to_policy"].eq(switch_events["true_policy"]).mean()
        ),
        "switches_by_target": {
            str(key): int(value)
            for key, value in switch_events["to_policy"].value_counts().sort_index().items()
        },
        "switches_by_true_policy": {
            str(key): int(value)
            for key, value in switch_events["true_policy"].value_counts().sort_index().items()
        },
    }


def _round_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """统一收敛小数位，让 CSV 更适合人工阅读。"""

    rounded = frame.copy()
    numeric_columns = rounded.select_dtypes(include=["float"]).columns
    rounded[numeric_columns] = rounded[numeric_columns].round(6)
    return rounded


def _first_record(frame: pd.DataFrame, sort_column: str) -> dict[str, Any] | None:
    """返回按某列升序后的第一条记录。"""

    if frame.empty:
        return None
    return _json_record(frame.sort_values(sort_column).head(1).iloc[0].to_dict())


def _episode_record(frame: pd.DataFrame) -> dict[str, Any] | None:
    """把单行 episode 记录转换为 JSON 友好的 dict。"""

    if frame.empty:
        return None
    keep = [
        "episode",
        "reward_gap",
        "ops_episode_reward",
        "baseline_episode_reward",
        "ops_interceptor_win",
        "baseline_interceptor_win",
        "ops_reason",
        "baseline_reason",
        "ops_final_intruder_policy",
        "outcome_pair",
    ]
    return _json_record(frame.iloc[0][keep].to_dict())


def _json_record(record: dict[str, Any]) -> dict[str, Any]:
    """清理 pandas/numpy 标量，确保可以稳定写入 JSON。"""

    cleaned: dict[str, Any] = {}
    for key, value in record.items():
        if pd.isna(value):
            cleaned[key] = None
        elif isinstance(value, bool):
            cleaned[key] = bool(value)
        elif isinstance(value, int):
            cleaned[key] = int(value)
        elif isinstance(value, float):
            cleaned[key] = _safe_float(value)
        else:
            cleaned[key] = value
    return cleaned


def _read_csv_optional(path: Path) -> pd.DataFrame:
    """读取可选 CSV；文件不存在或为空时返回空表。"""

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _safe_float(value: Any) -> float | None:
    """把数值转换为 JSON 友好的 float，过滤 NaN。"""

    if pd.isna(value):
        return None
    return float(value)


def _portable_path(path: Path) -> str:
    """输出跨平台可读路径，避免 Windows 反斜杠污染 JSON。"""

    return str(path).replace("\\", "/")


if __name__ == "__main__":
    main()
