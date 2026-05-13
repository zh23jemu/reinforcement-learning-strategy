"""分析连续二维拦截捕猎评估结果。

该脚本读取 `evaluate` 生成的连续场景运行目录，聚合 episode 级胜负、
回报、终止原因、响应策略准确率等指标，并输出到 `analysis/` 目录。
它的定位类似离散环境的 analyze 流程，但当前先保持为独立脚本，方便
后续 Slurm sweep 或人工拉回结果后直接复用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="分析连续 OPS-DeMo 评估结果")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="连续 evaluate 生成的目录，例如 runs/continuous_paper_like/20260513_032143",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="分析结果输出目录；默认写入 run-dir/analysis",
    )
    return parser


def main() -> None:
    """读取连续运行目录，生成诊断指标并打印核心结论。"""

    args = build_parser().parse_args()
    summary = analyze_continuous_run(args.run_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"连续场景分析结果已保存到: {summary['output_dir']}")


def analyze_continuous_run(run_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """分析一个连续场景 run 目录并写出结构化结果。

    参数:
        run_dir: `evaluate_continuous` 生成的运行目录。
        output_dir: 分析输出目录；为空时默认使用 `run_dir / "analysis"`。

    返回:
        包含核心指标、验收判断和输出文件路径的 summary 字典。
    """

    run_dir = run_dir.resolve()
    output_dir = (output_dir or run_dir / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    step_path = run_dir / "step_trace.csv"
    baseline_step_path = run_dir / "baseline_step_trace.csv"
    _require_files([summary_path, step_path, baseline_step_path])

    raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    step_trace = pd.read_csv(step_path)
    baseline_trace = pd.read_csv(baseline_step_path)

    episode_metrics = _build_episode_metrics(step_trace, baseline_trace)
    episode_metrics.to_csv(output_dir / "episode_metrics.csv", index=False)

    analysis_summary = _build_analysis_summary(
        raw_summary=raw_summary,
        step_trace=step_trace,
        baseline_trace=baseline_trace,
        episode_metrics=episode_metrics,
        output_dir=output_dir,
    )
    (output_dir / "continuous_analysis_summary.json").write_text(
        json.dumps(analysis_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return analysis_summary


def _require_files(paths: list[Path]) -> None:
    """检查输入文件是否齐全，尽早给出明确错误。"""

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"连续评估结果不完整，缺少文件:\n{joined}")


def _build_episode_metrics(step_trace: pd.DataFrame, baseline_trace: pd.DataFrame) -> pd.DataFrame:
    """生成 episode 级对比表。

    连续 OPS-DeMo trace 已经包含 `episode_reward_so_far`，因此每个 episode 取最后一行
    即可得到总回报和最终胜负；baseline trace 没有累计回报，需要按 episode 汇总 reward。
    """

    ops_last = step_trace.sort_values(["episode", "episode_step"]).groupby("episode").tail(1)
    baseline_last = baseline_trace.sort_values(["episode", "episode_step"]).groupby("episode").tail(1)
    baseline_rewards = baseline_trace.groupby("episode")["reward"].sum().rename("baseline_episode_reward")

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


def _build_analysis_summary(
    *,
    raw_summary: dict[str, Any],
    step_trace: pd.DataFrame,
    baseline_trace: pd.DataFrame,
    episode_metrics: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """汇总连续实验核心诊断指标。"""

    ops_reward = float(raw_summary["mean_episode_reward"])
    baseline_reward = float(raw_summary["baseline_mean_episode_reward"])
    ops_win_rate = float(raw_summary["interceptor_win_rate"])
    baseline_win_rate = float(raw_summary["baseline_interceptor_win_rate"])
    response_accuracy = float(raw_summary["response_policy_accuracy"])

    # 按入侵策略统计响应准确率，用来检查是否存在某一种入侵策略单独失效。
    accuracy_by_policy = (
        step_trace.groupby("intruder_policy")["response_policy_correct"]
        .mean()
        .sort_index()
        .to_dict()
    )
    policy_step_counts = step_trace["intruder_policy"].value_counts().sort_index().to_dict()

    engineering_pass = (
        response_accuracy >= 0.95
        and ops_reward > baseline_reward
        and ops_win_rate > baseline_win_rate
    )
    process_pass = _is_process_pass(raw_summary, step_trace, baseline_trace)

    return {
        "output_dir": str(output_dir),
        "episodes": int(raw_summary["episodes"]),
        "total_steps": int(len(step_trace)),
        "baseline_total_steps": int(len(baseline_trace)),
        "mean_episode_length": _safe_float(episode_metrics["ops_episode_len"].mean()),
        "baseline_mean_episode_length": _safe_float(episode_metrics["baseline_episode_len"].mean()),
        "mean_episode_reward": ops_reward,
        "baseline_mean_episode_reward": baseline_reward,
        "reward_improvement": _safe_float(ops_reward - baseline_reward),
        "interceptor_win_rate": ops_win_rate,
        "baseline_interceptor_win_rate": baseline_win_rate,
        "win_rate_improvement": _safe_float(ops_win_rate - baseline_win_rate),
        "response_policy_accuracy": response_accuracy,
        "accuracy_by_intruder_policy": {key: _safe_float(value) for key, value in accuracy_by_policy.items()},
        "intruder_policy_step_counts": {key: int(value) for key, value in policy_step_counts.items()},
        "ops_winner_counts": _value_counts(episode_metrics["ops_winner"]),
        "baseline_winner_counts": _value_counts(episode_metrics["baseline_winner"]),
        "ops_reason_counts": _value_counts(episode_metrics["ops_reason"]),
        "baseline_reason_counts": _value_counts(episode_metrics["baseline_reason"]),
        "episode_reward_quantiles": _quantiles(episode_metrics["ops_episode_reward"]),
        "baseline_episode_reward_quantiles": _quantiles(episode_metrics["baseline_episode_reward"]),
        "acceptance": {
            "process_pass": bool(process_pass),
            "engineering_pass": bool(engineering_pass),
            "paper_like_pass": bool(ops_win_rate >= 0.20 and engineering_pass),
            "judgement": _judge_result(process_pass, engineering_pass, ops_win_rate),
        },
    }


def _is_process_pass(
    raw_summary: dict[str, Any],
    step_trace: pd.DataFrame,
    baseline_trace: pd.DataFrame,
) -> bool:
    """判断是否达到“流程过关”标准。"""

    required_summary_keys = {
        "episodes",
        "mean_episode_reward",
        "interceptor_win_rate",
        "response_policy_accuracy",
        "baseline_mean_episode_reward",
        "baseline_interceptor_win_rate",
    }
    required_step_columns = {
        "episode",
        "reward",
        "intruder_policy",
        "response_policy",
        "response_policy_correct",
        "winner",
        "reason",
    }
    return (
        required_summary_keys.issubset(raw_summary)
        and required_step_columns.issubset(step_trace.columns)
        and not step_trace.empty
        and not baseline_trace.empty
    )


def _judge_result(process_pass: bool, engineering_pass: bool, ops_win_rate: float) -> str:
    """把指标转换成便于汇报的中文结论。"""

    if not process_pass:
        return "未达到流程过关：评估输出不完整或关键字段缺失。"
    if engineering_pass and ops_win_rate >= 0.20:
        return "达到接近论文效果的候选标准，可继续做多 seed 与参数敏感性验证。"
    if engineering_pass:
        return "达到工程复现主要标准：机制有效且优于 baseline，但绝对拦截胜率仍需优化。"
    return "仅达到流程过关：链路已跑通，但相对 baseline 的收益仍不足。"


def _value_counts(series: pd.Series) -> dict[str, int]:
    """把 pandas value_counts 转成 JSON 友好的普通字典。"""

    return {str(key): int(value) for key, value in series.fillna("").value_counts().sort_index().items()}


def _quantiles(series: pd.Series) -> dict[str, float]:
    """计算常用分位数，便于快速判断回报分布是否被极端值主导。"""

    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        "min": _safe_float(values.min()),
        "q25": _safe_float(values.quantile(0.25)),
        "median": _safe_float(values.quantile(0.50)),
        "q75": _safe_float(values.quantile(0.75)),
        "max": _safe_float(values.max()),
        "std": _safe_float(values.std(ddof=0)),
    }


def _safe_float(value: Any) -> float:
    """将 numpy/pandas 数值转为普通 float，并处理 NaN。"""

    result = float(value)
    if np.isnan(result):
        return 0.0
    return result


if __name__ == "__main__":
    main()
