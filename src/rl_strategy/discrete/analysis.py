"""离散 OPS-DeMo 实验结果分析与绘图。

该模块只读取 `evaluate` 已经保存的过程数据，不重新加载模型、不重新运行环境。
这样长训结束后可以反复调整图表和汇总逻辑，而不会影响实验结果本身。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_discrete_run(run_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """分析一次离散实验运行目录，并生成图表。

    参数:
        run_dir: `evaluate` 生成的运行目录，必须包含 `step_trace.csv`。
        output_dir: 图表和汇总输出目录；为空时使用 `run_dir / "analysis"`。

    返回:
        适合写入 JSON 的分析摘要。
    """

    run_dir = run_dir.resolve()
    output_dir = (output_dir or run_dir / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    step_trace = _read_csv_required(run_dir / "step_trace.csv")
    baseline_trace = _read_csv_optional(run_dir / "baseline_step_trace.csv")
    switch_events = _read_csv_optional(run_dir / "switch_events.csv")
    summary = _read_json_optional(run_dir / "summary.json")

    episode_metrics = _episode_metrics(step_trace, baseline_trace)
    analysis_summary = {
        "run_dir": str(run_dir),
        "episodes": int(step_trace["episode"].nunique()),
        "mean_episode_reward": float(episode_metrics["opsdemo_reward"].mean()),
        "std_episode_reward": float(episode_metrics["opsdemo_reward"].std(ddof=0)),
        "aop_accuracy": float(step_trace["assumption_correct"].mean())
        if "assumption_correct" in step_trace
        else float("nan"),
        "response_policy_accuracy": float(step_trace["response_policy_correct"].mean())
        if "response_policy_correct" in step_trace
        else float("nan"),
        "switch_count": int(len(switch_events)),
        "baseline_mean_episode_reward": _safe_mean(episode_metrics.get("baseline_reward")),
        "baseline_std_episode_reward": _safe_std(episode_metrics.get("baseline_reward")),
        "figures": [],
    }

    figure_paths = [
        _plot_running_errors(step_trace, switch_events, output_dir),
        _plot_aop_accuracy(step_trace, output_dir),
        _plot_reward_comparison(episode_metrics, output_dir),
        _plot_switch_timeline(step_trace, switch_events, output_dir),
    ]
    analysis_summary["figures"] = [str(path) for path in figure_paths if path is not None]

    episode_metrics.to_csv(output_dir / "episode_metrics.csv", index=False)
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(analysis_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return analysis_summary


def _read_csv_required(path: Path) -> pd.DataFrame:
    """读取必需 CSV，缺失时给出清晰错误。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少必需过程文件: {path}")
    return pd.read_csv(path)


def _read_csv_optional(path: Path) -> pd.DataFrame:
    """读取可选 CSV；文件缺失或为空时返回空表。"""

    if not path.exists() or path.stat().st_size <= 2:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json_optional(path: Path) -> dict[str, Any]:
    """读取可选 JSON 汇总文件。"""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_metrics(step_trace: pd.DataFrame, baseline_trace: pd.DataFrame) -> pd.DataFrame:
    """按 episode 汇总 OPS-DeMo 与 baseline 的回报。"""

    opsdemo = (
        step_trace.groupby("episode", as_index=False)["reward"]
        .sum()
        .rename(columns={"reward": "opsdemo_reward"})
    )
    if "response_policy_correct" in step_trace:
        response_accuracy = (
            step_trace.groupby("episode", as_index=False)["response_policy_correct"]
            .mean()
            .rename(columns={"response_policy_correct": "response_policy_accuracy"})
        )
        opsdemo = opsdemo.merge(response_accuracy, on="episode", how="left")
    if baseline_trace.empty:
        opsdemo["baseline_reward"] = np.nan
        return opsdemo

    baseline = (
        baseline_trace.groupby("episode", as_index=False)["reward"]
        .sum()
        .rename(columns={"reward": "baseline_reward"})
    )
    return opsdemo.merge(baseline, on="episode", how="left")


def _plot_running_errors(
    step_trace: pd.DataFrame,
    switch_events: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    """绘制两个候选策略的 running error 曲线。"""

    error_columns = [column for column in step_trace.columns if column.startswith("running_error_")]
    if not error_columns:
        return None

    fig, ax = plt.subplots(figsize=(11, 5))
    for column in error_columns:
        label = column.replace("running_error_", "")
        ax.plot(step_trace["global_step"], step_trace[column], label=label, linewidth=1.6)

    _draw_switch_lines(ax, switch_events)
    ax.set_title("OPS-DeMo running error")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Running error")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save_figure(fig, output_dir / "running_errors.png")


def _plot_aop_accuracy(step_trace: pd.DataFrame, output_dir: Path) -> Path | None:
    """绘制按 episode 统计的 AOP 准确率。"""

    if "assumption_correct" not in step_trace:
        return None

    per_episode = (
        step_trace.groupby("episode", as_index=False)["assumption_correct"]
        .mean()
        .rename(columns={"assumption_correct": "aop_accuracy"})
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(per_episode["episode"], per_episode["aop_accuracy"], marker="o", linewidth=1.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Assumed opponent policy accuracy")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.25)
    return _save_figure(fig, output_dir / "aop_accuracy.png")


def _plot_reward_comparison(episode_metrics: pd.DataFrame, output_dir: Path) -> Path:
    """绘制 OPS-DeMo 与普通 PPO baseline 的每 episode 回报对比。"""

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        episode_metrics["episode"],
        episode_metrics["opsdemo_reward"],
        label="OPS-DeMo + PPO",
        marker="o",
        linewidth=1.6,
    )
    if "baseline_reward" in episode_metrics and not episode_metrics["baseline_reward"].isna().all():
        ax.plot(
            episode_metrics["episode"],
            episode_metrics["baseline_reward"],
            label="Standalone PPO",
            marker="s",
            linewidth=1.6,
        )
    ax.set_title("Episodic accumulated reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save_figure(fig, output_dir / "reward_comparison.png")


def _plot_switch_timeline(
    step_trace: pd.DataFrame,
    switch_events: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """绘制真实策略与假设策略的时间线。"""

    policy_to_id = {name: idx for idx, name in enumerate(sorted(step_trace["true_policy"].unique()))}
    true_ids = step_trace["true_policy"].map(policy_to_id)
    assumed_ids = step_trace["assumed_policy"].map(policy_to_id)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.step(step_trace["global_step"], true_ids, where="post", label="True policy", linewidth=1.8)
    ax.step(
        step_trace["global_step"],
        assumed_ids,
        where="post",
        label="Assumed policy",
        linewidth=1.4,
        linestyle="--",
    )
    _draw_switch_lines(ax, switch_events)
    ax.set_yticks(list(policy_to_id.values()), list(policy_to_id.keys()))
    ax.set_title("True policy vs assumed policy")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Policy")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save_figure(fig, output_dir / "policy_timeline.png")


def _draw_switch_lines(ax: plt.Axes, switch_events: pd.DataFrame) -> None:
    """在图中标注检测到的切换事件。"""

    if switch_events.empty or "global_step" not in switch_events:
        return
    for step in switch_events["global_step"].dropna().astype(int):
        ax.axvline(step, color="tab:red", alpha=0.22, linewidth=1.0)


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    """保存图表并释放 matplotlib 资源。"""

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _safe_mean(series: pd.Series | None) -> float:
    """计算均值；缺失或全空时返回 NaN。"""

    if series is None or series.isna().all():
        return float("nan")
    return float(series.mean())


def _safe_std(series: pd.Series | None) -> float:
    """计算标准差；缺失或全空时返回 NaN。"""

    if series is None or series.isna().all():
        return float("nan")
    return float(series.std(ddof=0))

