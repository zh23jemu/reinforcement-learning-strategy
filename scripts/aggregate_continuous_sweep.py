"""汇总连续 OPS-DeMo sweep / confirm 结果。

读取 `runs/continuous_sweep_*/*/analysis/continuous_analysis_summary.json`
和 `runs/continuous_confirm_*/*/analysis/continuous_analysis_summary.json`，
提取参数、验收状态和核心效果指标，生成一个便于排序比较的 CSV。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_PREFIXES = (
    "continuous_sweep",
    "continuous_confirm",
    "continuous_sam_tune",
    "continuous_sam_multiseed_tune",
    "continuous_sam_geometry_narrow_tune",
    "continuous_sam_confirm",
    "continuous_sam_geometry_confirm",
    "continuous_oracle_compare",
    "continuous_response_focus",
)


SWEEP_PATTERN = re.compile(
    r"_is(?P<interceptor_speed>[^_]+)"
    r"_us(?P<intruder_speed>[^_]+)"
    r"_cr(?P<collision_radius>[^_]+)"
    r"_ts(?P<timesteps>[^_]+)"
    r"_s(?P<seed>[0-9]+)(?:_|$)"
)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="汇总连续 OPS-DeMo sweep 结果")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"), help="runs 根目录")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/continuous_sweep_summary.csv"),
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--run-prefix",
        action="append",
        dest="run_prefixes",
        default=None,
        help=(
            "要扫描的 runs 子目录前缀，可重复传入。默认同时扫描 "
            "continuous_sweep、continuous_confirm、continuous_sam_tune、"
            "continuous_sam_multiseed_tune、continuous_sam_geometry_narrow_tune、"
            "continuous_sam_confirm、continuous_sam_geometry_confirm 和 "
            "continuous_oracle_compare、continuous_response_focus。"
        ),
    )
    return parser


def main() -> None:
    """读取所有连续 sweep 分析摘要并写出总表。"""

    args = build_parser().parse_args()
    run_prefixes = tuple(args.run_prefixes or DEFAULT_RUN_PREFIXES)
    rows = [_flatten_summary(path) for path in _iter_summary_paths(args.runs_root, run_prefixes)]
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        frame = pd.DataFrame(rows)
        frame = frame.sort_values(
            ["engineering_pass", "win_rate_improvement", "reward_improvement"],
            ascending=[False, False, False],
        )
    else:
        frame = pd.DataFrame()
    frame.to_csv(output, index=False)
    print(f"已汇总 {len(rows)} 条连续 sweep 结果到: {output}")


def _iter_summary_paths(runs_root: Path, run_prefixes: tuple[str, ...]) -> list[Path]:
    """按运行目录前缀查找所有连续场景分析摘要。"""

    # 确认长训使用 continuous_confirm_* 前缀，300k 参数扫描使用
    # continuous_sweep_* 前缀；这里统一收集，避免长训完成后还需要换脚本。
    summary_paths: list[Path] = []
    for run_prefix in run_prefixes:
        summary_paths.extend(
            runs_root.glob(f"{run_prefix}_*/*/analysis/continuous_analysis_summary.json")
        )
    return sorted(set(summary_paths))


def _flatten_summary(summary_path: Path) -> dict[str, Any]:
    """把嵌套 JSON 摘要压平成 CSV 友好的单行记录。"""

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    experiment_name = summary_path.parents[2].name
    row: dict[str, Any] = {
        "experiment_name": experiment_name,
        # 统一使用 POSIX 风格路径，避免 Windows 本地自检时把已入库 CSV
        # 从 runs/foo/bar 改写成 runs\foo\bar，产生无意义的跨平台 diff。
        "run_dir": summary_path.parents[1].as_posix(),
        "episodes": data.get("episodes"),
        "total_steps": data.get("total_steps"),
        "mean_episode_reward": data.get("mean_episode_reward"),
        "baseline_mean_episode_reward": data.get("baseline_mean_episode_reward"),
        "reward_improvement": data.get("reward_improvement"),
        "interceptor_win_rate": data.get("interceptor_win_rate"),
        "baseline_interceptor_win_rate": data.get("baseline_interceptor_win_rate"),
        "win_rate_improvement": data.get("win_rate_improvement"),
        "response_policy_accuracy": data.get("response_policy_accuracy"),
        "switch_count": data.get("switch_count"),
        "process_pass": data.get("acceptance", {}).get("process_pass"),
        "engineering_pass": data.get("acceptance", {}).get("engineering_pass"),
        "paper_like_pass": data.get("acceptance", {}).get("paper_like_pass"),
        "judgement": data.get("acceptance", {}).get("judgement"),
    }
    row.update(_parse_experiment_name(experiment_name))
    return row


def _parse_experiment_name(experiment_name: str) -> dict[str, Any]:
    """从实验名解析 sweep 参数。"""

    match = SWEEP_PATTERN.search(experiment_name)
    if not match:
        return {}
    return {
        "interceptor_max_speed": _unslug_float(match.group("interceptor_speed")),
        "intruder_max_speed": _unslug_float(match.group("intruder_speed")),
        "collision_radius": _unslug_float(match.group("collision_radius")),
        "total_timesteps": int(match.group("timesteps")),
        "seed": int(match.group("seed")),
    }


def _unslug_float(value: str) -> float:
    """把目录名中的小数字符串还原为 float，例如 0p026 -> 0.026。"""

    return float(value.replace("p", "."))


if __name__ == "__main__":
    main()
