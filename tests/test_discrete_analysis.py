import json

import pandas as pd

from rl_strategy.discrete.analysis import analyze_discrete_run


def test_analyze_discrete_run_generates_figures_with_empty_switch_file(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    pd.DataFrame(
        [
            {
                "episode": 0,
                "global_step": 1,
                "episode_step": 1,
                "reward": -1.0,
                "true_policy": "chase_x",
                "response_policy": "chase_x",
                "response_policy_correct": True,
                "assumed_policy": "chase_x",
                "assumption_correct": True,
                "running_error_chase_x": 0.1,
                "running_error_chase_y": 0.3,
            },
            {
                "episode": 0,
                "global_step": 2,
                "episode_step": 2,
                "reward": 2.0,
                "true_policy": "chase_y",
                "response_policy": "chase_x",
                "response_policy_correct": False,
                "assumed_policy": "chase_x",
                "assumption_correct": False,
                "running_error_chase_x": 0.4,
                "running_error_chase_y": 0.2,
            },
        ]
    ).to_csv(run_dir / "step_trace.csv", index=False)
    pd.DataFrame(
        [
            {
                "episode": 0,
                "global_step": 1,
                "episode_step": 1,
                "reward": -2.0,
                "true_policy": "chase_x",
            }
        ]
    ).to_csv(run_dir / "baseline_step_trace.csv", index=False)
    (run_dir / "switch_events.csv").write_text("", encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"episodes": 1}), encoding="utf-8")

    summary = analyze_discrete_run(run_dir)

    analysis_dir = run_dir / "analysis"
    assert summary["episodes"] == 1
    assert summary["switch_count"] == 0
    assert summary["response_policy_accuracy"] == 0.5
    assert (analysis_dir / "analysis_summary.json").exists()
    assert (analysis_dir / "episode_metrics.csv").exists()
    assert (analysis_dir / "running_errors.png").exists()
    assert (analysis_dir / "reward_comparison.png").exists()

