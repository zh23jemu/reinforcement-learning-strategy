from argparse import Namespace
from pathlib import Path

from scripts.run_continuous_oracle_compare import (
    _apply_overrides,
    _build_experiment_name,
)


def test_oracle_compare_overrides_detector_and_paths():
    """oracle 对照脚本应只在运行时改配置，并写入独立目录。"""

    config = {
        "experiment": {
            "name": "base",
            "seed": 1,
            "artifact_dir": "artifacts/continuous",
        },
        "environment": {
            "interceptor_max_speed": 0.022,
            "intruder_max_speed": 0.018,
            "collision_radius": 0.08,
        },
        "ppo": {"total_timesteps": 1000},
        "evaluation": {"episodes": 10},
        "detector": {"method": "sam", "initial_policy": "direct"},
    }
    args = Namespace(
        interceptor_speed=0.03,
        intruder_speed=0.016,
        collision_radius=0.08,
        timesteps=1500,
        seed=43,
        episodes=800,
        artifact_dir=None,
    )

    _apply_overrides(config, args, "continuous_oracle_compare_is0p03_us0p016_cr0p08_ts1500_s43")

    assert config["experiment"]["name"] == "continuous_oracle_compare_is0p03_us0p016_cr0p08_ts1500_s43"
    assert config["experiment"]["seed"] == 43
    assert Path(config["experiment"]["artifact_dir"]).parts[-3:] == (
        "artifacts",
        "continuous_sweep",
        "continuous_oracle_compare_is0p03_us0p016_cr0p08_ts1500_s43",
    )
    assert config["environment"]["interceptor_max_speed"] == 0.03
    assert config["environment"]["intruder_max_speed"] == 0.016
    assert config["ppo"]["total_timesteps"] == 1500
    assert config["evaluation"]["episodes"] == 800
    assert config["detector"]["method"] == "oracle"
    assert config["detector"]["initial_policy"] == "direct"


def test_oracle_compare_accepts_explicit_artifact_dir():
    """显式 artifact-dir 用于复用已有模型时必须原样保留。"""

    config = {
        "experiment": {"name": "base", "seed": 1, "artifact_dir": "artifacts/continuous"},
        "environment": {
            "interceptor_max_speed": 0.022,
            "intruder_max_speed": 0.018,
            "collision_radius": 0.08,
        },
        "ppo": {"total_timesteps": 1000},
        "evaluation": {"episodes": 10},
        "detector": {"method": "sam"},
    }
    args = Namespace(
        interceptor_speed=0.03,
        intruder_speed=0.016,
        collision_radius=0.08,
        timesteps=1500,
        seed=42,
        episodes=None,
        artifact_dir=Path("artifacts/reuse"),
    )

    _apply_overrides(config, args, "oracle_reuse")

    assert Path(config["experiment"]["artifact_dir"]).parts == ("artifacts", "reuse")
    assert config["evaluation"]["episodes"] == 10
    assert config["detector"]["method"] == "oracle"
    assert config["detector"]["initial_policy"] == "direct"


def test_oracle_compare_experiment_name_matches_aggregator_pattern():
    """实验名沿用 sweep 命名格式，保证聚合脚本能解析速度、半径和 seed。"""

    name = _build_experiment_name(
        prefix="continuous_oracle_compare",
        interceptor_speed=0.03,
        intruder_speed=0.016,
        collision_radius=0.08,
        timesteps=1500000,
        seed=44,
    )

    assert name == "continuous_oracle_compare_is0p03_us0p016_cr0p08_ts1500000_s44"
