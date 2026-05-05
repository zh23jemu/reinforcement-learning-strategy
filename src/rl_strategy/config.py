"""配置读取工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """读取 YAML 配置文件。

    参数:
        path: 配置文件路径。

    返回:
        解析后的字典配置。
    """

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式无效: {path}")
    return config

