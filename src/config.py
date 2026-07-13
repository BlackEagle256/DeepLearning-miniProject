"""Central configuration loader.

All scripts and modules read settings from ``configs/config.yaml`` through
this module so that the whole project stays consistent and reproducible.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

# Repository root = two levels above this file (src/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: str | Path = CONFIG_PATH) -> dict:
    """Load and cache the global YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(key: str) -> Path:
    """Resolve a path declared in the ``paths`` section relative to the repo root."""
    cfg = load_config()
    return PROJECT_ROOT / cfg["paths"][key]


def get_seeds() -> list[int]:
    """Return the list of experiment seeds (5 seeds, per project spec)."""
    return list(load_config()["reproducibility"]["seeds"])


def get_base_seed() -> int:
    """Return the fixed base random seed (default 42)."""
    return int(load_config()["reproducibility"]["base_seed"])
