"""Single source of truth for configuration and paths.

Every path in ``configs/main.yaml`` is relative to the repository root and is
resolved here at runtime.  No absolute path is stored in the repository, so the
project runs unchanged on Windows, Colab and Kaggle.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "main.yaml"

# Machine-specific override; never stored in the config file.
HF_HOME_ENV = "HF_HOME"


def _resolve(value: str) -> Path:
    """Resolve a config path against the repo root.

    An absolute value is honoured (so a Kaggle working dir or an HF_HOME can be
    injected from outside) but never written into the repository.
    """
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p)


@lru_cache(maxsize=8)
def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["_config_path"] = str(cfg_path.relative_to(ROOT))
    cfg["paths"] = {k: _resolve(v) for k, v in cfg["paths"].items()}

    # HF_HOME from the environment wins over the config default.
    env_hf = os.environ.get(HF_HOME_ENV)
    if env_hf:
        cfg["paths"]["hf_cache"] = Path(env_hf)
    return cfg


def paths(config_path: str | None = None) -> dict[str, Path]:
    return load_config(config_path)["paths"]


def apply_hf_cache(config_path: str | None = None) -> Path:
    """Point HuggingFace at the configured cache before any model is loaded."""
    cache = paths(config_path)["hf_cache"]
    cache.mkdir(parents=True, exist_ok=True)
    os.environ[HF_HOME_ENV] = str(cache)
    return cache
