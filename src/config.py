"""Single source of truth for locating ``config/config.yaml``.

Notebooks run from ``notebooks/`` and tests run from the repository root;
resolving the project root once here removes the fragile ``../`` prefixes
and lets both entry points share identical paths.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

# src/config.py -> src -> repository root
ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "config" / "config.yaml"


def project_root() -> Path:
    """Absolute path to the repository root."""
    return ROOT


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    """Load the YAML config once per process.

    The returned mapping is cached; callers must treat it as read-only so a
    stray mutation in one notebook cell cannot silently change another's
    behaviour.
    """
    cfg_path = Path(path) if path is not None else CONFIG_PATH
    with open(cfg_path) as f:
        return yaml.safe_load(f)
