"""YAML config loading.

Step 2 version: minimal loader that returns parsed YAML as a dict.
Step 3 will add Pydantic validation and hot-reload.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    """Load and parse the router config YAML."""
    with path.open() as f:
        return yaml.safe_load(f)
