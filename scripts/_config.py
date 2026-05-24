"""Tiny YAML loader + deep merge for layered configs.

All scripts read configs via these helpers so we never depend on hydra/omegaconf.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(a: dict, b: dict) -> dict:
    """Return a new dict with b's keys overriding a's, recursing into dicts."""
    out = deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_layered(*paths: str | Path) -> dict[str, Any]:
    """Load and deep-merge YAML configs in order (later overrides earlier)."""
    out: dict[str, Any] = {}
    for p in paths:
        out = deep_merge(out, load_yaml(p))
    return out
