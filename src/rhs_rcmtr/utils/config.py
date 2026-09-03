"""Resolve immutable runtime and experiment configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be an object: {path}")
    return value


def resolve_config(runtime_path: Path, experiment_path: Path, initialization_path: Path) -> dict[str, Any]:
    runtime = _read_yaml(runtime_path)
    experiment = _read_yaml(experiment_path)
    initialization = _read_yaml(initialization_path)
    resolved = {"runtime": runtime, "experiment": experiment, "initialization": initialization}
    resolved["resolved_config_sha256"] = hashlib.sha256(
        json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return resolved
