"""Auditable checkpoint loading; checkpoint binaries stay cloud-only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


WEIGHT_ROOT = Path(chr(47), "root", "autodl-tmp", "weights")


def load_audited_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    expected_sha256: str,
    allowed_missing_prefixes: tuple[str, ...] = ("decoder.",),
    allow_synthetic_fixture: bool = False,
) -> dict[str, Any]:
    """Load a cloud checkpoint and return an explicit compatibility report."""
    resolved_path = checkpoint_path
    if not allow_synthetic_fixture and WEIGHT_ROOT not in resolved_path.parents:
        raise ValueError("checkpoint must remain under the cloud-only weight root")
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state dictionary")
    model_state = model.state_dict()
    shape_mismatches = [
        key for key, value in state.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    ]
    if shape_mismatches:
        raise ValueError(f"unexplained shape mismatch: {shape_mismatches}")
    compatible = {key: value for key, value in state.items() if key in model_state}
    missing = sorted(set(model_state) - set(compatible))
    unexpected = sorted(set(state) - set(model_state))
    unexplained_missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]
    if unexplained_missing or unexpected:
        raise ValueError(f"unexplained checkpoint keys: missing={unexplained_missing}, unexpected={unexpected}")
    model.load_state_dict(compatible, strict=False)
    return {
        "status": "pass",
        "checkpoint_sha256": digest,
        "architecture_match": True,
        "head_replacement_documented": bool(missing),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
    }


def verify_pretrained_audit(
    path: Path, expected_sha256: str, expected_initialization: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Verify the immutable cloud audit before constructing official backbones."""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("pretrained audit SHA256 mismatch")
    audit = json.loads(content)
    if audit.get("status") != "pass" or audit.get("protocol_ready") is not True:
        raise ValueError("pretrained audit is not protocol-ready")
    if audit.get("test_accessed") is not False:
        raise ValueError("pretrained audit must not access target test data")
    if expected_initialization is not None:
        expected_mode = str(expected_initialization.get("mode") or "")
        if audit.get("initialization_mode") != expected_mode:
            raise ValueError("pretrained audit initialization mode does not match resolved configuration")
        expected_sar_mode = str(expected_initialization.get("sar_initialization_mode") or "")
        if expected_sar_mode == "random_init_single_band":
            fallback = audit.get("sar_fallback")
            if not isinstance(fallback, dict) or fallback.get("status") != "pass":
                raise ValueError("resolved single-band fallback is not documented by the pretrained audit")
        input_spec = audit.get("input_spec")
        if not isinstance(input_spec, dict):
            raise ValueError("pretrained audit input_spec is required")
        if int(input_spec.get("optical_channels", -1)) != 3:
            raise ValueError("pretrained audit optical input channels do not match configuration")
        if expected_sar_mode == "random_init_single_band" and int(input_spec.get("sar_channels", -1)) != 1:
            raise ValueError("pretrained audit SAR input channels do not match single-band fallback")
    init_mode = str(audit.get("initialization_mode") or "")
    if init_mode == "hybrid_pretrained_with_sar_fallback":
        fallback = audit.get("sar_fallback")
        if not isinstance(fallback, dict) or fallback.get("status") != "pass" or fallback.get("reason") == "":
            raise ValueError("hybrid initialization requires a documented SAR fallback audit")
    for item in audit.get("selected_checkpoints", []):
        cloud_path = str(item.get("cloud_path") or "")
        if WEIGHT_ROOT not in Path(cloud_path).parents:
            raise ValueError("audited checkpoint escaped cloud weight root")
        if len(str(item.get("sha256") or "")) != 64:
            raise ValueError("audited checkpoint requires SHA256")
    return audit


def audited_checkpoint_hashes(audit: dict[str, Any], expected_paths: list[str]) -> dict[str, str]:
    """Bind configured checkpoint paths to audited metadata and live cloud bytes."""
    selected = {str(item.get("cloud_path") or ""): str(item.get("sha256") or "")
                for item in audit.get("selected_checkpoints", [])}
    missing = sorted(set(expected_paths) - set(selected))
    if missing:
        raise ValueError(f"configured checkpoint absent from audit: {missing}")
    result: dict[str, str] = {}
    for raw_path in expected_paths:
        path = Path(raw_path)
        expected = selected[raw_path]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"audited checkpoint bytes changed: {raw_path}")
        result[raw_path] = digest
    return result
