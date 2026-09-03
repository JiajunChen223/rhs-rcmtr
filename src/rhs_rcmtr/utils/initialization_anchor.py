"""Tensor-level common initialization anchors for fair candidate comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


ANCHOR_SCHEMA = "common-init-anchor-v1"
UNANCHORED_SUPPORT_PREFIXES = ("sar_auxiliary_decoder.", "auxiliary_head.")


def _is_common_trainable(name: str, parameter: nn.Parameter) -> bool:
    return bool(parameter.requires_grad) and not name.startswith("router.")


def common_parameter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return cloned CPU tensors for all non-mechanism trainable parameters."""

    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if _is_common_trainable(name, parameter)
    }


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def save_common_init_anchor(
    path: Path,
    model: nn.Module,
    *,
    common_init_seed: int,
    resolved_config_sha256: str,
    data_order_seed: int,
    augmentation_seed: int,
) -> dict[str, Any]:
    """Save only common trainable tensors; intended for the cloud host."""

    state = common_parameter_state(model)
    payload = {
        "artifact_type": "common_initialization_anchor",
        "schema_version": ANCHOR_SCHEMA,
        "common_parameters": state,
        "common_parameter_names": sorted(state),
        "common_parameter_state_sha256": _state_digest(state),
        "common_init_seed": int(common_init_seed),
        "data_order_seed": int(data_order_seed),
        "augmentation_seed": int(augmentation_seed),
        "resolved_config_sha256": resolved_config_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "artifact_type": "common_initialization_anchor_receipt",
        "schema_version": ANCHOR_SCHEMA,
        "status": "pass",
        "path": str(path),
        "sha256": digest,
        "common_parameter_state_sha256": payload["common_parameter_state_sha256"],
        "common_parameter_count": len(state),
        "common_parameter_names": sorted(state),
        "common_init_seed": int(common_init_seed),
        "data_order_seed": int(data_order_seed),
        "augmentation_seed": int(augmentation_seed),
        "resolved_config_sha256": resolved_config_sha256,
        "test_accessed": False,
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def load_common_init_anchor(
    path: Path,
    *,
    expected_sha256: str,
    model: nn.Module,
    expected_common_init_seed: int | None = None,
) -> dict[str, Any]:
    """Load an anchor and require exact common names/shapes before copying."""

    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError("common initialization anchor SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != ANCHOR_SCHEMA:
        raise ValueError("common initialization anchor schema mismatch")
    if expected_common_init_seed is not None:
        payload_seed = payload.get("common_init_seed")
        if payload_seed != int(expected_common_init_seed):
            raise ValueError(
                "common initialization anchor seed mismatch: "
                f"anchor={payload_seed} required={int(expected_common_init_seed)}"
            )
    state = payload.get("common_parameters")
    if not isinstance(state, dict) or not state:
        raise ValueError("common initialization anchor has no common parameters")
    expected = common_parameter_state(model)
    expected_names = set(expected)
    actual_names = set(state)
    # A unimodal diagnostic freezes one encoder branch.  The baseline anchor
    # legitimately contains that branch's trainable tensors, while the
    # diagnostic needs only the active trainable subset.  Require every active
    # name to be present, but allow anchor-only names to remain unused.
    missing = sorted(expected_names - actual_names)
    unsupported_missing = [
        name for name in missing
        if not name.startswith(UNANCHORED_SUPPORT_PREFIXES)
    ]
    if unsupported_missing:
        raise ValueError(
            "common initialization parameter names mismatch: "
            f"missing={unsupported_missing}"
        )
    for name in expected_names & actual_names:
        value = state[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(expected[name].shape):
            raise ValueError(f"common initialization parameter shape mismatch: {name}")
    expected_state_digest = _state_digest({name: state[name] for name in state})
    if expected_state_digest != payload.get("common_parameter_state_sha256"):
        raise ValueError("common initialization parameter state digest mismatch")
    for name, parameter in model.named_parameters():
        if _is_common_trainable(name, parameter) and name in state:
            parameter.data.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
            if not torch.equal(parameter.detach().cpu(), state[name].detach().cpu()):
                raise ValueError(f"common initialization tensor copy mismatch: {name}")
    return {
        "status": "pass",
        "path": str(path),
        "sha256": actual,
        "common_parameter_state_sha256": expected_state_digest,
        "common_parameter_count": len(expected),
        "common_parameter_names": sorted(expected),
        "anchor_parameter_count": len(state),
        "anchor_parameter_names": sorted(state),
        "common_init_seed": payload.get("common_init_seed"),
        "data_order_seed": payload.get("data_order_seed"),
        "augmentation_seed": payload.get("augmentation_seed"),
        "resolved_config_sha256": payload.get("resolved_config_sha256"),
    }


def reset_innovation_parameters(model: nn.Module, innovation_init_seed: int) -> None:
    """Reset only the internal mechanism under its dedicated RNG stream."""

    router = getattr(model, "router", None)
    if router is None:
        return
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(innovation_init_seed))
        for module in router.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
