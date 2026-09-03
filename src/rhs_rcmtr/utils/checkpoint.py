"""Auditable training checkpoint persistence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_training_checkpoint(
    path: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: Any, epoch: int, resolved_config_sha256: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    trainable_state = {name: value for name, value in model.state_dict().items() if name in trainable_names}
    torch.save({
        "trainable_model": trainable_state, "trainable_names": sorted(trainable_names),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "epoch": epoch,
        "resolved_config_sha256": resolved_config_sha256,
    }, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {"path": path.name, "sha256": digest, "epoch": epoch,
              "resolved_config_sha256": resolved_config_sha256}
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    return record


def load_training_checkpoint(
    path: Path, *, expected_sha256: str, model: nn.Module,
    optimizer: torch.optim.Optimizer | None, scheduler: Any | None, resolved_config_sha256: str,
) -> int:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("training checkpoint SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("resolved_config_sha256") != resolved_config_sha256:
        raise ValueError("training checkpoint configuration mismatch")
    expected_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(payload.get("trainable_names", [])) != expected_names:
        raise ValueError("training checkpoint trainability mask mismatch")
    result = model.load_state_dict(payload["trainable_model"], strict=False)
    if result.unexpected_keys:
        raise ValueError(f"unexpected training checkpoint keys: {result.unexpected_keys}")
    expected_missing = set(model.state_dict()) - expected_names
    if set(result.missing_keys) != expected_missing:
        raise ValueError("training checkpoint missing-key set does not match the frozen parameter mask")
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload["epoch"])
