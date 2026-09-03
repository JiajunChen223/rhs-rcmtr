import hashlib

import pytest
import torch

from rhs_rcmtr.models import ModelConfig, build_model
from rhs_rcmtr.utils.checkpoint import load_training_checkpoint, save_training_checkpoint


def test_trainable_only_checkpoint_round_trip(tmp_path) -> None:
    model = build_model(ModelConfig(enabled_mechanism_ids=("R5-C1-OA-SCRT",)))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    path = tmp_path / ("last" + "." + "pt")
    record = save_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, epoch=1,
        resolved_config_sha256="a" * 64,
    )
    assert set(torch.load(path, map_location="cpu", weights_only=True)) >= {"trainable_model", "trainable_names"}
    assert load_training_checkpoint(
        path, expected_sha256=record["sha256"], model=model, optimizer=optimizer,
        scheduler=scheduler, resolved_config_sha256="a" * 64,
    ) == 1


def test_checkpoint_config_mismatch_fails(tmp_path) -> None:
    model = build_model(ModelConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    path = tmp_path / ("last" + "." + "pt")
    save_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, epoch=1,
        resolved_config_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="configuration"):
        load_training_checkpoint(
            path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), model=model,
            optimizer=optimizer, scheduler=scheduler, resolved_config_sha256="b" * 64,
        )
