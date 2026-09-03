"""The shared end-to-end training/evaluation graph."""

from __future__ import annotations

import torch

from rhs_rcmtr.losses import segmentation_with_honesty_loss
from rhs_rcmtr.metrics import mean_iou
from rhs_rcmtr.models import ModelConfig, build_model


def run_synthetic_step(config: ModelConfig, *, train: bool, seed: int = 0) -> dict[str, float]:
    """Exercise the full graph on deterministic generated tensors only."""
    torch.manual_seed(seed)
    model = build_model(config)
    optical = torch.randn(2, config.optical_channels, 16, 16)
    sar = torch.randn(2, config.sar_channels, 16, 16)
    reliability = torch.rand(2, 2, 16, 16).add_(0.1)
    labels = torch.randint(0, config.num_classes, (2, 16, 16))
    outputs = model(optical, sar, reliability)
    loss, parts = segmentation_with_honesty_loss(
        outputs["logits"], labels, outputs["route_weights"], reliability,
        honesty_weight=0.1 if model.router is not None else 0.0,
    )
    if train:
        torch.optim.AdamW(model.parameters(), lr=1e-3).zero_grad(set_to_none=True)
        loss.backward()
    return {
        "loss": float(loss.detach()),
        "segmentation_loss": float(parts["segmentation"]),
        "honesty_loss": float(parts["honesty"]),
        "mean_iou": float(mean_iou(outputs["logits"], labels, config.num_classes)),
    }
