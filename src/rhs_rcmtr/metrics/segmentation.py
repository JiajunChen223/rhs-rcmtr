"""Leakage-free segmentation metrics."""

from __future__ import annotations

import torch
from torch import Tensor


def mean_iou(logits: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    predictions = logits.argmax(dim=1)
    valid_pixels = labels != 255
    values: list[Tensor] = []
    for class_id in range(num_classes):
        predicted = (predictions == class_id) & valid_pixels
        actual = (labels == class_id) & valid_pixels
        union = (predicted | actual).sum()
        if union.item() > 0:
            values.append((predicted & actual).sum().float() / union.float())
    return torch.stack(values).mean() if values else torch.zeros((), device=logits.device)
