"""Sensor-specific parameter-efficient adapters used inside a shared foundation model."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SensorAdapter(nn.Module):
    """Residual bottleneck adapter with an exact identity initialization."""

    def __init__(self, channels: int, bottleneck: int) -> None:
        super().__init__()
        if channels <= 0 or bottleneck <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.norm = nn.LayerNorm(channels)
        self.down = nn.Linear(channels, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.down.reset_parameters()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, patch_tokens: Tensor) -> Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError("SensorAdapter expects [B,N,C] patch tokens")
        return patch_tokens + self.up(self.act(self.down(self.norm(patch_tokens))))


class SensorAdapterPair(nn.Module):
    """Independent optical/SAR adapters with a shared architecture."""

    def __init__(self, channels: int, bottleneck: int) -> None:
        super().__init__()
        self.optical = SensorAdapter(channels, bottleneck)
        self.sar = SensorAdapter(channels, bottleneck)

    def reset_parameters(self) -> None:
        self.optical.reset_parameters()
        self.sar.reset_parameters()
