"""Semantic-conditioned SAR structural detail refinement for R7-S2FT."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SarStructuralRefinement(nn.Module):
    """Inject shallow SAR structure into the semantic decoder feature map.

    The refinement path is identity-initialized through a zero scalar so that
    enabling SDR cannot corrupt the semantic representation at initialization.
    """

    def __init__(self, semantic_channels: int, structure_channels: int = 64) -> None:
        super().__init__()
        if semantic_channels <= 0 or structure_channels <= 0:
            raise ValueError("SDR channel dimensions must be positive")
        self.semantic_channels = int(semantic_channels)
        self.structure_channels = int(structure_channels)
        self.shallow = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32),
            nn.Conv2d(32, structure_channels, 1),
            nn.GELU(),
        )
        self.structure_projection = nn.Conv2d(structure_channels * 2, semantic_channels, 1)
        self.semantic_gate = nn.Conv2d(semantic_channels, semantic_channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.shallow.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
        self.structure_projection.reset_parameters()
        self.semantic_gate.reset_parameters()
        with torch.no_grad():
            self.scale.zero_()

    def forward(self, semantic: Tensor, sar: Tensor, sar_mask: Tensor | None = None) -> Tensor:
        if semantic.ndim != 4:
            raise ValueError("semantic feature must be [B,C,H,W]")
        if sar.ndim != 4 or sar.shape[1] != 1 or sar.shape[0] != semantic.shape[0]:
            raise ValueError("SDR expects paired [B,1,H,W] SAR input")
        shallow = self.shallow(sar)
        if shallow.shape[2:] != semantic.shape[2:]:
            shallow = F.interpolate(shallow, size=semantic.shape[2:], mode="bilinear", align_corners=False)
        if sar_mask is not None:
            if sar_mask.ndim != 4 or sar_mask.shape[1] != 1:
                raise ValueError("SAR mask must be [B,1,H,W]")
            mask = sar_mask.to(shallow).clamp(0.0, 1.0)
            if mask.shape[2:] != shallow.shape[2:]:
                mask = F.interpolate(mask, size=shallow.shape[2:], mode="nearest")
            shallow = shallow * mask
        high = shallow - F.avg_pool2d(shallow, 3, stride=1, padding=1)
        structure = self.structure_projection(torch.cat((shallow, high), dim=1))
        gate = torch.sigmoid(self.semantic_gate(semantic))
        return semantic + self.scale * gate * structure
