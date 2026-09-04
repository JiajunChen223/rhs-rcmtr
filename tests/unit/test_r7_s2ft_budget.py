"""Architecture-budget guard for the full 768-dimensional R7-S2FT innovation."""

from __future__ import annotations

import torch
from torch import nn

from rhs_rcmtr.mechanisms.geo_local_interaction import GeoLocalDeformableCrossInteraction
from rhs_rcmtr.models.s2ft.config import S2FTConfig
from rhs_rcmtr.models.s2ft.decoder import S2FTDecoder
from rhs_rcmtr.models.s2ft.patch_transplant import SarFoundationPatchEmbed
from rhs_rcmtr.models.s2ft.sensor_adapter import SensorAdapterPair


class _DinoPatchFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 768, 14, stride=14)
        self.norm = nn.Identity()


def test_s2ft_declared_innovation_parameter_budget() -> None:
    config = S2FTConfig()
    modules = nn.ModuleList(
        [
            SarFoundationPatchEmbed(_DinoPatchFixture()),
            nn.ModuleDict(
                {
                    str(index): SensorAdapterPair(768, config.adapter_bottleneck)
                    for index in config.adapter_layers
                }
            ),
            nn.ModuleDict(
                {
                    str(index): GeoLocalDeformableCrossInteraction(
                        768,
                        interaction_dim=config.interaction_dim,
                        max_offset_tokens=config.max_offset_tokens,
                    )
                    for index in config.interaction_layers
                }
            ),
            S2FTDecoder(
                feature_channels=768,
                feature_layers=config.adapter_layers,
                decoder_channels=config.decoder_channels,
                structure_channels=config.structure_channels,
                num_classes=8,
                use_structural_refinement=config.use_structural_refinement,
            ),
        ]
    )
    trainable = sum(parameter.numel() for parameter in modules.parameters() if parameter.requires_grad)
    # Frozen v1 design budget: enough capacity for performance-oriented fusion,
    # while remaining far below a second trainable ViT-B/14 backbone.
    assert trainable <= 4_500_000
