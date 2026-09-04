"""Multi-stage semantic decoder for R7-S2FT."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from rhs_rcmtr.mechanisms.structural_refinement import SarStructuralRefinement


class _StageFusion(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(input_channels * 2, output_channels, 1),
            nn.GELU(),
        )

    def forward(self, optical: Tensor, sar: Tensor) -> Tensor:
        if optical.shape != sar.shape:
            raise ValueError("stage optical/SAR features must have identical shapes")
        return self.projection(__import__("torch").cat((optical, sar), dim=1))


class S2FTDecoder(nn.Module):
    """Fuse block-4/8/12 features and optionally refine them with SAR structure."""

    def __init__(
        self,
        *,
        feature_channels: int,
        feature_layers: Sequence[int],
        decoder_channels: int,
        structure_channels: int,
        num_classes: int,
        use_structural_refinement: bool,
    ) -> None:
        super().__init__()
        if not feature_layers:
            raise ValueError("S2FT decoder requires at least one feature layer")
        self.feature_layers = tuple(int(v) for v in feature_layers)
        self.stage_fusions = nn.ModuleDict(
            {
                str(index): _StageFusion(feature_channels, decoder_channels)
                for index in self.feature_layers
            }
        )
        self.post = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.GELU(),
        )
        self.structural_refinement = (
            SarStructuralRefinement(decoder_channels, structure_channels)
            if use_structural_refinement
            else None
        )
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)

    def reset_parameters(self) -> None:
        for module in self.stage_fusions.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
        for module in self.post.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
        if self.structural_refinement is not None:
            self.structural_refinement.reset_parameters()
        self.classifier.reset_parameters()

    def forward(
        self,
        optical_features: dict[int, Tensor],
        sar_features: dict[int, Tensor],
        sar_aligned: Tensor,
        sar_mask: Tensor | None,
    ) -> Tensor:
        semantic: Tensor | None = None
        for index in self.feature_layers:
            if index not in optical_features or index not in sar_features:
                raise ValueError(f"missing S2FT feature layer {index}")
            stage = self.stage_fusions[str(index)](
                optical_features[index], sar_features[index]
            )
            semantic = stage if semantic is None else semantic + stage
        if semantic is None:
            raise RuntimeError("S2FT decoder produced no semantic feature")
        semantic = self.post(semantic)
        if self.structural_refinement is not None:
            semantic = self.structural_refinement(semantic, sar_aligned, sar_mask)
        return self.classifier(semantic)
