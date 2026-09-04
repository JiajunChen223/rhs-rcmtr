"""End-to-end R7-S2FT segmentation model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rhs_rcmtr.models.s2ft.config import S2FTConfig
from rhs_rcmtr.models.s2ft.decoder import S2FTDecoder
from rhs_rcmtr.models.s2ft.shared_foundation import SharedDinoSensorBackbone


class S2FTSegmenter(nn.Module):
    """Sensor-Symmetric Foundation Transfer segmentation architecture."""

    def __init__(
        self,
        foundation: SharedDinoSensorBackbone,
        config: S2FTConfig,
        *,
        num_classes: int = 8,
        unimodal_mode: str = "multimodal",
    ) -> None:
        super().__init__()
        if unimodal_mode not in {"multimodal", "optical_only", "sar_only"}:
            raise ValueError(f"unsupported S2FT unimodal mode: {unimodal_mode}")
        self.config = config
        self.foundation = foundation
        self.unimodal_mode = unimodal_mode
        self.decoder = S2FTDecoder(
            feature_channels=foundation.output_channels,
            feature_layers=foundation.feature_layers,
            decoder_channels=config.decoder_channels,
            structure_channels=config.structure_channels,
            num_classes=num_classes,
            use_structural_refinement=config.use_structural_refinement,
        )

    @classmethod
    def official(
        cls,
        *,
        repo_path: Path,
        checkpoint_path: Path,
        expected_sha256: str,
        config: S2FTConfig,
        num_classes: int,
        unimodal_mode: str,
    ) -> "S2FTSegmenter":
        foundation = SharedDinoSensorBackbone.from_official(
            repo_path, checkpoint_path, expected_sha256, config
        )
        return cls(
            foundation, config, num_classes=num_classes, unimodal_mode=unimodal_mode
        )

    @classmethod
    def synthetic(
        cls,
        *,
        config: S2FTConfig,
        num_classes: int,
        unimodal_mode: str,
    ) -> "S2FTSegmenter":
        return cls(
            SharedDinoSensorBackbone.synthetic(config),
            config,
            num_classes=num_classes,
            unimodal_mode=unimodal_mode,
        )

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.foundation.dino.eval()
        return self

    def reset_innovation_parameters(self) -> None:
        self.foundation.reset_innovation_parameters()
        self.decoder.reset_parameters()

    def _effective_mask(
        self,
        optical: Tensor,
        modality_mask: Tensor | None,
    ) -> Tensor | None:
        if modality_mask is not None:
            if modality_mask.ndim not in {2, 4} or modality_mask.shape[0] != optical.shape[0]:
                raise ValueError("modality_mask must be [B,2] or [B,2,H,W]")
            if modality_mask.shape[1] != 2:
                raise ValueError("modality_mask must contain two modality channels")
        if self.unimodal_mode == "multimodal":
            return modality_mask
        forced = torch.zeros(optical.shape[0], 2, device=optical.device, dtype=optical.dtype)
        forced[:, 0 if self.unimodal_mode == "optical_only" else 1] = 1.0
        if modality_mask is None:
            return forced
        if modality_mask.ndim == 2:
            return modality_mask.to(forced) * forced
        return modality_mask.to(optical) * forced[:, :, None, None]

    @staticmethod
    def _compatibility_weights(
        optical: Tensor,
        modality_mask: Tensor | None,
    ) -> Tensor:
        b, _, h, w = optical.shape
        if modality_mask is None:
            return torch.full((b, 2, h, w), 0.5, device=optical.device, dtype=optical.dtype)
        if modality_mask.ndim == 2:
            weights = modality_mask[:, :, None, None].to(optical).expand(-1, -1, h, w)
        else:
            weights = modality_mask.to(optical)
            if weights.shape[2:] != (h, w):
                weights = F.interpolate(weights, size=(h, w), mode="nearest")
        weights = weights.clamp(0.0, 1.0)
        denom = weights.sum(dim=1, keepdim=True)
        fallback = torch.full_like(weights, 0.5)
        return torch.where(denom > 0, weights / denom.clamp_min(1e-6), fallback)

    def forward(
        self,
        optical: Tensor,
        sar: Tensor,
        reliability: Tensor | None = None,
        modality_mask: Tensor | None = None,
    ) -> dict[str, Tensor | str]:
        del reliability  # R7 forward is deliberately independent of reliability evidence.
        effective_mask = self._effective_mask(optical, modality_mask)
        optical_features, sar_features, sar_aligned, _optical_mask, sar_mask = self.foundation(
            optical, sar, effective_mask
        )
        logits = self.decoder(optical_features, sar_features, sar_aligned, sar_mask)
        if logits.shape[2:] != optical.shape[2:]:
            logits = F.interpolate(logits, size=optical.shape[2:], mode="bilinear", align_corners=False)
        weights = self._compatibility_weights(optical, effective_mask)
        return {
            "logits": logits,
            "route_weights": weights,
            "route_weights_semantics": "compatibility_constant_not_model_output",
        }

    def forward_sar_only(self, sar: Tensor) -> dict[str, Tensor | str]:
        if sar.ndim != 4 or sar.shape[1] != 1:
            raise ValueError("S2FT SAR-only path expects [B,1,H,W]")
        optical = torch.zeros(
            sar.shape[0], 3, sar.shape[2], sar.shape[3], device=sar.device, dtype=sar.dtype
        )
        mask = torch.zeros(sar.shape[0], 2, device=sar.device, dtype=sar.dtype)
        mask[:, 1] = 1.0
        output = self.forward(optical, sar, None, modality_mask=mask)
        return output
