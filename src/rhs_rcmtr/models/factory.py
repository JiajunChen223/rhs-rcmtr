"""One detector factory for baseline and internal-mechanism variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from rhs_rcmtr.models.backbones import (
    CromaSarBaseAdapter,
    DinoV2B14Adapter,
    RandomSarSingleBandAdapter,
    SarEncoderV2,
)
from rhs_rcmtr.models.s2ft import S2FTConfig, S2FTSegmenter
from rhs_rcmtr.mechanisms.reliability_router import EvScrtRouter, OpticalAnchoredSarResidualRouter


# R7-S2FT is an architecture-level internal delta, not a router.  Historical
# R5/R6 mechanisms remain available unchanged for frozen re-runs.
ALLOWED_MECHANISMS = frozenset({"R5-C1-OA-SCRT", "R6-C1-EVSCRT", "R7-S2FT"})


@dataclass(frozen=True)
class ModelConfig:
    optical_channels: int = 3
    sar_channels: int = 1
    hidden_channels: int = 16
    num_classes: int = 8
    enabled_mechanism_ids: tuple[str, ...] = ()
    backbone_mode: str = "synthetic_fixture"
    optical_repo_path: str = ""
    optical_checkpoint_path: str = ""
    optical_checkpoint_sha256: str = ""
    sar_repo_path: str = ""
    sar_checkpoint_path: str = ""
    sar_checkpoint_sha256: str = ""
    sar_initialization_mode: str = "random_init_single_band"
    sar_encoder_variant: str = "v1"
    image_resolution: int = 112
    unimodal_mode: str = "multimodal"
    auxiliary_sar_head: bool = False
    s2ft: S2FTConfig = S2FTConfig()


class MultimodalSegmenter(nn.Module):
    """Common training object used by every experimental row."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        unknown = set(config.enabled_mechanism_ids) - ALLOWED_MECHANISMS
        if unknown:
            raise ValueError(f"undeclared mechanism(s): {sorted(unknown)}")
        if len(config.enabled_mechanism_ids) > 1:
            raise ValueError("the training object supports at most one internal mechanism")
        self.config = config
        if config.unimodal_mode not in {"multimodal", "optical_only", "sar_only"}:
            raise ValueError(f"unsupported unimodal_mode: {config.unimodal_mode}")
        mechanism_id = config.enabled_mechanism_ids[0] if config.enabled_mechanism_ids else ""
        self.router: nn.Module | None = None
        self.innovation: nn.Module | None = None
        self.sar_auxiliary_decoder: nn.Module | None = None

        if mechanism_id == "R7-S2FT":
            if config.optical_channels != 3 or config.sar_channels != 1:
                raise ValueError("R7-S2FT requires three optical channels and exactly one SAR channel")
            if config.auxiliary_sar_head:
                raise ValueError("R7-S2FT v1 does not use the legacy SAR auxiliary head")
            if config.backbone_mode == "synthetic_fixture":
                self.innovation = S2FTSegmenter.synthetic(
                    config=config.s2ft,
                    num_classes=config.num_classes,
                    unimodal_mode=config.unimodal_mode,
                )
            elif config.backbone_mode in {"official_cloud_frozen", "official_cloud_hybrid"}:
                self.innovation = S2FTSegmenter.official(
                    repo_path=Path(config.optical_repo_path),
                    checkpoint_path=Path(config.optical_checkpoint_path),
                    expected_sha256=config.optical_checkpoint_sha256,
                    config=config.s2ft,
                    num_classes=config.num_classes,
                    unimodal_mode=config.unimodal_mode,
                )
            else:
                raise ValueError(f"unsupported backbone_mode for R7-S2FT: {config.backbone_mode}")
            return

        h = config.hidden_channels
        if config.backbone_mode == "synthetic_fixture":
            self.optical_encoder = nn.Sequential(nn.Conv2d(config.optical_channels, h, 3, padding=1), nn.GELU())
            if config.sar_encoder_variant == "v2":
                self.sar_encoder = SarEncoderV2(input_channels=config.sar_channels, hidden_channels=h)
                self.sar_projection = nn.Conv2d(self.sar_encoder.output_channels, h, 1)
            elif config.sar_encoder_variant == "v1":
                self.sar_encoder = nn.Sequential(nn.Conv2d(config.sar_channels, h, 3, padding=1), nn.GELU())
                self.sar_projection = nn.Identity()
            else:
                raise ValueError(f"unsupported sar_encoder_variant: {config.sar_encoder_variant}")
            self.optical_projection = nn.Identity()
        elif config.backbone_mode in {"official_cloud_frozen", "official_cloud_hybrid"}:
            self.optical_encoder = DinoV2B14Adapter(
                Path(config.optical_repo_path), Path(config.optical_checkpoint_path),
                config.optical_checkpoint_sha256,
            )
            if config.sar_initialization_mode == "random_init_single_band":
                if config.sar_encoder_variant == "v2":
                    self.sar_encoder = SarEncoderV2(input_channels=config.sar_channels)
                elif config.sar_encoder_variant == "v1":
                    self.sar_encoder = RandomSarSingleBandAdapter(input_channels=config.sar_channels)
                else:
                    raise ValueError(f"unsupported sar_encoder_variant: {config.sar_encoder_variant}")
            elif config.sar_initialization_mode == "pretrained_croma_sar":
                self.sar_encoder = CromaSarBaseAdapter(
                    Path(config.sar_repo_path), Path(config.sar_checkpoint_path), config.image_resolution,
                    config.sar_checkpoint_sha256,
                )
            else:
                raise ValueError(f"unsupported SAR initialization mode: {config.sar_initialization_mode}")
            self.optical_projection = nn.Conv2d(self.optical_encoder.output_channels, h, 1)
            self.sar_projection = nn.Conv2d(self.sar_encoder.output_channels, h, 1)
        else:
            raise ValueError(f"unsupported backbone_mode: {config.backbone_mode}")
        if mechanism_id == "R5-C1-OA-SCRT":
            self.router = OpticalAnchoredSarResidualRouter(h)
        elif mechanism_id == "R6-C1-EVSCRT":
            self.router = EvScrtRouter(h)
        self.decoder = nn.Sequential(nn.Conv2d(h, h, 3, padding=1), nn.GELU(), nn.Conv2d(h, config.num_classes, 1))
        self.sar_auxiliary_decoder = (
            nn.Sequential(nn.Conv2d(h, h, 3, padding=1), nn.GELU(), nn.Conv2d(h, config.num_classes, 1))
            if config.auxiliary_sar_head
            else None
        )
        if config.unimodal_mode == "optical_only":
            self.sar_encoder.requires_grad_(False)
            self.sar_projection.requires_grad_(False)
        elif config.unimodal_mode == "sar_only":
            self.optical_encoder.requires_grad_(False)
            self.optical_projection.requires_grad_(False)

    def forward(
        self, optical: Tensor, sar: Tensor, reliability: Tensor,
        modality_mask: Tensor | None = None,
    ) -> dict[str, Tensor | str]:
        if self.innovation is not None:
            return self.innovation(optical, sar, reliability, modality_mask=modality_mask)  # type: ignore[operator]
        if self.config.unimodal_mode == "optical_only":
            optical_features = self.optical_projection(self.optical_encoder(optical))
            fused = optical_features
            weights = torch.zeros(
                (optical.shape[0], 2, optical_features.shape[2], optical_features.shape[3]),
                device=optical.device,
                dtype=optical.dtype,
            )
            weights[:, 0] = 1.0
            logits = self.decoder(fused)
            if logits.shape[2:] != optical.shape[2:]:
                logits = torch.nn.functional.interpolate(logits, size=optical.shape[2:], mode="bilinear", align_corners=False)
            if weights.shape[2:] != optical.shape[2:]:
                weights = torch.nn.functional.interpolate(weights, size=optical.shape[2:], mode="bilinear", align_corners=False)
            output: dict[str, Tensor | str] = {"logits": logits, "route_weights": weights}
            if self.training and self.sar_auxiliary_decoder is not None:
                output["sar_auxiliary_logits"] = self.sar_auxiliary_decoder(optical_features)
            return output
        if self.config.unimodal_mode == "sar_only":
            sar_features = self.sar_projection(self.sar_encoder(sar))
            fused = sar_features
            weights = torch.zeros(
                (sar.shape[0], 2, sar_features.shape[2], sar_features.shape[3]),
                device=sar.device,
                dtype=sar.dtype,
            )
            weights[:, 1] = 1.0
            logits = self.decoder(fused)
            if logits.shape[2:] != sar.shape[2:]:
                logits = torch.nn.functional.interpolate(logits, size=sar.shape[2:], mode="bilinear", align_corners=False)
            if weights.shape[2:] != sar.shape[2:]:
                weights = torch.nn.functional.interpolate(weights, size=sar.shape[2:], mode="bilinear", align_corners=False)
            output = {"logits": logits, "route_weights": weights}
            if self.training and self.sar_auxiliary_decoder is not None:
                output["sar_auxiliary_logits"] = self.sar_auxiliary_decoder(sar_features)
            return output
        optical_features = self.optical_projection(self.optical_encoder(optical))
        sar_features = self.sar_projection(self.sar_encoder(sar))
        if optical_features.shape[2:] != sar_features.shape[2:]:
            sar_features = torch.nn.functional.interpolate(
                sar_features, size=optical_features.shape[2:], mode="bilinear", align_corners=False
            )
        if modality_mask is not None:
            if modality_mask.ndim not in {2, 4} or modality_mask.shape[0] != optical.shape[0]:
                raise ValueError("modality_mask must be [B,2] or [B,2,H,W]")
            if modality_mask.ndim == 2:
                modality_mask = modality_mask[:, :, None, None]
            if modality_mask.shape[1] != 2:
                raise ValueError("modality_mask must have two modality channels")
            modality_mask = modality_mask.to(optical_features).clamp(0.0, 1.0)
            if modality_mask.shape[2:] != optical_features.shape[2:]:
                modality_mask = torch.nn.functional.interpolate(
                    modality_mask, size=optical_features.shape[2:], mode="nearest"
                )
            optical_features = optical_features * modality_mask[:, 0:1]
            sar_features = sar_features * modality_mask[:, 1:2]
        if reliability.shape[2:] != optical_features.shape[2:]:
            reliability = torch.nn.functional.interpolate(
                reliability, size=optical_features.shape[2:], mode="bilinear", align_corners=False
            )
        if self.router is None:
            weights = torch.full(
                (optical.shape[0], 2, optical.shape[2], optical.shape[3]),
                0.5,
                device=optical.device,
                dtype=optical.dtype,
            )
            fused = 0.5 * (optical_features + sar_features)
        else:
            fused, weights = self.router(optical_features, sar_features, reliability)
        logits = self.decoder(fused)
        target_size = optical.shape[2:]
        if logits.shape[2:] != target_size:
            logits = torch.nn.functional.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        if weights.shape[2:] != target_size:
            weights = torch.nn.functional.interpolate(weights, size=target_size, mode="bilinear", align_corners=False)
        output = {"logits": logits, "route_weights": weights}
        gate = getattr(self.router, "gate", None)
        if gate is not None:
            if gate.shape[2:] != target_size:
                gate = torch.nn.functional.interpolate(
                    gate, size=target_size, mode="bilinear", align_corners=False
                )
            output["gate"] = gate
        if self.training and self.sar_auxiliary_decoder is not None:
            auxiliary = self.sar_auxiliary_decoder(sar_features)
            if auxiliary.shape[2:] != target_size:
                auxiliary = torch.nn.functional.interpolate(
                    auxiliary, size=target_size, mode="bilinear", align_corners=False
                )
            output["sar_auxiliary_logits"] = auxiliary
        return output

    def forward_sar_only(self, sar: Tensor) -> dict[str, Tensor | str]:
        """Run the explicitly declared external SAR-only evaluation path."""
        if self.innovation is not None:
            return self.innovation.forward_sar_only(sar)  # type: ignore[union-attr]
        sar_features = self.sar_projection(self.sar_encoder(sar))
        logits = self.decoder(sar_features)
        if logits.shape[2:] != sar.shape[2:]:
            logits = torch.nn.functional.interpolate(logits, size=sar.shape[2:], mode="bilinear", align_corners=False)
        weights = torch.zeros((sar.shape[0], 2, sar.shape[2], sar.shape[3]), device=sar.device, dtype=sar.dtype)
        weights[:, 1] = 1.0
        return {"logits": logits, "route_weights": weights}


def build_model(config: ModelConfig) -> MultimodalSegmenter:
    """Build the shared end-to-end detector training object."""
    return MultimodalSegmenter(config)
