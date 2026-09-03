"""Frozen official VFM adapters; all repositories and weights are cloud-only."""

from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F



def _verify_checkpoint(path: Path, expected_sha256: str) -> None:
    if len(expected_sha256) != 64:
        raise ValueError(f"missing audited checkpoint SHA256: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"checkpoint SHA256 mismatch: {path}")


def dino_input_grid(images: Tensor, patch_size: int = 14) -> Tensor:
    """Resize optical rasters to the nearest lower patch-aligned grid."""
    if images.ndim != 4:
        raise ValueError("DINOv2 optical input must be a four-dimensional tensor")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    height, width = images.shape[-2:]
    target_height = max(patch_size, (height // patch_size) * patch_size)
    target_width = max(patch_size, (width // patch_size) * patch_size)
    if (target_height, target_width) == (height, width):
        return images
    return F.interpolate(
        images, size=(target_height, target_width), mode="bilinear", align_corners=False,
    )


class DinoV2B14Adapter(nn.Module):
    """Official local-repository DINOv2 ViT-B/14 patch-token adapter."""

    output_channels = 768

    def __init__(self, repo_path: Path, checkpoint_path: Path, expected_sha256: str) -> None:
        super().__init__()
        _verify_checkpoint(checkpoint_path, expected_sha256)
        model = torch.hub.load(str(repo_path), "dinov2_vitb14", source="local", pretrained=False)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state, strict=True)
        model.requires_grad_(False).eval()
        self.model = model

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            result = self.model.forward_features(dino_input_grid(images))
            tokens = result["x_norm_patchtokens"]
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise ValueError("DINOv2 patch-token count is not a square grid")
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)


class RandomSarSingleBandAdapter(nn.Module):
    """Explicit one-band SAR fallback when no compatible two-band VFM exists."""

    output_channels = 64

    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        if input_channels != 1:
            raise ValueError("the random SAR fallback is defined for exactly one input band")
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, self.output_channels, 3, padding=1), nn.GELU(),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.shape[1] != 1:
            raise ValueError("random SAR fallback requires one SAR channel")
        return self.model(images)


class _ResBlock(nn.Module):
    """Residual block used by the R6 SAR encoder (stride-2 stream supported)."""

    def __init__(self, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()
        if stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(channels, channels, 1, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv2(self.act(self.conv1(x))) + self.shortcut(x))


class SarEncoderV2(nn.Module):
    """R6 SAR-DE encoder: four-layer ResBlock backbone with a stride-2 multi-scale
    stream fused back to the shared grid (one input band, budget < 1.5M params)."""

    output_channels = 64

    def __init__(self, input_channels: int = 1, hidden_channels: int = 64) -> None:
        super().__init__()
        if input_channels != 1:
            raise ValueError("SarEncoderV2 is defined for exactly one input band")
        self.stem = nn.Sequential(nn.Conv2d(input_channels, hidden_channels, 3, padding=1), nn.GELU())
        self.block_low = _ResBlock(hidden_channels, stride=2)
        self.block_main = _ResBlock(hidden_channels, stride=1)
        self.tail = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_channels, self.output_channels, 1),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.shape[1] != 1:
            raise ValueError("SarEncoderV2 requires one SAR channel")
        h = self.stem(images)
        low = self.block_low(h)
        low_up = F.interpolate(low, size=h.shape[2:], mode="bilinear", align_corners=False)
        main = self.block_main(h)
        return self.tail(torch.cat((main, low_up), dim=1))


class CromaSarBaseAdapter(nn.Module):
    """Official CROMA base SAR encoder adapter using the project's use_croma.py."""

    output_channels = 768

    def __init__(
        self, repo_path: Path, checkpoint_path: Path, image_resolution: int, expected_sha256: str
    ) -> None:
        super().__init__()
        _verify_checkpoint(checkpoint_path, expected_sha256)
        source = repo_path / "use_croma.py"
        spec = importlib.util.spec_from_file_location("official_croma_use", source)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load official CROMA adapter: {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.PretrainedCROMA(
            pretrained_path=str(checkpoint_path), size="base", modality="SAR",
            image_resolution=image_resolution,
        )
        model.requires_grad_(False).eval()
        self.model = model

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.shape[1] != 2:
            raise ValueError("official CROMA SAR requires two SAR channels")
        with torch.no_grad():
            tokens = self.model(SAR_images=images)["SAR_encodings"]
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise ValueError("CROMA patch-token count is not a square grid")
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
