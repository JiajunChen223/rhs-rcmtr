"""Shared optical/SAR foundation backbone for R7-S2FT."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from rhs_rcmtr.mechanisms.geo_local_interaction import GeoLocalDeformableCrossInteraction
from rhs_rcmtr.models.backbones import _verify_checkpoint, dino_input_grid
from rhs_rcmtr.models.s2ft.config import S2FTConfig
from rhs_rcmtr.models.s2ft.patch_transplant import SarFoundationPatchEmbed
from rhs_rcmtr.models.s2ft.sensor_adapter import SensorAdapterPair


class _TinyPatchEmbed(nn.Module):
    def __init__(self, embed_dim: int, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class _TinyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class _TinyDinoLike(nn.Module):
    """Synthetic-only DINO-shaped fixture; never used as scientific evidence."""

    def __init__(self, embed_dim: int = 64, depth: int = 12, patch_size: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = _TinyPatchEmbed(embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = None
        self.num_register_tokens = 0
        self.blocks = nn.ModuleList([_TinyBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def interpolate_pos_encoding(self, x: Tensor, w: int, h: int) -> Tensor:
        return torch.zeros_like(x)


class SharedDinoSensorBackbone(nn.Module):
    """One frozen DINO transformer, two sensor tokenizations, sensor PEFT, and GLCI.

    DINO parameters are frozen but its forward pass is intentionally *not*
    wrapped in ``torch.no_grad``.  Gradients must propagate through frozen blocks
    so the trainable SAR patch embedding and sensor adapters can learn.
    """

    def __init__(self, dino: nn.Module, config: S2FTConfig, *, patch_size: int) -> None:
        super().__init__()
        config.validate()
        if not hasattr(dino, "patch_embed") or not hasattr(dino, "blocks") or not hasattr(dino, "norm"):
            raise TypeError("foundation model does not expose the DINO block interface")
        self.config = config
        self.dino = dino
        self.dino.requires_grad_(False).eval()
        self.patch_size = int(patch_size)
        embed_dim = int(getattr(dino, "embed_dim", getattr(dino.patch_embed.proj, "out_channels")))
        self.output_channels = embed_dim
        self.sar_patch_embed = SarFoundationPatchEmbed(dino.patch_embed)
        self.adapters = nn.ModuleDict(
            {
                str(index): SensorAdapterPair(embed_dim, config.adapter_bottleneck)
                for index in config.adapter_layers
            }
        )
        self.interactions = nn.ModuleDict(
            {
                str(index): GeoLocalDeformableCrossInteraction(
                    embed_dim,
                    interaction_dim=config.interaction_dim,
                    max_offset_tokens=config.max_offset_tokens,
                )
                for index in config.interaction_layers
            }
        )
        depth = len(self.dino.blocks)
        if max(config.adapter_layers) >= depth:
            raise ValueError(f"S2FT adapter layer exceeds foundation depth {depth}")
        self.feature_layers = tuple(config.adapter_layers)

    @classmethod
    def from_official(
        cls,
        repo_path: Path,
        checkpoint_path: Path,
        expected_sha256: str,
        config: S2FTConfig,
    ) -> "SharedDinoSensorBackbone":
        _verify_checkpoint(checkpoint_path, expected_sha256)
        dino = torch.hub.load(str(repo_path), "dinov2_vitb14", source="local", pretrained=False)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        dino.load_state_dict(state, strict=True)
        patch = getattr(dino.patch_embed, "patch_size", (14, 14))
        patch_size = int(patch[0] if isinstance(patch, tuple) else patch)
        return cls(dino, config, patch_size=patch_size)

    @classmethod
    def synthetic(cls, config: S2FTConfig) -> "SharedDinoSensorBackbone":
        return cls(_TinyDinoLike(), config, patch_size=4)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.dino.eval()
        return self

    @property
    def patch_token_start(self) -> int:
        return 1 + int(getattr(self.dino, "num_register_tokens", 0))

    def reset_innovation_parameters(self) -> None:
        """Reset only R7 trainable modules; never reset the frozen DINO weights."""
        self.sar_patch_embed.reset_from_source()
        for pair in self.adapters.values():
            pair.reset_parameters()
        for interaction in self.interactions.values():
            interaction.reset_parameters()

    def _prepare_tokens(self, images: Tensor, patch_embed: Callable[[Tensor], Tensor]) -> Tensor:
        b, _, w, h = images.shape
        x = patch_embed(images)
        cls = self.dino.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.dino.interpolate_pos_encoding(x, w, h)
        register_tokens = getattr(self.dino, "register_tokens", None)
        if register_tokens is not None:
            x = torch.cat((x[:, :1], register_tokens.expand(b, -1, -1), x[:, 1:]), dim=1)
        return x

    def _resolve_masks(
        self,
        modality_mask: Tensor | None,
        *,
        batch: int,
        input_size: tuple[int, int],
        token_size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        if modality_mask is None:
            base = torch.ones(batch, 2, *token_size, device=device, dtype=dtype)
        else:
            if modality_mask.ndim == 2:
                if modality_mask.shape != (batch, 2):
                    raise ValueError("modality_mask [B,2] shape mismatch")
                base = modality_mask[:, :, None, None].to(device=device, dtype=dtype)
                base = base.expand(-1, -1, *token_size)
            elif modality_mask.ndim == 4:
                if modality_mask.shape[0] != batch or modality_mask.shape[1] != 2:
                    raise ValueError("modality_mask must have shape [B,2,H,W]")
                base = modality_mask.to(device=device, dtype=dtype)
                if base.shape[2:] != token_size:
                    base = F.interpolate(base, size=token_size, mode="nearest")
            else:
                raise ValueError("modality_mask must be [B,2] or [B,2,H,W]")
            base = base.clamp(0.0, 1.0)
        return base[:, 0:1], base[:, 1:2]

    def _mask_patch_tokens(self, x: Tensor, mask: Tensor) -> Tensor:
        start = self.patch_token_start
        patch = x[:, start:]
        b, n, c = patch.shape
        if n != mask.shape[2] * mask.shape[3]:
            raise ValueError("token mask grid does not match patch-token count")
        flat = mask.flatten(2).transpose(1, 2)
        masked = patch * flat
        return torch.cat((x[:, :start], masked), dim=1)

    def _replace_patch_tokens(self, x: Tensor, patch: Tensor) -> Tensor:
        return torch.cat((x[:, : self.patch_token_start], patch), dim=1)

    def _run_block(self, block: nn.Module, x: Tensor) -> Tensor:
        if self.training and self.config.checkpoint_frozen_blocks and x.requires_grad:
            return activation_checkpoint(block, x, use_reentrant=False)
        return block(x)

    def _capture(self, x: Tensor, grid: tuple[int, int]) -> Tensor:
        normalized = self.dino.norm(x)
        patch = normalized[:, self.patch_token_start :]
        b, n, c = patch.shape
        if n != grid[0] * grid[1]:
            raise ValueError("captured patch-token count does not match token grid")
        return patch.transpose(1, 2).reshape(b, c, grid[0], grid[1])

    def forward(
        self,
        optical: Tensor,
        sar: Tensor,
        modality_mask: Tensor | None = None,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor], Tensor, Tensor, Tensor]:
        if optical.ndim != 4 or optical.shape[1] != 3:
            raise ValueError("S2FT optical input must be [B,3,H,W]")
        if sar.ndim != 4 or sar.shape[1] != 1 or sar.shape[0] != optical.shape[0]:
            raise ValueError("S2FT SAR input must be paired [B,1,H,W]")
        optical_aligned = dino_input_grid(optical, self.patch_size)
        target_size = optical_aligned.shape[2:]
        sar_aligned = sar
        if sar_aligned.shape[2:] != target_size:
            sar_aligned = F.interpolate(sar_aligned, size=target_size, mode="bilinear", align_corners=False)
        grid = (target_size[0] // self.patch_size, target_size[1] // self.patch_size)
        optical_mask, sar_mask = self._resolve_masks(
            modality_mask,
            batch=optical.shape[0],
            input_size=optical.shape[2:],
            token_size=grid,
            device=optical.device,
            dtype=optical.dtype,
        )
        xo = self._prepare_tokens(optical_aligned, self.dino.patch_embed)
        xs = self._prepare_tokens(sar_aligned, self.sar_patch_embed)
        xo = self._mask_patch_tokens(xo, optical_mask)
        xs = self._mask_patch_tokens(xs, sar_mask)
        optical_outputs: dict[int, Tensor] = {}
        sar_outputs: dict[int, Tensor] = {}
        for index, block in enumerate(self.dino.blocks):
            xo = self._run_block(block, xo)
            xs = self._run_block(block, xs)
            # Re-mask after every shared transformer block.  This prevents CLS,
            # register-token, and affine-bias leakage into an explicitly absent modality.
            xo = self._mask_patch_tokens(xo, optical_mask)
            xs = self._mask_patch_tokens(xs, sar_mask)
            key = str(index)
            if key in self.adapters:
                xo_patch = self.adapters[key].optical(xo[:, self.patch_token_start :])
                xs_patch = self.adapters[key].sar(xs[:, self.patch_token_start :])
                xo = self._replace_patch_tokens(xo, xo_patch)
                xs = self._replace_patch_tokens(xs, xs_patch)
                xo = self._mask_patch_tokens(xo, optical_mask)
                xs = self._mask_patch_tokens(xs, sar_mask)
            if key in self.interactions:
                fo = xo[:, self.patch_token_start :].transpose(1, 2).reshape(
                    optical.shape[0], self.output_channels, *grid
                )
                fs = xs[:, self.patch_token_start :].transpose(1, 2).reshape(
                    optical.shape[0], self.output_channels, *grid
                )
                fo, fs = self.interactions[key](
                    fo, fs, optical_mask=optical_mask, sar_mask=sar_mask,
                )
                xo = self._replace_patch_tokens(xo, fo.flatten(2).transpose(1, 2))
                xs = self._replace_patch_tokens(xs, fs.flatten(2).transpose(1, 2))
                xo = self._mask_patch_tokens(xo, optical_mask)
                xs = self._mask_patch_tokens(xs, sar_mask)
            if index in self.feature_layers:
                optical_outputs[index] = self._capture(xo, grid) * optical_mask
                sar_outputs[index] = self._capture(xs, grid) * sar_mask
        if set(optical_outputs) != set(self.feature_layers) or set(sar_outputs) != set(self.feature_layers):
            raise RuntimeError("S2FT did not capture every declared feature layer")
        return optical_outputs, sar_outputs, sar_aligned, optical_mask, sar_mask
