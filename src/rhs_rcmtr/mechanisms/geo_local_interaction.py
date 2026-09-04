"""Geo-local deformable cross-sensor interaction for R7-S2FT."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _GeoLocalDirection(nn.Module):
    """One directional local cross-attention with bounded deformable sampling."""

    def __init__(self, channels: int, interaction_dim: int, max_offset_tokens: float) -> None:
        super().__init__()
        if channels <= 0 or interaction_dim <= 0:
            raise ValueError("interaction dimensions must be positive")
        if max_offset_tokens <= 0:
            raise ValueError("max_offset_tokens must be positive")
        self.channels = int(channels)
        self.interaction_dim = int(interaction_dim)
        self.max_offset_tokens = float(max_offset_tokens)
        self.num_points = 9
        self.q_proj = nn.Conv2d(channels, interaction_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(channels, interaction_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(channels, interaction_dim, 1, bias=False)
        self.offset = nn.Conv2d(channels * 2, self.num_points * 2, 3, padding=1)
        self.out_proj = nn.Conv2d(interaction_dim, channels, 1, bias=False)
        self.relative_bias = nn.Parameter(torch.zeros(self.num_points))
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "base_offsets",
            torch.tensor(
                [
                    [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                    [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                    [1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.out_proj.reset_parameters()
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.relative_bias)
        with torch.no_grad():
            self.gamma.zero_()

    @staticmethod
    def _normalize_mask(mask: Tensor | None, reference: Tensor) -> Tensor:
        if mask is None:
            return torch.ones(
                reference.shape[0], 1, reference.shape[2], reference.shape[3],
                device=reference.device, dtype=reference.dtype,
            )
        if mask.ndim != 4 or mask.shape[0] != reference.shape[0] or mask.shape[1] != 1:
            raise ValueError("interaction mask must be [B,1,H,W]")
        mask = mask.to(reference).clamp(0.0, 1.0)
        if mask.shape[2:] != reference.shape[2:]:
            mask = F.interpolate(mask, size=reference.shape[2:], mode="nearest")
        return mask

    def bounded_offsets(self, target: Tensor, source: Tensor) -> Tensor:
        raw = self.offset(torch.cat((target, source), dim=1))
        b, _, h, w = raw.shape
        raw = raw.view(b, self.num_points, 2, h, w)
        return self.max_offset_tokens * torch.tanh(raw)

    def _sample(self, feature: Tensor, offsets: Tensor) -> Tensor:
        """Return [B,K,C,H,W] samples from a local 3x3 deformable stencil."""
        b, c, h, w = feature.shape
        if offsets.shape != (b, self.num_points, 2, h, w):
            raise ValueError("offset tensor shape mismatch")
        yy, xx = torch.meshgrid(
            torch.arange(h, device=feature.device, dtype=feature.dtype),
            torch.arange(w, device=feature.device, dtype=feature.dtype),
            indexing="ij",
        )
        centers = torch.stack((yy, xx), dim=-1)[None, None]
        base = self.base_offsets.to(device=feature.device, dtype=feature.dtype)[None, :, None, None]
        learned = offsets.permute(0, 1, 3, 4, 2)
        coords = centers + base + learned
        y = coords[..., 0]
        x = coords[..., 1]
        if w > 1:
            x = 2.0 * x / float(w - 1) - 1.0
        else:
            x = torch.zeros_like(x)
        if h > 1:
            y = 2.0 * y / float(h - 1) - 1.0
        else:
            y = torch.zeros_like(y)
        grid = torch.stack((x, y), dim=-1).reshape(b * self.num_points, h, w, 2)
        tiled = feature[:, None].expand(-1, self.num_points, -1, -1, -1).reshape(
            b * self.num_points, c, h, w
        )
        sampled = F.grid_sample(
            tiled, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        return sampled.reshape(b, self.num_points, c, h, w)

    def forward(
        self,
        target: Tensor,
        source: Tensor,
        *,
        target_mask: Tensor | None = None,
        source_mask: Tensor | None = None,
    ) -> Tensor:
        if target.shape != source.shape or target.ndim != 4:
            raise ValueError("GLCI directional inputs must share [B,C,H,W]")
        target_mask = self._normalize_mask(target_mask, target)
        source_mask = self._normalize_mask(source_mask, source)
        target_visible = target * target_mask
        source_visible = source * source_mask
        q = self.q_proj(target_visible) * target_mask
        k = self.k_proj(source_visible) * source_mask
        v = self.v_proj(source_visible) * source_mask
        offsets = self.bounded_offsets(target_visible, source_visible)
        sampled_k = self._sample(k, offsets)
        sampled_v = self._sample(v, offsets)
        sampled_mask = self._sample(source_mask, offsets)
        sampled_k = sampled_k * sampled_mask
        sampled_v = sampled_v * sampled_mask
        score = (q[:, None] * sampled_k).sum(dim=2) / math.sqrt(float(self.interaction_dim))
        score = score + self.relative_bias[None, :, None, None]
        valid = sampled_mask.squeeze(2) > 1e-6
        score = score.masked_fill(~valid, -1e4)
        attention = torch.softmax(score, dim=1)
        attention = attention * valid.to(attention.dtype)
        denom = attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
        attention = attention / denom
        context = (attention[:, :, None] * sampled_v).sum(dim=1)
        delta = self.out_proj(context) * target_mask
        return target + self.gamma * delta


class GeoLocalDeformableCrossInteraction(nn.Module):
    """Bidirectional optical/SAR interaction constrained to a local geo-neighborhood."""

    def __init__(self, channels: int, interaction_dim: int = 96, max_offset_tokens: float = 1.5) -> None:
        super().__init__()
        self.sar_to_optical = _GeoLocalDirection(channels, interaction_dim, max_offset_tokens)
        self.optical_to_sar = _GeoLocalDirection(channels, interaction_dim, max_offset_tokens)

    def reset_parameters(self) -> None:
        self.sar_to_optical.reset_parameters()
        self.optical_to_sar.reset_parameters()

    def forward(
        self,
        optical: Tensor,
        sar: Tensor,
        *,
        optical_mask: Tensor | None = None,
        sar_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        optical_out = self.sar_to_optical(
            optical, sar, target_mask=optical_mask, source_mask=sar_mask,
        )
        sar_out = self.optical_to_sar(
            sar, optical, target_mask=sar_mask, source_mask=optical_mask,
        )
        return optical_out, sar_out
