"""Single-band SAR patch embedding initialized from the optical foundation filters."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SarFoundationPatchEmbed(nn.Module):
    """DINO-compatible one-band patch embed initialized from RGB patch filters.

    The RGB filters are collapsed with 1/sqrt(3) scaling.  This preserves filter
    energy when the three source-channel kernels are identical and avoids the
    amplitude shrinkage produced by a plain channel mean.
    """

    def __init__(self, source_patch_embed: nn.Module) -> None:
        super().__init__()
        source_proj = getattr(source_patch_embed, "proj", None)
        if not isinstance(source_proj, nn.Conv2d):
            raise TypeError("source patch embedding must expose a Conv2d .proj")
        if source_proj.in_channels != 3:
            raise ValueError("R7-S2FT v1 expects a three-channel optical patch embedding")
        self.source_patch_embed = source_patch_embed
        self.proj = nn.Conv2d(
            1,
            source_proj.out_channels,
            kernel_size=source_proj.kernel_size,
            stride=source_proj.stride,
            padding=source_proj.padding,
            dilation=source_proj.dilation,
            groups=1,
            bias=source_proj.bias is not None,
        )
        source_norm = getattr(source_patch_embed, "norm", nn.Identity())
        if isinstance(source_norm, nn.LayerNorm):
            self.norm: nn.Module = nn.LayerNorm(source_norm.normalized_shape, eps=source_norm.eps)
            with torch.no_grad():
                if source_norm.elementwise_affine:
                    self.norm.weight.copy_(source_norm.weight)
                    self.norm.bias.copy_(source_norm.bias)
            self.norm.requires_grad_(False)
        else:
            self.norm = nn.Identity()
        self.reset_from_source()

    @property
    def patch_size(self) -> tuple[int, int]:
        value = self.proj.kernel_size
        return int(value[0]), int(value[1])

    def reset_from_source(self) -> None:
        source_proj = getattr(self.source_patch_embed, "proj")
        with torch.no_grad():
            collapsed = source_proj.weight.detach().sum(dim=1, keepdim=True) / math.sqrt(3.0)
            self.proj.weight.copy_(collapsed)
            if source_proj.bias is not None and self.proj.bias is not None:
                self.proj.bias.copy_(source_proj.bias.detach())

    def forward(self, sar: Tensor) -> Tensor:
        if sar.ndim != 4 or sar.shape[1] != 1:
            raise ValueError("SarFoundationPatchEmbed requires [B,1,H,W]")
        patch_h, patch_w = self.patch_size
        if sar.shape[-2] % patch_h or sar.shape[-1] % patch_w:
            raise ValueError("SAR input must be aligned to the foundation patch grid")
        x = self.proj(sar)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)
