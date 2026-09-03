# Cleaned by two-zone cleanup 2026-09-02: rejected mechanism classes moved to
# 20_HISTORY/02_legacy_code_pkgs/rejected_mechanisms_20260902/ (rejected_routers_pool.py).
"""Optical-anchor fusion with a bounded SAR-required residual payload (R5-C1-OA-SCRT).

This module is the sole remaining mechanism after R5 rapid screening; it is the
verified formal candidate (SAR null hypotheses passed, effect marginal).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class OpticalAnchoredSarResidualRouter(nn.Module):
    """Optical-anchor fusion with a bounded SAR-required residual payload."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.residual_head = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, channels, 1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, channels, 1), nn.GELU(),
            nn.Conv2d(channels, 1, 1), nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))
        self.last_diagnostics: dict[str, Tensor] = {}

    def forward(self, optical: Tensor, sar: Tensor, reliability: Tensor) -> tuple[Tensor, Tensor]:
        if optical.shape != sar.shape or reliability.shape[1] != 2:
            raise ValueError("OA-SCRT requires aligned optical, SAR and two-channel reliability")
        context = torch.cat((optical, sar, reliability), dim=1)
        residual = self.residual_head(context) * torch.tanh(self.residual_scale)
        gate = self.gate_head(context)
        fused = optical + gate * residual
        sar_mass = torch.sigmoid(reliability[:, 1:2])
        weights = torch.cat((1.0 - sar_mass, sar_mass), dim=1)
        self.last_diagnostics = {
            "sar_residual": residual,
            "residual_gate": gate,
            "syndrome_proxy": (sar - optical).abs(),
        }
        return fused, weights


class EvScrtRouter(nn.Module):
    """Evidence-conditioned bounded residual correction (R6-C1-EVSCRT, SAR-DE v2).

    Optical-anchor fusion identical in spirit to OA-SCRT, with two R6 fixes:
    - ``residual_scale`` starts at 0.0 (tanh(0) = 0), so initialization is a true
      optical identity (z == z_o; the Sigmoid gate sits at 0.5 but does not move
      the anchor output while the residual is zero);
    - the gate is the object of ``evidence_response_gate_loss`` (soft box
      (1 - u_o) <= gate <= q_s), i.e. the model learns to respond to the
      reliability evidence instead of ignoring it.
    No ``syndrome`` terminology: the residual is a direct learned correction.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.residual_head = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, channels, 1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, channels, 1), nn.GELU(),
            nn.Conv2d(channels, 1, 1), nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.0))
        self.last_diagnostics: dict[str, Tensor] = {}
        self.gate: Tensor | None = None

    def forward(self, optical: Tensor, sar: Tensor, reliability: Tensor) -> tuple[Tensor, Tensor]:
        if optical.shape != sar.shape or reliability.shape[1] != 2:
            raise ValueError("EVSCRT requires aligned optical, SAR and two-channel reliability")
        context = torch.cat((optical, sar, reliability), dim=1)
        residual = self.residual_head(context) * torch.tanh(self.residual_scale)
        gate = self.gate_head(context)
        fused = optical + gate * residual
        sar_mass = torch.sigmoid(reliability[:, 1:2])
        weights = torch.cat((1.0 - sar_mass, sar_mass), dim=1)
        self.gate = gate
        self.last_diagnostics = {
            "sar_residual": residual,
            "residual_gate": gate,
        }
        return fused, weights

    def reset_parameters(self) -> None:
        """Explicit R6 reset contract: re-initialize the mechanism heads and
        pin ``residual_scale`` to 0.0 (optical-identity start survives any
        innovation-seed reset, regardless of which submodules gain a
        ``reset_parameters`` in the future)."""
        for module in self.modules():
            reset = getattr(module, "reset_parameters", None)
            if reset is not None and module is not self:
                reset()
        self.residual_scale.data.fill_(0.0)


