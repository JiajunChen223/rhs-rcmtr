"""Segmentation and route-honesty objectives."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def segmentation_with_honesty_loss(
    logits: Tensor,
    labels: Tensor,
    route_weights: Tensor,
    reliability: Tensor,
    honesty_weight: float,
    *,
    segmentation_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    segmentation = F.cross_entropy(logits, labels, ignore_index=255)
    target = reliability.clamp_min(0)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    token_kl = F.kl_div(route_weights.clamp_min(1e-6).log(), target, reduction="none").sum(dim=1)
    honesty = token_kl.mean()
    total = float(segmentation_weight) * segmentation + float(honesty_weight) * honesty
    return total, {"segmentation": segmentation.detach(), "honesty": honesty.detach()}


def counterfactual_monotonicity_loss(
    original_weights: Tensor,
    degraded_weights: Tensor,
    *,
    modality_index: int,
    margin: float = 0.0,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Penalize a modality becoming more trusted after its evidence drops."""

    if original_weights.shape != degraded_weights.shape:
        raise ValueError("counterfactual route tensors must have identical shapes")
    if original_weights.ndim != 4 or original_weights.shape[1] != 2:
        raise ValueError("counterfactual route tensors must be [B,2,H,W]")
    if modality_index not in (0, 1):
        raise ValueError("modality_index must be 0 (optical) or 1 (SAR)")
    violation = degraded_weights[:, modality_index] - original_weights[:, modality_index] + float(margin)
    penalty = F.relu(violation)
    if valid_mask is not None:
        if valid_mask.shape != penalty.shape:
            raise ValueError("valid_mask must match the selected route-weight shape")
        mask = valid_mask.to(dtype=penalty.dtype)
        return (penalty * mask).sum() / mask.sum().clamp_min(1.0)
    return penalty.mean()


def evidence_response_gate_loss(
    gate: Tensor,
    reliability: Tensor,
    *,
    weight_up: float = 0.5,
    weight_dn: float = 0.25,
    margin: float = 0.05,
    conflict_eps: float = 0.05,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Soft box constraint (1 - u_o) <= gate <= q_s with conflict masking (R6-C1-EVSCRT).

    Physical assumption: SAR residual corrections are only trustworthy while the
    SAR quality evidence ``q_s`` is high, and may (not must) increase when the
    optical evidence ``u_o`` drops. Pixels whose box is empty (joint degradation,
    q_s < 1 - u_o - eps) are masked out and left to the segmentation loss; the
    constraint is a soft mean-field penalty, not a hard per-pixel bound.
    """
    if gate.ndim != 4 or gate.shape[1] != 1:
        raise ValueError("gate must be [B,1,H,W]")
    if reliability.ndim != 4 or reliability.shape[1] != 2:
        raise ValueError("reliability must be [B,2,H,W]")
    if gate.shape[2:] != reliability.shape[2:]:
        raise ValueError("gate and reliability must share the feature grid")
    u_o = reliability[:, 0:1]
    q_s = reliability[:, 1:2]
    conflict = q_s < (1.0 - u_o) - float(conflict_eps)
    active = (~conflict).to(dtype=gate.dtype)
    up = F.relu(gate - q_s + float(margin)) * active
    dn = F.relu((1.0 - u_o) - gate + float(margin)) * active
    denom = active.sum().clamp_min(1.0)
    up_mean = up.sum() / denom
    dn_mean = dn.sum() / denom
    violation = ((gate > q_s).to(dtype=gate.dtype) * active).sum() / denom
    conflict_fraction = conflict.to(dtype=gate.dtype).mean()
    total = float(weight_up) * up_mean + float(weight_dn) * dn_mean
    return total, {
        "evidence_up": up_mean.detach(),
        "evidence_dn": dn_mean.detach(),
        "violation_fraction": violation.detach(),
        "conflict_fraction": conflict_fraction.detach(),
    }
