"""Deterministic P1-audited sensor interventions for CRM-v2 and CF-S."""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F


OPTICAL_DEGRADATIONS = ("under_exposure", "over_exposure", "blur", "radiometric_noise")
SAR_DEGRADATIONS = ("speckle_increase", "intensity_noise", "local_dropout")
MODALITY_DROPOUT_POLICIES = ("none", "random_single_modality_p0.5", "sar_p0.5")


def modality_dropout_assignment(sample_ids: Iterable[str], policy: str) -> int | None:
    """Return the deterministic dropped modality, or ``None`` for no drop."""
    policy = str(policy or "none")
    if policy not in MODALITY_DROPOUT_POLICIES:
        raise ValueError(f"unsupported modality dropout policy: {policy}")
    if policy == "none":
        return None
    first = next(iter(sample_ids), "")
    digest = hashlib.sha256(f"modality-dropout:{first}".encode("utf-8")).digest()
    if policy == "sar_p0.5" and digest[0] % 2 == 0:
        return None
    return int(digest[0] % 2) if policy != "sar_p0.5" else 1


def _pattern(reference: Tensor) -> Tensor:
    """Return a deterministic zero-mean spatial pattern with no RNG state."""

    height, width = reference.shape[-2:]
    yy = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    return torch.sin(3.17 * grid_x + 1.73 * grid_y).view(1, 1, height, width)


def degrade_sensor(value: Tensor, modality_index: int, family: str, severity: float) -> Tensor:
    """Apply one bounded deterministic sensor degradation to BCHW tensors."""

    if value.ndim != 4:
        raise ValueError("sensor tensor must be BCHW")
    if modality_index not in (0, 1):
        raise ValueError("modality_index must be 0 (optical) or 1 (SAR)")
    severity = float(severity)
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0,1]")
    allowed = OPTICAL_DEGRADATIONS if modality_index == 0 else SAR_DEGRADATIONS
    if family not in allowed:
        raise ValueError(f"degradation {family!r} is invalid for modality {modality_index}")
    if severity == 0.0:
        return value.clone()
    if modality_index == 0:
        # Inputs are ImageNet-normalized RGB. The exposure operations are
        # applied in the normalized tensor while remaining bounded and invertible.
        if family == "under_exposure":
            return value * (1.0 - 0.55 * severity)
        if family == "over_exposure":
            return value + (1.0 - value) * (0.55 * severity)
        if family == "blur":
            blurred = F.avg_pool2d(value, kernel_size=3, stride=1, padding=1)
            return (1.0 - 0.75 * severity) * value + (0.75 * severity) * blurred
        return value + 0.08 * severity * _pattern(value)
    if family == "speckle_increase":
        return value * (1.0 + 0.35 * severity * _pattern(value))
    if family == "intensity_noise":
        return value + 0.08 * severity * _pattern(value)
    mask = (_pattern(value).abs() > (0.12 + 0.35 * (1.0 - severity))).to(value.dtype)
    return value * mask


def degrade_pair(
    optical: Tensor, sar: Tensor, modality_index: int, family: str, severity: float
) -> tuple[Tensor, Tensor]:
    """Apply an intervention to exactly one modality."""

    if modality_index == 0:
        return degrade_sensor(optical, 0, family, severity), sar.clone()
    if modality_index == 1:
        return optical.clone(), degrade_sensor(sar, 1, family, severity)
    raise ValueError("modality_index must be 0 or 1")


def apply_modality_dropout(
    optical: Tensor,
    sar: Tensor,
    reliability: Tensor | None,
    sample_ids: Iterable[str],
    policy: str,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Apply a deterministic training-only modality dropout intervention.

    The dropout decision is derived from the first sample ID in the batch so
    that worker count and process scheduling cannot change the intervention.
    When a modality is dropped, its corresponding reliability evidence is
    zeroed as well.  Evaluation never calls this helper because the entrypoint
    gates it on ``model.training``.
    """

    drop = modality_dropout_assignment(sample_ids, policy)
    if drop is None:
        return optical, sar, reliability
    optical_out, sar_out = optical.clone(), sar.clone()
    reliability_out = reliability.clone() if reliability is not None else None
    if drop == 0:
        optical_out.zero_()
    else:
        sar_out.zero_()
    if reliability_out is not None and reliability_out.ndim == 4 and reliability_out.shape[1] == 2:
        reliability_out[:, drop : drop + 1] = 0.0
    return optical_out, sar_out, reliability_out


def _denormalize_optical(optical: Tensor) -> Tensor:
    mean = optical.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = optical.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (optical * std + mean).clamp(0.0, 1.0)


def _local_mean(value: Tensor, kernel: int = 5) -> Tensor:
    return F.avg_pool2d(value, kernel, stride=1, padding=kernel // 2)


def recompute_reliability_evidence(optical: Tensor, sar: Tensor) -> Tensor:
    """Recompute differentiable P1-aligned evidence maps from sensor tensors.

    This is an operational tensor implementation of the P1 proxy definitions:
    optical radiometric usability combines exposure/clipping and low-weight
    residual/gradient terms; SAR stability uses a bounded local-CV mapping.
    """

    if optical.ndim != 4 or optical.shape[1] != 3:
        raise ValueError("optical must be BCHW with three channels")
    if sar.ndim != 4 or sar.shape[1] != 1:
        raise ValueError("sar must be BCHW with one channel")
    rgb = _denormalize_optical(optical)
    luminance = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]).clamp(0.0, 1.0)
    clipped = ((rgb <= 0.02) | (rgb >= 0.98)).any(dim=1, keepdim=True).to(rgb.dtype)
    exposure = torch.exp(-8.0 * (luminance - 0.5).square())
    clipping = 1.0 - clipped
    smooth = _local_mean(luminance)
    residual = (luminance - smooth).abs()
    noise_usability = torch.exp(-residual / 0.05)
    gx = luminance[..., :, 1:] - luminance[..., :, :-1]
    gy = luminance[..., 1:, :] - luminance[..., :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    gradient = torch.sqrt(gx.square() + gy.square())
    gradient_usability = gradient / (gradient + 0.04)
    optical_signal = gradient_usability.clamp_min(1e-6).pow(0.5) * noise_usability.clamp_min(1e-6).pow(0.5)
    optical_evidence = (exposure.clamp_min(1e-6) * clipping.clamp_min(1e-6)).pow(0.6)
    optical_evidence = optical_evidence * optical_signal.clamp_min(1e-6).pow(0.4)

    sar_value = sar.clamp(0.0, 1.0)
    local_mean = _local_mean(sar_value)
    local_second = _local_mean(sar_value.square())
    local_std = (local_second - local_mean.square()).clamp_min(0.0).sqrt()
    local_cv = local_std / (local_mean.abs() + 1e-3)
    sar_evidence = torch.exp(-1.5 * local_cv.clamp_min(0.0)).clamp(0.0, 1.0)
    return torch.cat((optical_evidence, sar_evidence), dim=1)


def select_crm_intervention(sample_ids: Iterable[str], *, salt: str = "crm-v2") -> tuple[int, str, float]:
    """Select a deterministic pseudo-random P1 family/severity per batch."""

    first = next(iter(sample_ids), "")
    digest = hashlib.sha256(f"{salt}:{first}".encode("utf-8")).digest()
    modality = digest[0] % 2
    families = OPTICAL_DEGRADATIONS if modality == 0 else SAR_DEGRADATIONS
    family = families[digest[1] % len(families)]
    severity = 0.25 + 0.5 * (digest[2] / 255.0)
    return modality, family, float(severity)


# R6 training-time degradation injection (handoff v2 §2-B2, frozen).
# Five severity steps; per-sample deterministic selection by sample_id hash;
# applied to a configurable fraction of each batch (default 30%).
R6_SEVERITY_LADDER = (0.10, 0.25, 0.40, 0.55, 0.70)


def select_r6_injections(
    sample_ids: Iterable[str], *, fraction: float = 0.3
) -> list[tuple[int, int, str, float]]:
    """Return [(batch_index, modality, family, severity)] for injected samples.

    Deterministic: derived only from sample ids (worker/scheduling invariant),
    mirroring the modality-dropout contract. ``fraction`` in [0, 1].
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("r6 injection fraction must be in [0, 1]")
    injections: list[tuple[int, int, str, float]] = []
    for index, sample_id in enumerate(sample_ids):
        digest = hashlib.sha256(f"r6-inject-v1:{sample_id}".encode("utf-8")).digest()
        if digest[0] % 100 >= int(round(fraction * 100)):
            continue
        modality = digest[1] % 2
        families = OPTICAL_DEGRADATIONS if modality == 0 else SAR_DEGRADATIONS
        family = families[digest[2] % len(families)]
        severity = R6_SEVERITY_LADDER[digest[3] % len(R6_SEVERITY_LADDER)]
        injections.append((index, modality, family, float(severity)))
    return injections


def apply_r6_train_injection(
    optical: Tensor,
    sar: Tensor,
    reliability: Tensor | None,
    sample_ids: Iterable[str],
    *,
    fraction: float = 0.3,
) -> tuple[Tensor, Tensor, Tensor | None, list[tuple[int, int, str, float]]]:
    """R6 frozen training-time injection: degrade a deterministic per-sample
    subset, then recompute the reliability evidence under the conservative
    contract ``q_after = min(q_after, q_before * (1 - 0.05 * severity))``.
    """
    ids = list(sample_ids)
    injections = select_r6_injections(ids, fraction=fraction)
    if not injections:
        return optical, sar, reliability, []
    optical_out = optical.clone()
    sar_out = sar.clone()
    for index, modality, family, severity in injections:
        degraded_optical, degraded_sar = degrade_pair(
            optical[index : index + 1], sar[index : index + 1], modality, family, severity
        )
        optical_out[index : index + 1] = degraded_optical
        sar_out[index : index + 1] = degraded_sar
    if reliability is None:
        return optical_out, sar_out, None, injections
    reliability_out = recompute_reliability_evidence(optical_out, sar_out)
    if reliability_out.shape[2:] != reliability.shape[2:]:
        reliability_out = F.interpolate(
            reliability_out, size=reliability.shape[2:], mode="bilinear", align_corners=False
        )
    for index, modality, _family, severity in injections:
        q_before = reliability[index, modality : modality + 1]
        q_after = reliability_out[index, modality : modality + 1]
        reliability_out[index, modality : modality + 1] = torch.minimum(
            q_after, q_before * (1.0 - 0.05 * severity)
        )
    return optical_out, sar_out, reliability_out, injections
