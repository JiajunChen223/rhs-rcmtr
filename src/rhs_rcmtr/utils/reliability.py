"""Observable modality-quality proxies used by the P1 validation audit.

These functions deliberately return evidence maps, not ground-truth quality
labels.  They are small NumPy-only utilities so the same definitions can be
used by the cloud derivation script, the validation-only audit, and synthetic
unit tests without adding a new runtime dependency.
"""

from __future__ import annotations

import numpy as np


def _float01(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
    if float(np.nanmax(value)) > 1.5:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0)


def _window_view(array: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        raise ValueError("radius must be positive")
    value = np.asarray(array, dtype=np.float32)
    padded = np.pad(value, ((radius, radius), (radius, radius)), mode="reflect")
    return np.lib.stride_tricks.sliding_window_view(
        padded, (2 * radius + 1, 2 * radius + 1)
    )


def local_mean(array: np.ndarray, radius: int = 2) -> np.ndarray:
    """Return a reflect-padded local mean on a two-dimensional grid."""

    return _window_view(array, radius).mean(axis=(-1, -2), dtype=np.float32)


def local_cv(
    array: np.ndarray,
    radius: int = 2,
    *,
    robust: bool = False,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """Return local coefficient of variation or a robust median/MAD variant."""

    value = _float01(np.asarray(array))
    if value.ndim != 2:
        raise ValueError("SAR reliability expects a two-dimensional grid")
    windows = _window_view(value, radius)
    if robust:
        center = np.median(windows, axis=(-1, -2)).astype(np.float32)
        dispersion = np.median(np.abs(windows - center[..., None, None]), axis=(-1, -2))
        dispersion = 1.4826 * dispersion
    else:
        center = windows.mean(axis=(-1, -2), dtype=np.float32)
        second = (windows * windows).mean(axis=(-1, -2), dtype=np.float32)
        dispersion = np.sqrt(np.maximum(second - center * center, 0.0))
    return dispersion / (np.abs(center) + float(epsilon))


def optical_radiometric_usability(
    optical: np.ndarray,
    *,
    gradient_weight: float = 0.20,
    noise_weight: float = 0.20,
    radius: int = 1,
) -> np.ndarray:
    """Compute a conservative optical radiometric-usability evidence map.

    Exposure validity and clipping validity carry most of the score.  A small
    signal-usability term responds to blur and high-frequency noise, but is
    intentionally low-weight so semantic edges are not treated as quality
    truth.
    """

    value = _float01(np.asarray(optical))
    if value.ndim == 3:
        if value.shape[0] != 3:
            raise ValueError("optical input must have three channels")
        luminance = (
            0.299 * value[0] + 0.587 * value[1] + 0.114 * value[2]
        ).astype(np.float32)
        clipped = np.any((value <= 0.02) | (value >= 0.98), axis=0)
    elif value.ndim == 2:
        luminance = value
        clipped = (value <= 0.02) | (value >= 0.98)
    else:
        raise ValueError("optical input must be [3,H,W] or [H,W]")
    if not 0.0 <= gradient_weight <= 1.0 or not 0.0 <= noise_weight <= 1.0:
        raise ValueError("signal weights must be in [0,1]")
    if gradient_weight + noise_weight > 1.0:
        raise ValueError("gradient_weight + noise_weight must be <= 1")

    # The peak is deliberately centred on a normally exposed luminance and
    # falls off toward both under- and over-exposure.
    exposure = np.exp(-8.0 * np.square(luminance - 0.5)).astype(np.float32)
    clipping = np.where(clipped, 0.0, 1.0).astype(np.float32)
    smooth = local_mean(luminance, radius=max(1, radius))
    residual = np.abs(luminance - smooth)
    # A conservative residual scale keeps the signal term sensitive to
    # radiometric noise while its overall contribution remains low-weight.
    noise_usability = np.exp(-residual / 0.05).astype(np.float32)
    gx = np.diff(luminance, axis=1, append=luminance[:, -1:])
    gy = np.diff(luminance, axis=0, append=luminance[-1:, :])
    gradient = np.sqrt(gx * gx + gy * gy)
    gradient_usability = (gradient / (gradient + 0.04)).astype(np.float32)

    signal_weight = float(gradient_weight + noise_weight)
    if signal_weight == 0.0:
        signal = np.ones_like(exposure)
    else:
        # Use a weighted geometric mean so either blur or noise can lower the
        # signal term while keeping it a deliberately secondary contribution.
        gw = float(gradient_weight / signal_weight)
        nw = float(noise_weight / signal_weight)
        signal = np.power(np.clip(gradient_usability, 1e-6, 1.0), gw)
        signal *= np.power(np.clip(noise_usability, 1e-6, 1.0), nw)
    core = np.clip(exposure * clipping, 0.0, 1.0)
    score = np.power(np.clip(core, 1e-6, 1.0), 1.0 - signal_weight)
    score *= np.power(np.clip(signal, 1e-6, 1.0), signal_weight)
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def sar_local_stability(
    sar: np.ndarray,
    *,
    alpha: float = 1.5,
    radius: int = 2,
    robust: bool = False,
) -> np.ndarray:
    """Compute bounded local SAR stability evidence from local CV or MAD/CV."""

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    cv = local_cv(sar, radius=radius, robust=robust)
    return np.exp(-float(alpha) * np.maximum(cv, 0.0)).clip(0.0, 1.0).astype(np.float32)


def proxy_scalar(proxy: np.ndarray) -> float:
    """Return a finite image-level summary for validation-only diagnostics."""

    value = np.asarray(proxy, dtype=np.float32)
    result = float(np.nanmean(np.clip(value, 0.0, 1.0)))
    return result if np.isfinite(result) else 0.0
