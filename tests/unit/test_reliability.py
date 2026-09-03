import numpy as np

from rhs_rcmtr.utils.reliability import (
    local_cv,
    optical_radiometric_usability,
    sar_local_stability,
)


def test_optical_radiometric_usability_prefers_valid_exposure():
    normal = np.full((3, 32, 32), 128, dtype=np.uint8)
    dark = np.zeros_like(normal)
    saturated = np.full_like(normal, 255)
    normal_score = float(optical_radiometric_usability(normal).mean())
    dark_score = float(optical_radiometric_usability(dark).mean())
    saturated_score = float(optical_radiometric_usability(saturated).mean())
    assert normal_score > dark_score
    assert normal_score > saturated_score


def test_sar_local_cv_and_robust_stability_are_bounded():
    smooth = np.full((32, 32), 0.5, dtype=np.float32)
    noisy = np.clip(smooth + np.random.default_rng(0).normal(0, 0.2, smooth.shape), 0, 1)
    cv = local_cv(noisy)
    robust_cv = local_cv(noisy, robust=True)
    assert cv.shape == robust_cv.shape == smooth.shape
    assert np.isfinite(cv).all() and np.isfinite(robust_cv).all()
    assert float(sar_local_stability(smooth).mean()) > float(sar_local_stability(noisy).mean())
    assert 0.0 <= float(sar_local_stability(noisy).min()) <= 1.0
    assert 0.0 <= float(sar_local_stability(noisy).max()) <= 1.0
