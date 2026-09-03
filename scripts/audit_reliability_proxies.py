"""Validation-only audit of observable optical/SAR reliability evidence.

The script reads only optical and SAR assets from the declared internal
validation split.  It never constructs a dataset, touches labels, trains a
router, or opens the sealed official test role.  Each degradation is applied
in memory and the proxy is recomputed so the severity-to-evidence direction
can be audited independently of model performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
from rasterio.enums import Resampling

from rhs_rcmtr.utils.reliability import (
    local_mean,
    optical_radiometric_usability,
    proxy_scalar,
    sar_local_stability,
)


SEVERITIES = (1, 2, 3, 4)
MIN_ABS_SPEARMAN_RHO = 0.7


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    """Compute tie-aware Spearman correlation without SciPy."""

    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.size != right.size or left.size < 3:
        return 0.0
    left_rank = _rank(left)
    right_rank = _rank(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(np.sqrt(np.dot(left_rank, left_rank) * np.dot(right_rank, right_rank)))
    return float(np.dot(left_rank, right_rank) / denominator) if denominator > 0 else 0.0


def _read_grid(path: str, *, bands: int, grid_size: int) -> np.ndarray:
    with rasterio.open(path) as source:
        return source.read(
            out_shape=(bands, grid_size, grid_size),
            resampling=Resampling.average,
            out_dtype="float32",
        )


def _sample_seed(sample_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) ^ int(base_seed)


def _blur(optical: np.ndarray, severity: int) -> np.ndarray:
    # The audit operates on a 32x32 evidence grid; doubling the box radius
    # keeps the four levels visibly distinct after downsampling.
    radius = max(1, 2 * int(severity))
    return np.stack([local_mean(channel, radius=radius) for channel in optical], axis=0)


def _optical_degradation(
    optical: np.ndarray, name: str, severity: int, rng: np.random.Generator
) -> np.ndarray:
    value = np.asarray(optical, dtype=np.float32) / 255.0
    if name == "under_exposure":
        value = value * (1.0 - 0.17 * severity)
    elif name == "over_exposure":
        value = value + 0.18 * severity
    elif name == "blur":
        value = _blur(value, severity)
    elif name == "radiometric_noise":
        value = value + rng.normal(0.0, 0.06 * severity, size=value.shape).astype(np.float32)
    else:
        raise ValueError(f"unknown optical degradation: {name}")
    return np.clip(value, 0.0, 1.0)


def _sar_degradation(
    sar: np.ndarray, name: str, severity: int, rng: np.random.Generator
) -> np.ndarray:
    value = np.asarray(sar, dtype=np.float32)[0] / 255.0
    if name == "speckle_increase":
        sigma = 0.08 * severity
        shape = max(1.0, 1.0 / (sigma * sigma))
        value = value * rng.gamma(shape, 1.0 / shape, size=value.shape).astype(np.float32)
    elif name == "intensity_noise":
        value = value + rng.normal(0.0, 0.025 * severity, size=value.shape).astype(np.float32)
    elif name == "local_dropout":
        drop_probability = 0.04 * severity
        value = value.copy()
        value[rng.random(value.shape) < drop_probability] = 0.0
    else:
        raise ValueError(f"unknown SAR degradation: {name}")
    return np.clip(value, 0.0, 1.0)


def _monotonic_pair_fraction(values_by_sample: list[list[float]]) -> float:
    if not values_by_sample:
        return 0.0
    passing = 0
    for values in values_by_sample:
        passing += int(all(values[index + 1] <= values[index] + 1e-6 for index in range(len(values) - 1)))
    return passing / len(values_by_sample)


def _audit_one(
    samples: list[dict[str, Any]],
    *,
    modality: str,
    degradation: str,
    grid_size: int,
    base_seed: int,
) -> dict[str, Any]:
    severity_values: list[float] = []
    proxy_values: list[float] = []
    values_by_sample: list[list[float]] = []
    for item in samples:
        sample_id = str(item["sample_id"])
        rng = np.random.default_rng(_sample_seed(sample_id, base_seed))
        if modality == "optical":
            original = _read_grid(str(item["optical_path"]), bands=3, grid_size=grid_size)
            baseline = proxy_scalar(optical_radiometric_usability(original))
            transform: Callable[[int], np.ndarray] = lambda level: _optical_degradation(
                original, degradation, level, rng
            )
            proxy_fn: Callable[[np.ndarray], np.ndarray] = optical_radiometric_usability
        else:
            original = _read_grid(str(item["sar_path"]), bands=1, grid_size=grid_size)
            baseline = proxy_scalar(sar_local_stability(original[0]))
            transform = lambda level: _sar_degradation(original, degradation, level, rng)
            proxy_fn = sar_local_stability
        sample_values: list[float] = []
        for severity in SEVERITIES:
            degraded = transform(severity)
            proxy = proxy_fn(degraded)
            value = proxy_scalar(proxy)
            # Paired ratio removes most between-scene semantic/content
            # variation while retaining the direction of the degradation.
            relative = value / max(baseline, 1e-6)
            severity_values.append(float(severity))
            proxy_values.append(float(relative))
            sample_values.append(float(relative))
        values_by_sample.append(sample_values)
    medians = [float(np.median([values[index] for values in values_by_sample])) for index in range(4)]
    pooled_rho = spearman(severity_values, proxy_values)
    # The primary gate is computed on the four paired severity medians.  A
    # pooled scene-level correlation is retained diagnostically, but semantic
    # scene content can dominate it even when every scene has the expected
    # degradation direction.
    median_rho = spearman(np.asarray(SEVERITIES, dtype=np.float64), np.asarray(medians))
    pair_fraction = _monotonic_pair_fraction(values_by_sample)
    return {
        "modality": modality,
        "degradation": degradation,
        "severity_levels": list(SEVERITIES),
        "sample_count": len(samples),
        "spearman_rho": median_rho,
        "absolute_spearman_rho": abs(median_rho),
        "pooled_scene_spearman_rho": pooled_rho,
        "expected_direction": "decrease",
        "direction_correct": median_rho <= -MIN_ABS_SPEARMAN_RHO and pair_fraction >= 0.8,
        "median_relative_proxy_by_severity": medians,
        "per_sample_monotonic_fraction": pair_fraction,
    }


def run_audit(manifest_path: Path, output_path: Path, *, grid_size: int, max_samples: int, seed: int) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("host") != "cloud" or payload.get("transfer_policy") != "never_to_local":
        raise ValueError("reliability audit requires the cloud-only data manifest")
    samples = [item for item in payload.get("samples", []) if item.get("split") == "validation"]
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError("no validation samples available for reliability audit")
    if any(str(item.get("split")) == "test" for item in payload.get("samples", [])):
        raise ValueError("test split is forbidden in the P1 reliability audit")
    required = ("sample_id", "optical_path", "sar_path")
    if any(not all(str(item.get(field) or "").strip() for field in required) for item in samples):
        raise ValueError("validation samples must provide optical and SAR paths")

    results: list[dict[str, Any]] = []
    for degradation in ("under_exposure", "over_exposure", "blur", "radiometric_noise"):
        results.append(
            _audit_one(
                samples,
                modality="optical",
                degradation=degradation,
                grid_size=grid_size,
                base_seed=seed,
            )
        )
    for degradation in ("speckle_increase", "intensity_noise", "local_dropout"):
        results.append(
            _audit_one(
                samples,
                modality="sar",
                degradation=degradation,
                grid_size=grid_size,
                base_seed=seed,
            )
        )
    gate_pass = all(bool(result["direction_correct"]) for result in results)
    report = {
        "artifact_type": "reliability_validation_audit",
        "schema_version": 1,
        "status": "pass" if gate_pass else "fail",
        "execution_context": "cloud",
        "audit_mode": "validation_only_no_training",
        "split": "validation",
        "sample_count": len(samples),
        "grid_size": grid_size,
        "proxy_definitions": {
            "optical": "radiometric_usability(exposure_validity * clipping_validity with low-weight signal/noise usability)",
            "sar": "local_sar_stability_evidence(exp(-alpha * local_CV)); robust median/MAD implementation available",
        },
        "degradation_ladder": {
            "optical": ["under_exposure", "over_exposure", "blur", "radiometric_noise"],
            "sar": ["speckle_increase", "intensity_noise", "local_dropout"],
        },
        "results": results,
        "gate": {
            "minimum_absolute_spearman_rho": MIN_ABS_SPEARMAN_RHO,
            "direction_must_match": True,
            "all_degradations_pass": gate_pass,
        },
        "data_manifest": str(manifest_path),
        "test_seal_status": "sealed",
        "test_accessed": False,
        "training_started": False,
        "router_training_authorized": gate_pass,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run_audit(
        args.manifest,
        args.output,
        grid_size=args.grid_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "sample_count": report["sample_count"], "output": str(args.output)}))
    raise SystemExit(0 if report["status"] == "pass" else 3)


if __name__ == "__main__":
    main()
