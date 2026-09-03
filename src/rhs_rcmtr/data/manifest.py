"""Fail-closed cloud data and split-manifest validation."""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
CLOUD_ROOT = PurePosixPath(chr(47), "root", "autodl-tmp")


@dataclass(frozen=True)
class PairedSample:
    sample_id: str
    group_id: str
    split: str
    optical_path: str
    sar_path: str
    label_path: str
    reliability_path: str


@dataclass(frozen=True)
class SarExternalSample:
    """One official validation SAR/label pair used only for sealed external evaluation."""

    sample_id: str
    sar_path: str
    label_path: str


def _cloud_path(value: Any, cloud_root: str) -> str:
    raw = str(value or "")
    normalized = posixpath.normpath(raw)
    path = PurePosixPath(normalized)
    root = PurePosixPath(posixpath.normpath(cloud_root))
    if normalized != raw or not path.is_absolute() or path != root and root not in path.parents:
        raise ValueError(f"path must remain under cloud data root: {raw}")
    return raw


def validate_split_integrity(samples: list[PairedSample]) -> None:
    ids: set[str] = set()
    group_splits: dict[str, str] = {}
    path_splits: dict[str, str] = {}
    for sample in samples:
        if sample.split not in ALLOWED_SPLITS:
            raise ValueError(f"unknown split: {sample.split}")
        if not sample.sample_id or sample.sample_id in ids:
            raise ValueError(f"duplicate or empty sample_id: {sample.sample_id}")
        ids.add(sample.sample_id)
        previous = group_splits.setdefault(sample.group_id, sample.split)
        if previous != sample.split:
            raise ValueError(f"group crosses splits: {sample.group_id}")
        if not sample.group_id:
            raise ValueError(f"empty group_id: {sample.sample_id}")
        for path in (sample.optical_path, sample.sar_path, sample.label_path, sample.reliability_path):
            previous_path = path_splits.setdefault(path, sample.split)
            if previous_path != sample.split:
                raise ValueError(f"asset crosses splits: {path}")


def load_data_manifest(path: Path, *, expected_sha256: str) -> tuple[dict[str, Any], list[PairedSample]]:
    import hashlib

    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != expected_sha256:
        raise ValueError("data manifest SHA256 mismatch")
    payload = json.loads(payload_bytes)
    if payload.get("host") != "cloud" or payload.get("transfer_policy") != "never_to_local":
        raise ValueError("real data manifest must be cloud-only and never_to_local")
    cloud_root = str(payload.get("cloud_data_root") or "")
    declared_root = PurePosixPath(cloud_root)
    if declared_root != CLOUD_ROOT and CLOUD_ROOT not in declared_root.parents:
        raise ValueError("unexpected cloud data root")
    if payload.get("target_test_excluded_from_training") is not True:
        raise ValueError("target-test exclusion is not declared")
    label_policy = payload.get("label_policy")
    if not isinstance(label_policy, dict) or label_policy.get("encoding") not in {
        "one_based_1_to_8_with_zero_ignore", "zero_based_0_to_7_with_255_ignore",
    }:
        raise ValueError("data manifest must declare a supported label policy")
    if payload.get("internal_split_policy") != "sha256_sample_id_seed0_80_20_of_official_train_triplets":
        raise ValueError("data manifest split policy is not the approved seed-0 contract")
    if payload.get("split_seed") != 0:
        raise ValueError("data manifest split_seed must be 0")
    if payload.get("group_id_policy") != "sample_id_as_trainarea_stem":
        raise ValueError("data manifest group_id policy is not approved")
    samples = [
        PairedSample(
            sample_id=str(item.get("sample_id") or ""),
            group_id=str(item.get("group_id") or ""),
            split=str(item.get("split") or ""),
            optical_path=_cloud_path(item.get("optical_path"), cloud_root),
            sar_path=_cloud_path(item.get("sar_path"), cloud_root),
            label_path=_cloud_path(item.get("label_path"), cloud_root),
            reliability_path=_cloud_path(item.get("reliability_path"), cloud_root),
        )
        for item in payload.get("samples", [])
    ]
    if not samples:
        raise ValueError("data manifest contains no samples")
    if payload.get("sample_count") != len(samples):
        raise ValueError("data manifest sample_count does not match samples")
    validate_split_integrity(samples)
    return payload, samples


def load_sar_external_manifest(path: Path, *, expected_sha256: str) -> tuple[dict[str, Any], list[SarExternalSample]]:
    """Load the official DFC25 val SAR-only manifest without inventing optical pairs."""
    import hashlib

    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("external SAR manifest SHA256 mismatch")
    payload = json.loads(content)
    if payload.get("host") != "cloud" or payload.get("transfer_policy") != "never_to_local":
        raise ValueError("external SAR manifest must be cloud-only")
    if payload.get("evaluation_role") != "official_dfc25_val_sar_only_external":
        raise ValueError("external SAR manifest has an unexpected evaluation role")
    cloud_root = str(payload.get("cloud_data_root") or "")
    normalized_root = PurePosixPath(posixpath.normpath(cloud_root))
    if normalized_root != CLOUD_ROOT and CLOUD_ROOT not in normalized_root.parents:
        raise ValueError("external SAR manifest escaped cloud data root")
    if payload.get("target_test_excluded_from_training") is not True:
        raise ValueError("external SAR manifest must exclude target test")
    label_policy = payload.get("label_policy")
    if not isinstance(label_policy, dict) or label_policy.get("encoding") != "one_based_1_to_8_with_zero_ignore":
        raise ValueError("external SAR manifest must declare one-based label policy")
    items: list[SarExternalSample] = []
    seen: set[str] = set()
    for item in payload.get("samples", []):
        sample_id = str(item.get("sample_id") or "")
        if not sample_id or sample_id in seen:
            raise ValueError(f"duplicate or empty external sample_id: {sample_id}")
        seen.add(sample_id)
        sar_path = _cloud_path(item.get("sar_path"), cloud_root)
        label_path = _cloud_path(item.get("label_path"), cloud_root)
        items.append(SarExternalSample(sample_id, sar_path, label_path))
    if not items:
        raise ValueError("external SAR manifest contains no samples")
    if payload.get("sample_count") != len(items) or len(items) != 210:
        raise ValueError("external SAR manifest must contain exactly 210 official validation samples")
    return payload, items
