"""Immutable run-manifest loading and runtime authorization checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rhs_rcmtr.utils.test_seal import assert_test_access_allowed


def load_verified_run_manifest(
    path: Path, expected_sha256: str, requested_split: str, *,
    expected_data_manifest_sha256: str, expected_pretrained_audit_sha256: str,
    expected_code_manifest_sha256: str, expected_resolved_config_sha256: str,
    required_evaluation_role: str = "",
) -> dict[str, Any]:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("run manifest SHA256 mismatch")
    value = json.loads(content)
    if value.get("status") not in {"prepared", "running", "completed"}:
        raise ValueError("run manifest is not executable")
    if value.get("data_location") != "cloud":
        raise ValueError("real experiment run manifest must use cloud data")
    if value.get("router_plan_approval_bound") is not True:
        raise ValueError("run manifest is not bound to PLAN approval")
    bindings = {
        "data_manifest_sha256": expected_data_manifest_sha256,
        "pretrained_audit_sha256": expected_pretrained_audit_sha256,
        "code_manifest_sha256": expected_code_manifest_sha256,
        "resolved_config_sha256": expected_resolved_config_sha256,
    }
    for field, expected in bindings.items():
        if len(expected) != 64 or value.get(field) != expected:
            raise ValueError(f"run manifest binding mismatch: {field}")
    if required_evaluation_role and value.get("evaluation_role") != required_evaluation_role:
        raise ValueError("run manifest evaluation role is not authorized for this entry point")
    if required_evaluation_role == "official_dfc25_val_sar_only_external":
        frozen_flags = (
            "model_frozen", "configuration_frozen", "accepted_hypotheses_frozen",
            "checkpoint_selection_rule_frozen",
        )
        if any(value.get(flag) is not True for flag in frozen_flags):
            raise ValueError("SAR-only external evaluation requires all selection/freeze flags")
    assert_test_access_allowed(value, requested_split)
    value["authorized_split"] = requested_split
    return value
