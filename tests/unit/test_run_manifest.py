import hashlib
import json

import pytest

from rhs_rcmtr.utils.run_manifest import load_verified_run_manifest


def _manifest(tmp_path, **changes):
    value = {"status": "prepared", "data_location": "cloud", "router_plan_approval_bound": True,
             "execution_scale": "baseline", "test_seal_status": "sealed",
             "data_manifest_sha256": "d" * 64, "pretrained_audit_sha256": "p" * 64,
             "code_manifest_sha256": "c" * 64, "resolved_config_sha256": "r" * 64}
    value.update(changes)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_validation_manifest_passes(tmp_path) -> None:
    path, digest = _manifest(tmp_path)
    assert load_verified_run_manifest(
        path, digest, "validation", expected_data_manifest_sha256="d" * 64,
        expected_pretrained_audit_sha256="p" * 64,
        expected_code_manifest_sha256="c" * 64,
        expected_resolved_config_sha256="r" * 64,
    )["status"] == "prepared"


def test_forged_yaml_cannot_unseal_test(tmp_path) -> None:
    path, digest = _manifest(tmp_path, execution_scale="baseline", test_seal_status="final_test")
    with pytest.raises(RuntimeError):
        load_verified_run_manifest(
            path, digest, "test", expected_data_manifest_sha256="d" * 64,
            expected_pretrained_audit_sha256="p" * 64,
            expected_code_manifest_sha256="c" * 64,
            expected_resolved_config_sha256="r" * 64,
        )


def test_manifest_hash_mismatch_fails(tmp_path) -> None:
    path, _ = _manifest(tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        load_verified_run_manifest(
            path, "0" * 64, "validation", expected_data_manifest_sha256="d" * 64,
            expected_pretrained_audit_sha256="p" * 64,
            expected_code_manifest_sha256="c" * 64,
            expected_resolved_config_sha256="r" * 64,
        )


def test_cross_manifest_substitution_fails(tmp_path) -> None:
    path, digest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="binding mismatch"):
        load_verified_run_manifest(
            path, digest, "validation", expected_data_manifest_sha256="x" * 64,
            expected_pretrained_audit_sha256="p" * 64,
            expected_code_manifest_sha256="c" * 64,
            expected_resolved_config_sha256="r" * 64,
        )
