import hashlib
import json

import pytest

from rhs_rcmtr.utils.initialization import verify_pretrained_audit


def test_pretrained_audit_is_bound_to_hybrid_initialization(tmp_path) -> None:
    payload = {
        "status": "pass", "protocol_ready": True, "test_accessed": False,
        "initialization_mode": "hybrid_pretrained_with_sar_fallback",
        "input_spec": {"optical_channels": 3, "sar_channels": 1},
        "sar_fallback": {"status": "pass", "reason": "official two-band mismatch"},
        "selected_checkpoints": [],
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = verify_pretrained_audit(path, digest, {
        "mode": "hybrid_pretrained_with_sar_fallback",
        "sar_initialization_mode": "random_init_single_band",
    })
    assert result["status"] == "pass"


def test_pretrained_audit_mode_mismatch_fails(tmp_path) -> None:
    payload = {
        "status": "pass", "protocol_ready": True, "test_accessed": False,
        "initialization_mode": "pretrained", "input_spec": {"optical_channels": 3, "sar_channels": 1},
        "sar_fallback": {"status": "pass", "reason": "fallback"}, "selected_checkpoints": [],
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="initialization mode"):
        verify_pretrained_audit(path, hashlib.sha256(path.read_bytes()).hexdigest(), {
            "mode": "hybrid_pretrained_with_sar_fallback",
            "sar_initialization_mode": "random_init_single_band",
        })
