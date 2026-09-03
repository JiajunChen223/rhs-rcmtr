import hashlib

import pytest
import torch

from rhs_rcmtr.models import ModelConfig, build_model
from rhs_rcmtr.utils.initialization import load_audited_checkpoint


def test_checkpoint_hash_and_compatibility_report(tmp_path) -> None:
    model = build_model(ModelConfig())
    path = tmp_path / "synthetic_checkpoint.fixture"
    torch.save({"state_dict": model.state_dict()}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report = load_audited_checkpoint(model, path, expected_sha256=digest, allow_synthetic_fixture=True)
    assert report["status"] == "pass"
    assert report["shape_mismatches"] == []


def test_wrong_hash_is_rejected(tmp_path) -> None:
    path = tmp_path / "synthetic_checkpoint.fixture"
    torch.save({"state_dict": build_model(ModelConfig()).state_dict()}, path)
    with pytest.raises(ValueError, match="SHA256"):
        load_audited_checkpoint(build_model(ModelConfig()), path, expected_sha256="0" * 64, allow_synthetic_fixture=True)


def test_local_checkpoint_is_rejected_by_default(tmp_path) -> None:
    path = tmp_path / "synthetic_checkpoint.fixture"
    torch.save({"state_dict": build_model(ModelConfig()).state_dict()}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="cloud-only"):
        load_audited_checkpoint(build_model(ModelConfig()), path, expected_sha256=digest)
