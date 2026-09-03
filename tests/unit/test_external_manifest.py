import hashlib
import json

import pytest

from rhs_rcmtr.data import load_sar_external_manifest


def test_external_sar_manifest_is_explicit_and_cloud_only(tmp_path) -> None:
    separator = chr(47)
    root = separator + "root" + separator + "autodl-tmp" + separator + "project"
    suffix = chr(46) + "tif"
    payload = {
        "host": "cloud", "transfer_policy": "never_to_local", "cloud_data_root": root,
        "evaluation_role": "official_dfc25_val_sar_only_external",
        "target_test_excluded_from_training": True,
        "sample_count": 210,
        "label_policy": {"encoding": "one_based_1_to_8_with_zero_ignore", "ignore_index": 255},
        "samples": [
            {"sample_id": f"ValArea_{i:03d}", "sar_path": root + separator + f"s{i}" + suffix,
             "label_path": root + separator + f"l{i}" + suffix}
            for i in range(1, 211)
        ],
    }
    path = tmp_path / "external.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _, samples = load_sar_external_manifest(path, expected_sha256=digest)
    assert samples[0].sample_id == "ValArea_001"


def test_external_sar_manifest_rejects_wrong_role(tmp_path) -> None:
    separator = chr(47)
    root = separator + "root" + separator + "autodl-tmp" + separator + "p"
    payload = {"host": "cloud", "transfer_policy": "never_to_local", "cloud_data_root": root, "samples": []}
    path = tmp_path / "external.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation role"):
        load_sar_external_manifest(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_external_sar_manifest_requires_exact_official_sample_count(tmp_path) -> None:
    separator = chr(47)
    root = separator + "root" + separator + "autodl-tmp" + separator + "p"
    suffix = chr(46) + "tif"
    payload = {
        "host": "cloud", "transfer_policy": "never_to_local", "cloud_data_root": root,
        "evaluation_role": "official_dfc25_val_sar_only_external",
        "target_test_excluded_from_training": True, "sample_count": 1,
        "label_policy": {"encoding": "one_based_1_to_8_with_zero_ignore", "ignore_index": 255},
        "samples": [{"sample_id": "ValArea_001", "sar_path": root + separator + "s" + suffix, "label_path": root + separator + "l" + suffix}],
    }
    path = tmp_path / "external.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 210"):
        load_sar_external_manifest(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
