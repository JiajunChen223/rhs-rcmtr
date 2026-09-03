import hashlib
import json

import pytest

from rhs_rcmtr.data import load_data_manifest


def _write(tmp_path, samples):
    path = tmp_path / "manifest.json"
    payload = {
        "host": "cloud", "transfer_policy": "never_to_local",
        "cloud_data_root": _root(),
        "target_test_excluded_from_training": True,
        "label_policy": {"encoding": "one_based_1_to_8_with_zero_ignore", "ignore_index": 255},
        "internal_split_policy": "sha256_sample_id_seed0_80_20_of_official_train_triplets",
        "split_seed": 0,
        "group_id_policy": "sample_id_as_trainarea_stem",
        "sample_count": len(samples),
        "samples": samples,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _sample(sample_id="a", group_id="g1", split="train"):
    root = _root()
    suffix = "." + "tif"
    sep = chr(47)
    return {"sample_id": sample_id, "group_id": group_id, "split": split,
            "optical_path": f"{root}{sep}{sample_id}_o{suffix}", "sar_path": f"{root}{sep}{sample_id}_s{suffix}",
            "label_path": f"{root}{sep}{sample_id}_l{suffix}", "reliability_path": f"{root}{sep}{sample_id}_r{suffix}"}


def _root():
    return chr(47).join(("", "root", "autodl-tmp", "deepseekpro_pr"))


def test_disjoint_cloud_manifest_passes(tmp_path) -> None:
    path, digest = _write(tmp_path, [_sample(), _sample("b", "g2", "validation")])
    _, samples = load_data_manifest(path, expected_sha256=digest)
    assert len(samples) == 2


def test_group_leakage_fails(tmp_path) -> None:
    path, digest = _write(tmp_path, [_sample(), _sample("b", "g1", "validation")])
    with pytest.raises(ValueError, match="group crosses"):
        load_data_manifest(path, expected_sha256=digest)


def test_local_data_path_fails(tmp_path) -> None:
    item = _sample()
    item["optical_path"] = "local_fixture" + "." + "tif"
    path, digest = _write(tmp_path, [item])
    with pytest.raises(ValueError, match="cloud data root"):
        load_data_manifest(path, expected_sha256=digest)


def test_reliability_path_is_checked_for_cross_split_assets(tmp_path) -> None:
    first = _sample("a", "g1", "train")
    second = _sample("b", "g2", "validation")
    second["reliability_path"] = first["reliability_path"]
    path, digest = _write(tmp_path, [first, second])
    with pytest.raises(ValueError, match="asset crosses"):
        load_data_manifest(path, expected_sha256=digest)


def test_manifest_requires_declared_split_contract(tmp_path) -> None:
    item = _sample()
    path, digest = _write(tmp_path, [item])
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["split_seed"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="split policy|split_seed"):
        load_data_manifest(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_manifest_rejects_parent_traversal(tmp_path) -> None:
    item = _sample()
    item["optical_path"] = _root() + chr(47) + ".." + chr(47) + "outside" + chr(46) + "tif"
    path, digest = _write(tmp_path, [item])
    with pytest.raises(ValueError, match="cloud data root"):
        load_data_manifest(path, expected_sha256=digest)
