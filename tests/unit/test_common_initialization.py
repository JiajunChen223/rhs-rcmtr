import torch
import pytest

from rhs_rcmtr.models import ModelConfig, build_model
from rhs_rcmtr.utils.initialization_anchor import (
    common_parameter_state,
    load_common_init_anchor,
    reset_innovation_parameters,
    save_common_init_anchor,
)


def test_common_initialization_anchor_makes_common_tensors_equal(tmp_path) -> None:
    torch.manual_seed(0)
    baseline = build_model(ModelConfig())
    anchor = tmp_path / "COMMON_INIT_SEED0.pt"
    record = save_common_init_anchor(
        anchor,
        baseline,
        common_init_seed=0,
        resolved_config_sha256="a" * 64,
        data_order_seed=0,
        augmentation_seed=0,
    )
    torch.manual_seed(1000)
    candidate = build_model(ModelConfig(enabled_mechanism_ids=("R5-C1-OA-SCRT",)))
    loaded = load_common_init_anchor(anchor, expected_sha256=record["sha256"], model=candidate)
    assert loaded["status"] == "pass"
    baseline_state = common_parameter_state(baseline)
    candidate_state = common_parameter_state(candidate)
    assert baseline_state.keys() == candidate_state.keys()
    assert all(torch.equal(baseline_state[name], candidate_state[name]) for name in baseline_state)


def test_innovation_seed_changes_only_router_after_common_anchor(tmp_path) -> None:
    torch.manual_seed(0)
    baseline = build_model(ModelConfig())
    anchor = tmp_path / "COMMON_INIT_SEED0.pt"
    record = save_common_init_anchor(
        anchor,
        baseline,
        common_init_seed=0,
        resolved_config_sha256="b" * 64,
        data_order_seed=0,
        augmentation_seed=0,
    )
    torch.manual_seed(123)
    candidate_a = build_model(ModelConfig(enabled_mechanism_ids=("R5-C1-OA-SCRT",)))
    load_common_init_anchor(anchor, expected_sha256=record["sha256"], model=candidate_a)
    reset_innovation_parameters(candidate_a, 1000)
    torch.manual_seed(456)
    candidate_b = build_model(ModelConfig(enabled_mechanism_ids=("R5-C1-OA-SCRT",)))
    load_common_init_anchor(anchor, expected_sha256=record["sha256"], model=candidate_b)
    reset_innovation_parameters(candidate_b, 1001)
    assert all(
        torch.equal(parameter_a, parameter_b)
        for (name_a, parameter_a), (name_b, parameter_b)
        in zip(candidate_a.named_parameters(), candidate_b.named_parameters())
        if not name_a.startswith("router.") and name_a == name_b
    )


def test_anchor_rejects_common_parameter_shape_mismatch(tmp_path) -> None:
    torch.manual_seed(0)
    baseline = build_model(ModelConfig())
    anchor = tmp_path / "COMMON_INIT_SEED0.pt"
    record = save_common_init_anchor(
        anchor,
        baseline,
        common_init_seed=0,
        resolved_config_sha256="c" * 64,
        data_order_seed=0,
        augmentation_seed=0,
    )
    candidate = build_model(ModelConfig(hidden_channels=8))
    try:
        load_common_init_anchor(anchor, expected_sha256=record["sha256"], model=candidate)
    except ValueError as exc:
        assert "shape mismatch" in str(exc) or "names mismatch" in str(exc)
    else:
        raise AssertionError("shape mismatch must be rejected")


@pytest.mark.parametrize("mode", ["optical_only", "sar_only"])
def test_common_anchor_accepts_unimodal_diagnostic_masks(tmp_path, mode: str) -> None:
    torch.manual_seed(0)
    baseline = build_model(ModelConfig())
    anchor = tmp_path / f"COMMON_INIT_{mode}.pt"
    record = save_common_init_anchor(
        anchor,
        baseline,
        common_init_seed=0,
        resolved_config_sha256="d" * 64,
        data_order_seed=0,
        augmentation_seed=0,
    )
    diagnostic = build_model(ModelConfig(unimodal_mode=mode))
    loaded = load_common_init_anchor(anchor, expected_sha256=record["sha256"], model=diagnostic)
    assert loaded["status"] == "pass"


def test_common_anchor_allows_unanchored_training_support_head(tmp_path) -> None:
    torch.manual_seed(0)
    baseline = build_model(ModelConfig())
    anchor = tmp_path / "COMMON_INIT_aux.pt"
    record = save_common_init_anchor(
        anchor,
        baseline,
        common_init_seed=0,
        resolved_config_sha256="e" * 64,
        data_order_seed=0,
        augmentation_seed=0,
    )
    diagnostic = build_model(ModelConfig(auxiliary_sar_head=True))
    loaded = load_common_init_anchor(anchor, expected_sha256=record["sha256"], model=diagnostic)
    assert loaded["status"] == "pass"
    assert set(loaded["anchor_parameter_names"]) <= set(loaded["common_parameter_names"])
    assert any(name.startswith("sar_auxiliary_decoder.") for name in loaded["common_parameter_names"])
