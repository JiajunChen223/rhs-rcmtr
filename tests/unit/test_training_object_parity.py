import copy

import pytest

from rhs_rcmtr.utils.contracts import (
    assert_composition_row,
    assert_extended_training_object_parity,
    assert_training_object_parity,
    matched_protocol_budget_hash,
)
from rhs_rcmtr.models import ModelConfig, build_model


BASELINE = {
    "pretrained_initialization": "frozen_vfm_v1", "data_split": "train_validation",
    "preprocessing": "grid_v1", "augmentation": "non_cloud_v1", "sampler": "group_v1",
    "micro_batch": 4, "effective_batch": 8,
    "optimizer": "adamw", "learning_rate": 1e-4, "scheduler": "cosine",
    "evaluator": "segmentation_v1", "nms": "not_applicable", "seed": 0,
    "epochs": 20, "gradient_accumulation_steps": 2, "weight_decay": 0.01, "num_workers": 4,
    "enabled_mechanism_ids": [],
    "external_trainable_component": False,
}


def test_single_internal_delta_and_hash_match() -> None:
    candidate = copy.deepcopy(BASELINE)
    candidate["enabled_mechanism_ids"] = ["R5-C1-OA-SCRT"]
    assert_training_object_parity(BASELINE, candidate)
    assert matched_protocol_budget_hash(BASELINE) == matched_protocol_budget_hash(candidate)


@pytest.mark.parametrize("mechanism", ["R5-C1-OA-SCRT"])
def test_screening_mechanism_ids_are_single_internal_deltas(mechanism: str) -> None:
    candidate = copy.deepcopy(BASELINE)
    candidate["enabled_mechanism_ids"] = [mechanism]
    assert_training_object_parity(BASELINE, candidate)
    assert matched_protocol_budget_hash(BASELINE) == matched_protocol_budget_hash(candidate)


def test_external_wrapper_is_rejected() -> None:
    candidate = copy.deepcopy(BASELINE)
    candidate.update(enabled_mechanism_ids=["R5-C1-OA-SCRT"], external_trainable_component=True)
    with pytest.raises(ValueError):
        assert_training_object_parity(BASELINE, candidate)


def test_common_protocol_change_is_rejected() -> None:
    candidate = copy.deepcopy(BASELINE)
    candidate.update(enabled_mechanism_ids=["R5-C1-OA-SCRT"], learning_rate=2e-4)
    with pytest.raises(ValueError):
        assert_training_object_parity(BASELINE, candidate)


@pytest.mark.parametrize("field", ["pretrained_initialization", "data_split", "sampler", "nms"])
def test_every_protected_field_is_hashed_and_checked(field: str) -> None:
    candidate = copy.deepcopy(BASELINE)
    candidate.update(enabled_mechanism_ids=["R5-C1-OA-SCRT"])
    candidate[field] = "changed"
    with pytest.raises(ValueError):
        assert_training_object_parity(BASELINE, candidate)
    assert matched_protocol_budget_hash(BASELINE) != matched_protocol_budget_hash(candidate)


def test_common_parameter_trainability_mask_is_preserved() -> None:
    baseline = build_model(ModelConfig())
    candidate = build_model(ModelConfig(enabled_mechanism_ids=("R5-C1-OA-SCRT",)))
    baseline_mask = {name: parameter.requires_grad for name, parameter in baseline.named_parameters()}
    candidate_mask = {name: parameter.requires_grad for name, parameter in candidate.named_parameters() if not name.startswith("router.")}
    assert baseline_mask == candidate_mask


def test_composition_parent_and_undeclared_delta_are_guarded() -> None:
    composition = copy.deepcopy(BASELINE)
    composition.update(enabled_mechanism_ids=["R5-C1-OA-SCRT"], parent_row="baseline")
    assert_composition_row(BASELINE, composition)
    composition["parent_row"] = "candidate_x"
    with pytest.raises(ValueError, match="parent_row"):
        assert_composition_row(BASELINE, composition)
    composition.update(parent_row="baseline", extra_budget=True)
    with pytest.raises(ValueError, match="undeclared"):
        assert_composition_row(BASELINE, composition)


def test_extended_training_object_fields_are_explicitly_guarded() -> None:
    candidate = copy.deepcopy(BASELINE)
    assert_extended_training_object_parity(BASELINE, candidate)
    candidate["feature_normalization_signature"] = "modality_specific_v1"
    with pytest.raises(ValueError, match="extended training-object"):
        assert_extended_training_object_parity(BASELINE, candidate)
    assert_extended_training_object_parity(
        BASELINE, candidate, {"feature_normalization_signature"}
    )
