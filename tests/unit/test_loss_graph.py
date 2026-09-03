import copy

import pytest

from rhs_rcmtr.utils.contracts import assert_loss_graph_delta, loss_graph_evidence
from rhs_rcmtr.utils.loss_graph import loss_graph_differences, loss_graph_signature


def _row() -> dict:
    return {
        "loss": {
            "segmentation_weight": 1.0,
            "honesty_weight": 0.1,
            "counterfactual_weight": 0.0,
        },
        "modality_dropout_policy": "none",
        "degradation_sampler_version": "none",
    }


def test_c3_to_crm_is_one_loss_graph_delta() -> None:
    parent = _row()
    child = copy.deepcopy(parent)
    child["loss"]["counterfactual_weight"] = 0.1
    child["degradation_sampler_version"] = "crm_v2"
    assert loss_graph_signature(parent) == "segmentation+honesty"
    assert loss_graph_signature(child) == "segmentation+honesty+crm_v2"
    assert loss_graph_differences(parent, child) == [
        "counterfactual_weight",
        "degradation_sampler_version",
    ]
    assert_loss_graph_delta(parent, child, {"counterfactual_weight", "degradation_sampler_version"})


def test_mixed_honesty_and_crm_change_is_rejected_as_single_delta() -> None:
    parent = _row()
    child = copy.deepcopy(parent)
    child["loss"]["honesty_weight"] = 0.0
    child["loss"]["counterfactual_weight"] = 0.1
    child["degradation_sampler_version"] = "crm_v2"
    with pytest.raises(ValueError, match="outside declared"):
        assert_loss_graph_delta(parent, child, {"counterfactual_weight", "degradation_sampler_version"})


def test_loss_graph_evidence_is_explicit() -> None:
    row = _row()
    evidence = loss_graph_evidence(row)
    assert evidence["signature"] == "segmentation+honesty"
    assert evidence["degradation_sampler_version"] == "none"
