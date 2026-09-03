import torch

from rhs_rcmtr.losses import counterfactual_monotonicity_loss
from rhs_rcmtr.utils.interventions import (
    degrade_pair,
    recompute_reliability_evidence,
    select_crm_intervention,
)


def test_degrade_pair_changes_exactly_one_modality() -> None:
    optical = torch.ones(2, 3, 8, 8)
    sar = torch.ones(2, 1, 8, 8)
    degraded_optical, degraded_sar = degrade_pair(optical, sar, 0, "blur", 0.5)
    assert not torch.equal(degraded_optical, optical)
    assert torch.equal(degraded_sar, sar)
    same_optical, degraded_sar = degrade_pair(optical, sar, 1, "intensity_noise", 0.5)
    assert torch.equal(same_optical, optical)
    assert not torch.equal(degraded_sar, sar)


def test_recomputed_evidence_is_two_channel_and_bounded() -> None:
    optical = torch.randn(2, 3, 8, 8)
    sar = torch.rand(2, 1, 8, 8)
    evidence = recompute_reliability_evidence(optical, sar)
    assert evidence.shape == (2, 2, 8, 8)
    assert float(evidence.min()) >= 0.0
    assert float(evidence.max()) <= 1.0


def test_crm_selection_is_deterministic_and_masked_penalty_is_finite() -> None:
    first = select_crm_intervention(["sample-a", "sample-b"])
    second = select_crm_intervention(["sample-a", "sample-b"])
    assert first == second
    original = torch.tensor([[[[0.7]], [[0.3]]]])
    degraded = torch.tensor([[[[0.8]], [[0.2]]]])
    mask = torch.ones(1, 1, 1, dtype=torch.bool)
    value = counterfactual_monotonicity_loss(
        original, degraded, modality_index=0, valid_mask=mask
    )
    assert torch.isfinite(value)
    assert value.item() > 0.0


def test_crm_v2_batch_loss_records_intervention_metadata() -> None:
    from rhs_rcmtr.engine.entrypoint import _batch_loss
    from rhs_rcmtr.models import ModelConfig, build_model

    model = build_model(ModelConfig(hidden_channels=4, enabled_mechanism_ids=("R5-C1-OA-SCRT",)))
    batch = {
        "sample_id": ["sample-a", "sample-b"],
        "optical": torch.randn(2, 3, 8, 8),
        "sar": torch.rand(2, 1, 8, 8),
        "reliability": torch.rand(2, 2, 8, 8),
        "label": torch.zeros(2, 8, 8, dtype=torch.long),
    }
    stats: dict = {"by_intervention": {}, "examples": []}
    loss, _ = _batch_loss(
        model,
        batch,
        torch.device("cpu"),
        sar_only_external=False,
        loss_config={
            "segmentation_weight": 1.0,
            "honesty_weight": 0.1,
            "counterfactual_weight": 0.1,
            "degradation_sampler_version": "crm_v2",
        },
        crm_stats=stats,
    )
    loss.backward()
    assert stats["batch_count"] == 1
    assert stats["by_intervention"]
    assert stats["examples"][0]["degradation"]
