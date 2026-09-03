import torch

from rhs_rcmtr.models import ModelConfig, build_model
from rhs_rcmtr.utils.interventions import apply_modality_dropout


def test_sar_auxiliary_head_is_training_only() -> None:
    model = build_model(ModelConfig(hidden_channels=4, unimodal_mode="sar_only", auxiliary_sar_head=True))
    batch = (torch.randn(2, 3, 8, 8), torch.rand(2, 1, 8, 8), torch.rand(2, 2, 8, 8))
    model.train()
    training_output = model(*batch)
    assert "sar_auxiliary_logits" in training_output
    model.eval()
    eval_output = model(*batch)
    assert "sar_auxiliary_logits" not in eval_output


def test_modality_dropout_is_deterministic_and_zeroes_matching_evidence() -> None:
    optical = torch.ones(2, 3, 4, 4)
    sar = torch.ones(2, 1, 4, 4)
    evidence = torch.ones(2, 2, 4, 4)
    first = apply_modality_dropout(optical, sar, evidence, ["sample-a"], "random_single_modality_p0.5")
    second = apply_modality_dropout(optical, sar, evidence, ["sample-a"], "random_single_modality_p0.5")
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert (first[0] == 0).all() or (first[1] == 0).all()
    dropped = 0 if (first[0] == 0).all() else 1
    assert (first[2][:, dropped] == 0).all()
