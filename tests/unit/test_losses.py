import torch

from rhs_rcmtr.losses import counterfactual_monotonicity_loss, segmentation_with_honesty_loss


def _honesty_for_size(size: int) -> float:
    logits = torch.zeros(1, 2, size, size)
    labels = torch.zeros(1, size, size, dtype=torch.long)
    route = torch.full((1, 2, size, size), 0.5)
    reliability = torch.zeros(1, 2, size, size)
    reliability[:, 0] = 0.9
    reliability[:, 1] = 0.1
    _, parts = segmentation_with_honesty_loss(logits, labels, route, reliability, 1.0)
    return float(parts["honesty"])


def test_honesty_is_token_mean_not_area_sum() -> None:
    assert abs(_honesty_for_size(8) - _honesty_for_size(16)) < 1e-6


def test_counterfactual_monotonicity_penalty_only_counts_trust_increases() -> None:
    original = torch.tensor([[[[0.7]], [[0.3]]]])
    degraded = torch.tensor([[[[0.5]], [[0.5]]]])
    assert counterfactual_monotonicity_loss(original, degraded, modality_index=0).item() == 0.0
    worsened = torch.tensor([[[[0.8]], [[0.2]]]])
    assert counterfactual_monotonicity_loss(original, worsened, modality_index=0).item() > 0.0
