import torch

from rhs_rcmtr.utils.diagnostics import confusion_metrics, route_entropy, spearman


def test_spearman_is_directional_and_tie_safe() -> None:
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) < -0.99
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) > 0.99


def test_route_entropy_and_confusion_summary_are_stable() -> None:
    weights = torch.tensor([[[[0.5, 1.0]], [[0.5, 0.0]]]])
    entropy = route_entropy(weights)
    assert entropy.shape == (1,)
    assert float(entropy) >= 0.0
    summary = confusion_metrics(torch.tensor([[3, 1], [0, 2]]))
    assert summary["mIoU"] > 0.0
    assert summary["mF1"] > 0.0
