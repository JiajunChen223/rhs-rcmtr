import torch

from rhs_rcmtr.data.dataset import _map_labels


def test_one_based_train_and_external_val_mapping() -> None:
    result = _map_labels(torch.tensor([[0, 1, 8, 9]]).numpy(), {
        "encoding": "one_based_1_to_8_with_zero_ignore", "ignore_index": 255,
    })
    assert result.tolist() == [[255, 0, 7, 255]]
