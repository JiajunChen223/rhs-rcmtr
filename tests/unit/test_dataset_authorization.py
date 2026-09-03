import pytest

from rhs_rcmtr.data import CloudPairedRasterDataset, PairedSample


def test_dataset_rejects_unverified_split_authorization() -> None:
    sample = PairedSample("a", "g", "train", "o", "s", "l", "r")
    with pytest.raises(RuntimeError, match="authorization"):
        CloudPairedRasterDataset([sample], "train", authorization={})
