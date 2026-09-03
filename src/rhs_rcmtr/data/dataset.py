"""Cloud paired-raster dataset; never instantiated for local synthetic tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from rhs_rcmtr.data.manifest import PairedSample, SarExternalSample


def _read_raster(path: str) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("cloud raster execution requires rasterio") from exc
    with rasterio.open(path) as source:
        return source.read()


def _float_tensor(array: np.ndarray) -> Tensor:
    value = torch.from_numpy(np.asarray(array)).float()
    finite = torch.isfinite(value)
    if not finite.all():
        value = torch.where(finite, value, torch.zeros_like(value))
    return value


def _optical_tensor(array: np.ndarray) -> Tensor:
    value = _float_tensor(array)
    if value.shape[0] != 3:
        raise ValueError(f"optical RGB input must have three bands, got {value.shape[0]}")
    # OpenEarthMap RGB is uint8; retain an explicit ImageNet normalization contract for DINOv2.
    value = value / 255.0
    mean = value.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = value.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (value - mean) / std


def _sar_tensor(array: np.ndarray) -> Tensor:
    value = _float_tensor(array)
    if value.shape[0] != 1:
        raise ValueError(f"single-band SAR fallback expects one band, got {value.shape[0]}")
    return value / 255.0


def _map_labels(array: np.ndarray, policy: dict | None) -> Tensor:
    raw = torch.from_numpy(np.asarray(array)).long()
    policy = policy or {"encoding": "zero_based_0_to_7", "ignore_index": 255}
    encoding = str(policy.get("encoding", "zero_based_0_to_7"))
    ignore_index = int(policy.get("ignore_index", 255))
    if encoding == "one_based_1_to_8_with_zero_ignore":
        mapped = torch.full_like(raw, ignore_index)
        valid = (raw >= 1) & (raw <= 8)
        mapped[valid] = raw[valid] - 1
        return mapped
    if encoding == "zero_based_0_to_7_with_255_ignore":
        return raw
    raise ValueError(f"unsupported label encoding: {encoding}")


class CloudPairedRasterDataset(Dataset[dict[str, Tensor | str]]):
    """Read paired optical/SAR/label/reliability assets listed by the cloud manifest."""

    def __init__(
        self, samples: list[PairedSample], split: str, *, authorization: dict,
        label_policy: dict | None = None,
        transform: Callable | None = None,
    ) -> None:
        if authorization.get("authorized_split") != split:
            raise RuntimeError("dataset construction requires a verified run-manifest split authorization")
        self.samples = [sample for sample in samples if sample.split == split]
        if not self.samples:
            raise ValueError(f"no samples for requested split: {split}")
        self.label_policy = label_policy
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        item: dict[str, Tensor | str] = {
            "sample_id": sample.sample_id,
            "optical": _optical_tensor(_read_raster(sample.optical_path)),
            "sar": _sar_tensor(_read_raster(sample.sar_path)),
            "label": _map_labels(_read_raster(sample.label_path)[0], self.label_policy),
            "reliability": _float_tensor(_read_raster(sample.reliability_path)),
        }
        return self.transform(item) if self.transform else item


class CloudSarOnlyRasterDataset(Dataset[dict[str, Tensor | str]]):
    """Read official validation SAR/labels without constructing optical pairs."""

    def __init__(
        self, samples: list[SarExternalSample], *, authorization: dict,
        label_policy: dict | None = None,
    ) -> None:
        if authorization.get("authorized_split") != "validation":
            raise RuntimeError("SAR-only external dataset requires a verified validation authorization")
        if not samples:
            raise ValueError("no external SAR samples")
        self.samples = samples
        self.label_policy = label_policy

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        return {
            "sample_id": sample.sample_id,
            "sar": _sar_tensor(_read_raster(sample.sar_path)),
            "label": _map_labels(_read_raster(sample.label_path)[0], self.label_policy),
        }
