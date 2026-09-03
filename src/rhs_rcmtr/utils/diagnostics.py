"""Numerical helpers for auditable route-behaviour summaries."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

import torch
from torch import Tensor


def rankdata(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    sorted_values = array[order]
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(left: Iterable[float], right: Iterable[float]) -> float:
    """Return tie-aware Spearman correlation, or zero for a constant input."""

    x = rankdata(left)
    y = rankdata(right)
    if x.size != y.size or x.size < 3:
        return 0.0
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator > 0 else 0.0


def route_entropy(weights: Tensor) -> Tensor:
    """Return the mean categorical entropy per sample for [B,2,H,W] weights."""

    if weights.ndim != 4 or weights.shape[1] != 2:
        raise ValueError("route weights must be [B,2,H,W]")
    entropy = -(weights.clamp_min(1e-6) * weights.clamp_min(1e-6).log()).sum(dim=1)
    return entropy.flatten(1).mean(dim=1)


def confusion_metrics(confusion: Tensor) -> dict[str, object]:
    """Convert an integer confusion matrix into mIoU/mF1 and per-class values."""

    matrix = confusion.detach().cpu().double()
    true_positive = torch.diag(matrix)
    row = matrix.sum(dim=1)
    column = matrix.sum(dim=0)
    iou_denominator = row + column - true_positive
    f1_denominator = row + column
    iou = torch.where(iou_denominator > 0, true_positive / iou_denominator, torch.nan)
    f1 = torch.where(f1_denominator > 0, 2.0 * true_positive / f1_denominator, torch.nan)
    return {
        "mIoU": float(torch.nanmean(iou)),
        "mF1": float(torch.nanmean(f1)),
        "per_class_IoU": [None if math.isnan(float(value)) else float(value) for value in iou],
        "per_class_F1": [None if math.isnan(float(value)) else float(value) for value in f1],
        "confusion_matrix": matrix.to(torch.int64).tolist(),
    }
