"""Shared per-sample evaluation helpers for the R6 cloud audit scripts.

The evaluation loop is a pure-torch function over generic batch dicts so that
it is locally unit-testable with synthetic loaders; the cloud scripts build the
real (sealed) validation loader and feed it here. ``mean_iou`` is called per
sample so the audit gets one value per paired observation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import torch

from rhs_rcmtr.metrics import mean_iou

if TYPE_CHECKING:
    from torch.nn import Module


def per_sample_miou(
    model: Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    reliability_permute: bool = False,
    forward_sar_only: bool = False,
) -> list[float]:
    """Evaluate one batch, returning one mIoU (0..1) per sample.

    ``reliability_permute`` swaps the two evidence channels ([u_o, q_s] ->
    [q_s, u_o]) before the forward pass; used by the q-permutation audit.
    """
    model.eval()
    optical = batch["optical"].to(device)
    sar = batch["sar"].to(device)
    label = batch["label"].to(device)
    reliability = batch.get("reliability")
    with torch.no_grad():
        if forward_sar_only:
            output = model.forward_sar_only(sar)
        else:
            if reliability is None:
                reliability = torch.ones(
                    optical.shape[0], 2, optical.shape[2], optical.shape[3],
                    device=device, dtype=optical.dtype,
                )
            else:
                reliability = reliability.to(device)
                if reliability_permute:
                    reliability = reliability[:, [1, 0]]
            output = model(optical, sar, reliability)
        logits = output["logits"]
        return [
            float(mean_iou(logits[idx : idx + 1], label[idx : idx + 1], model.config.num_classes))
            for idx in range(logits.shape[0])
        ]


def evaluate_loader_per_sample(
    model: Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    reliability_permute: bool = False,
    forward_sar_only: bool = False,
) -> list[float]:
    """Per-sample mIoU across an entire loader (order preserved)."""
    values: list[float] = []
    for batch in loader:
        values.extend(
            per_sample_miou(
                model, batch, device,
                reliability_permute=reliability_permute,
                forward_sar_only=forward_sar_only,
            )
        )
    return values


def spearman(ordered_pairs: Iterable[tuple[float, float]]) -> float | None:
    """Simple rank correlation guarded for ties; used by the degradation-ladder
    monotonicity audit (frozen G1 criterion: spearman <= -0.7)."""
    import numpy as np

    pairs = [(float(x), float(y)) for x, y in ordered_pairs]
    if len(pairs) < 2:
        return None
    xs = np.asarray([p[0] for p in pairs])
    ys = np.asarray([p[1] for p in pairs])
    if np.std(xs) == 0.0 or np.std(ys) == 0.0:
        return None
    return float(np.corrcoef(_ranks(xs), _ranks(ys))[0, 1])


def _ranks(values: "np.ndarray") -> "np.ndarray":
    """Average ranks with tie handling (equivalent to scipy.stats.rankdata)."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    order = values.argsort(kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=float)
    uniq, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    averaged = (cumulative - (counts - 1) / 2.0)[inverse]
    return averaged


def sar_ladder_miou(
    model: Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    family: str = "speckle_increase",
    severities: tuple[float, ...] = (0.10, 0.25, 0.40, 0.55, 0.70),
) -> dict[str, float]:
    """SAR-only mIoU across a severity ladder (frozen G1 monotonicity check).

    Returns mean per-sample mIoU per severity step plus the rank correlation
    (spearman) between severity and quality. The P1-audited expectation is a
    monotone decline with severity (spearman <= -0.7).
    """
    from rhs_rcmtr.utils.interventions import degrade_sensor

    means: list[float] = []
    for severity in severities:
        values: list[float] = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                sar = batch["sar"].to(device)
                label = batch["label"].to(device)
                degraded = degrade_sensor(sar, 1, family, float(severity))
                output = model.forward_sar_only(degraded)
                for idx in range(output["logits"].shape[0]):
                    values.append(
                        float(mean_iou(output["logits"][idx : idx + 1], label[idx : idx + 1], model.config.num_classes))
                    )
        means.append(sum(values) / max(len(values), 1))
    correlation = spearman(list(zip(severities, means)))
    return {"severities": list(severities), "mean_miou": means, "spearman": correlation}