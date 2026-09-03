"""Declared training losses."""

from .objectives import (
    counterfactual_monotonicity_loss,
    evidence_response_gate_loss,
    segmentation_with_honesty_loss,
)

__all__ = [
    "counterfactual_monotonicity_loss",
    "evidence_response_gate_loss",
    "segmentation_with_honesty_loss",
]
