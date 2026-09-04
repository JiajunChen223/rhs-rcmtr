"""Predeclared R7 architecture-row and protocol-parity tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rhs_rcmtr.models import ModelConfig, build_model
from rhs_rcmtr.models.s2ft import R7_S2FT_VARIANTS
from rhs_rcmtr.utils.contracts import assert_training_object_parity, matched_protocol_budget_hash


@pytest.mark.parametrize(
    ("mechanism_id", "expected_interactions", "expected_sdr"),
    [
        ("R7-A-SAFT", (), False),
        ("R7-B-GLCI12", (11,), False),
        ("R7-C-GLCI-MULTI", (3, 7, 11), False),
        ("R7-S2FT", (3, 7, 11), True),
    ],
)
def test_r7_predeclared_variant_builds_exact_architecture(
    mechanism_id: str,
    expected_interactions: tuple[int, ...],
    expected_sdr: bool,
) -> None:
    assert mechanism_id in R7_S2FT_VARIANTS
    model = build_model(ModelConfig(enabled_mechanism_ids=(mechanism_id,)))
    innovation = model.innovation
    assert innovation is not None
    assert tuple(innovation.config.interaction_layers) == expected_interactions
    assert (innovation.decoder.structural_refinement is not None) is expected_sdr
    assert tuple(int(key) for key in innovation.foundation.interactions.keys()) == expected_interactions


@pytest.mark.parametrize(
    "filename",
    [
        "r7_a_saft.yaml",
        "r7_b_glci12.yaml",
        "r7_c_glci_multi.yaml",
        "r7_s2ft_full.yaml",
    ],
)
def test_every_r7_candidate_row_matches_parent_protocol(filename: str) -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "configs/experiment/r7"
    baseline = yaml.safe_load((folder / "baseline.yaml").read_text(encoding="utf-8"))
    candidate = yaml.safe_load((folder / filename).read_text(encoding="utf-8"))
    assert_training_object_parity(baseline, candidate)
    assert matched_protocol_budget_hash(baseline) == matched_protocol_budget_hash(candidate)
    assert baseline["loss"] == candidate["loss"]
    assert baseline["loss_graph_signature"] == candidate["loss_graph_signature"]
    differing = {
        key for key in set(baseline) | set(candidate)
        if baseline.get(key) != candidate.get(key)
    }
    assert differing == {"enabled_mechanism_ids"}
