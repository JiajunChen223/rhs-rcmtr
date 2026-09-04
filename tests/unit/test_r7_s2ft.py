"""Synthetic-only scientific invariant tests for R7-S2FT."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from rhs_rcmtr.mechanisms.geo_local_interaction import GeoLocalDeformableCrossInteraction
from rhs_rcmtr.mechanisms.structural_refinement import SarStructuralRefinement
from rhs_rcmtr.models import ModelConfig, build_model
from rhs_rcmtr.models.s2ft.patch_transplant import SarFoundationPatchEmbed
from rhs_rcmtr.models.s2ft.sensor_adapter import SensorAdapter
from rhs_rcmtr.utils.contracts import (
    assert_training_object_parity,
    matched_protocol_budget_hash,
    trainable_parameter_audit,
)
from rhs_rcmtr.utils.initialization_anchor import reset_innovation_parameters


class _SourcePatch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 8, 4, stride=4, bias=True)
        self.norm = nn.Identity()


def _r7_model() -> nn.Module:
    torch.manual_seed(7)
    return build_model(ModelConfig(enabled_mechanism_ids=("R7-S2FT",)))


def test_s2ft_single_band_patch_transplant() -> None:
    embed = SarFoundationPatchEmbed(_SourcePatch())
    output = embed(torch.randn(2, 1, 16, 16))
    assert output.shape == (2, 16, 8)
    with pytest.raises(ValueError):
        embed(torch.randn(2, 2, 16, 16))


def test_s2ft_patch_transplant_energy_preserving() -> None:
    source = _SourcePatch()
    with torch.no_grad():
        source.proj.weight.fill_(2.0)
        source.proj.bias.fill_(0.25)
    embed = SarFoundationPatchEmbed(source)
    expected = torch.full_like(embed.proj.weight, 2.0 * math.sqrt(3.0))
    assert torch.allclose(embed.proj.weight, expected)
    assert torch.allclose(embed.proj.bias, source.proj.bias)
    # Non-persistent source snapshots must not register a duplicate source model.
    assert not any("source_patch_embed" in key for key in embed.state_dict())


def test_s2ft_adapter_identity_init() -> None:
    adapter = SensorAdapter(32, 8)
    x = torch.randn(2, 9, 32)
    assert torch.equal(adapter(x), x)


def test_s2ft_glci_identity_and_bounded_offsets() -> None:
    glci = GeoLocalDeformableCrossInteraction(32, interaction_dim=8, max_offset_tokens=1.5)
    optical = torch.randn(2, 32, 5, 6)
    sar = torch.randn(2, 32, 5, 6)
    out_o, out_s = glci(optical, sar)
    assert torch.equal(out_o, optical)
    assert torch.equal(out_s, sar)
    offsets = glci.sar_to_optical.bounded_offsets(optical, sar)
    assert float(offsets.abs().max()) <= 1.5 + 1e-6
    with torch.no_grad():
        glci.sar_to_optical.offset.bias.fill_(100.0)
    offsets = glci.sar_to_optical.bounded_offsets(optical, sar)
    assert float(offsets.abs().max()) <= 1.5 + 1e-6


def test_s2ft_sdr_identity_init() -> None:
    sdr = SarStructuralRefinement(32, 16)
    semantic = torch.randn(2, 32, 6, 6)
    sar = torch.randn(2, 1, 24, 24)
    assert torch.equal(sdr(semantic, sar), semantic)


def test_s2ft_reliability_is_not_a_model_input() -> None:
    model = _r7_model().eval()
    optical = torch.randn(2, 3, 16, 16)
    sar = torch.randn(2, 1, 16, 16)
    rel_a = torch.zeros(2, 2, 16, 16)
    rel_b = torch.rand(2, 2, 16, 16)
    with torch.no_grad():
        out_a = model(optical, sar, rel_a)["logits"]
        out_b = model(optical, sar, rel_b)["logits"]
    assert torch.equal(out_a, out_b)


def test_s2ft_modality_mask_has_no_feature_leakage() -> None:
    model = _r7_model().eval()
    innovation = model.innovation
    assert innovation is not None
    foundation = innovation.foundation
    optical = torch.randn(2, 3, 16, 16)
    sar = torch.randn(2, 1, 16, 16)
    sar_off = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    with torch.no_grad():
        _opt, sar_features, _sar, _om, sar_mask = foundation(optical, sar, sar_off)
    assert torch.count_nonzero(sar_mask) == 0
    for feature in sar_features.values():
        assert torch.count_nonzero(feature) == 0


def test_s2ft_frozen_foundation_allows_adapter_gradient() -> None:
    model = _r7_model().train()
    innovation = model.innovation
    assert innovation is not None
    foundation = innovation.foundation
    assert all(not parameter.requires_grad for parameter in foundation.dino.parameters())
    optical = torch.randn(2, 3, 16, 16)
    sar = torch.randn(2, 1, 16, 16)
    reliability = torch.rand(2, 2, 16, 16)
    logits = model(optical, sar, reliability)["logits"]
    logits.mean().backward()
    adapter_grads = [
        parameter.grad
        for name, parameter in foundation.adapters.named_parameters()
        if name.endswith("up.weight")
    ]
    assert adapter_grads and any(grad is not None for grad in adapter_grads)
    assert foundation.sar_patch_embed.proj.weight.grad is not None
    assert all(parameter.grad is None for parameter in foundation.dino.parameters())


def test_s2ft_innovation_reset_preserves_foundation() -> None:
    model = _r7_model()
    innovation = model.innovation
    assert innovation is not None
    before = {
        name: tensor.detach().clone()
        for name, tensor in innovation.foundation.dino.state_dict().items()
    }
    with torch.no_grad():
        next(iter(innovation.decoder.parameters())).add_(1.0)
    reset_innovation_parameters(model, 20260904)
    after = innovation.foundation.dino.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_s2ft_factory_shape_and_compatibility_weights() -> None:
    model = _r7_model().eval()
    output = model(
        torch.randn(2, 3, 17, 19),
        torch.randn(2, 1, 17, 19),
        torch.rand(2, 2, 17, 19),
    )
    assert output["logits"].shape == (2, 8, 17, 19)
    assert output["route_weights"].shape == (2, 2, 17, 19)
    assert torch.allclose(output["route_weights"].sum(1), torch.ones(2, 17, 19))
    assert output["route_weights_semantics"] == "compatibility_constant_not_model_output"


def test_s2ft_parameter_audit_separates_innovation() -> None:
    model = _r7_model()
    audit = trainable_parameter_audit(model)
    assert audit["innovation_trainable_names"]
    assert all(name.startswith("innovation.") for name in audit["innovation_trainable_names"])
    assert audit["common_trainable_names"] == []
    dino_keys = [key for key in model.state_dict() if ".foundation.dino." in key]
    assert dino_keys
    assert not any("source_patch_embed" in key for key in model.state_dict())


def test_s2ft_candidate_keeps_parent_protocol_and_loss_graph() -> None:
    root = Path(__file__).resolve().parents[2]
    baseline = yaml.safe_load((root / "configs/experiment/r7/baseline.yaml").read_text(encoding="utf-8"))
    candidate = yaml.safe_load((root / "configs/experiment/r7/r7_s2ft_full.yaml").read_text(encoding="utf-8"))
    assert_training_object_parity(baseline, candidate)
    assert matched_protocol_budget_hash(baseline) == matched_protocol_budget_hash(candidate)
    assert baseline["loss"] == candidate["loss"]
    assert baseline["loss_graph_signature"] == candidate["loss_graph_signature"]
    differing = {
        key for key in set(baseline) | set(candidate)
        if baseline.get(key) != candidate.get(key)
    }
    assert differing == {"enabled_mechanism_ids"}
