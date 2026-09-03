import pytest
import torch

from rhs_rcmtr.models import ModelConfig, build_model


def test_logits_and_router_weights_match_input_resolution() -> None:
    import torch

    model = build_model(ModelConfig(enabled_mechanism_ids=("R5-C1-OA-SCRT",)))
    optical = torch.randn(2, 3, 17, 19)
    sar = torch.randn(2, 1, 17, 19)
    reliability = torch.rand(2, 2, 17, 19)
    output = model(optical, sar, reliability)
    assert output["logits"].shape[2:] == (17, 19)
    assert output["route_weights"].shape[2:] == (17, 19)


@pytest.mark.parametrize("mechanisms", [(), ("R5-C1-OA-SCRT",)])
def test_common_factory_shapes(mechanisms: tuple[str, ...]) -> None:
    model = build_model(ModelConfig(enabled_mechanism_ids=mechanisms))
    result = model(torch.randn(2, 3, 8, 8), torch.randn(2, 1, 8, 8), torch.rand(2, 2, 8, 8))
    assert result["logits"].shape == (2, 8, 8, 8)
    assert result["route_weights"].shape == (2, 2, 8, 8)
    assert torch.allclose(result["route_weights"].sum(1), torch.ones(2, 8, 8), atol=1e-5)


def test_r5_c1_is_internal_and_differentiable() -> None:
    model = build_model(ModelConfig(enabled_mechanism_ids=("R5-C1-OA-SCRT",)))
    result = model(
        torch.randn(2, 3, 8, 8),
        torch.randn(2, 1, 8, 8),
        torch.rand(2, 2, 8, 8),
    )
    result["logits"].mean().backward()
    router_parameters = [p for name, p in model.named_parameters() if name.startswith("router.")]
    assert router_parameters
    assert all(parameter.grad is not None for parameter in router_parameters)


def test_undeclared_mechanism_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_model(ModelConfig(enabled_mechanism_ids=("external_router",)))


def test_r5_c1_builds_and_preserves_shapes() -> None:
    optical = torch.randn(2, 3, 32, 32)
    sar = torch.randn(2, 1, 32, 32)
    reliability = torch.rand(2, 2, 32, 32)
    model = build_model(ModelConfig(hidden_channels=8, enabled_mechanism_ids=("R5-C1-OA-SCRT",)))
    result = model(optical, sar, reliability)
    assert result["logits"].shape == (2, 8, 32, 32)
    assert result["route_weights"].shape == (2, 2, 32, 32)


def test_feature_level_modality_mask_is_explicit() -> None:
    optical = torch.randn(1, 3, 32, 32)
    sar = torch.randn(1, 1, 32, 32)
    reliability = torch.rand(1, 2, 32, 32)
    model = build_model(ModelConfig(hidden_channels=8, enabled_mechanism_ids=("R5-C1-OA-SCRT",)))
    full = model(optical, sar, reliability)
    masked = model(optical, sar, reliability, modality_mask=torch.tensor([[1.0, 0.0]]))
    assert full["logits"].shape == masked["logits"].shape
    assert not torch.allclose(full["logits"], masked["logits"])


@pytest.mark.parametrize(
    ("mode", "active_module", "inactive_module"),
    [
        ("optical_only", "optical_encoder", "sar_encoder"),
        ("sar_only", "sar_encoder", "optical_encoder"),
    ],
)
def test_unimodal_modes_freeze_inactive_encoder(
    mode: str, active_module: str, inactive_module: str
) -> None:
    model = build_model(ModelConfig(unimodal_mode=mode))
    active_parameters = list(getattr(model, active_module).parameters())
    inactive_parameters = list(getattr(model, inactive_module).parameters())
    assert active_parameters and any(parameter.requires_grad for parameter in active_parameters)
    assert inactive_parameters and all(not parameter.requires_grad for parameter in inactive_parameters)


def test_random_single_band_sar_fallback_has_one_channel() -> None:
    import torch
    from rhs_rcmtr.models.backbones import RandomSarSingleBandAdapter

    adapter = RandomSarSingleBandAdapter(input_channels=1)
    output = adapter(torch.randn(1, 1, 16, 16))
    assert output.shape == (1, 64, 16, 16)


def test_dino_input_grid_is_patch_aligned_without_changing_aligned_inputs() -> None:
    from rhs_rcmtr.models.backbones import dino_input_grid

    unaligned = torch.randn(1, 3, 1024, 1001)
    aligned = dino_input_grid(unaligned)
    assert aligned.shape[-2:] == (1022, 994)
    exact = torch.randn(1, 3, 28, 42)
    assert dino_input_grid(exact) is exact
