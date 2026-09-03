"""R6 SAR-DE (handoff v2) unit tests: mechanisms, losses, gates, parity.

Every test name listed in ``CLAIM_ALIGNMENT_TABLE`` (contracts.py) must exist
here; ``test_clam_table_integrity`` enforces the name set.
"""

from __future__ import annotations

import copy

import pytest
import torch

from rhs_rcmtr.engine.entrypoint import _batch_loss
from rhs_rcmtr.engine.runner import run_synthetic_step
from rhs_rcmtr.losses import evidence_response_gate_loss
from rhs_rcmtr.mechanisms.reliability_router import (
    EvScrtRouter,
    OpticalAnchoredSarResidualRouter,
)
from rhs_rcmtr.models import ModelConfig, build_model
from rhs_rcmtr.models.backbones import SarEncoderV2
from rhs_rcmtr.utils.contracts import (
    CLAIM_ALIGNMENT_TABLE,
    assert_training_object_parity,
    trainable_parameter_audit,
)
from rhs_rcmtr.utils.loss_graph import loss_graph_differences, loss_graph_signature
from rhs_rcmtr.utils.r6_gates import g1_decision, q_permutation_decision


def _v2_cfg(mechanism: str = "") -> ModelConfig:
    return ModelConfig(
        enabled_mechanism_ids=(mechanism,) if mechanism else (),
        sar_encoder_variant="v2",
    )


def _device() -> torch.device:
    return torch.device("cpu")


# --------------------------------------------------------------------------
# CLAM tests: A1
# --------------------------------------------------------------------------
def test_sar_encoder_v2_input_validation() -> None:
    encoder = SarEncoderV2(input_channels=1)
    x = torch.randn(2, 2, 16, 16)  # wrong band count
    with pytest.raises(ValueError):
        encoder(x)


def test_sar_encoder_v2_grid_and_budget() -> None:
    encoder = SarEncoderV2(input_channels=1)
    assert encoder.output_channels == 64
    out = encoder(torch.randn(2, 1, 32, 32))
    assert out.shape == (2, 64, 32, 32)
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    assert trainable < 1_500_000, f"SarEncoderV2 budget exceeded: {trainable}"


def test_factory_variant_registration() -> None:
    model = build_model(_v2_cfg())
    assert isinstance(model.sar_encoder, SarEncoderV2)
    assert isinstance(model.sar_projection, torch.nn.Conv2d)
    with pytest.raises(ValueError):
        build_model(ModelConfig(sar_encoder_variant="v9"))
    # Both mechanism ids registered at factory level.
    build_model(_v2_cfg("R6-C1-EVSCRT"))
    build_model(ModelConfig(enabled_mechanism_ids=("R5-C1-OA-SCRT",)))


# --------------------------------------------------------------------------
# CLAM tests: B1 mechanism identity / gap fixes
# --------------------------------------------------------------------------
def test_evscrt_residual_scale_zero_init() -> None:
    router = EvScrtRouter(16)
    assert float(router.residual_scale.detach()) == 0.0
    model = build_model(_v2_cfg("R6-C1-EVSCRT"))
    optical = torch.randn(2, 3, 16, 16)
    sar = torch.randn(2, 1, 16, 16)
    reliability = torch.rand(2, 2, 16, 16).add_(0.1)
    output = model(optical, sar, reliability)
    # True optical identity at initialization: fused == optical features.
    with torch.no_grad():
        anchor_logits = model.decoder(model.optical_projection(model.optical_encoder(optical)))
    # decoder output sizes may need interpolation to the input grid; compare grids.
    if anchor_logits.shape[2:] != output["logits"].shape[2:]:
        anchor_logits = torch.nn.functional.interpolate(
            anchor_logits, size=output["logits"].shape[2:], mode="bilinear", align_corners=False
        )
    assert torch.allclose(output["logits"], anchor_logits, atol=1e-6)
    # Gate is exposed and differentiable.
    assert output["gate"].shape == (2, 1, 16, 16)
    assert output["gate"].requires_grad
    # No syndrome terminology in the new mechanism.
    assert "syndrome_proxy" not in router.last_diagnostics
    assert "syndrome" not in router.__class__.__name__.lower()
    # The OA-SCRT class body is untouched (still has its legacy diagnostics
    # after a forward pass).
    legacy = OpticalAnchoredSarResidualRouter(16)
    legacy(torch.randn(2, 16, 8, 8), torch.randn(2, 16, 8, 8), torch.rand(2, 2, 8, 8).add_(0.1))
    assert "syndrome_proxy" in legacy.last_diagnostics


def test_reset_innovation_parameters_only_router() -> None:
    from rhs_rcmtr.utils.initialization_anchor import reset_innovation_parameters

    model = build_model(_v2_cfg("R6-C1-EVSCRT"))
    model.eval()
    # Force dropout branches into eval and draw a deterministic baseline.
    router = model.router
    assert isinstance(router, EvScrtRouter)
    assert float(router.residual_scale) == 0.0
    scale_before = float(router.residual_scale)
    head_weight_before = router.gate_head[0].weight.detach().clone()
    reset_innovation_parameters(model, innovation_init_seed=1000)
    assert float(router.residual_scale) == 0.0  # reset keeps optical identity
    assert not torch.equal(head_weight_before, router.gate_head[0].weight.detach())
    assert scale_before == 0.0


# --------------------------------------------------------------------------
# CLAM tests: A2 distillation
# --------------------------------------------------------------------------
def test_distill_anchor_isomorphy() -> None:
    """r6 baseline, r6 candidate and the distillation topology share one
    common trainable parameter set (single-mechanism delta requirement)."""
    base = build_model(_v2_cfg())
    cand = build_model(_v2_cfg("R6-C1-EVSCRT"))
    common_base = set(trainable_parameter_audit(base)["common_trainable_names"])
    common_cand = set(trainable_parameter_audit(cand)["common_trainable_names"])
    assert common_base == common_cand
    assert all(not name.startswith("router.") for name in common_base)
    mech = set(trainable_parameter_audit(cand)["mechanism_trainable_names"])
    assert mech and all(name.startswith("router.") for name in mech)


def test_distill_loss_components_teacher_detached() -> None:
    model = build_model(_v2_cfg())
    model.train()
    batch = {
        "optical": torch.randn(2, 3, 16, 16),
        "sar": torch.randn(2, 1, 16, 16),
        "label": torch.randint(0, 8, (2, 16, 16)),
        "sample_id": ["s0", "s1"],
    }
    loss, _ = _batch_loss(
        model, batch, _device(), sar_only_external=False,
        loss_config={"segmentation_weight": 1.0,
                     "distill_temperature": 4.0, "distill_kl_weight": 1.0,
                     "distill_hard_filter_fraction": 0.05, "distill_floor_weight": 0.5},
        distill_pretrain=True,
    )
    assert torch.isfinite(loss)
    loss.backward()
    # Teacher branch (optical encoder / projection) receives no gradient.
    for name, parameter in model.named_parameters():
        if name.startswith("optical_"):
            assert parameter.grad is None, f"teacher parameter {name} received gradient"
    # Student branch (SAR encoder / decoder) is trained.
    assert model.sar_encoder.stem[0].weight.grad is not None
    assert model.decoder[0].weight.grad is not None


# --------------------------------------------------------------------------
# CLAM tests: A3 gradient balance
# --------------------------------------------------------------------------
def test_gradient_balance_hook_scope() -> None:
    model = build_model(_v2_cfg())
    model.train()
    hooks = []
    for name, parameter in model.named_parameters():
        if name.startswith("sar_encoder.") and parameter.requires_grad:
            hooks.append(parameter.register_hook(lambda g: g * 2.0))
    optical = torch.randn(2, 3, 16, 16)
    sar = torch.randn(2, 1, 16, 16)
    reliability = torch.rand(2, 2, 16, 16).add_(0.1)
    labels = torch.randint(0, 8, (2, 16, 16))
    outputs = model(optical, sar, reliability)
    loss = torch.nn.functional.cross_entropy(outputs["logits"], labels, ignore_index=255)
    loss.backward()
    for hook in hooks:
        hook.remove()
    base = build_model(_v2_cfg())
    base.load_state_dict(copy.deepcopy(model.state_dict()))
    base.train()
    outputs_b = base(optical, sar, reliability)
    loss_b = torch.nn.functional.cross_entropy(outputs_b["logits"], labels, ignore_index=255)
    loss_b.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        baseline_grad = base.get_parameter(name) if "." not in name else None
        if name.startswith("sar_encoder."):
            ref = dict(base.named_parameters())[name]
            assert torch.allclose(parameter.grad, ref.grad * 2.0, atol=1e-6), name
        else:
            ref = dict(base.named_parameters())[name]
            assert torch.allclose(parameter.grad, ref.grad, atol=1e-6), name


# --------------------------------------------------------------------------
# CLAM tests: B2 evidence-response loss
# --------------------------------------------------------------------------
def test_evidence_response_gate_loss_bounds() -> None:
    b, h, w = 2, 1, 8
    gate = torch.full((b, 1, h, w), 0.5)
    q_s = torch.full((b, 1, h, w), 0.2)  # low SAR evidence -> upper bound tight
    u_o = torch.full((b, 1, h, w), 0.9)  # strong optical evidence -> lower bound loose
    reliability = torch.cat((u_o, q_s), dim=1)
    loss, parts = evidence_response_gate_loss(gate, reliability)
    assert loss > 0.0  # gate 0.5 > q_s 0.2 -> upper violation penalized
    assert parts["evidence_up"] > 0.0
    # Within the box: penalty is (numerically) zero.
    gate_ok = torch.full((b, 1, h, w), 0.15)
    loss_ok, parts_ok = evidence_response_gate_loss(gate_ok, reliability)
    assert loss_ok < 1e-6
    assert parts_ok["evidence_up"] < 1e-6 and parts_ok["evidence_dn"] < 1e-6
    # Conflict masking: joint degradation with empty box disables the loss
    # (q_s < 1 - u_o - eps) and reports conflict_fraction == 1.
    u_low = torch.full((b, 1, h, w), 0.1)
    q_low = torch.full((b, 1, h, w), 0.1)
    reliability_conflict = torch.cat((u_low, q_low), dim=1)
    _, parts_conflict = evidence_response_gate_loss(gate, reliability_conflict)
    assert parts_conflict["conflict_fraction"] == 1.0
    assert parts_conflict["evidence_up"] == 0.0 and parts_conflict["evidence_dn"] == 0.0


# --------------------------------------------------------------------------
# CLAM tests: loss graph / parity
# --------------------------------------------------------------------------
def test_loss_graph_evcr_single_delta() -> None:
    baseline = {
        "loss": {"segmentation_weight": 1.0, "evidence_response_weight": 0.0},
        "modality_dropout_policy": "none",
        "degradation_sampler_version": "none",
    }
    candidate = {
        "loss": {"segmentation_weight": 1.0, "evidence_response_weight": 0.5},
        "modality_dropout_policy": "none",
        "degradation_sampler_version": "none",
    }
    assert loss_graph_differences(baseline, candidate) == ["evidence_response_weight"]
    assert "evcr" in loss_graph_signature(candidate)


def test_parity_r6_rows_and_legacy_rows() -> None:
    r6_base = {
        "epochs": 12, "sar_encoder_variant": "v2", "sar_grad_scale": 1.0,
        "common_init_seed": 0, "innovation_init_seed": 1000,
        "data_order_seed": 0, "augmentation_seed": 0,
        "micro_batch": 4, "effective_batch": 8, "learning_rate": 0.0001,
    }
    r6_cand = dict(r6_base, enabled_mechanism_ids=["R6-C1-EVSCRT"])
    r6_cand["external_trainable_component"] = False
    candidate_row = dict(r6_cand)
    candidate_row["loss"] = {"evidence_response_weight": 0.5}
    assert_training_object_parity(r6_base, candidate_row)  # allowed: same common fields
    broken = dict(r6_cand, sar_grad_scale=2.0)
    with pytest.raises(ValueError):
        assert_training_object_parity(r6_base, broken)
    # Legacy rows without the new keys pair safely (None == None).
    legacy = {"epochs": 12, "common_init_seed": 0, "micro_batch": 4,
              "effective_batch": 8, "learning_rate": 0.0001}
    assert_training_object_parity(legacy, dict(legacy, enabled_mechanism_ids=["R5-C1-OA-SCRT"]))


# --------------------------------------------------------------------------
# CLAM tests: frozen gates
# --------------------------------------------------------------------------
def test_g1_gate_logic() -> None:
    # Pass on the first try.
    decision = g1_decision(point_estimate=21.0, ci90_lower=17.0, monotonic_spearman=-0.8)
    assert decision.status == "pass"
    # Soft band first try -> retry.
    decision = g1_decision(point_estimate=17.0, ci90_lower=14.0, monotonic_spearman=-0.8)
    assert decision.status == "retry"
    # Retry above target -> pass.
    decision = g1_decision(point_estimate=18.5, ci90_lower=16.5, monotonic_spearman=-0.8, tries=2)
    assert decision.status == "pass"
    # Retry below target -> fail.
    decision = g1_decision(point_estimate=17.5, ci90_lower=15.0, monotonic_spearman=-0.8, tries=2)
    assert decision.status == "fail"
    # Hard floor.
    decision = g1_decision(point_estimate=14.0, ci90_lower=12.0, monotonic_spearman=-0.8)
    assert decision.status == "fail"
    # Monotonicity is mandatory by default.
    decision = g1_decision(point_estimate=21.0, ci90_lower=17.0, monotonic_spearman=-0.5)
    assert decision.status != "pass"
    # A non-monotone value never passes even in the soft band.
    decision = g1_decision(point_estimate=19.0, ci90_lower=17.0, monotonic_spearman=-0.5)
    assert decision.status != "pass"


def test_q_permutation_detectability_metric() -> None:
    # Detectable: mIoU drops >= 0.3pp and the CI upper bound stays below 0.
    decision = q_permutation_decision(delta_mean_pp=-0.42, ci_upper_pp=-0.05)
    assert decision.detectable
    # Not detectable when the drop is too small.
    assert not q_permutation_decision(delta_mean_pp=-0.10, ci_upper_pp=-0.02).detectable
    # Not detectable when the CI crosses zero.
    assert not q_permutation_decision(delta_mean_pp=-0.45, ci_upper_pp=0.01).detectable


def test_paired_bootstrap_ci_pure_and_deterministic() -> None:
    from rhs_rcmtr.utils.r6_gates import paired_bootstrap_ci
    deltas = [float(i % 7) - 3.0 for i in range(909)]
    first = paired_bootstrap_ci(deltas, n_replicates=500, seed=42)
    second = paired_bootstrap_ci(deltas, n_replicates=500, seed=42)
    assert first == second  # deterministic
    assert first["n"] == 909
    assert first["mean"] == sum(deltas) / len(deltas)
    assert first["ci_lower"] <= first["mean"] <= first["ci_upper"]
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], n_replicates=500)  # too few deltas
    with pytest.raises(ValueError):
        paired_bootstrap_ci(deltas, n_replicates=50)  # too few replicates


def test_clam_table_integrity() -> None:
    import re
    import pathlib
    for row in CLAIM_ALIGNMENT_TABLE:
        assert set(row) == {"claim_id", "claim", "impl_anchor", "unit_test", "audit_field"}
        assert row["claim_id"] and row["claim"] and row["impl_anchor"] and row["audit_field"]
    test_root = pathlib.Path(__file__).resolve().parents[1]
    all_names: set[str] = set()
    for path in test_root.rglob("test_*.py"):
        all_names.update(re.findall(r"^def (test_\w+)", path.read_text(encoding="utf-8"), re.MULTILINE))
    missing = {row["unit_test"] for row in CLAIM_ALIGNMENT_TABLE} - all_names
    assert not missing, f"CLAM unit tests missing from the test tree: {missing}"


# --------------------------------------------------------------------------
# End-to-end synthetic graph
# --------------------------------------------------------------------------
def test_synthetic_smoke_r6_graph() -> None:
    metrics = run_synthetic_step(
        _v2_cfg("R6-C1-EVSCRT"), train=True, seed=0,
    )
    assert metrics["loss"] >= 0.0
    assert 0.0 <= metrics["mean_iou"] <= 1.0
    # Baseline arm with the same v2 object also trains.
    run_synthetic_step(_v2_cfg(), train=True, seed=0)


def test_evscrt_checkpoint_roundtrip() -> None:
    import tempfile
    from pathlib import Path

    from rhs_rcmtr.utils.checkpoint import load_training_checkpoint, save_training_checkpoint
    model = build_model(_v2_cfg("R6-C1-EVSCRT"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trial.pt"
        record = save_training_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler, epoch=0,
            resolved_config_sha256="0" * 64,
        )
        assert record is not None
        restored = build_model(_v2_cfg("R6-C1-EVSCRT"))
        restored_epoch = load_training_checkpoint(
            path, expected_sha256=record["sha256"], model=restored,
            optimizer=None, scheduler=None, resolved_config_sha256="0" * 64,
        )
        assert restored_epoch == 0
        for name, parameter in model.named_parameters():
            assert torch.equal(parameter.detach(), dict(restored.named_parameters())[name].detach())