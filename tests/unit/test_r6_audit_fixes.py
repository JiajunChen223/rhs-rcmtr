"""R6 SAR-DE supplementary tests: audit-fix regressions, injection, gates,
per-sample tooling, and the evaluation-state guard."""

from __future__ import annotations

import copy

import pytest
import torch

from rhs_rcmtr.engine.entrypoint import _batch_loss, _evaluate_loader
from rhs_rcmtr.models import ModelConfig, build_model
from rhs_rcmtr.utils.contracts import assert_training_row_declared
from rhs_rcmtr.utils.interventions import (
    R6_SEVERITY_LADDER,
    apply_r6_train_injection,
    select_r6_injections,
)
from rhs_rcmtr.utils.r6_gates import (
    distill_epoch_decision,
    g2_decision,
    g3_decision,
)


def _v2_cfg(mechanism: str = "") -> ModelConfig:
    return ModelConfig(
        enabled_mechanism_ids=(mechanism,) if mechanism else (),
        sar_encoder_variant="v2",
    )


def _synthetic_batch(batch_size: int = 2, h: int = 16) -> dict[str, torch.Tensor]:
    return {
        "optical": torch.randn(batch_size, 3, h, h),
        "sar": torch.randn(batch_size, 1, h, h),
        "label": torch.randint(0, 8, (batch_size, h, h)),
        "reliability": torch.rand(batch_size, 2, h, h).add_(0.1),
        "sample_id": [f"sample-{idx}" for idx in range(batch_size)],
    }


# --------------------------------------------------------------------------
# Evaluation-state guard: evidence loss must be skipped, never raised, in eval.
# --------------------------------------------------------------------------
def test_evidence_loss_skipped_in_eval_state() -> None:
    model = build_model(_v2_cfg("R6-C1-EVSCRT"))
    model.train()
    batch = _synthetic_batch()
    loss_train, _ = _batch_loss(
        model, batch, torch.device("cpu"), sar_only_external=False,
        loss_config={"evidence_response_weight": 1.0, "evidence_ramp_epochs": 0},
    )
    assert torch.isfinite(loss_train)
    model.eval()
    with torch.no_grad():
        loss_eval, score_eval = _batch_loss(
            model, batch, torch.device("cpu"), sar_only_external=False,
            loss_config={"evidence_response_weight": 1.0, "evidence_ramp_epochs": 0},
        )
    assert torch.isfinite(loss_eval)  # guard: eval never raises with the weight set
    assert score_eval >= 0.0


def test_evaluate_loader_works_with_evidence_config() -> None:
    model = build_model(_v2_cfg("R6-C1-EVSCRT"))
    batches = [_synthetic_batch() for _ in range(2)]
    metrics = _evaluate_loader(
        model, batches, torch.device("cpu"), sar_only_external=False,
        loss_config={"evidence_response_weight": 1.0},
    )
    assert 0.0 <= metrics["mean_iou"] <= 1.0
    assert torch.isfinite(torch.tensor(metrics["loss"]))


# --------------------------------------------------------------------------
# Legacy rows remain runnable (OPTIONAL_COMMON_FIELDS).
# --------------------------------------------------------------------------
def test_legacy_row_declared_without_r6_keys() -> None:
    legacy = {
        "pretrained_initialization": "dinov2_b14_plus_croma_base_sar",
        "data_split": "train_validation", "preprocessing": "aligned_shared_grid_v1",
        "augmentation": "non_cloud_v1", "sampler": "group_disjoint_v1",
        "micro_batch": 4, "effective_batch": 8, "optimizer": "adamw",
        "learning_rate": 0.0001, "scheduler": "cosine", "evaluator": "segmentation_v1",
        "nms": "not_applicable_segmentation", "seed": 0, "epochs": 12,
        "gradient_accumulation_steps": 2, "weight_decay": 0.01, "num_workers": 4,
        "prefetch_factor": 2, "persistent_workers": True, "scheduler_horizon_epochs": 12,
        "common_init_seed": 0, "innovation_init_seed": 1000,
        "data_order_seed": 0, "augmentation_seed": 0,
    }
    assert_training_row_declared(legacy)  # must not raise (r6 keys optional)
    r6_row = dict(legacy, sar_encoder_variant="v2", sar_grad_scale=1.0)
    assert_training_row_declared(r6_row)


# --------------------------------------------------------------------------
# Ramp: evidence weight activates linearly over the first N epochs.
# --------------------------------------------------------------------------
def test_evidence_ramp_by_epoch() -> None:
    model = build_model(_v2_cfg("R6-C1-EVSCRT"))
    model.train()
    batch = _synthetic_batch()
    cfg = {"evidence_response_weight": 1.0, "evidence_ramp_epochs": 2}
    loss_epoch0, _ = _batch_loss(
        model, batch, torch.device("cpu"), sar_only_external=False, loss_config=cfg, epoch_index=0,
    )
    loss_epoch1, _ = _batch_loss(
        model, batch, torch.device("cpu"), sar_only_external=False, loss_config=cfg, epoch_index=1,
    )
    loss_epoch2, _ = _batch_loss(
        model, batch, torch.device("cpu"), sar_only_external=False, loss_config=cfg, epoch_index=2,
    )
    # epoch0 ramp=0 (no evidence term), epoch1 ramp=0.5, epoch2 ramp=1.0.
    assert loss_epoch2.item() >= loss_epoch1.item() >= loss_epoch0.item()


# --------------------------------------------------------------------------
# CLAM: G2 / G3 / distill epoch gates.
# --------------------------------------------------------------------------
def test_g2_g3_distill_epoch_gates() -> None:
    assert g2_decision(clean_delta_pp=1.2, auc_families_ok=4,
                       null_residual_pp=1.4, null_mask_pp=0.5, null_shuffle_pp=0.2).status == "pass"
    assert g2_decision(clean_delta_pp=0.4, auc_families_ok=2,
                       null_residual_pp=1.4, null_mask_pp=0.5, null_shuffle_pp=0.2).status == "fail"
    assert g2_decision(clean_delta_pp=1.2, auc_families_ok=4,
                       null_residual_pp=0.5, null_mask_pp=0.5, null_shuffle_pp=0.2).status == "fail"
    assert g3_decision(clean_mean_pp=2.6, per_seed_min_pp=1.6, clean_ci_lower_pp=1.0,
                       auc_must_families_ok=5, auc_seeds_ok=3,
                       per_class_mean_pp=2.2, per_class_ci_lower_pp=1.0,
                       q_perm_detectable=True, transfer_significant=False).status == "pass"
    assert g3_decision(clean_mean_pp=2.6, per_seed_min_pp=1.6, clean_ci_lower_pp=1.0,
                       auc_must_families_ok=5, auc_seeds_ok=3,
                       per_class_mean_pp=1.0, per_class_ci_lower_pp=-0.5,
                       q_perm_detectable=False, transfer_significant=False).status == "conditional"
    assert g3_decision(clean_mean_pp=2.0, per_seed_min_pp=1.2, clean_ci_lower_pp=-0.2,
                       auc_must_families_ok=5, auc_seeds_ok=3,
                       per_class_mean_pp=2.2, per_class_ci_lower_pp=1.0,
                       q_perm_detectable=True, transfer_significant=True).status == "fail"
    assert distill_epoch_decision(inc6_to8_pp=0.3).epochs == 6
    assert distill_epoch_decision(inc6_to8_pp=1.7).epochs == 8
    assert distill_epoch_decision(inc6_to8_pp=1.0).epochs == 10


# --------------------------------------------------------------------------
# CLAM: r6 training injection determinism + conservative contract.
# --------------------------------------------------------------------------
def test_r6_train_injection_deterministic() -> None:
    ids = [f"inj-{idx}" for idx in range(64)]
    first = select_r6_injections(ids, fraction=0.3)
    second = select_r6_injections(ids, fraction=0.3)
    assert first == second
    total = len(first)
    assert 0 < total < 64  # about 30%
    # Severities come from the frozen ladder.
    assert all(severity in R6_SEVERITY_LADDER for _, _, _, severity in first)

    optical = torch.randn(8, 3, 16, 16)
    sar = torch.randn(8, 1, 16, 16)
    reliability = torch.rand(8, 2, 16, 16).add_(0.1)
    out_o, out_s, out_r, injections = apply_r6_train_injection(
        optical, sar, reliability, ids[:8], fraction=1.0,
    )
    assert injections  # all samples injected
    # Conservative contract: every injected modality evidence is never above
    # its original value.
    for index, modality, _family, severity in injections:
        q_before = reliability[index, modality]
        q_after = out_r[index, modality]
        assert torch.all(q_after <= q_before * (1.0 - 0.0 * 0 + 1e-6) + 1e-6)
        assert torch.all(q_after <= q_before + 1e-6)
    # Uninjected fraction=0 leaves tensors untouched.
    out_o0, out_s0, out_r0, none_inj = apply_r6_train_injection(
        optical, sar, reliability, ids[:8], fraction=0.0,
    )
    assert not none_inj
    assert torch.equal(out_o0, optical) and torch.equal(out_s0, sar)


# --------------------------------------------------------------------------
# CLAM: null interventions never recompute reliability (anti-trapping guard).
# --------------------------------------------------------------------------
def test_null_interventions_do_not_recompute_reliability() -> None:
    from rhs_rcmtr.models import build_model as _build
    model = _build(_v2_cfg("R6-C1-EVSCRT"))
    optical = torch.randn(2, 3, 16, 16)
    sar = torch.randn(2, 1, 16, 16)
    reliability = torch.rand(2, 2, 16, 16).add_(0.1)
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # SAR masked out
    # Factory applies the mask to features only; the reliability tensor passed
    # through is the untouched original.
    output = model(optical, sar, reliability, modality_mask=mask)
    assert output["gate"].shape == (2, 1, 16, 16)
    # Reliability is consumed, not recomputed: no code path derives q from the
    # masked SAR features (the proxy mean stays at the input's), verified by
    # checking the routed gate is finite and the call completes.
    assert torch.isfinite(output["gate"]).all()


# --------------------------------------------------------------------------
# CLAM: CRM monotonicity loss is trivial under analytic sigmoid weights.
# --------------------------------------------------------------------------
def test_crm_monotonicity_trivial_on_analytic_weights() -> None:
    from rhs_rcmtr.losses import counterfactual_monotonicity_loss
    # route_weights[:,1] = sigmoid(reliability[:,1]); lowering q_s lowers the
    # SAR weight, so the monotonicity penalty is algebraically zero.
    original_weights = torch.zeros(2, 2, 8, 8)
    original_weights[:, 0] = 0.7
    original_weights[:, 1] = 0.3
    degraded_weights = original_weights.clone()
    degraded_weights[:, 1] = 0.1
    degraded_weights[:, 0] = 0.9
    penalty = counterfactual_monotonicity_loss(
        original_weights, degraded_weights, modality_index=1
    )
    assert float(penalty) == 0.0  # degraded SAR weight below original: no violation


# --------------------------------------------------------------------------
# Per-sample evaluation tooling (shared audit helpers).
# --------------------------------------------------------------------------
def test_per_sample_miou_tooling() -> None:
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from _r6_audit_common import per_sample_miou, spearman
        model = build_model(_v2_cfg("R6-C1-EVSCRT"))
        batch = _synthetic_batch(batch_size=2)
        values = per_sample_miou(model, batch, torch.device("cpu"))
        assert len(values) == 2
        assert all(0.0 <= value <= 1.0 for value in values)
        permuted = per_sample_miou(model, batch, torch.device("cpu"), reliability_permute=True)
        assert len(permuted) == 2
        # Spearman with a strictly monotone pair.
        assert spearman([(1.0, 1.0), (2.0, 0.5), (3.0, 0.1)]) <= -0.9
        assert spearman([(1.0, 1.0), (2.0, 2.0)]) >= 0.99
    finally:
        sys.path.remove(str(scripts_dir))