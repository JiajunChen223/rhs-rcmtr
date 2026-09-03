"""Configuration parity and immutable budget contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from rhs_rcmtr.utils.loss_graph import assert_single_loss_graph_delta, loss_graph_signature


# R6 SAR-DE (handoff v2): public training-object fields shared by both arms.
# These enter parity/comparisons but are NOT required keys of legacy rows: the
# historical frozen configs (baseline.yaml etc.) predate them, and requiring
# their presence would break re-runs of the frozen R5 chain. Missing keys on a
# legacy row behave as their semantic defaults (v1 / 1.0).
COMMON_FIELDS = (
    "pretrained_initialization", "data_split", "preprocessing", "augmentation",
    "sampler", "micro_batch", "effective_batch", "optimizer", "learning_rate",
    "scheduler", "evaluator", "nms", "seed",
    "epochs", "gradient_accumulation_steps", "weight_decay", "num_workers",
    "prefetch_factor", "persistent_workers", "scheduler_horizon_epochs",
    "common_init_seed", "innovation_init_seed", "data_order_seed", "augmentation_seed",
    "sar_encoder_variant", "sar_grad_scale",
)
# New keys that legacy (frozen) rows are allowed to omit; they default to the
# R5 semantic no-op values. Only enforced as \"present\" for new rows.
OPTIONAL_COMMON_FIELDS = frozenset({"sar_encoder_variant", "sar_grad_scale"})

# R6 SAR-DE claim-alignment registry (CLAM). Each mechanism-level claim of the
# frozen spec (plan_revision_20260902/innovation_route_handoff_v2_SAR_DE_v1.0.md)
# must resolve to a code anchor, a unit test, and an audit output field. The
# `test_claim_alignment.py` CI guard iterates this table; a dangling entry fails
# the pipeline. This is the enforcement mechanism against claim-vs-impl gaps.
CLAIM_ALIGNMENT_TABLE: tuple[dict[str, str], ...] = (
    {"claim_id": "A1-1", "claim": "SarEncoderV2 accepts exactly one SAR band",
     "impl_anchor": "rhs_rcmtr.models.backbones:SarEncoderV2.forward", "unit_test": "test_sar_encoder_v2_input_validation",
     "audit_field": "sar_encoder.input_bands"},
    {"claim_id": "A1-2", "claim": "SarEncoderV2 budget below 1.5M trainable params and 64 output channels",
     "impl_anchor": "rhs_rcmtr.models.backbones:SarEncoderV2", "unit_test": "test_sar_encoder_v2_grid_and_budget",
     "audit_field": "parameter_audit.trainable_numel"},
    {"claim_id": "A2-1", "claim": "Distillation teacher path is fully detached (no gradient)",
     "impl_anchor": "rhs_rcmtr.engine.entrypoint:_batch_loss", "unit_test": "test_distill_loss_components_teacher_detached",
     "audit_field": "run_record.loss"},
    {"claim_id": "A2-2", "claim": "Distillation student is SAR encoder + shared decoder (forward_sar_only)",
     "impl_anchor": "rhs_rcmtr.engine.entrypoint:_batch_loss", "unit_test": "test_distill_anchor_isomorphy",
     "audit_field": "run_record.parameter_audit"},
    {"claim_id": "A3-1", "claim": "Gradient balance scales only sar_encoder.* gradients",
     "impl_anchor": "rhs_rcmtr.engine.entrypoint:execute", "unit_test": "test_gradient_balance_hook_scope",
     "audit_field": "run_record.sar_grad_scale"},
    {"claim_id": "B1-1", "claim": "EVSCRT residual_scale initializes to 0 (true optical identity start)",
     "impl_anchor": "rhs_rcmtr.mechanisms.reliability_router:EvScrtRouter", "unit_test": "test_evscrt_residual_scale_zero_init",
     "audit_field": "parameter_audit"},
    {"claim_id": "B1-2", "claim": "EVSCRT init forward equals the optical anchor (residual == 0)",
     "impl_anchor": "rhs_rcmtr.mechanisms.reliability_router:EvScrtRouter.forward", "unit_test": "test_evscrt_residual_scale_zero_init",
     "audit_field": "mechanism_diagnostics"},
    {"claim_id": "B1-3", "claim": "EVSCRT has no syndrome terminology or proxy field",
     "impl_anchor": "rhs_rcmtr.mechanisms.reliability_router:EvScrtRouter", "unit_test": "test_evscrt_residual_scale_zero_init",
     "audit_field": "mechanism_diagnostics"},
    {"claim_id": "B2-1", "claim": "L_ev encodes the soft box (1-u_o) <= gate <= q_s with conflict masking",
     "impl_anchor": "rhs_rcmtr.losses.objectives:evidence_response_gate_loss", "unit_test": "test_evidence_response_gate_loss_bounds",
     "audit_field": "run_record.evidence_up/evidence_dn/violation_fraction"},
    {"claim_id": "B2-2", "claim": "The gate tensor is exposed to the training loop (differentiable)",
     "impl_anchor": "rhs_rcmtr.models.factory:MultimodalSegmenter.forward", "unit_test": "test_loss_graph_evcr_single_delta",
     "audit_field": "output.gate"},
    {"claim_id": "G1-1", "claim": "G1 gate logic is a pure function with the frozen thresholds (20/16/15/18)",
     "impl_anchor": "rhs_rcmtr.utils.r6_gates:g1_decision", "unit_test": "test_g1_gate_logic",
     "audit_field": "g1_decision.status"},
    {"claim_id": "Q-1", "claim": "q-permutation detectability metric is defined on end-to-end mIoU delta",
     "impl_anchor": "rhs_rcmtr.utils.r6_gates:q_permutation_decision", "unit_test": "test_q_permutation_detectability_metric",
     "audit_field": "q_permutation.observed"},
    {"claim_id": "G2-1", "claim": "G2 rapid gate is a pure function (dual channel + calibrated nulls)",
     "impl_anchor": "rhs_rcmtr.utils.r6_gates:g2_decision", "unit_test": "test_g2_g3_distill_epoch_gates",
     "audit_field": "g2_decision.status"},
    {"claim_id": "G3-1", "claim": "G3 formal gate is hierarchical (G3a/G3b mandatory, G3c 2-of-3)",
     "impl_anchor": "rhs_rcmtr.utils.r6_gates:g3_decision", "unit_test": "test_g2_g3_distill_epoch_gates",
     "audit_field": "g3_decision.status"},
    {"claim_id": "A2-3", "claim": "Distillation epoch rule frozen (6/8 increments)",
     "impl_anchor": "rhs_rcmtr.utils.r6_gates:distill_epoch_decision", "unit_test": "test_g2_g3_distill_epoch_gates",
     "audit_field": "distill_epoch_decision.epochs"},
    {"claim_id": "B2-3", "claim": "Null interventions (mask/shuffle) never recompute reliability evidence",
     "impl_anchor": "rhs_rcmtr.models.factory:MultimodalSegmenter.forward", "unit_test": "test_null_interventions_do_not_recompute_reliability",
     "audit_field": "strict_null_audit"},
    {"claim_id": "B2-4", "claim": "r6 training-time degradation injection is deterministic and conservative",
     "impl_anchor": "rhs_rcmtr.utils.interventions:apply_r6_train_injection", "unit_test": "test_r6_train_injection_deterministic",
     "audit_field": "run_record.r6_injection_stats"},
    {"claim_id": "B1-4", "claim": "CRM monotonicity loss is trivial under analytic sigmoid weights (never cited as evidence)",
     "impl_anchor": "rhs_rcmtr.mechanisms.reliability_router:EvScrtRouter.forward", "unit_test": "test_crm_monotonicity_trivial_on_analytic_weights",
     "audit_field": "mechanism_diagnostics"},
)

# These fields are not part of the fixed optimization/split budget, but they
# change the training object and therefore must be explicitly audited when a
# candidate is compared with its declared parent.  Legacy rows use the
# defaults below so their historical records remain readable.
EXTENDED_PARITY_FIELDS = (
    "modality_dropout_policy",
    "auxiliary_head_presence",
    "feature_normalization_signature",
    "decoder_adapter_signature",
    "evidence_response_weight",
)
EXTENDED_PARITY_DEFAULTS = {
    "modality_dropout_policy": "none",
    "auxiliary_head_presence": "none",
    "feature_normalization_signature": "none",
    "decoder_adapter_signature": "none",
    "evidence_response_weight": 0.0,
}


def trainable_parameter_audit(model: Any) -> dict[str, Any]:
    """Emit stable trainability evidence split into common and mechanism parameters."""
    entries = [{"name": name, "requires_grad": parameter.requires_grad, "numel": parameter.numel()}
               for name, parameter in model.named_parameters()]
    common = [item for item in entries if not item["name"].startswith("router.")]
    mechanism = [item for item in entries if item["name"].startswith("router.")]
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "status": "pass",
        "all_parameter_mask_sha256": hashlib.sha256(payload).hexdigest(),
        "common_trainable_names": [item["name"] for item in common if item["requires_grad"]],
        "common_frozen_names": [item["name"] for item in common if not item["requires_grad"]],
        "mechanism_trainable_names": [item["name"] for item in mechanism if item["requires_grad"]],
        "trainable_numel": sum(item["numel"] for item in entries if item["requires_grad"]),
        "total_numel": sum(item["numel"] for item in entries),
    }


def assert_training_object_parity(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    differences = [key for key in COMMON_FIELDS if baseline.get(key) != candidate.get(key)]
    if differences:
        raise ValueError(f"common training-object fields changed: {differences}")
    enabled = candidate.get("enabled_mechanism_ids", [])
    if not isinstance(enabled, list) or len(enabled) != 1 or not str(enabled[0]).strip():
        raise ValueError("candidate delta must be exactly one declared internal mechanism")
    if candidate.get("external_trainable_component", False):
        raise ValueError("external trainable components are forbidden")


def extended_training_object_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Return explicit architecture/augmentation fields for strict audits."""

    loss = config.get("loss") if isinstance(config.get("loss"), dict) else {}
    payload: dict[str, Any] = {}
    for field, default in EXTENDED_PARITY_DEFAULTS.items():
        value = config.get(field)
        if value is None and field == "modality_dropout_policy":
            value = loss.get(field)
        payload[field] = default if value is None else value
    return payload


def assert_extended_training_object_parity(
    parent: dict[str, Any], child: dict[str, Any], allowed_fields: set[str] | None = None
) -> None:
    """Fail closed when compatibility/augmentation fields change unexpectedly."""

    allowed = set(allowed_fields or set())
    left = extended_training_object_payload(parent)
    right = extended_training_object_payload(child)
    differences = {field for field in EXTENDED_PARITY_FIELDS if left[field] != right[field]}
    if differences != allowed:
        raise ValueError(
            "extended training-object fields changed outside declared delta: "
            f"observed={sorted(differences)}, allowed={sorted(allowed)}"
        )


def assert_composition_row(baseline: dict[str, Any], composition: dict[str, Any]) -> None:
    """Require an attributable parent and only the declared internal mechanism delta."""
    assert_training_object_parity(baseline, composition)
    if composition.get("parent_row") != "baseline":
        raise ValueError("composition parent_row must be baseline")
    allowed_differences = {"enabled_mechanism_ids", "parent_row"}
    keys = set(baseline) | set(composition)
    undeclared = sorted(
        key for key in keys if key not in allowed_differences and baseline.get(key) != composition.get(key)
    )
    if undeclared:
        raise ValueError(f"undeclared composition differences: {undeclared}")


def assert_training_row_declared(row: dict[str, Any]) -> None:
    """Fail closed when a run row omits or changes protected training fields."""
    if row.get("external_trainable_component", False):
        raise ValueError("external trainable components are forbidden")
    enabled = row.get("enabled_mechanism_ids", [])
    if enabled != [] and (not isinstance(enabled, list) or len(enabled) != 1 or not str(enabled[0]).strip()):
        raise ValueError("training row must declare baseline or exactly one internal mechanism")
    missing = [field for field in COMMON_FIELDS if field not in row and field not in OPTIONAL_COMMON_FIELDS]
    if missing:
        raise ValueError(f"training row omits protected fields: {missing}")
    if row.get("parent_row") is not None and row.get("parent_row") != "baseline":
        raise ValueError("composition parent_row must be baseline")


def matched_protocol_budget_hash(config: dict[str, Any]) -> str:
    """Hash the frozen budget fields with semantic defaults for missing keys.

    Optional new keys default to their no-op values (v1 / 1.0) so that legacy
    rows hash consistently with their recorded values; note that hashes computed
    by pre-R6 code (before these keys existed) are not recomputable and are
    treated as frozen historical records.
    """
    defaults = {"sar_encoder_variant": "v1", "sar_grad_scale": 1.0}
    payload = {
        key: config.get(key, defaults.get(key)) for key in COMMON_FIELDS
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assert_loss_graph_delta(
    parent: dict[str, Any], child: dict[str, Any], allowed_fields: set[str]
) -> None:
    """Validate that objective/intervention changes are exactly declared."""

    assert_single_loss_graph_delta(parent, child, allowed_fields)


def loss_graph_evidence(config: dict[str, Any]) -> dict[str, Any]:
    """Return a stable loss-graph record for run manifests and parity audits."""

    payload = extended_training_object_payload(config)
    return {
        "signature": loss_graph_signature(config),
        "declared_loss": config.get("loss", {}),
        "modality_dropout_policy": payload["modality_dropout_policy"],
        "degradation_sampler_version": (
            config.get("degradation_sampler_version")
            or (config.get("loss", {}) or {}).get("degradation_sampler_version", "none")
        ),
        "auxiliary_head_presence": payload["auxiliary_head_presence"],
        "feature_normalization_signature": payload["feature_normalization_signature"],
        "decoder_adapter_signature": payload["decoder_adapter_signature"],
    }
