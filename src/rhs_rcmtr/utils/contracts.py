"""Configuration parity and immutable budget contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from rhs_rcmtr.utils.loss_graph import assert_single_loss_graph_delta, loss_graph_signature


COMMON_FIELDS = (
    "pretrained_initialization", "data_split", "preprocessing", "augmentation",
    "sampler", "micro_batch", "effective_batch", "optimizer", "learning_rate",
    "scheduler", "evaluator", "nms", "seed",
    "epochs", "gradient_accumulation_steps", "weight_decay", "num_workers",
    "prefetch_factor", "persistent_workers", "scheduler_horizon_epochs",
    "common_init_seed", "innovation_init_seed", "data_order_seed", "augmentation_seed",
    "sar_encoder_variant", "sar_grad_scale",
)
OPTIONAL_COMMON_FIELDS = frozenset({"sar_encoder_variant", "sar_grad_scale"})
INNOVATION_PREFIXES = ("router.", "innovation.")


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
    {"claim_id": "R7-A1", "claim": "S2FT consumes exactly one raw SAR band",
     "impl_anchor": "rhs_rcmtr.models.s2ft.patch_transplant:SarFoundationPatchEmbed.forward", "unit_test": "test_s2ft_single_band_patch_transplant",
     "audit_field": "r7_audit.sar_input_bands"},
    {"claim_id": "R7-A2", "claim": "SAR patch filters are initialized by energy-preserving collapse of DINO RGB filters",
     "impl_anchor": "rhs_rcmtr.models.s2ft.patch_transplant:SarFoundationPatchEmbed.reset_from_source", "unit_test": "test_s2ft_patch_transplant_energy_preserving",
     "audit_field": "r7_audit.patch_transplant"},
    {"claim_id": "R7-A3", "claim": "The DINO transformer is shared and frozen while gradients pass through it to trainable adapters",
     "impl_anchor": "rhs_rcmtr.models.s2ft.shared_foundation:SharedDinoSensorBackbone", "unit_test": "test_s2ft_frozen_foundation_allows_adapter_gradient",
     "audit_field": "r7_audit.foundation"},
    {"claim_id": "R7-A4", "claim": "Sensor adapters have exact identity initialization",
     "impl_anchor": "rhs_rcmtr.models.s2ft.sensor_adapter:SensorAdapter", "unit_test": "test_s2ft_adapter_identity_init",
     "audit_field": "r7_audit.adapter_identity"},
    {"claim_id": "R7-B1", "claim": "GLCI is bidirectional, geo-local, and uses bounded deformable offsets",
     "impl_anchor": "rhs_rcmtr.mechanisms.geo_local_interaction:GeoLocalDeformableCrossInteraction", "unit_test": "test_s2ft_glci_identity_and_bounded_offsets",
     "audit_field": "r7_audit.glci"},
    {"claim_id": "R7-B2", "claim": "GLCI starts as an exact identity through zero residual scales",
     "impl_anchor": "rhs_rcmtr.mechanisms.geo_local_interaction:GeoLocalDeformableCrossInteraction.forward", "unit_test": "test_s2ft_glci_identity_and_bounded_offsets",
     "audit_field": "r7_audit.glci_identity"},
    {"claim_id": "R7-C1", "claim": "SAR structural refinement is semantic-conditioned and identity initialized",
     "impl_anchor": "rhs_rcmtr.mechanisms.structural_refinement:SarStructuralRefinement", "unit_test": "test_s2ft_sdr_identity_init",
     "audit_field": "r7_audit.sdr"},
    {"claim_id": "R7-P1", "claim": "R7 logits are independent of legacy reliability evidence",
     "impl_anchor": "rhs_rcmtr.models.s2ft.model:S2FTSegmenter.forward", "unit_test": "test_s2ft_reliability_is_not_a_model_input",
     "audit_field": "r7_audit.reliability_independence"},
    {"claim_id": "R7-P2", "claim": "Explicit modality masking prevents foundation-token leakage from an absent modality",
     "impl_anchor": "rhs_rcmtr.models.s2ft.shared_foundation:SharedDinoSensorBackbone.forward", "unit_test": "test_s2ft_modality_mask_has_no_feature_leakage",
     "audit_field": "r7_audit.modality_mask"},
    {"claim_id": "R7-P3", "claim": "Innovation reset never resets the shared frozen foundation",
     "impl_anchor": "rhs_rcmtr.utils.initialization_anchor:reset_innovation_parameters", "unit_test": "test_s2ft_innovation_reset_preserves_foundation",
     "audit_field": "r7_audit.innovation_reset"},
)

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
    """Emit stable trainability evidence split into common and innovation parameters."""
    entries = [{"name": name, "requires_grad": parameter.requires_grad, "numel": parameter.numel()}
               for name, parameter in model.named_parameters()]
    common = [item for item in entries if not item["name"].startswith(INNOVATION_PREFIXES)]
    innovation = [item for item in entries if item["name"].startswith(INNOVATION_PREFIXES)]
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    innovation_trainable = [item["name"] for item in innovation if item["requires_grad"]]
    return {
        "status": "pass",
        "all_parameter_mask_sha256": hashlib.sha256(payload).hexdigest(),
        "common_trainable_names": [item["name"] for item in common if item["requires_grad"]],
        "common_frozen_names": [item["name"] for item in common if not item["requires_grad"]],
        "mechanism_trainable_names": innovation_trainable,
        "innovation_trainable_names": innovation_trainable,
        "trainable_numel": sum(item["numel"] for item in entries if item["requires_grad"]),
        "innovation_trainable_numel": sum(item["numel"] for item in innovation if item["requires_grad"]),
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
    defaults = {"sar_encoder_variant": "v1", "sar_grad_scale": 1.0}
    payload = {key: config.get(key, defaults.get(key)) for key in COMMON_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assert_loss_graph_delta(
    parent: dict[str, Any], child: dict[str, Any], allowed_fields: set[str]
) -> None:
    assert_single_loss_graph_delta(parent, child, allowed_fields)


def loss_graph_evidence(config: dict[str, Any]) -> dict[str, Any]:
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
