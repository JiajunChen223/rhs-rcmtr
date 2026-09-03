"""Loss-graph signatures used to enforce single-mechanism attribution."""

from __future__ import annotations

from typing import Any


LOSS_GRAPH_FIELDS = (
    "segmentation_weight",
    "honesty_weight",
    "counterfactual_weight",
    "auxiliary_sar_weight",
    "modality_dropout_policy",
    "degradation_sampler_version",
    "evidence_response_weight",
)


def loss_graph_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Return only objective/augmentation fields that affect the loss graph."""

    loss = config.get("loss") if isinstance(config.get("loss"), dict) else {}
    sampler = config.get("degradation_sampler_version")
    if sampler is None:
        sampler = loss.get("degradation_sampler_version", "none")
    return {
        "segmentation_weight": float(loss.get("segmentation_weight", 1.0)),
        "honesty_weight": float(loss.get("honesty_weight", 0.0)),
        "counterfactual_weight": float(loss.get("counterfactual_weight", 0.0)),
        "auxiliary_sar_weight": float(loss.get("auxiliary_sar_weight", 0.0)),
        "modality_dropout_policy": str(config.get("modality_dropout_policy", "none")),
        "degradation_sampler_version": str(sampler),
        "evidence_response_weight": float(loss.get("evidence_response_weight", 0.0)),
    }


def loss_graph_signature(config: dict[str, Any]) -> str:
    """Return a stable human-readable loss-graph signature."""

    explicit = str(config.get("loss_graph_signature", "")).strip()
    if explicit:
        return explicit
    payload = loss_graph_payload(config)
    parts = ["segmentation"]
    if payload["honesty_weight"] != 0.0:
        parts.append("honesty")
    if payload["counterfactual_weight"] != 0.0:
        sampler = payload["degradation_sampler_version"]
        parts.append("crm_v2" if sampler == "crm_v2" else "crm")
    if payload["auxiliary_sar_weight"] != 0.0:
        parts.append("sar_aux")
    if payload["modality_dropout_policy"] != "none":
        parts.append("modality_dropout")
    if payload["evidence_response_weight"] != 0.0:
        parts.append("evcr")
    return "+".join(parts)


def loss_graph_differences(parent: dict[str, Any], child: dict[str, Any]) -> list[str]:
    """Return objective fields that differ between two configuration rows."""

    left = loss_graph_payload(parent)
    right = loss_graph_payload(child)
    return [field for field in LOSS_GRAPH_FIELDS if left[field] != right[field]]


def assert_single_loss_graph_delta(
    parent: dict[str, Any], child: dict[str, Any], allowed_fields: set[str]
) -> None:
    """Fail closed unless loss-graph changes are exactly the declared fields."""

    differences = set(loss_graph_differences(parent, child))
    if differences != set(allowed_fields):
        raise ValueError(
            "loss graph changed outside declared mechanism delta: "
            f"observed={sorted(differences)}, allowed={sorted(allowed_fields)}"
        )
