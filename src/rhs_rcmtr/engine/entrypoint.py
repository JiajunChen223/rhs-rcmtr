"""Unified baseline/candidate/composition execution and run tracking."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from rhs_rcmtr.data import CloudPairedRasterDataset, CloudSarOnlyRasterDataset, load_data_manifest, load_sar_external_manifest
from rhs_rcmtr.engine.runner import run_synthetic_step
from rhs_rcmtr.losses import (
    counterfactual_monotonicity_loss,
    evidence_response_gate_loss,
    segmentation_with_honesty_loss,
)
from rhs_rcmtr.metrics import mean_iou
from rhs_rcmtr.models import ModelConfig, build_model
from rhs_rcmtr.utils.config import resolve_config
from rhs_rcmtr.utils.contracts import (
    assert_composition_row, assert_training_object_parity, assert_training_row_declared,
    matched_protocol_budget_hash, trainable_parameter_audit,
)
from rhs_rcmtr.utils.checkpoint import load_training_checkpoint, save_training_checkpoint
from rhs_rcmtr.utils.run_manifest import load_verified_run_manifest
from rhs_rcmtr.utils.test_seal import assert_test_access_allowed
from rhs_rcmtr.utils.initialization import audited_checkpoint_hashes, verify_pretrained_audit
from rhs_rcmtr.utils.data_order import DeterministicEpochSampler, seed_worker
from rhs_rcmtr.utils.initialization_anchor import (
    load_common_init_anchor,
    reset_innovation_parameters,
)
from rhs_rcmtr.utils.contracts import loss_graph_evidence
from rhs_rcmtr.utils.interventions import (
    apply_modality_dropout,
    apply_r6_train_injection,
    degrade_pair,
    modality_dropout_assignment,
    recompute_reliability_evidence,
    select_crm_intervention,
)


def model_config(resolved: dict[str, Any], checkpoint_hashes: dict[str, str] | None = None) -> ModelConfig:
    experiment = resolved["experiment"]
    runtime = resolved["runtime"]
    initialization = resolved["initialization"]["initialization"]
    hashes = checkpoint_hashes or {}
    sar_mode = str(initialization.get("sar_initialization_mode", "random_init_single_band"))
    backbone_mode = str(runtime["backbone_mode"])
    return ModelConfig(
        sar_channels=2 if sar_mode == "pretrained_croma_sar" else 1,
        enabled_mechanism_ids=tuple(experiment.get("enabled_mechanism_ids", [])),
        backbone_mode=backbone_mode,
        optical_repo_path=str(initialization.get("optical_repo_path", "")),
        optical_checkpoint_path=str(initialization.get("optical_checkpoint_path", "")),
        optical_checkpoint_sha256=hashes.get(str(initialization.get("optical_checkpoint_path", "")), ""),
        sar_repo_path=str(initialization.get("sar_repo_path", "")),
        sar_checkpoint_path=str(initialization.get("sar_checkpoint_path", "")),
        sar_checkpoint_sha256=hashes.get(str(initialization.get("sar_checkpoint_path", "")), ""),
        sar_initialization_mode=sar_mode,
        unimodal_mode=str(experiment.get("unimodal_mode", "multimodal")),
        auxiliary_sar_head=bool(experiment.get("auxiliary_sar_head", False)),
        sar_encoder_variant=str(experiment.get("sar_encoder_variant", "v1")),
    )


def _batch_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    sar_only_external: bool,
    loss_config: dict[str, Any],
    crm_stats: dict[str, Any] | None = None,
    distill_pretrain: bool = False,
    evidence_stats: dict[str, float] | None = None,
    injection_stats: dict[str, Any] | None = None,
    epoch_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    optical = batch["optical"].to(device)
    sar = batch["sar"].to(device)
    if distill_pretrain:
        # R6 SAR-DE Phase A2: distillation pre-training short-circuit. Student =
        # SAR encoder + shared decoder (forward_sar_only); teacher = optical
        # branch fully detached (no_grad), so the teacher path never receives
        # gradient and the parameter mask (thus anchor isomorphism) is untouched.
        import torch.nn.functional as _F
        label = batch["label"].to(device)
        with torch.no_grad():
            teacher_feats = model.optical_projection(model.optical_encoder(optical))
            teacher_logits = model.decoder(teacher_feats)
            if teacher_logits.shape[2:] != label.shape[1:]:
                teacher_logits = _F.interpolate(
                    teacher_logits, size=tuple(label.shape[1:]), mode="bilinear", align_corners=False
                )
        student_output = model.forward_sar_only(sar)
        student_logits = student_output["logits"]
        seg_distill = _F.cross_entropy(student_logits, label, ignore_index=255)
        temperature = float(loss_config.get("distill_temperature", 4.0))
        kl_weight = float(loss_config.get("distill_kl_weight", 1.0))
        valid = (label != 255).to(student_logits.dtype)
        student_logp = _F.log_softmax(student_logits / temperature, dim=1)
        teacher_prob = _F.softmax(teacher_logits.detach() / temperature, dim=1)
        per_pixel_kl = (teacher_prob * (teacher_prob.log() - student_logp)).sum(dim=1)
        # Hard top-5% filtering with a 0.5 floor: drop tokens most disagreed with
        # the teacher instead of continuous down-weighting (protect hard classes).
        grad = per_pixel_kl.detach()
        if valid.sum() > 0:
            # Hard top-5% filtering with a 0.5 floor: drop tokens most disagreed
            # with the teacher instead of continuous down-weighting.
            drop_fraction = float(loss_config.get("distill_hard_filter_fraction", 0.05))
            floor = float(loss_config.get("distill_floor_weight", 0.5))
            cutoff = torch.quantile(grad[valid > 0].clamp_min(0), 1.0 - drop_fraction)
            keep_mask = (grad <= cutoff) | (valid == 0)
            token_weight = torch.where(
                keep_mask, torch.ones_like(grad), torch.full_like(grad, floor)
            )
            token_weight = token_weight * valid
            kl_denom = valid.sum()
            kl = (per_pixel_kl * token_weight).sum() / kl_denom
        else:
            kl = per_pixel_kl.sum() * 0.0  # all-ignore batch: no distillation signal
        # Standard temperature-squared scaling of the KL term.
        loss = seg_distill + kl_weight * (temperature ** 2) * kl
        score = mean_iou(student_logits.detach(), label, 8)
        return loss, score
    reliability_input = batch.get("reliability")
    if reliability_input is not None:
        reliability_input = reliability_input.to(device)
    if not sar_only_external and model.training and str(
        loss_config.get("degradation_sampler_version", "none")
    ) == "r6_train_inject_v1":
        fraction = float(loss_config.get("degradation_inject_fraction", 0.3))
        optical, sar, reliability_input, injections = apply_r6_train_injection(
            optical, sar, reliability_input, batch.get("sample_id", []), fraction=fraction,
        )
        if injection_stats is not None and injections:
            from collections import Counter
            per_family = Counter(
                (("optical" if modality == 0 else "sar"), family)
                for _index, modality, family, _severity in injections
            )
            injection_stats["batches"] = int(injection_stats.get("batches", 0)) + 1
            injection_stats["injected_samples"] = int(
                injection_stats.get("injected_samples", 0) + len(injections)
            )
            injection_stats["total_samples"] = int(
                injection_stats.get("total_samples", 0) + int(optical.shape[0])
            )
            for key, count in per_family.items():
                flat = f"{key[0]}:{key[1]}"
                injection_stats[flat] = int(injection_stats.get(flat, 0)) + count
    if not sar_only_external and model.training:
        optical, sar, reliability_input = apply_modality_dropout(
            optical,
            sar,
            reliability_input,
            batch.get("sample_id", []),
            str(loss_config.get("modality_dropout_policy", "none")),
        )
    if sar_only_external:
        output = model.forward_sar_only(sar)
    else:
        output = model(
            optical,
            sar,
            reliability_input if reliability_input is not None else torch.ones(
                optical.shape[0], 2, optical.shape[2], optical.shape[3], device=device,
                dtype=optical.dtype,
            ),
        )
    reliability = reliability_input
    if reliability is None:
        reliability = torch.ones(
            output["route_weights"].shape[0],
            output["route_weights"].shape[1],
            output["route_weights"].shape[2],
            output["route_weights"].shape[3],
            device=device,
            dtype=output["route_weights"].dtype,
        )
    else:
        reliability = reliability.to(device)
        if reliability.shape[2:] != output["route_weights"].shape[2:]:
            reliability = torch.nn.functional.interpolate(
                reliability,
                size=output["route_weights"].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
    segmentation_weight = float(loss_config.get("segmentation_weight", 1.0))
    honesty_weight = float(loss_config.get("honesty_weight", 0.0))
    counterfactual_weight = float(loss_config.get("counterfactual_weight", 0.0))
    auxiliary_sar_weight = float(loss_config.get("auxiliary_sar_weight", 0.0))
    loss, _ = segmentation_with_honesty_loss(
        output["logits"],
        batch["label"].to(device),
        output["route_weights"],
        reliability,
        honesty_weight if model.router is not None and not sar_only_external else 0.0,
        segmentation_weight=segmentation_weight,
    )
    if distill_pretrain:
        # Distillation handled by the short-circuit above; never reach here.
        raise AssertionError("distill_pretrain must be short-circuited in _batch_loss")
    if auxiliary_sar_weight != 0.0:
        auxiliary_logits = output.get("sar_auxiliary_logits")
        if auxiliary_logits is None:
            if model.training:
                raise ValueError("auxiliary_sar_weight requires a training-time SAR auxiliary head")
        else:
            auxiliary_loss = torch.nn.functional.cross_entropy(
                auxiliary_logits,
                batch["label"].to(device),
                ignore_index=255,
            )
            loss = loss + auxiliary_sar_weight * auxiliary_loss
    if counterfactual_weight != 0.0:
        if sar_only_external or model.router is None:
            raise ValueError("counterfactual_weight requires a routed multimodal training row")
        # The same (possibly deterministically dropped) tensors used for the
        # primary forward are the parent of the counterfactual intervention.
        sampler_version = str(loss_config.get("degradation_sampler_version", "none"))
        if sampler_version == "crm_v2":
            ids = batch.get("sample_id", [])
            if isinstance(ids, str):
                ids = [ids]
            modality_index, family, severity = select_crm_intervention(ids)
            degraded_optical, degraded_sar = degrade_pair(
                optical, sar, modality_index, family, severity
            )
            degraded_reliability = recompute_reliability_evidence(
                degraded_optical, degraded_sar
            )
            if degraded_reliability.shape[2:] != reliability.shape[2:]:
                degraded_reliability = torch.nn.functional.interpolate(
                    degraded_reliability,
                    size=reliability.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            # Keep the intervention contract conservative: if the differentiable
            # recomputation is numerically tied, impose a small evidence decrease.
            q_before = reliability[:, modality_index : modality_index + 1]
            q_after = degraded_reliability[:, modality_index : modality_index + 1]
            q_after = torch.minimum(q_after, q_before * (1.0 - 0.05 * severity))
            degraded_reliability = degraded_reliability.clone()
            degraded_reliability[:, modality_index : modality_index + 1] = q_after
            degraded_output = model(
                degraded_optical, degraded_sar, degraded_reliability
            )
            valid_mask = (q_after < q_before - 1e-7).squeeze(1)
            crm_penalty = counterfactual_monotonicity_loss(
                output["route_weights"],
                degraded_output["route_weights"],
                modality_index=modality_index,
                valid_mask=valid_mask,
            )
            contribution = float(counterfactual_weight) * crm_penalty
            loss = loss + contribution
            if crm_stats is not None:
                key = f"{modality_index}:{family}"
                bucket = crm_stats.setdefault("by_intervention", {}).setdefault(
                    key,
                    {
                        "modality": "optical" if modality_index == 0 else "sar",
                        "degradation": family,
                        "count": 0,
                        "severity_sum": 0.0,
                        "valid_token_fraction_sum": 0.0,
                        "loss_contribution_sum": 0.0,
                    },
                )
                valid_fraction = float(valid_mask.float().mean().detach())
                bucket["count"] += 1
                bucket["severity_sum"] += float(severity)
                bucket["valid_token_fraction_sum"] += valid_fraction
                bucket["loss_contribution_sum"] += float(contribution.detach())
                crm_stats["batch_count"] = int(crm_stats.get("batch_count", 0)) + 1
                crm_stats["valid_token_fraction_sum"] = float(
                    crm_stats.get("valid_token_fraction_sum", 0.0) + valid_fraction
                )
                crm_stats["loss_contribution_sum"] = float(
                    crm_stats.get("loss_contribution_sum", 0.0) + float(contribution.detach())
                )
                examples = crm_stats.setdefault("examples", [])
                if len(examples) < 32:
                    examples.append(
                        {
                            "modality": "optical" if modality_index == 0 else "sar",
                            "degradation": family,
                            "severity": float(severity),
                            "valid_token_fraction": valid_fraction,
                            "loss_contribution": float(contribution.detach()),
                        }
                    )
        else:
            for modality_index in (0, 1):
                degraded_reliability = reliability.clone()
                degraded_reliability[:, modality_index : modality_index + 1] *= 0.5
                degraded_output = model(
                    optical * (0.5 if modality_index == 0 else 1.0),
                    sar * (0.5 if modality_index == 1 else 1.0),
                    degraded_reliability,
                )
                loss = loss + float(counterfactual_weight) * counterfactual_monotonicity_loss(
                    output["route_weights"],
                    degraded_output["route_weights"],
                    modality_index=modality_index,
                )
    evidence_weight = float(loss_config.get("evidence_response_weight", 0.0))
    if evidence_weight != 0.0:
        if sar_only_external:
            raise ValueError("evidence_response_weight is incompatible with the SAR-only external evaluation")
        if model.training:
            gate = output.get("gate")
            if gate is None:
                raise ValueError("evidence_response_weight requires a routed row exposing a gate (R6-C1-EVSCRT)")
            ramp_epochs = int(loss_config.get("evidence_ramp_epochs", 0))
            ramp = 1.0 if ramp_epochs <= 0 else min(1.0, max(epoch_index, 0) / float(ramp_epochs))
            if ramp > 0.0:
                effective = evidence_weight * ramp
                evidence_loss, evidence_parts = evidence_response_gate_loss(gate, reliability)
                loss = loss + float(effective) * evidence_loss
                if evidence_stats is not None:
                    for key in ("evidence_up", "evidence_dn", "violation_fraction", "conflict_fraction"):
                        evidence_stats[key] = evidence_stats.get(key, 0.0) + float(
                            evidence_parts[key] * ramp if key.startswith("evidence") else evidence_parts[key]
                        )
                    evidence_stats["batches"] = int(evidence_stats.get("batches", 0)) + 1
    return loss, mean_iou(output["logits"], batch["label"].to(device), 8)


def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    sar_only_external: bool,
    loss_config: dict[str, Any],
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals = {"loss": 0.0, "mean_iou": 0.0}
    with torch.no_grad():
        for batch in loader:
            loss, score = _batch_loss(
                model,
                batch,
                device,
                sar_only_external=sar_only_external,
                loss_config=loss_config,
            )
            totals["loss"] += float(loss)
            totals["mean_iou"] += float(score)
    if was_training:
        model.train()
    return {key: value / len(loader) for key, value in totals.items()}


def _write_record(
    output: Path, resolved: dict[str, Any], metrics: dict[str, float], mode: str,
    *, parameter_audit: dict[str, Any] | None = None, evidence: dict[str, Any] | None = None,
) -> None:
    (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    record = {
        "status": "completed", "mode": mode, "seed": resolved["experiment"]["seed"],
        "formal_epochs": resolved["experiment"].get("epochs"),
        "execution_epochs": resolved["experiment"].get("execution_epochs", resolved["experiment"].get("epochs")),
        "scheduler_horizon_epochs": resolved["experiment"].get("scheduler_horizon_epochs", resolved["experiment"].get("epochs")),
        "loss": resolved["experiment"].get("loss", {}),
        "loss_graph": loss_graph_evidence(resolved["experiment"]),
        "common_init_seed": resolved["experiment"].get("common_init_seed"),
        "innovation_init_seed": resolved["experiment"].get("innovation_init_seed"),
        "data_order_seed": resolved["experiment"].get("data_order_seed"),
        "augmentation_seed": resolved["experiment"].get("augmentation_seed"),
        "resolved_config_sha256": resolved["resolved_config_sha256"],
        "matched_protocol_budget_hash": matched_protocol_budget_hash(resolved["experiment"]),
        "enabled_mechanism_ids": resolved["experiment"].get("enabled_mechanism_ids", []),
        "synthetic_only": bool(resolved["runtime"]["synthetic_only"]),
        "parameter_audit": "parameter_audit.json" if parameter_audit else None,
        "evidence": evidence or {},
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    if parameter_audit:
        (output / "parameter_audit.json").write_text(
            json.dumps(parameter_audit, indent=2, sort_keys=True), encoding="utf-8"
        )


def execute(
    *, runtime_config: Path, experiment_config: Path, initialization_config: Path,
    run_id: str, mode: str, run_manifest: Path | None = None, run_manifest_sha256: str = "",
    data_manifest: Path | None = None, data_manifest_sha256: str = "",
    pretrained_audit: Path | None = None, pretrained_audit_sha256: str = "",
    checkpoint: Path | None = None, checkpoint_sha256: str = "",
    code_manifest: Path | None = None, code_manifest_sha256: str = "",
) -> dict[str, Any]:
    resolved = resolve_config(runtime_config, experiment_config, initialization_config)
    runtime = resolved["runtime"]
    experiment = resolved["experiment"]
    loss_config = dict(experiment.get("loss", {}))
    if not isinstance(loss_config, dict):
        raise ValueError("loss must be a configuration object")
    if "modality_dropout_policy" not in loss_config and "modality_dropout_policy" in experiment:
        loss_config["modality_dropout_policy"] = experiment["modality_dropout_policy"]
    for loss_name in (
        "segmentation_weight",
        "honesty_weight",
        "counterfactual_weight",
        "auxiliary_sar_weight",
        "evidence_response_weight",
        "evidence_ramp_epochs",
        "distill_temperature",
        "distill_kl_weight",
        "distill_hard_filter_fraction",
        "distill_floor_weight",
    ):
        try:
            loss_value = float(loss_config.get(loss_name, 0.0 if loss_name != "segmentation_weight" else 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"loss.{loss_name} must be numeric") from exc
        if loss_value < 0:
            raise ValueError(f"loss.{loss_name} must be non-negative")
    if float(loss_config.get("distill_temperature", 4.0)) == 0.0:
        raise ValueError("loss.distill_temperature must be nonzero")
    assert_training_row_declared(experiment)
    baseline_path = experiment_config.parent / "baseline.yaml"
    if baseline_path.is_file() and experiment.get("enabled_mechanism_ids"):
        import yaml
        baseline_row = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
        assert_training_object_parity(baseline_row, experiment)
        # Enforce loss-graph single-delta at runtime (handoff v2 §3): the only
        # permitted objective change for a candidate is its declared mechanism.
        from rhs_rcmtr.utils.loss_graph import assert_single_loss_graph_delta
        enabled_mechanism = str(experiment["enabled_mechanism_ids"][0])
        allowed_loss_fields = (
            {"evidence_response_weight", "evidence_ramp_epochs"}
            if enabled_mechanism == "R6-C1-EVSCRT" else set()
        )
        assert_single_loss_graph_delta(baseline_row, experiment, allowed_loss_fields)
        if experiment.get("parent_row") is not None:
            assert_composition_row(baseline_row, experiment)
    sar_only_external = mode == "sar_only_external"
    distill_pretrain = bool(experiment.get("distill_pretrain", False))
    if distill_pretrain and mode != "train":
        raise ValueError("distill_pretrain requires mode=train (run 1 of the two-run distillation chain)")
    if distill_pretrain and experiment.get("enabled_mechanism_ids"):
        raise ValueError("distill_pretrain must run without a mechanism (run 1 is protocol, not a comparison row)")
    requested_split = "train" if mode == "train" else "validation"
    if runtime["synthetic_only"]:
        manifest = {"execution_scale": "smoke", "test_seal_status": "sealed"}
        assert_test_access_allowed(manifest, requested_split)
        metrics = run_synthetic_step(model_config(resolved), train=mode == "train", seed=int(experiment["seed"]))
        output = Path(runtime["output_root"]) / run_id
        output.mkdir(parents=True, exist_ok=False)
        audit = trainable_parameter_audit(build_model(model_config(resolved)))
        _write_record(output, resolved, metrics, mode, parameter_audit=audit)
        return {"status": "pass", "synthetic_only": True, "metrics": metrics, "output": str(output)}

    if run_manifest is None or data_manifest is None or pretrained_audit is None or code_manifest is None:
        raise ValueError("cloud execution requires immutable run, code, data, and pretrained-audit manifests")
    code_manifest_bytes = code_manifest.read_bytes()
    if hashlib.sha256(code_manifest_bytes).hexdigest() != code_manifest_sha256:
        raise ValueError("code manifest SHA256 mismatch")
    code_manifest_value = json.loads(code_manifest_bytes)
    if code_manifest_value.get("status") != "pass" or code_manifest_value.get("credentials_included") is not False:
        raise ValueError("code manifest is not clean-sync ready")
    verified_run = load_verified_run_manifest(
        run_manifest, run_manifest_sha256, requested_split,
        expected_data_manifest_sha256=data_manifest_sha256,
        expected_pretrained_audit_sha256=pretrained_audit_sha256,
        expected_code_manifest_sha256=code_manifest_sha256,
        expected_resolved_config_sha256=resolved["resolved_config_sha256"],
        required_evaluation_role="official_dfc25_val_sar_only_external" if sar_only_external else "",
    )
    validation_authorization: dict[str, Any] | None = None
    if mode == "train" and not sar_only_external:
        validation_authorization = load_verified_run_manifest(
            run_manifest, run_manifest_sha256, "validation",
            expected_data_manifest_sha256=data_manifest_sha256,
            expected_pretrained_audit_sha256=pretrained_audit_sha256,
            expected_code_manifest_sha256=code_manifest_sha256,
            expected_resolved_config_sha256=resolved["resolved_config_sha256"],
        )
    audit_metadata = verify_pretrained_audit(
        pretrained_audit, pretrained_audit_sha256, resolved["initialization"]["initialization"]
    )
    initialization = resolved["initialization"]["initialization"]
    selected_paths = [str(initialization["optical_checkpoint_path"])]
    if str(initialization.get("sar_initialization_mode")) == "pretrained_croma_sar":
        selected_paths.append(str(initialization["sar_checkpoint_path"]))
    checkpoint_hashes = audited_checkpoint_hashes(audit_metadata, selected_paths)
    if sar_only_external:
        manifest_payload, external_samples = load_sar_external_manifest(
            data_manifest, expected_sha256=data_manifest_sha256
        )
        dataset = CloudSarOnlyRasterDataset(
            external_samples, authorization=verified_run,
            label_policy=manifest_payload.get("label_policy"),
        )
    else:
        manifest_payload, samples = load_data_manifest(data_manifest, expected_sha256=data_manifest_sha256)
        dataset = CloudPairedRasterDataset(
            samples, requested_split, authorization=verified_run,
            label_policy=manifest_payload.get("label_policy"),
        )
    seed = int(experiment["seed"])
    common_init_seed = int(experiment.get("common_init_seed", seed))
    innovation_init_seed = int(experiment.get("innovation_init_seed", 1000))
    data_order_seed = int(experiment.get("data_order_seed", seed))
    augmentation_seed = int(experiment.get("augmentation_seed", seed))
    torch.manual_seed(common_init_seed)
    torch.cuda.manual_seed_all(common_init_seed)
    generator = torch.Generator().manual_seed(augmentation_seed)
    num_workers = int(experiment["num_workers"])
    loader_options: dict[str, Any] = {
        "batch_size": int(experiment["micro_batch"]),
        "num_workers": num_workers,
        "pin_memory": True,
        "generator": generator,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        loader_options["prefetch_factor"] = int(experiment.get("prefetch_factor", 2))
        loader_options["persistent_workers"] = bool(experiment.get("persistent_workers", True))
    train_sampler: DeterministicEpochSampler | None = None
    if mode == "train":
        train_sampler = DeterministicEpochSampler(
            [sample.sample_id for sample in dataset.samples], data_order_seed
        )
    loader = DataLoader(
        dataset,
        shuffle=False if train_sampler is not None else False,
        sampler=train_sampler,
        **loader_options,
    )
    validation_loader: DataLoader | None = None
    if mode == "train" and not sar_only_external and not distill_pretrain:
        validation_dataset = CloudPairedRasterDataset(
            samples, "validation", authorization=validation_authorization or verified_run,
            label_policy=manifest_payload.get("label_policy"),
        )
        validation_options = dict(loader_options)
        validation_options["generator"] = torch.Generator().manual_seed(augmentation_seed + 1)
        validation_loader = DataLoader(
            validation_dataset, shuffle=False, **validation_options,
        )
    device = torch.device("cuda")
    model = build_model(model_config(resolved, checkpoint_hashes))
    anchor_path_text = str(initialization.get("common_init_anchor_path") or "").strip()
    anchor_sha256 = str(initialization.get("common_init_anchor_sha256") or "").strip()
    anchor_metadata: dict[str, Any] | None = None
    if not runtime["synthetic_only"]:
        if not anchor_path_text or len(anchor_sha256) != 64:
            raise ValueError(
                "cloud execution requires a hash-bound common_init_anchor_path and common_init_anchor_sha256"
            )
        anchor_metadata = load_common_init_anchor(
            Path(anchor_path_text), expected_sha256=anchor_sha256, model=model,
            expected_common_init_seed=int(experiment.get("common_init_seed", 0)),
        )
    reset_innovation_parameters(model, innovation_init_seed)
    model = model.to(device)
    torch.manual_seed(augmentation_seed)
    torch.cuda.manual_seed_all(augmentation_seed)
    parameter_audit = trainable_parameter_audit(model)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=float(experiment["learning_rate"]),
        weight_decay=float(experiment["weight_decay"]),
    )
    # R6 gradient balance: scale only the SAR-encoder gradients (applied inside
    # backward via parameter hooks, before optimizer.step). The same public scale
    # is used by both comparison arms; 1.0 is a no-op.
    sar_grad_scale = float(experiment.get("sar_grad_scale", 1.0))
    grad_balance_hooks: list[torch.utils.hooks.RemovableHandle] = []
    if mode == "train" and not sar_only_external and sar_grad_scale != 1.0:
        if sar_grad_scale <= 0.0:
            raise ValueError("sar_grad_scale must be positive")
        for name, parameter in model.named_parameters():
            if name.startswith("sar_encoder.") and parameter.requires_grad:
                grad_balance_hooks.append(
                    parameter.register_hook(lambda grad, scale=sar_grad_scale: grad * scale)
                )
    formal_epochs = int(experiment["epochs"]) if mode == "train" else 1
    execution_epochs = int(experiment.get("execution_epochs", formal_epochs)) if mode == "train" else 1
    scheduler_horizon_epochs = int(experiment.get("scheduler_horizon_epochs", formal_epochs)) if mode == "train" else 1
    if mode == "train" and not 0 < execution_epochs <= formal_epochs:
        raise ValueError("execution_epochs must be positive and no greater than formal epochs")
    if mode == "train" and scheduler_horizon_epochs != formal_epochs:
        raise ValueError("scheduler_horizon_epochs must equal the formal epoch horizon")
    epochs = execution_epochs
    rapid_screen_epochs = experiment.get("rapid_screen_horizon_epochs")
    if mode == "train" and runtime.get("execution_scale") == "innovation_screening_seed0":
        diagnostic_only = bool(experiment.get("diagnostic_only", False))
        if not isinstance(rapid_screen_epochs, int) or not 0 < rapid_screen_epochs <= execution_epochs:
            raise ValueError(
                "innovation rapid_screen_horizon_epochs must be positive and no greater than execution_epochs"
            )
        if not diagnostic_only and rapid_screen_epochs != execution_epochs:
            raise ValueError(
                "rapid innovation execution_epochs must equal the predeclared rapid_screen_horizon_epochs"
            )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(scheduler_horizon_epochs, 1)
    )
    restored_epoch = None
    if mode != "train":
        if checkpoint is None or len(checkpoint_sha256) != 64:
            raise ValueError("cloud evaluation requires an immutable training checkpoint")
        restored_epoch = load_training_checkpoint(
            checkpoint, expected_sha256=checkpoint_sha256, model=model,
            optimizer=None, scheduler=None, resolved_config_sha256=resolved["resolved_config_sha256"],
        )
    metrics: dict[str, float] = {}
    validation_metrics: dict[str, float] = {}
    epoch_curve: list[dict[str, Any]] = []
    rapid_checkpoint_record: dict[str, Any] | None = None
    rapid_validation_metrics: dict[str, float] | None = None
    epoch1_checkpoint_record: dict[str, Any] | None = None
    best_validation_checkpoint_record: dict[str, Any] | None = None
    best_validation_metrics: dict[str, float] | None = None
    output_root = Path(runtime["output_root"]) / run_id
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    rapid_horizon = experiment.get("rapid_screen_horizon_epochs")
    if mode == "train" and runtime.get("execution_scale") == "baseline":
        if execution_epochs != formal_epochs:
            raise ValueError("baseline execution_epochs must equal formal epochs")
        if not isinstance(rapid_horizon, int) or not 0 < rapid_horizon < formal_epochs:
            raise ValueError(
                "baseline training requires a predeclared rapid_screen_horizon_epochs shorter than epochs"
            )
    elif mode == "train" and runtime.get("execution_scale") == "distillation_pretrain":
        # R6 run 1: protocol run, no rapid checkpoint, no mechanism, no gate.
        if execution_epochs != formal_epochs:
            raise ValueError("distillation execution_epochs must equal formal epochs")
        rapid_horizon = None
    model.train(mode == "train")
    accumulation = int(experiment["gradient_accumulation_steps"])
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    sample_order_hashes: list[dict[str, int | str]] = []
    dropout_stats: dict[str, int] = {"no_drop": 0, "optical_drop": 0, "sar_drop": 0, "batches": 0}
    crm_stats: dict[str, Any] | None = None
    if str(loss_config.get("degradation_sampler_version", "none")) == "crm_v2":
        crm_stats = {
            "sampler_version": "crm_v2",
            "batch_count": 0,
            "valid_token_fraction_sum": 0.0,
            "loss_contribution_sum": 0.0,
            "by_intervention": {},
            "examples": [],
        }
    injection_stats: dict[str, Any] | None = None
    if mode == "train" and not sar_only_external and not distill_pretrain:
        if str(loss_config.get("degradation_sampler_version", "none")) == "r6_train_inject_v1":
            injection_stats = {"version": "r6_train_inject_v1", "batches": 0, "injected_samples": 0, "total_samples": 0}
    evidence_stats: dict[str, float] | None = None
    if mode == "train" and not sar_only_external and not distill_pretrain:
        if float(loss_config.get("evidence_response_weight", 0.0)) != 0.0:
            evidence_stats = {"batches": 0}
    for _epoch in range(epochs):
        if evidence_stats is not None:
            evidence_stats.clear()
            evidence_stats["batches"] = 0
        if train_sampler is not None:
            train_sampler.set_epoch(_epoch)
            sample_order_hashes.append(
                {"epoch": _epoch + 1, "sample_order_hash": train_sampler.order_hash()}
            )
        totals = {"loss": 0.0, "mean_iou": 0.0}
        if mode == "train":
            optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            if mode == "train" and not sar_only_external:
                dropped = modality_dropout_assignment(
                    batch.get("sample_id", []), str(loss_config.get("modality_dropout_policy", "none"))
                )
                dropout_stats["batches"] += 1
                key = "no_drop" if dropped is None else ("optical_drop" if dropped == 0 else "sar_drop")
                dropout_stats[key] += 1
            loss, score = _batch_loss(
                model,
                batch,
                device,
                sar_only_external=sar_only_external,
                loss_config=loss_config,
                crm_stats=crm_stats,
                distill_pretrain=distill_pretrain,
                evidence_stats=evidence_stats,
                injection_stats=injection_stats,
                epoch_index=_epoch,
            )
            if mode == "train":
                (loss / accumulation).backward()
                if step % accumulation == 0 or step == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            totals["loss"] += float(loss.detach())
            totals["mean_iou"] += float(score)
        metrics = {key: value / len(loader) for key, value in totals.items()}
        if evidence_stats is not None and len(loader) > 0:
            for key in ("evidence_up", "evidence_dn", "violation_fraction", "conflict_fraction"):
                metrics[key] = evidence_stats.get(key, 0.0) / len(loader)
        if mode == "train":
            scheduler.step()
            if validation_loader is not None:
                validation_metrics = _evaluate_loader(
                    model,
                    validation_loader,
                    device,
                    sar_only_external=False,
                    loss_config=loss_config,
                )
                if (
                    best_validation_metrics is None
                    or validation_metrics["mean_iou"] > best_validation_metrics["mean_iou"]
                ):
                    best_validation_checkpoint_record = save_training_checkpoint(
                        output_root / "checkpoints" / "best_validation.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=_epoch + 1,
                        resolved_config_sha256=resolved["resolved_config_sha256"],
                    )
                    best_validation_metrics = dict(validation_metrics)
            if rapid_horizon == _epoch + 1:
                rapid_checkpoint_record = save_training_checkpoint(
                    output_root / "checkpoints" / (f"rapid_epoch_{_epoch + 1}" + "." + "pt"),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=_epoch + 1,
                    resolved_config_sha256=resolved["resolved_config_sha256"],
                )
                rapid_validation_metrics = dict(validation_metrics)
            if _epoch == 0:
                epoch1_checkpoint_record = save_training_checkpoint(
                    output_root / "checkpoints" / "epoch_1.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=1,
                    resolved_config_sha256=resolved["resolved_config_sha256"],
                )
        epoch_entry: dict[str, float | int | str | None] = {
            "epoch": _epoch + 1,
            "execution_epochs": epochs,
            "scheduler_horizon_epochs": scheduler_horizon_epochs,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "sample_order_hash": sample_order_hashes[-1]["sample_order_hash"] if sample_order_hashes else None,
            **metrics,
        }
        if validation_metrics:
            epoch_entry.update(
                {
                    "validation_loss": validation_metrics["loss"],
                    "validation_mean_iou": validation_metrics["mean_iou"],
                }
            )
        epoch_curve.append(epoch_entry)
    (output_root / "validation_curve.json").write_text(
        json.dumps(
            {
                "artifact_type": "training_curve",
                "schema_version": 1,
                "status": "pass",
                "run_id": run_id,
                "mode": mode,
                "split": "train_validation" if validation_loader is not None else requested_split,
                "training_split": "train" if mode == "train" else None,
                "validation_split": "validation" if validation_loader is not None else requested_split,
                "test_seal_status": "sealed",
                "test_accessed": False,
                "formal_epochs": formal_epochs,
                "execution_epochs": epochs,
                "scheduler_horizon_epochs": scheduler_horizon_epochs,
                "common_init_seed": common_init_seed,
                "innovation_init_seed": innovation_init_seed,
                "data_order_seed": data_order_seed,
                "augmentation_seed": augmentation_seed,
                "loss": loss_config,
                "crm_v2_intervention_stats": crm_stats,
        "r6_injection_stats": injection_stats,
                "modality_dropout_stats": dropout_stats,
                "sample_order_hashes": sample_order_hashes,
                "epochs": epoch_curve,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_root / "sample_order_manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "sample_order_manifest",
                "schema_version": 1,
                "status": "pass",
                "run_id": run_id,
                "data_order_seed": data_order_seed,
                "sample_count": len(dataset),
                "epochs": sample_order_hashes,
                "test_accessed": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    checkpoint_record = None
    if mode == "train":
        checkpoint_record = save_training_checkpoint(
            output_root / "checkpoints" / ("last" + "." + "pt"), model=model, optimizer=optimizer,
            scheduler=scheduler, epoch=epochs, resolved_config_sha256=resolved["resolved_config_sha256"],
        )
        if rapid_checkpoint_record is not None:
            (output_root / "rapid_validation_metrics.json").write_text(
                json.dumps(
                    {
                        "artifact_type": "rapid_baseline_validation_metrics",
                        "schema_version": 1,
                        "status": "pass",
                        "run_id": run_id,
                        "epoch": rapid_horizon,
                        "metrics": rapid_validation_metrics or {},
                        "checkpoint": rapid_checkpoint_record,
                        "test_seal_status": "sealed",
                        "test_accessed": False,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        if best_validation_checkpoint_record is not None:
            (output_root / "best_validation_metrics.json").write_text(
                json.dumps(
                    {
                        "artifact_type": "best_validation_metrics",
                        "schema_version": 1,
                        "status": "pass",
                        "run_id": run_id,
                        "epoch": best_validation_checkpoint_record["epoch"],
                        "metrics": best_validation_metrics or {},
                        "checkpoint": best_validation_checkpoint_record,
                        "test_seal_status": "sealed",
                        "test_accessed": False,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
    evidence = {
        "run_manifest_sha256": run_manifest_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "pretrained_audit_sha256": pretrained_audit_sha256,
        "source_checkpoint_sha256": checkpoint_sha256 or None,
        "source_checkpoint_epoch": restored_epoch,
        "code_manifest_path": str(code_manifest),
        "code_manifest_sha256": code_manifest_sha256,
        "data_manifest_path": str(data_manifest),
        "pretrained_audit_path": str(pretrained_audit),
        "checkpoint": checkpoint_record,
        "rapid_checkpoint": rapid_checkpoint_record,
        "rapid_validation_metrics": rapid_validation_metrics,
        "best_validation_checkpoint": best_validation_checkpoint_record,
        "best_validation_metrics": best_validation_metrics,
        "epoch1_checkpoint": epoch1_checkpoint_record,
        "common_init_anchor": anchor_metadata,
        "common_init_seed": common_init_seed,
        "innovation_init_seed": innovation_init_seed,
        "data_order_seed": data_order_seed,
        "augmentation_seed": augmentation_seed,
        "execution_epochs": epochs,
        "scheduler_horizon_epochs": scheduler_horizon_epochs,
        "loss": loss_config,
        "crm_v2_intervention_stats": crm_stats,
        "r6_injection_stats": injection_stats,
        "modality_dropout_stats": dropout_stats,
        "sample_order_hashes": sample_order_hashes,
    }
    _write_record(output_root, resolved, metrics, mode, parameter_audit=parameter_audit, evidence=evidence)
    (output_root / "verified_run_manifest.json").write_text(json.dumps(verified_run, indent=2), encoding="utf-8")
    for hook in grad_balance_hooks:
        hook.remove()
    return {"status": "pass", "synthetic_only": False, "metrics": metrics, "output": str(output_root)}
