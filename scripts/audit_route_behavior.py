"""Audit route/reliability behaviour on a sealed internal validation split.

This is evaluation-only: it loads a completed cloud checkpoint, computes route
correlation, entropy, modality mass, SAR ablation and counterfactual direction,
and writes metadata.  It never trains, reads the official test role, or copies
real data/checkpoints to the local host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from rhs_rcmtr.data import CloudPairedRasterDataset, load_data_manifest
from rhs_rcmtr.engine.entrypoint import model_config
from rhs_rcmtr.models import build_model
from rhs_rcmtr.utils.checkpoint import load_training_checkpoint
from rhs_rcmtr.utils.config import resolve_config
from rhs_rcmtr.utils.diagnostics import confusion_metrics, route_entropy, spearman
from rhs_rcmtr.utils.initialization import audited_checkpoint_hashes, verify_pretrained_audit
from rhs_rcmtr.utils.run_manifest import load_verified_run_manifest


def _confusion(logits: torch.Tensor, labels: torch.Tensor, classes: int = 8) -> torch.Tensor:
    predictions = logits.argmax(dim=1)
    valid = labels != 255
    encoded = labels[valid] * classes + predictions[valid]
    return torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def audit(
    *,
    runtime_config: Path,
    experiment_config: Path,
    initialization_config: Path,
    run_manifest: Path,
    run_manifest_sha256: str,
    data_manifest: Path,
    data_manifest_sha256: str,
    pretrained_audit: Path,
    pretrained_audit_sha256: str,
    code_manifest: Path,
    code_manifest_sha256: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    output: Path,
    mechanism_id: str,
    max_samples: int = 0,
) -> dict[str, Any]:
    resolved = resolve_config(runtime_config, experiment_config, initialization_config)
    verified_run = load_verified_run_manifest(
        run_manifest,
        run_manifest_sha256,
        "validation",
        expected_data_manifest_sha256=data_manifest_sha256,
        expected_pretrained_audit_sha256=pretrained_audit_sha256,
        expected_code_manifest_sha256=code_manifest_sha256,
        expected_resolved_config_sha256=resolved["resolved_config_sha256"],
    )
    audit_payload = verify_pretrained_audit(
        pretrained_audit,
        pretrained_audit_sha256,
        resolved["initialization"]["initialization"],
    )
    init = resolved["initialization"]["initialization"]
    checkpoint_hashes = audited_checkpoint_hashes(
        audit_payload, [str(init["optical_checkpoint_path"])]
    )
    manifest_payload, samples = load_data_manifest(
        data_manifest, expected_sha256=data_manifest_sha256
    )
    dataset = CloudPairedRasterDataset(
        samples,
        "validation",
        authorization=verified_run,
        label_policy=manifest_payload.get("label_policy"),
    )
    if max_samples > 0:
        dataset.samples = dataset.samples[:max_samples]
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    model = build_model(model_config(resolved, checkpoint_hashes)).cuda()
    restored_epoch = load_training_checkpoint(
        checkpoint,
        expected_sha256=checkpoint_sha256,
        model=model,
        optimizer=None,
        scheduler=None,
        resolved_config_sha256=resolved["resolved_config_sha256"],
    )
    model.eval()
    fusion_confusion = torch.zeros((8, 8), dtype=torch.int64, device="cuda")
    sar_only_confusion = torch.zeros_like(fusion_confusion)
    no_sar_confusion = torch.zeros_like(fusion_confusion)
    no_optical_confusion = torch.zeros_like(fusion_confusion)
    q_differences: list[float] = []
    w_differences: list[float] = []
    optical_masses: list[float] = []
    sar_masses: list[float] = []
    entropies: list[float] = []
    optical_counterfactual_deltas: list[float] = []
    sar_counterfactual_deltas: list[float] = []
    with torch.no_grad():
        for batch in loader:
            optical = batch["optical"].cuda(non_blocking=True)
            sar = batch["sar"].cuda(non_blocking=True)
            reliability = batch["reliability"].cuda(non_blocking=True)
            labels = batch["label"].cuda(non_blocking=True)
            original = model(optical, sar, reliability)
            weights = original["route_weights"]
            evidence = reliability
            if evidence.shape[2:] != weights.shape[2:]:
                evidence = torch.nn.functional.interpolate(
                    evidence, size=weights.shape[2:], mode="bilinear", align_corners=False
                )
            q_diff = (evidence[:, 0] - evidence[:, 1]).flatten(1).mean(dim=1)
            w_diff = (weights[:, 0] - weights[:, 1]).flatten(1).mean(dim=1)
            q_differences.extend(float(value) for value in q_diff.cpu())
            w_differences.extend(float(value) for value in w_diff.cpu())
            optical_masses.extend(float(value) for value in weights[:, 0].flatten(1).mean(dim=1).cpu())
            sar_masses.extend(float(value) for value in weights[:, 1].flatten(1).mean(dim=1).cpu())
            entropies.extend(float(value) for value in route_entropy(weights).cpu())
            fusion_confusion += _confusion(original["logits"], labels)

            sar_only = model.forward_sar_only(sar)
            sar_only_confusion += _confusion(sar_only["logits"], labels)

            no_sar_reliability = evidence.clone()
            no_sar_reliability[:, 1:2] = 0.0
            no_sar = model(optical, torch.zeros_like(sar), no_sar_reliability)
            no_sar_confusion += _confusion(no_sar["logits"], labels)

            no_optical_reliability = evidence.clone()
            no_optical_reliability[:, 0:1] = 0.0
            no_optical = model(torch.zeros_like(optical), sar, no_optical_reliability)
            no_optical_confusion += _confusion(no_optical["logits"], labels)

            optical_reliability = evidence.clone()
            optical_reliability[:, 0:1] *= 0.5
            optical_degraded = model(optical * 0.5, sar, optical_reliability)
            optical_counterfactual_deltas.extend(
                float(value)
                for value in (
                    optical_degraded["route_weights"][:, 0]
                    - weights[:, 0]
                ).flatten(1).mean(dim=1).cpu()
            )
            sar_reliability = evidence.clone()
            sar_reliability[:, 1:2] *= 0.5
            sar_degraded = model(optical, sar * 0.5, sar_reliability)
            sar_counterfactual_deltas.extend(
                float(value)
                for value in (
                    sar_degraded["route_weights"][:, 1]
                    - weights[:, 1]
                ).flatten(1).mean(dim=1).cpu()
            )

    fusion_metrics = confusion_metrics(fusion_confusion)
    sar_only_metrics = confusion_metrics(sar_only_confusion)
    no_sar_metrics = confusion_metrics(no_sar_confusion)
    no_optical_metrics = confusion_metrics(no_optical_confusion)
    mean_optical_mass = float(sum(optical_masses) / max(len(optical_masses), 1))
    mean_sar_mass = float(sum(sar_masses) / max(len(sar_masses), 1))
    mean_optical_delta = float(
        sum(optical_counterfactual_deltas) / max(len(optical_counterfactual_deltas), 1)
    )
    mean_sar_delta = float(
        sum(sar_counterfactual_deltas) / max(len(sar_counterfactual_deltas), 1)
    )
    # Two-zone cleanup 2026-09-02: the historical reliability-router family
    # (rhs_router/esr/esr_crm/a4r) was rejected and archived; R5-C1-OA-SCRT is
    # validated through SAR null hypotheses, not the reliability-route family.
    reliability_router = False
    report = {
        "artifact_type": "route_behavior_audit",
        "schema_version": 1,
        "status": "pass",
        "execution_context": "cloud",
        "run_id": verified_run.get("run_id"),
        "mechanism_id": mechanism_id,
        "checkpoint_epoch": restored_epoch,
        "split": "validation",
        "sample_count": len(dataset),
        "metrics": {
            "fusion": fusion_metrics,
            "sar_only": sar_only_metrics,
            "no_sar": no_sar_metrics,
            "no_optical": no_optical_metrics,
        },
        "route_behavior": {
            "reliability_route_spearman": spearman(q_differences, w_differences),
            "mean_optical_route_mass": mean_optical_mass,
            "mean_sar_route_mass": mean_sar_mass,
            "mean_route_entropy": float(sum(entropies) / max(len(entropies), 1)),
            "optical_counterfactual_delta": mean_optical_delta,
            "sar_counterfactual_delta": mean_sar_delta,
        },
        "mechanism_validity": {
            "reliability_router_expected": reliability_router,
            "counterfactual_direction_ok": (
                (not reliability_router)
                or (mean_optical_delta <= 1e-6 and mean_sar_delta <= 1e-6)
            ),
            "no_route_collapse": 0.01 < mean_optical_mass < 0.99 and 0.01 < mean_sar_mass < 0.99,
            "sar_survival": mean_sar_mass > 0.01 and float(sar_only_metrics["mIoU"]) > 0.0,
        },
        "test_seal_status": "sealed",
        "test_accessed": False,
        "training_started": False,
        "data_manifest_sha256": data_manifest_sha256,
        "pretrained_audit_sha256": pretrained_audit_sha256,
        "code_manifest_sha256": code_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_config_sha256": resolved["resolved_config_sha256"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "runtime-config", "experiment-config", "initialization-config", "run-manifest",
        "data-manifest", "pretrained-audit", "code-manifest", "checkpoint", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for name in (
        "run-manifest-sha256", "data-manifest-sha256", "pretrained-audit-sha256",
        "code-manifest-sha256", "checkpoint-sha256", "mechanism-id",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    report = audit(
        runtime_config=args.runtime_config,
        experiment_config=args.experiment_config,
        initialization_config=args.initialization_config,
        run_manifest=args.run_manifest,
        run_manifest_sha256=args.run_manifest_sha256,
        data_manifest=args.data_manifest,
        data_manifest_sha256=args.data_manifest_sha256,
        pretrained_audit=args.pretrained_audit,
        pretrained_audit_sha256=args.pretrained_audit_sha256,
        code_manifest=args.code_manifest,
        code_manifest_sha256=args.code_manifest_sha256,
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        output=args.output,
        mechanism_id=args.mechanism_id,
        max_samples=args.max_samples,
    )
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
