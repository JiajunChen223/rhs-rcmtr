"""Audit CF-E/CF-S and Conditional Route Elasticity on cloud validation only.

The audit deliberately separates two questions that were previously mixed:

* CF-E changes only the evidence tensor while holding both sensor images fixed.
* CF-S changes one sensor with a deterministic P1-audited degradation and
  recomputes evidence from the changed images.

No training is performed and the official test role is never constructed.  A
static-fusion checkpoint may optionally be supplied to measure degradation
robustness AUC relative to the frozen comparator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rhs_rcmtr.data import CloudPairedRasterDataset, load_data_manifest  # noqa: E402
from rhs_rcmtr.engine.entrypoint import model_config  # noqa: E402
from rhs_rcmtr.models import build_model  # noqa: E402
from rhs_rcmtr.utils.checkpoint import load_training_checkpoint  # noqa: E402
from rhs_rcmtr.utils.config import resolve_config  # noqa: E402
from rhs_rcmtr.utils.diagnostics import confusion_metrics, spearman  # noqa: E402
from rhs_rcmtr.utils.initialization import audited_checkpoint_hashes, verify_pretrained_audit  # noqa: E402
from rhs_rcmtr.utils.interventions import (  # noqa: E402
    OPTICAL_DEGRADATIONS,
    SAR_DEGRADATIONS,
    degrade_pair,
    recompute_reliability_evidence,
)
from rhs_rcmtr.utils.run_manifest import load_verified_run_manifest  # noqa: E402


CF_LEVELS = (0.25, 0.5, 0.75)
CF_S_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_ABS_RHO = 0.70
MIN_DIRECTION_FRACTION = 0.80
EPS = 1e-6


def _confusion(logits: torch.Tensor, labels: torch.Tensor, classes: int = 8) -> torch.Tensor:
    predictions = logits.argmax(dim=1)
    valid = labels != 255
    encoded = labels[valid] * classes + predictions[valid]
    return torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def _sample_route_mass(weights: torch.Tensor) -> torch.Tensor:
    return weights.flatten(2).mean(dim=2)


def _fraction(values: list[bool]) -> float:
    return float(sum(values) / max(len(values), 1))


def _monotonic_fraction(values: list[list[float]]) -> float:
    if not values:
        return 0.0
    passing = 0
    for row in values:
        passing += int(all(row[index + 1] <= row[index] + EPS for index in range(len(row) - 1)))
    return float(passing / len(values))


def _normalized_auc(values: list[float], levels: tuple[float, ...] = CF_S_LEVELS) -> float:
    if len(values) != len(levels) or len(values) < 2:
        return 0.0
    area = 0.0
    for left, right, x0, x1 in zip(values[:-1], values[1:], levels[:-1], levels[1:]):
        area += 0.5 * (left + right) * (x1 - x0)
    return float(area / max(levels[-1] - levels[0], EPS))


def _trapz(values: list[float], levels: tuple[float, ...]) -> float:
    if len(values) != len(levels) or len(values) < 2:
        return 0.0
    area = 0.0
    for left, right, x0, x1 in zip(values[:-1], values[1:], levels[:-1], levels[1:]):
        area += 0.5 * (left + right) * (x1 - x0)
    return float(area / max(levels[-1] - levels[0], EPS))


def _load_candidate(
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
) -> tuple[dict[str, Any], dict[str, Any], list[Any], Any, torch.nn.Module]:
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
    initialization = resolved["initialization"]["initialization"]
    checkpoint_hashes = audited_checkpoint_hashes(
        audit_payload,
        [
            str(initialization["optical_checkpoint_path"]),
            *(
                [str(initialization["sar_checkpoint_path"])]
                if str(initialization.get("sar_initialization_mode")) == "pretrained_croma_sar"
                else []
            ),
        ],
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
    model = build_model(model_config(resolved, checkpoint_hashes)).cuda()
    epoch = load_training_checkpoint(
        checkpoint,
        expected_sha256=checkpoint_sha256,
        model=model,
        optimizer=None,
        scheduler=None,
        resolved_config_sha256=resolved["resolved_config_sha256"],
    )
    model.eval()
    return resolved, verified_run, dataset, epoch, model


def _load_reference_model(
    *,
    runtime_config: Path,
    experiment_config: Path,
    initialization_config: Path,
    pretrained_audit: Path,
    pretrained_audit_sha256: str,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> torch.nn.Module:
    """Load the frozen static-fusion comparator without opening another split."""

    resolved = resolve_config(runtime_config, experiment_config, initialization_config)
    audit_payload = verify_pretrained_audit(
        pretrained_audit,
        pretrained_audit_sha256,
        resolved["initialization"]["initialization"],
    )
    initialization = resolved["initialization"]["initialization"]
    checkpoint_hashes = audited_checkpoint_hashes(
        audit_payload,
        [
            str(initialization["optical_checkpoint_path"]),
            *(
                [str(initialization["sar_checkpoint_path"])]
                if str(initialization.get("sar_initialization_mode")) == "pretrained_croma_sar"
                else []
            ),
        ],
    )
    model = build_model(model_config(resolved, checkpoint_hashes)).cuda()
    load_training_checkpoint(
        checkpoint,
        expected_sha256=checkpoint_sha256,
        model=model,
        optimizer=None,
        scheduler=None,
        resolved_config_sha256=resolved["resolved_config_sha256"],
    )
    model.eval()
    return model


def _clean_metrics(model: torch.nn.Module, loader: DataLoader) -> dict[str, Any]:
    confusion = torch.zeros((8, 8), dtype=torch.int64, device="cuda")
    with torch.no_grad():
        for batch in loader:
            optical = batch["optical"].cuda(non_blocking=True)
            sar = batch["sar"].cuda(non_blocking=True)
            reliability = batch["reliability"].cuda(non_blocking=True)
            labels = batch["label"].cuda(non_blocking=True)
            confusion += _confusion(model(optical, sar, reliability)["logits"], labels)
    return confusion_metrics(confusion)


def _cf_e(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    max_batches: int = 0,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    for modality, name in ((0, "optical"), (1, "sar")):
        direction: list[bool] = []
        paired_values: list[list[float]] = []
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if max_batches > 0 and batch_index >= max_batches:
                    break
                optical = batch["optical"].cuda(non_blocking=True)
                sar = batch["sar"].cuda(non_blocking=True)
                reliability = batch["reliability"].cuda(non_blocking=True)
                before = _sample_route_mass(model(optical, sar, reliability)["route_weights"])
                per_sample: list[list[float]] = [[] for _ in range(optical.shape[0])]
                for level in CF_LEVELS:
                    changed = reliability.clone()
                    changed[:, modality : modality + 1] *= 1.0 - level
                    after = _sample_route_mass(model(optical, sar, changed)["route_weights"])
                    if modality == 0:
                        direction.extend(
                            bool((a[0] <= b[0] + EPS).item() and (a[1] >= b[1] - EPS).item())
                            for a, b in zip(after, before)
                        )
                        values = after[:, 0].detach().cpu().tolist()
                    else:
                        direction.extend(
                            bool((a[1] <= b[1] + EPS).item() and (a[0] >= b[0] - EPS).item())
                            for a, b in zip(after, before)
                        )
                        values = after[:, 1].detach().cpu().tolist()
                    for index, value in enumerate(values):
                        per_sample[index].append(float(value))
                paired_values.extend(per_sample)
        results[name] = {
            "levels": list(CF_LEVELS),
            "correct_direction_fraction": _fraction(direction),
            "sample_count": len(direction),
            "route_mass_monotonic_fraction": _monotonic_fraction(paired_values),
            "direction_definition": (
                "lowered evidence must not increase target trust and must not decrease the other trust"
            ),
        }
    return results


def _cf_s(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    max_batches: int = 0,
    reference_model: torch.nn.Module | None = None,
) -> tuple[dict[str, Any], dict[str, list[float]], dict[str, list[float]]]:
    all_results: dict[str, Any] = {}
    robustness: dict[str, list[float]] = {}
    reference_robustness: dict[str, list[float]] = {}
    families = ((0, OPTICAL_DEGRADATIONS), (1, SAR_DEGRADATIONS))
    for modality, names in families:
        for family in names:
            route_rows: list[list[float]] = []
            q_drop: list[bool] = []
            miou_by_level: list[list[float]] = [[] for _ in CF_S_LEVELS]
            reference_miou_by_level: list[list[float]] = [[] for _ in CF_S_LEVELS]
            for batch_index, batch in enumerate(loader):
                if max_batches > 0 and batch_index >= max_batches:
                    break
                optical = batch["optical"].cuda(non_blocking=True)
                sar = batch["sar"].cuda(non_blocking=True)
                reliability = batch["reliability"].cuda(non_blocking=True)
                labels = batch["label"].cuda(non_blocking=True)
                per_sample: list[list[float]] = [[] for _ in range(optical.shape[0])]
                baseline_q = reliability[:, modality : modality + 1]
                for level_index, level in enumerate(CF_S_LEVELS):
                    if level == 0.0:
                        degraded_optical, degraded_sar, degraded_q = optical, sar, reliability
                    else:
                        degraded_optical, degraded_sar = degrade_pair(
                            optical, sar, modality, family, level
                        )
                        degraded_q = recompute_reliability_evidence(degraded_optical, degraded_sar)
                        if degraded_q.shape[2:] != reliability.shape[2:]:
                            degraded_q = torch.nn.functional.interpolate(
                                degraded_q, size=reliability.shape[2:], mode="bilinear", align_corners=False
                            )
                    output = model(degraded_optical, degraded_sar, degraded_q)
                    masses = _sample_route_mass(output["route_weights"])
                    target_mass = masses[:, modality]
                    for index, value in enumerate(target_mass.detach().cpu().tolist()):
                        per_sample[index].append(float(value))
                    q_after = degraded_q[:, modality : modality + 1]
                    q_drop.extend(
                        bool(value)
                        for value in (q_after.flatten(1).mean(dim=1) < baseline_q.flatten(1).mean(dim=1) - EPS).cpu()
                    )
                    if modality == 0:
                        # Optical degradation robustness is a primary CRE utility curve.
                        confusion = _confusion(output["logits"], labels)
                        miou_by_level[level_index].append(float(confusion_metrics(confusion)["mIoU"]))
                        if reference_model is not None:
                            reference_output = reference_model(
                                degraded_optical, degraded_sar, degraded_q
                            )
                            reference_confusion = _confusion(reference_output["logits"], labels)
                            reference_miou_by_level[level_index].append(
                                float(confusion_metrics(reference_confusion)["mIoU"])
                            )
                route_rows.extend(per_sample)
            medians = [
                float(torch.tensor([row[index] for row in route_rows]).median())
                for index in range(len(CF_S_LEVELS))
            ] if route_rows else []
            rho = spearman(list(CF_S_LEVELS), medians) if medians else 0.0
            monotonic = _monotonic_fraction(route_rows)
            key = f"{('optical' if modality == 0 else 'sar')}:{family}"
            all_results[key] = {
                "modality": "optical" if modality == 0 else "sar",
                "degradation": family,
                "severity_levels": list(CF_S_LEVELS),
                "sample_count": len(route_rows),
                "target_route_mass_medians": medians,
                "severity_median_spearman_rho": float(rho),
                "absolute_spearman_rho": abs(float(rho)),
                "monotonic_fraction": monotonic,
                "q_drop_fraction": _fraction(q_drop),
                "direction_correct": monotonic >= MIN_DIRECTION_FRACTION and rho <= -MIN_ABS_RHO,
            }
            if modality == 0:
                robustness[key] = [
                    float(sum(values) / max(len(values), 1)) for values in miou_by_level
                ]
                if reference_model is not None:
                    reference_robustness[key] = [
                        float(sum(values) / max(len(values), 1))
                        for values in reference_miou_by_level
                    ]
    return all_results, robustness, reference_robustness


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
    max_samples: int = 256,
    batch_size: int = 4,
    static_experiment_config: Path | None = None,
    static_checkpoint: Path | None = None,
    static_checkpoint_sha256: str = "",
) -> dict[str, Any]:
    resolved, verified_run, dataset, epoch, model = _load_candidate(
        runtime_config=runtime_config,
        experiment_config=experiment_config,
        initialization_config=initialization_config,
        run_manifest=run_manifest,
        run_manifest_sha256=run_manifest_sha256,
        data_manifest=data_manifest,
        data_manifest_sha256=data_manifest_sha256,
        pretrained_audit=pretrained_audit,
        pretrained_audit_sha256=pretrained_audit_sha256,
        code_manifest=code_manifest,
        code_manifest_sha256=code_manifest_sha256,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    if max_samples > 0:
        dataset.samples = dataset.samples[:max_samples]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    static_model = None
    if any(item is not None for item in (static_experiment_config, static_checkpoint)):
        if static_experiment_config is None or static_checkpoint is None or len(static_checkpoint_sha256) != 64:
            raise ValueError("static comparator requires config, checkpoint and a 64-character SHA256")
        static_model = _load_reference_model(
            runtime_config=runtime_config,
            experiment_config=static_experiment_config,
            initialization_config=initialization_config,
            pretrained_audit=pretrained_audit,
            pretrained_audit_sha256=pretrained_audit_sha256,
            checkpoint=static_checkpoint,
            checkpoint_sha256=static_checkpoint_sha256,
        )
    clean = _clean_metrics(model, loader)
    cf_e = _cf_e(model, loader)
    cf_s, optical_curves, static_curves = _cf_s(model, loader, reference_model=static_model)
    direction_e = all(
        result["correct_direction_fraction"] >= MIN_DIRECTION_FRACTION for result in cf_e.values()
    )
    direction_s = all(bool(result["direction_correct"]) for result in cf_s.values())
    report = {
        "artifact_type": "conditional_route_elasticity_audit",
        "schema_version": 1,
        "status": "pass" if direction_e and direction_s else "fail",
        "execution_context": "cloud",
        "run_id": verified_run.get("run_id"),
        "checkpoint_epoch": epoch,
        "split": "validation",
        "sample_count": len(dataset),
        "cf_e": {
            "definition": "sensor tensors fixed; only evidence q is changed",
            "results": cf_e,
            "all_modalities_pass": direction_e,
        },
        "cf_s": {
            "definition": "one sensor is deterministically degraded and evidence is recomputed",
            "results": cf_s,
            "all_degradations_pass": direction_s,
        },
        "clean_metrics": clean,
        "robustness": {
            "optical_degradation_miou_curves": optical_curves,
            "optical_degradation_auc": {
                key: _normalized_auc(values) for key, values in optical_curves.items()
            },
            "auc_definition": "normalized trapezoidal AUC over severity levels 0..1; static comparator supplied separately",
            "static_comparator": "supplied" if static_model is not None else "not_supplied",
            "static_optical_degradation_miou_curves": static_curves,
            "static_optical_degradation_auc": {
                key: _normalized_auc(values) for key, values in static_curves.items()
            },
            "auc_gain_candidate_minus_static": {
                key: _normalized_auc(values) - _normalized_auc(static_curves[key])
                for key, values in optical_curves.items()
                if key in static_curves
            },
        },
        "gates": {
            "minimum_absolute_spearman_rho": MIN_ABS_RHO,
            "minimum_correct_direction_fraction": MIN_DIRECTION_FRACTION,
            "cf_e_pass": direction_e,
            "cf_s_pass": direction_s,
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
        "code-manifest-sha256", "checkpoint-sha256",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--static-experiment-config", type=Path)
    parser.add_argument("--static-checkpoint", type=Path)
    parser.add_argument("--static-checkpoint-sha256", default="")
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
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        static_experiment_config=args.static_experiment_config,
        static_checkpoint=args.static_checkpoint,
        static_checkpoint_sha256=args.static_checkpoint_sha256,
    )
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
