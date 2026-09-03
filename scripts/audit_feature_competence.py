"""Cloud-only feature, decoder and gradient competence audit.

The input is a JSON case registry.  Each case points to immutable cloud
manifests and one or more cloud checkpoint snapshots (for example epoch 1 and
epoch 6).  The registry is metadata only; this script never copies real
rasters or checkpoint binaries to the local host and never opens the test role.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rhs_rcmtr.data import CloudPairedRasterDataset, load_data_manifest  # noqa: E402
from rhs_rcmtr.engine.entrypoint import _batch_loss, model_config  # noqa: E402
from rhs_rcmtr.models import build_model  # noqa: E402
from rhs_rcmtr.utils.checkpoint import load_training_checkpoint  # noqa: E402
from rhs_rcmtr.utils.config import resolve_config  # noqa: E402
from rhs_rcmtr.utils.initialization import audited_checkpoint_hashes, verify_pretrained_audit  # noqa: E402
from rhs_rcmtr.utils.initialization_anchor import load_common_init_anchor, reset_innovation_parameters  # noqa: E402
from rhs_rcmtr.utils.run_manifest import load_verified_run_manifest  # noqa: E402


def _norm_summary(values: list[torch.Tensor]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}
    merged = torch.cat(values)
    return {
        "mean": float(merged.mean()),
        "median": float(merged.median()),
        "std": float(merged.std(unbiased=False)),
    }


def _module_grad_norm(module: torch.nn.Module) -> float:
    squared = [parameter.grad.detach().float().square().sum() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.stack(squared).sum().sqrt()) if squared else 0.0


def _module_parameter_norms(module: torch.nn.Module) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, parameter in module.named_parameters():
        result[name] = float(parameter.detach().float().norm())
    result["total"] = float(
        sum(value * value for key, value in result.items() if key != "total") ** 0.5
    )
    return result


def _load_snapshot(case: dict[str, Any], snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[Any], torch.nn.Module, int]:
    resolved = resolve_config(
        Path(case["runtime_config"]),
        Path(case["experiment_config"]),
        Path(case["initialization_config"]),
    )
    snapshot_sha = str(case.get("resolved_config_sha256") or "").strip()
    if snapshot_sha:
        if len(snapshot_sha) != 64:
            raise ValueError(f"case {case.get('id')} has an invalid resolved_config_sha256")
        # Historical runs are bound to the immutable resolved snapshot recorded
        # at launch; later diagnostic metadata in YAML does not rewrite it.
        resolved["resolved_config_sha256"] = snapshot_sha
    run_path = Path(case["run_manifest"])
    data_path = Path(case["data_manifest"])
    audit_path = Path(case["pretrained_audit"])
    verified = load_verified_run_manifest(
        run_path,
        str(case["run_manifest_sha256"]),
        "validation",
        expected_data_manifest_sha256=str(case["data_manifest_sha256"]),
        expected_pretrained_audit_sha256=str(case["pretrained_audit_sha256"]),
        expected_code_manifest_sha256=str(case["code_manifest_sha256"]),
        expected_resolved_config_sha256=resolved["resolved_config_sha256"],
    )
    audit_payload = verify_pretrained_audit(
        audit_path,
        str(case["pretrained_audit_sha256"]),
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
        data_path, expected_sha256=str(case["data_manifest_sha256"])
    )
    dataset = CloudPairedRasterDataset(
        samples,
        "validation",
        authorization=verified,
        label_policy=manifest_payload.get("label_policy"),
    )
    # Keep the complete paired sample list and a separate train authorization
    # on the in-memory dataset object for the gradient half of D3.  Both are
    # derived from the same verified cloud manifest.
    dataset.all_samples = samples
    dataset.train_authorization = load_verified_run_manifest(
        run_path,
        str(case["run_manifest_sha256"]),
        "train",
        expected_data_manifest_sha256=str(case["data_manifest_sha256"]),
        expected_pretrained_audit_sha256=str(case["pretrained_audit_sha256"]),
        expected_code_manifest_sha256=str(case["code_manifest_sha256"]),
        expected_resolved_config_sha256=resolved["resolved_config_sha256"],
    )
    model = build_model(model_config(resolved, checkpoint_hashes)).cuda()
    role = str(snapshot.get("role", "")).casefold()
    if role == "initial" or not str(snapshot.get("path") or "").strip():
        anchor_path = str(initialization.get("common_init_anchor_path") or "").strip()
        anchor_sha = str(initialization.get("common_init_anchor_sha256") or "").strip()
        if not anchor_path or len(anchor_sha) != 64:
            raise ValueError("initial D3 snapshot requires a hash-bound common initialization anchor")
        load_common_init_anchor(Path(anchor_path), expected_sha256=anchor_sha, model=model)
        reset_innovation_parameters(model, int(resolved["experiment"].get("innovation_init_seed", 1000)))
        epoch = 0
    else:
        epoch = load_training_checkpoint(
            Path(snapshot["path"]),
            expected_sha256=str(snapshot["sha256"]),
            model=model,
            optimizer=None,
            scheduler=None,
            resolved_config_sha256=resolved["resolved_config_sha256"],
        )
    model.eval()
    return resolved, verified, dataset, model, epoch


def _collect_features(model: torch.nn.Module, loader: DataLoader) -> dict[str, Any]:
    optical_values: list[torch.Tensor] = []
    sar_values: list[torch.Tensor] = []
    fused_values: list[torch.Tensor] = []
    cosine_values: list[torch.Tensor] = []
    feature_stats: dict[str, list[dict[str, float]]] = {"optical": [], "sar": [], "fused": []}
    latest: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if name == "fused":
                tensor = inputs[0].detach().float()
                fused_values.append(torch.linalg.vector_norm(tensor, dim=1).flatten().cpu())
                feature_stats["fused"].append({
                    "mean": float(tensor.mean()),
                    "std": float(tensor.std(unbiased=False)),
                    "channel_variance_mean": float(tensor.flatten(2).var(dim=2, unbiased=False).mean()),
                })
            elif name == "optical":
                tensor = output.detach().float()
                latest["optical"] = tensor
                optical_values.append(torch.linalg.vector_norm(tensor, dim=1).flatten().cpu())
                feature_stats["optical"].append({
                    "mean": float(tensor.mean()),
                    "std": float(tensor.std(unbiased=False)),
                    "channel_variance_mean": float(tensor.flatten(2).var(dim=2, unbiased=False).mean()),
                })
            else:
                tensor = output.detach().float()
                latest["sar"] = tensor
                sar_values.append(torch.linalg.vector_norm(tensor, dim=1).flatten().cpu())
                feature_stats["sar"].append({
                    "mean": float(tensor.mean()),
                    "std": float(tensor.std(unbiased=False)),
                    "channel_variance_mean": float(tensor.flatten(2).var(dim=2, unbiased=False).mean()),
                })
        return hook

    def capture_fused(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        tensor = inputs[0].detach().float()
        fused_values.append(torch.linalg.vector_norm(tensor, dim=1).flatten().cpu())
        feature_stats["fused"].append({
            "mean": float(tensor.mean()),
            "std": float(tensor.std(unbiased=False)),
            "channel_variance_mean": float(tensor.flatten(2).var(dim=2, unbiased=False).mean()),
        })

    handles = [
        model.optical_projection.register_forward_hook(capture("optical")),
        model.sar_projection.register_forward_hook(capture("sar")),
        model.decoder[0].register_forward_pre_hook(capture_fused),
    ]
    with torch.no_grad():
        for batch in loader:
            optical = batch["optical"].cuda(non_blocking=True)
            sar = batch["sar"].cuda(non_blocking=True)
            reliability = batch["reliability"].cuda(non_blocking=True)
            # Hooks retain projected tensors for this forward.  Cosine uses the
            # corresponding projected maps and is collected by explicit hooks
            # below when both tensors are available.
            model(optical, sar, reliability)
            if "optical" in latest and "sar" in latest:
                left = latest["optical"]
                right = latest["sar"]
                if right.shape[2:] != left.shape[2:]:
                    right = F.interpolate(right, size=left.shape[2:], mode="bilinear", align_corners=False)
                cosine_values.append(F.cosine_similarity(left, right, dim=1).flatten().cpu())
    for handle in handles:
        handle.remove()
    optical_summary = _norm_summary(optical_values)
    sar_summary = _norm_summary(sar_values)
    fused_summary = _norm_summary(fused_values)
    cosine = torch.cat(cosine_values) if cosine_values else torch.zeros(1)
    stats_summary = {
        name: {
            key: float(sum(item[key] for item in entries) / max(len(entries), 1))
            for key in ("mean", "std", "channel_variance_mean")
        }
        for name, entries in feature_stats.items()
    }
    return {
        "projection_optical_l2": optical_summary,
        "projection_sar_l2": sar_summary,
        "decoder_input_l2": fused_summary,
        "feature_statistics": stats_summary,
        "projection_parameter_norms": {
            "optical": _module_parameter_norms(model.optical_projection),
            "sar": _module_parameter_norms(model.sar_projection),
        },
        "optical_sar_l2_ratio_median": float(
            optical_summary["median"] / max(sar_summary["median"], 1e-8)
        ),
        "optical_sar_cosine_proxy": {
            "mean": float(cosine.mean()),
            "std": float(cosine.std(unbiased=False)),
            "median": float(cosine.median()),
        },
    }


def _collect_gradients(model: torch.nn.Module, resolved: dict[str, Any], dataset: Any, max_samples: int) -> dict[str, Any]:
    # Use a deterministic prefix of the declared train split only.  A fresh
    # loader is constructed with the same batch size; no optimizer step occurs.
    train_samples = [sample for sample in dataset.all_samples if sample.split == "train"]
    if max_samples > 0:
        train_samples = train_samples[:max_samples]
    if not train_samples:
        return {"status": "missing_train_samples"}
    # The dataset instance is validation-authorized by _load_snapshot.  Build
    # a narrow view with the same verified authorization carried by the object.
    train_dataset = CloudPairedRasterDataset(
        dataset.all_samples,
        "train",
        authorization=dataset.train_authorization,
        label_policy=dataset.label_policy,
    )
    train_dataset.samples = train_samples
    loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
    model.train()
    unimodal_mode = str(resolved["experiment"].get("unimodal_mode", "multimodal"))
    values: list[dict[str, float]] = []
    for batch in loader:
        model.zero_grad(set_to_none=True)
        loss, _ = _batch_loss(
            model,
            batch,
            torch.device("cuda"),
            sar_only_external=False,
            loss_config=resolved["experiment"].get("loss", {}),
        )
        loss.backward()
        optical_norm = _module_grad_norm(model.optical_encoder) + _module_grad_norm(model.optical_projection)
        sar_norm = _module_grad_norm(model.sar_encoder) + _module_grad_norm(model.sar_projection)
        decoder_norm = _module_grad_norm(model.decoder)
        router_norm = _module_grad_norm(model.router) if model.router is not None else 0.0
        values.append({
            "optical_encoder_projection": optical_norm,
            "sar_encoder_projection": sar_norm,
            "decoder": decoder_norm,
            "router": router_norm,
            "sar_to_optical_ratio": sar_norm / max(optical_norm, 1e-8),
        })
        if len(values) * 4 >= max_samples:
            break
    model.eval()
    if not values:
        return {"status": "no_gradient_batches"}
    keys = tuple(values[0])
    summary = {
        key: {
            "mean": float(sum(item[key] for item in values) / len(values)),
            "max": float(max(item[key] for item in values)),
        }
        for key in keys
    }
    ratio = summary["sar_to_optical_ratio"]["mean"]
    if unimodal_mode == "optical_only":
        ratio_flag = "unimodal_ratio_not_interpretable_inactive_sar"
    elif unimodal_mode == "sar_only":
        ratio_flag = "unimodal_ratio_not_interpretable_inactive_optical"
    else:
        ratio_flag = (
            "severe_domination" if ratio > 10.0 else "scale_domination" if ratio > 3.0 else "within_3_to_1"
        )
    return {
        "status": "pass",
        "unimodal_mode": unimodal_mode,
        "batch_count": len(values),
        "summary": summary,
        "diagnostic_flag": ratio_flag,
        "ratio_interpretation": (
            "raw ratio retained for traceability; a single active branch makes cross-branch gradient domination undefined"
            if unimodal_mode != "multimodal"
            else "sar gradient norm divided by optical encoder+projection gradient norm"
        ),
    }


def audit(case_registry: Path, output: Path, *, max_validation_samples: int, max_train_samples: int) -> dict[str, Any]:
    registry = json.loads(case_registry.read_text(encoding="utf-8"))
    cases = registry.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("case registry must contain a non-empty cases list")
    reports: list[dict[str, Any]] = []
    common = registry.get("common") if isinstance(registry.get("common"), dict) else {}
    for raw_case in cases:
        case = dict(common)
        case.update(raw_case)
        snapshots = case.get("checkpoints") or []
        if not snapshots:
            raise ValueError(f"case {case.get('id')} has no checkpoint snapshots")
        for snapshot in snapshots:
            resolved, _verified, dataset, model, epoch = _load_snapshot(case, snapshot)
            dataset.samples = dataset.samples[:max_validation_samples] if max_validation_samples > 0 else dataset.samples
            loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
            feature_report = _collect_features(model, loader)
            gradient_report = _collect_gradients(model, resolved, dataset, max_train_samples)
            reports.append({
                "case_id": str(case["id"]),
                "snapshot_role": str(snapshot.get("role", "unspecified")),
                "checkpoint_epoch": epoch,
                "feature": feature_report,
                "gradient": gradient_report,
                "resolved_config_sha256": resolved["resolved_config_sha256"],
                "test_seal_status": "sealed",
                "test_accessed": False,
            })
    report = {
        "artifact_type": "feature_competence_audit",
        "schema_version": 1,
        "status": "pass",
        "execution_context": "cloud",
        "validation_sample_cap": max_validation_samples,
        "train_sample_cap": max_train_samples,
        "diagnostic_only": True,
        "scale_domination_threshold": 3.0,
        "severe_domination_threshold": 10.0,
        "reports": reports,
        "test_seal_status": "sealed",
        "test_accessed": False,
        "training_started": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-validation-samples", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=128)
    args = parser.parse_args()
    report = audit(
        args.case_registry,
        args.output,
        max_validation_samples=args.max_validation_samples,
        max_train_samples=args.max_train_samples,
    )
    print(json.dumps({"status": report["status"], "reports": len(report["reports"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
