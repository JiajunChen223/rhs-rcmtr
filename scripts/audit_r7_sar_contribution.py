"""R7 SAR-contribution mask audit (cloud, read-only evaluation).

Loads a finished R7-S2FT checkpoint and evaluates the validation split under
three modality masks: [1,1] full, [1,0] optical-only, [0,1] SAR-only. The
difference full - optical-only quantifies the marginal contribution of the SAR
flow to the R7 architecture. Deterministic: same loader order as training
(shuffle=False), same seeds, per-sample mIoU pairing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from rhs_rcmtr.data import CloudPairedRasterDataset, load_data_manifest
from rhs_rcmtr.engine.entrypoint import model_config
from rhs_rcmtr.models import build_model
from rhs_rcmtr.utils.checkpoint import load_training_checkpoint
from rhs_rcmtr.utils.config import resolve_config
from rhs_rcmtr.utils.initialization import audited_checkpoint_hashes, verify_pretrained_audit
from rhs_rcmtr.utils.run_manifest import load_verified_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--initialization-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--run-manifest-sha256", required=True)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--pretrained-audit", required=True, type=Path)
    parser.add_argument("--pretrained-audit-sha256", required=True)
    parser.add_argument("--code-manifest", required=True, type=Path)
    parser.add_argument("--code-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("full", "optical_only", "sar_only"), default="full")
    args = parser.parse_args()

    resolved = resolve_config(args.runtime_config, args.experiment_config, args.initialization_config)
    verified_run = load_verified_run_manifest(
        args.run_manifest, args.run_manifest_sha256, "validation",
        expected_data_manifest_sha256=args.data_manifest_sha256,
        expected_pretrained_audit_sha256=args.pretrained_audit_sha256,
        expected_code_manifest_sha256=args.code_manifest_sha256,
        expected_resolved_config_sha256=resolved["resolved_config_sha256"],
    )
    initialization = resolved["initialization"]["initialization"]
    audit = verify_pretrained_audit(
        args.pretrained_audit, args.pretrained_audit_sha256, initialization,
    )
    selected = [str(initialization["optical_checkpoint_path"])]
    checkpoint_hashes = audited_checkpoint_hashes(audit, selected)
    payload, samples = load_data_manifest(args.data_manifest, expected_sha256=args.data_manifest_sha256)
    dataset = CloudPairedRasterDataset(
        samples, "validation", authorization=verified_run,
        label_policy=payload.get("label_policy"),
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)
    device = torch.device("cuda")
    model = build_model(model_config(resolved, checkpoint_hashes)).to(device)
    load_training_checkpoint(
        args.checkpoint, expected_sha256=args.checkpoint_sha256, model=model,
        optimizer=None, scheduler=None,
        resolved_config_sha256=resolved["resolved_config_sha256"],
    )
    model.eval()

    from rhs_rcmtr.metrics import mean_iou
    mask_tensor = {"full": None, "optical_only": torch.tensor([[1.0, 0.0]]), "sar_only": torch.tensor([[0.0, 1.0]])}[args.mode]
    # Match the training evaluation exactly: batch-level mIoU (multi-image merged
    # confusion, as _batch_loss does), averaged over batches -- NOT per-image mean.
    batch_scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            optical = batch["optical"].to(device)
            sar = batch["sar"].to(device)
            label = batch["label"].to(device)
            reliability = torch.ones(optical.shape[0], 2, optical.shape[2], optical.shape[3],
                                     device=device, dtype=optical.dtype)
            mask = None
            if mask_tensor is not None:
                mask = mask_tensor.to(device).expand(optical.shape[0], -1)
            output = model(optical, sar, reliability, modality_mask=mask)
            batch_scores.append(float(mean_iou(output["logits"], label, model.config.num_classes)))
    mean = sum(batch_scores) / len(batch_scores)
    receipt = {
        "artifact_type": "r7_sar_contribution_mask_audit",
        "schema_version": 2,
        "mode": args.mode,
        "checkpoint_sha256": args.checkpoint_sha256,
        "n_batches": len(batch_scores),
        "mean_miou": mean,
        "aggregation": "batch_level_miou_averaged_over_batches (matches training _evaluate_loader)",
        "test_accessed": False,
        "note": "optical_only masks the SAR flow ([1,0]); sar_only masks optical ([0,1])."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()