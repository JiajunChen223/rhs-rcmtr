"""G1 pre-training gate audit (R6 SAR-DE handoff v2 §4).

Cloud execution: loads the distillation run's last checkpoint, evaluates
SAR-only mode on the validation split, collects per-sample mIoU, computes the
90% one-sided CI via the frozen paired bootstrap, runs the SAR degradation
ladder monotonicity check (spearman <= -0.7), and writes a receipt through the
frozen ``g1_decision`` pure function.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rhs_rcmtr.utils.r6_gates import (  # noqa: E402
    G1_MONOTONIC_SPEARMAN,
    g1_decision,
    paired_bootstrap_ci,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--tries", type=int, default=1)
    parser.add_argument("--pre-train-epochs", type=float, default=None,
                        help="overrides the run record epoch count (default reads run record)")
    return parser


def _build_model_and_loader(args: argparse.Namespace):
    import torch
    from torch.utils.data import DataLoader

    from rhs_rcmtr.data import CloudPairedRasterDataset, load_data_manifest
    from rhs_rcmtr.engine.entrypoint import model_config
    from rhs_rcmtr.models import build_model
    from rhs_rcmtr.utils.checkpoint import load_training_checkpoint
    from rhs_rcmtr.utils.config import resolve_config
    from rhs_rcmtr.utils.initialization import audited_checkpoint_hashes, verify_pretrained_audit
    from rhs_rcmtr.utils.run_manifest import load_verified_run_manifest

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
    if str(initialization.get("sar_initialization_mode")) == "pretrained_croma_sar":
        selected.append(str(initialization["sar_checkpoint_path"]))
    checkpoint_hashes = audited_checkpoint_hashes(audit, selected)
    payload, samples = load_data_manifest(args.data_manifest, expected_sha256=args.data_manifest_sha256)
    dataset = CloudPairedRasterDataset(
        samples, "validation", authorization=verified_run,
        label_policy=payload.get("label_policy"),
    )
    device = torch.device("cuda")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)
    model = build_model(model_config(resolved, checkpoint_hashes)).to(device)
    load_training_checkpoint(
        args.checkpoint, expected_sha256=args.checkpoint_sha256, model=model,
        optimizer=None, scheduler=None,
        resolved_config_sha256=resolved["resolved_config_sha256"],
    )
    return model, loader, device


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run the G1 audit on the cloud (real checkpoints and val split)."""
    from _r6_audit_common import evaluate_loader_per_sample, sar_ladder_miou

    model, loader, device = _build_model_and_loader(args)
    values = evaluate_loader_per_sample(model, loader, device, forward_sar_only=True)
    point_estimate = float(sum(values) / len(values)) * 100.0
    deltas_pp = [value * 100.0 for value in values]
    ci = paired_bootstrap_ci(deltas_pp, n_replicates=2000, seed=args.seed, alpha=0.10)
    ladder = sar_ladder_miou(model, loader, device)
    decision = g1_decision(
        point_estimate=point_estimate, ci90_lower=ci["ci_lower"],
        monotonic_spearman=ladder["spearman"], tries=args.tries,
    )
    receipt = {
        "artifact_type": "r6_g1_pretrain_gate_audit",
        "schema_version": 2,
        "status": decision.status,
        "g1_decision": {
            "status": decision.status,
            "point_estimate": decision.point_estimate,
            "ci90_lower": decision.ci90_lower,
            "monotonic_spearman": decision.monotonic_spearman,
            "monotonic_required": True,
            "thresholds": {"point": 20.0, "ci90": 16.0, "soft_floor": 15.0, "retry_target": 18.0,
                           "monotonic": G1_MONOTONIC_SPEARMAN},
            "tries": decision.tries,
            "reason": decision.reason,
        },
        "sar_ladder": ladder,
        "bootstrap": ci,
        "execution": {"mode": "sar_only", "checkpoint_sha256": args.checkpoint_sha256,
                      "pre_train_epochs": args.pre_train_epochs},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def main() -> None:
    args = build_arg_parser().parse_args()
    receipt = evaluate(args)
    print(f"G1 status: {receipt['g1_decision']['status']}")


if __name__ == "__main__":
    main()