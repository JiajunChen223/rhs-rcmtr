"""q-permutation detectability audit (R6 SAR-DE handoff v2 §4).

Cloud execution: loads the formal checkpoint, evaluates the validation split
twice (original vs swapped [u_o, q_s] reliability channels, optionally under
the frozen degradation ladder), computes per-sample mIoU deltas, boots them
(Bonferroni alpha / n_families) and writes a receipt through the frozen
``q_permutation_decision`` pure function.
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
    QPERM_MIN_ABS_DROP_PP,
    paired_bootstrap_ci,
    q_permutation_decision,
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
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--n-families", type=int, default=8,
                        help="Bonferroni families (7 degradation + 1 joint); alpha is divided by this")
    parser.add_argument("--degraded-condition", action="store_true",
                        help="inject the frozen degradation ladder while auditing "
                             "(mandatory: clean evidence is flat and cannot detect swaps)")
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
    """Run the q-permutation audit on the cloud (real checkpoints and val split)."""
    from _r6_audit_common import evaluate_loader_per_sample

    model, loader, device = _build_model_and_loader(args)
    original = evaluate_loader_per_sample(model, loader, device)
    permuted = evaluate_loader_per_sample(model, loader, device, reliability_permute=True)
    if len(original) != len(permuted) or not original:
        raise ValueError("per-sample IoU lists missing or length-mismatched")
    deltas_pp = [(float(perm) - float(orig)) * 100.0 for perm, orig in zip(permuted, original)]
    alpha = 0.05 / max(int(args.n_families), 1)
    ci = paired_bootstrap_ci(deltas_pp, n_replicates=2000, seed=args.seed, alpha=alpha)
    decision = q_permutation_decision(delta_mean_pp=ci["mean"], ci_upper_pp=ci["ci_upper"])
    receipt = {
        "artifact_type": "r6_q_permutation_detectability_audit",
        "schema_version": 2,
        "status": "pass" if decision.detectable else "inconclusive",
        "q_permutation": {
            "detectable": decision.detectable,
            "delta_mean_pp": decision.delta_mean_pp,
            "ci_upper_pp": decision.ci_upper_pp,
            "min_abs_drop_pp": QPERM_MIN_ABS_DROP_PP,
            "degraded_condition": bool(args.degraded_condition),
            "bonferroni_alpha": alpha,
            "reason": decision.reason,
        },
        "bootstrap": ci,
        "execution": {"checkpoint_sha256": args.checkpoint_sha256},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def main() -> None:
    args = build_arg_parser().parse_args()
    receipt = evaluate(args)
    print(f"q-permutation detectable: {receipt['q_permutation']['detectable']}")


if __name__ == "__main__":
    main()