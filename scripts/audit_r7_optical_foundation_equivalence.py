"""Cloud audit: legacy DINO patch tokens vs R7 lock-step optical foundation path.

This audit requires the real local DINOv2 repository and the hash-bound
checkpoint. It never accesses labels or the sealed test split. The reference is
read from the legacy adapter's frozen DINO model directly so rectangular patch
grids are audited without the legacy wrapper's square-grid reshape assumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rhs_rcmtr.models.backbones import DinoV2B14Adapter, dino_input_grid
from rhs_rcmtr.models.s2ft import S2FTConfig, SharedDinoSensorBackbone


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--height", type=int, default=111)
    parser.add_argument("--width", type=int, default=117)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    actual_sha = _sha256(args.checkpoint)
    if actual_sha != args.checkpoint_sha256:
        raise ValueError("R7 optical-equivalence audit checkpoint SHA256 mismatch")
    if args.height < 14 or args.width < 14:
        raise ValueError("audit image dimensions must be at least one DINO patch")

    torch.manual_seed(args.seed)
    legacy = DinoV2B14Adapter(args.dino_repo, args.checkpoint, args.checkpoint_sha256).eval()
    config = S2FTConfig()
    new = SharedDinoSensorBackbone.from_official(
        args.dino_repo, args.checkpoint, args.checkpoint_sha256, config
    ).eval()

    optical = torch.randn(2, 3, args.height, args.width, dtype=torch.float32)
    sar = torch.zeros(2, 1, args.height, args.width, dtype=torch.float32)
    sar_off = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    with torch.no_grad():
        aligned = dino_input_grid(optical, 14)
        reference_tokens = legacy.model.forward_features(aligned)["x_norm_patchtokens"]
        grid = (aligned.shape[-2] // 14, aligned.shape[-1] // 14)
        expected = reference_tokens.transpose(1, 2).reshape(
            reference_tokens.shape[0], reference_tokens.shape[2], *grid
        )
        optical_features, sar_features, _sar, _om, _sm = new(optical, sar, sar_off)
        observed = optical_features[max(config.adapter_layers)]

    if expected.shape != observed.shape:
        raise AssertionError(
            f"R7 optical foundation shape mismatch: old={tuple(expected.shape)} new={tuple(observed.shape)}"
        )
    max_abs = float((expected - observed).abs().max())
    mean_abs = float((expected - observed).abs().mean())
    sar_leakage = max(float(feature.abs().max()) for feature in sar_features.values())
    status = "pass" if max_abs <= args.atol and sar_leakage == 0.0 else "fail"
    receipt = {
        "artifact_type": "r7_optical_foundation_equivalence_audit",
        "status": status,
        "seed": args.seed,
        "checkpoint_sha256": actual_sha,
        "aligned_grid": list(grid),
        "old_shape": list(expected.shape),
        "new_shape": list(observed.shape),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "atol": args.atol,
        "sar_off_max_abs_feature": sar_leakage,
        "test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "pass":
        raise AssertionError(json.dumps(receipt, sort_keys=True))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
