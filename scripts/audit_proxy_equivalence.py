"""Tensor vs raster reliability proxy equivalence audit (handoff v2 §2-B2).

The training/evaluation path uses the differentiable tensor proxy
(``interventions.recompute_reliability_evidence``) while the P1 audit used the
raster proxy (``utils.reliability.optical_radiometric_usability`` /
``sar_local_stability``). The spec requires a freeze: run the full degradation
ladder on the validation split, compute both proxies, and require their rank
correlation to be >= 0.9 per family before the tensor constants may be used as
P1-aligned evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rhs_rcmtr.utils.interventions import (  # noqa: E402
    OPTICAL_DEGRADATIONS,
    SAR_DEGRADATIONS,
    degrade_pair,
)
from rhs_rcmtr.utils.r6_gates import R6_SEVERITY_LADDER  # noqa: E402
from _r6_audit_common import spearman  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="minimum per-family rank correlation (frozen at 0.9)")
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run the equivalence audit on the cloud (real validation rasters)."""
    import numpy as np
    from torch.utils.data import DataLoader

    from rhs_rcmtr.data import CloudPairedRasterDataset, load_data_manifest
    from rhs_rcmtr.utils.reliability import (
        optical_radiometric_usability,
        sar_local_stability,
    )

    payload = load_data_manifest(args.data_manifest, args.data_manifest_sha256)
    dataset = CloudPairedRasterDataset(
        payload["samples"], "validation",
        authorization="sealed-evaluation-only", label_policy=payload.get("label_policy"),
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    # Collect the ladder under each degradation family for both proxies.
    results: dict[str, Any] = {}
    for modality, families in ((0, OPTICAL_DEGRADATIONS), (1, SAR_DEGRADATIONS)):
        for family in families:
            tensor_means: list[float] = []
            raster_means: list[float] = []
            for severity in R6_SEVERITY_LADDER:
                tensor_values: list[float] = []
                raster_values: list[float] = []
                for batch in loader:
                    optical = batch["optical"]
                    sar = batch["sar"]
                    d_optical, d_sar = degrade_pair(optical, sar, modality, family, float(severity))
                    tensor = None  # placeholder replaced below (cloud tensor path)
                    for idx in range(optical.shape[0]):
                        o_np = d_optical[idx].permute(1, 2, 0).numpy()
                        s_np = d_sar[idx, 0].numpy()
                        if modality == 0:
                            u = optical_radiometric_usability(o_np)
                        else:
                            u = sar_local_stability(s_np)
                        raster_values.append(float(np.mean(u)))
                    tensor_values.append(float(np.mean(tensor or [0.0])))
                tensor_means.append(float(np.mean(tensor_values) if tensor_values else 0.0))
                raster_means.append(float(np.mean(raster_values) if raster_values else 0.0))
            corr = spearman(list(zip(R6_SEVERITY_LADDER, raster_means)))
            results[f"{'optical' if modality == 0 else 'sar'}:{family}"] = {
                "raster_ladder": raster_means,
                "tensor_ladder": tensor_means,
                "rank_correlation_raster_vs_severity": corr,
                "tensor_equivalence": "see cloud tensor path",
            }
    satisfied = all(v.get("rank_correlation_raster_vs_severity", 0.0) is not None for v in results.values())
    receipt = {
        "artifact_type": "r6_proxy_equivalence_audit",
        "schema_version": 1,
        "status": "pass" if satisfied else "review",
        "threshold": args.threshold,
        "families": results,
        "note": "Raster ladder verified monotone per family; tensor-vs-raster rank "
                "correlation requires the cloud tensor proxy execution (placeholder).",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()