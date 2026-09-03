"""Create the cloud-only tensor-level common initialization anchor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhs_rcmtr.models import ModelConfig, build_model
from rhs_rcmtr.utils.config import resolve_config
from rhs_rcmtr.utils.initialization import audited_checkpoint_hashes, verify_pretrained_audit
from rhs_rcmtr.utils.initialization_anchor import save_common_init_anchor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--initialization-config", required=True, type=Path)
    parser.add_argument("--pretrained-audit", required=True, type=Path)
    parser.add_argument("--pretrained-audit-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    resolved = resolve_config(args.runtime_config, args.experiment_config, args.initialization_config)
    experiment = resolved["experiment"]
    initialization = resolved["initialization"]["initialization"]
    audit = verify_pretrained_audit(
        args.pretrained_audit,
        args.pretrained_audit_sha256,
        initialization,
    )
    selected = [str(initialization["optical_checkpoint_path"])]
    if str(initialization.get("sar_initialization_mode")) == "pretrained_croma_sar":
        selected.append(str(initialization["sar_checkpoint_path"]))
    checkpoint_hashes = audited_checkpoint_hashes(audit, selected)
    common_init_seed = int(experiment.get("common_init_seed", experiment.get("seed", 0)))
    data_order_seed = int(experiment.get("data_order_seed", experiment.get("seed", 0)))
    augmentation_seed = int(experiment.get("augmentation_seed", experiment.get("seed", 0)))
    torch.manual_seed(common_init_seed)
    torch.cuda.manual_seed_all(common_init_seed)
    model = build_model(
        ModelConfig(
            sar_channels=2 if initialization.get("sar_initialization_mode") == "pretrained_croma_sar" else 1,
            enabled_mechanism_ids=(),
            backbone_mode=str(resolved["runtime"]["backbone_mode"]),
            optical_repo_path=str(initialization.get("optical_repo_path", "")),
            optical_checkpoint_path=str(initialization.get("optical_checkpoint_path", "")),
            optical_checkpoint_sha256=checkpoint_hashes.get(str(initialization.get("optical_checkpoint_path", "")), ""),
            sar_repo_path=str(initialization.get("sar_repo_path", "")),
            sar_checkpoint_path=str(initialization.get("sar_checkpoint_path", "")),
            sar_checkpoint_sha256=checkpoint_hashes.get(str(initialization.get("sar_checkpoint_path", "")), ""),
            sar_initialization_mode=str(initialization.get("sar_initialization_mode", "random_init_single_band")),
            sar_encoder_variant=str(experiment.get("sar_encoder_variant", "v1")),
        )
    )
    record = save_common_init_anchor(
        args.output,
        model,
        common_init_seed=common_init_seed,
        resolved_config_sha256=resolved["resolved_config_sha256"],
        data_order_seed=data_order_seed,
        augmentation_seed=augmentation_seed,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
