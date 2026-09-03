"""Create the R6 pre-training anchor from a distillation run (handoff v2 §2-A2).

Run 1 (`distill_pretrain`) produces a normal training checkpoint (`last.pt`);
this script converts its common trainable tensors into the anchor schema so
that runs 2 (r6 baseline / R6-C1-EVSCRT) can load it as their
``common_init_anchor_path``. Metadata (seeds, resolved config hash) is taken
from the run record so the anchor is fully traceable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhs_rcmtr.engine.entrypoint import model_config  # noqa: E402
from rhs_rcmtr.models import build_model  # noqa: E402
from rhs_rcmtr.utils.checkpoint import load_training_checkpoint  # noqa: E402
from rhs_rcmtr.utils.config import resolve_config  # noqa: E402
from rhs_rcmtr.utils.initialization_anchor import save_common_init_anchor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--initialization-config", required=True, type=Path)
    parser.add_argument("--pretrained-audit", required=True, type=Path)
    parser.add_argument("--pretrained-audit-sha256", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path, help="distillation run last.pt")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--run-record", required=True, type=Path, help="distillation run run_record.json")
    parser.add_argument("--output", required=True, type=Path, help="output anchor .pt path")
    args = parser.parse_args()

    resolved = resolve_config(args.runtime_config, args.experiment_config, args.initialization_config)
    initialization = resolved["initialization"]["initialization"]
    from rhs_rcmtr.utils.initialization import audited_checkpoint_hashes, verify_pretrained_audit
    audit = verify_pretrained_audit(
        args.pretrained_audit, args.pretrained_audit_sha256, initialization,
    )
    selected = [str(initialization["optical_checkpoint_path"])]
    if str(initialization.get("sar_initialization_mode")) == "pretrained_croma_sar":
        selected.append(str(initialization["sar_checkpoint_path"]))
    checkpoint_hashes = audited_checkpoint_hashes(audit, selected)
    run_record = json.loads(args.run_record.read_text(encoding="utf-8"))
    common_init_seed = int(run_record.get("common_init_seed", 0))
    data_order_seed = int(run_record.get("data_order_seed", 0))
    augmentation_seed = int(run_record.get("augmentation_seed", 0))
    model = build_model(model_config(resolved, checkpoint_hashes))
    # The distillation run's own resolved hash is expected in the checkpoint.
    restored_epoch = load_training_checkpoint(
        args.checkpoint, expected_sha256=args.checkpoint_sha256, model=model,
        optimizer=None, scheduler=None,
        resolved_config_sha256=run_record.get("resolved_config_sha256", resolved["resolved_config_sha256"]),
    )
    record = save_common_init_anchor(
        args.output,
        model,
        common_init_seed=common_init_seed,
        resolved_config_sha256=run_record.get("resolved_config_sha256", resolved["resolved_config_sha256"]),
        data_order_seed=data_order_seed,
        augmentation_seed=augmentation_seed,
    )
    record["distill_checkpoint_sha256"] = args.checkpoint_sha256
    record["distill_epoch"] = restored_epoch
    record["distill_run_record"] = str(args.run_record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())