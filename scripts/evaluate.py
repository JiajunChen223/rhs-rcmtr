"""Single configuration-driven evaluation entry point."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhs_rcmtr.engine.entrypoint import execute
from rhs_rcmtr.utils.test_seal import assert_test_access_allowed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--initialization-config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--run-manifest-sha256", default="")
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--data-manifest-sha256", default="")
    parser.add_argument("--pretrained-audit", type=Path)
    parser.add_argument("--pretrained-audit-sha256", default="")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--code-manifest", type=Path)
    parser.add_argument("--code-manifest-sha256", default="")
    parser.add_argument("--sar-only-external", action="store_true")
    args = parser.parse_args()
    assert_test_access_allowed({"execution_scale": "smoke", "test_seal_status": "sealed"}, "validation")
    result = execute(runtime_config=args.runtime_config, experiment_config=args.experiment_config,
                     initialization_config=args.initialization_config, run_id=args.run_id,
                     mode="sar_only_external" if args.sar_only_external else "evaluate",
                     run_manifest=args.run_manifest, run_manifest_sha256=args.run_manifest_sha256,
                     data_manifest=args.data_manifest, data_manifest_sha256=args.data_manifest_sha256,
                     pretrained_audit=args.pretrained_audit, pretrained_audit_sha256=args.pretrained_audit_sha256,
                     checkpoint=args.checkpoint, checkpoint_sha256=args.checkpoint_sha256,
                     code_manifest=args.code_manifest, code_manifest_sha256=args.code_manifest_sha256)
    print(json.dumps(result, sort_keys=True))
