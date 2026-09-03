"""Apply the predeclared two-gate survivor rule to a route-behaviour audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate(payload: dict[str, Any], *, baseline_miou: float) -> dict[str, Any]:
    if payload.get("status") != "pass":
        raise ValueError("route-behaviour audit must have status=pass")
    if payload.get("test_accessed") is not False or payload.get("test_seal_status") != "sealed":
        raise ValueError("survivor validation requires test_accessed=false and sealed test status")
    metrics = payload.get("metrics")
    behavior = payload.get("route_behavior")
    validity = payload.get("mechanism_validity")
    if not isinstance(metrics, dict) or not isinstance(behavior, dict) or not isinstance(validity, dict):
        raise ValueError("route-behaviour audit is missing metrics, route_behavior or mechanism_validity")
    fusion = metrics.get("fusion")
    if not isinstance(fusion, dict) or not isinstance(fusion.get("mIoU"), (int, float)):
        raise ValueError("route-behaviour audit is missing fusion mIoU")
    candidate_miou = float(fusion["mIoU"])
    delta_pp = (candidate_miou - float(baseline_miou)) * 100.0
    reliability_expected = bool(validity.get("reliability_router_expected"))
    mechanism_checks = {
        "counterfactual_direction_ok": bool(validity.get("counterfactual_direction_ok")),
        "no_route_collapse": bool(validity.get("no_route_collapse")),
        "sar_survival": bool(validity.get("sar_survival")),
    }
    if reliability_expected:
        mechanism_valid = all(mechanism_checks.values()) and float(behavior.get("reliability_route_spearman", 0.0)) > 0.0
    else:
        mechanism_valid = True
    task_gate = delta_pp >= -0.3
    decision = "survivor" if task_gate and mechanism_valid else "does_not_survive"
    return {
        "artifact_type": "mechanism_survivor_validation",
        "schema_version": 1,
        "status": "pass",
        "run_id": payload.get("run_id"),
        "mechanism_id": payload.get("mechanism_id"),
        "baseline_mIoU": float(baseline_miou),
        "candidate_mIoU": candidate_miou,
        "delta_mIoU_pp": delta_pp,
        "task_gate": {"threshold_delta_mIoU_pp": -0.3, "pass": task_gate},
        "mechanism_gate": {
            "reliability_router_expected": reliability_expected,
            "checks": mechanism_checks,
            "route_reliability_spearman": float(behavior.get("reliability_route_spearman", 0.0)),
            "pass": mechanism_valid,
        },
        "decision": decision,
        "test_accessed": False,
        "test_seal_status": "sealed",
        "scientific_result": True,
        "note": "The decision requires both task performance and mechanism validity; a single mIoU value is insufficient.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--baseline-miou", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.audit.read_text(encoding="utf-8"))
    result = validate(payload, baseline_miou=args.baseline_miou)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
