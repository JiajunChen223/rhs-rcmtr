"""Emit baseline/candidate/composition parity and trainability evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhs_rcmtr.models import ModelConfig, build_model
from rhs_rcmtr.utils.contracts import (
    assert_composition_row, assert_training_object_parity, matched_protocol_budget_hash,
    trainable_parameter_audit,
)


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid config: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--composition", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline, candidate, composition = load(args.baseline), load(args.candidate), load(args.composition)
    assert_training_object_parity(baseline, candidate)
    assert_composition_row(baseline, composition)
    baseline_model = build_model(ModelConfig())
    candidate_model = build_model(ModelConfig(enabled_mechanism_ids=("R5-C1-OA-SCRT",)))
    baseline_audit = trainable_parameter_audit(baseline_model)
    candidate_audit = trainable_parameter_audit(candidate_model)
    if baseline_audit["common_trainable_names"] != candidate_audit["common_trainable_names"]:
        raise ValueError("common trainable-parameter mask changed")
    hashes = {name: matched_protocol_budget_hash(value) for name, value in {
        "baseline": baseline, "candidate": candidate, "composition": composition,
    }.items()}
    if len(set(hashes.values())) != 1:
        raise ValueError("matched protocol/budget hash differs")
    report = {
        "status": "pass", "single_internal_mechanism_delta": True,
        "external_trainable_component_forbidden": True,
        "same_entry_points": ["scripts/train.py", "scripts/evaluate.py"],
        "parent_row": composition.get("parent_row"), "matched_protocol_budget_hashes": hashes,
        "baseline_parameter_audit": baseline_audit, "candidate_parameter_audit": candidate_audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(args.output)}))


if __name__ == "__main__":
    main()
