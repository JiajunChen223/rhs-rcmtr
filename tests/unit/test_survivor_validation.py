from pathlib import Path
import importlib.util


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_mechanism_survivor.py"
_SPEC = importlib.util.spec_from_file_location("survivor", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _audit(miou: float = 0.520) -> dict:
    return {
        "status": "pass",
        "run_id": "x",
        "mechanism_id": "R5-C1-OA-SCRT",
        "metrics": {"fusion": {"mIoU": miou}},
        "route_behavior": {"reliability_route_spearman": 0.4},
        "mechanism_validity": {
            "reliability_router_expected": False,
            "counterfactual_direction_ok": True,
            "no_route_collapse": True,
            "sar_survival": True,
        },
        "test_accessed": False,
        "test_seal_status": "sealed",
    }


def test_two_gate_survivor_uses_task_gate_for_sar_null_mechanism() -> None:
    # R5-C1-OA-SCRT is validated through SAR null hypotheses, not the
    # historical reliability-route family, so the task gate drives the decision.
    result = _MODULE.validate(_audit(), baseline_miou=0.515)
    assert result["decision"] == "survivor"
    below_task_gate = _audit(miou=0.500)
    assert _MODULE.validate(below_task_gate, baseline_miou=0.515)["decision"] == "does_not_survive"
