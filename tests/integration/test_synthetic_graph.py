from rhs_rcmtr.engine import run_synthetic_step
from rhs_rcmtr.models import ModelConfig


def test_baseline_and_candidate_use_complete_graph() -> None:
    baseline = run_synthetic_step(ModelConfig(), train=True)
    candidate = run_synthetic_step(ModelConfig(enabled_mechanism_ids=("R5-C1-OA-SCRT",)), train=True)
    assert baseline["loss"] > 0
    assert candidate["loss"] > 0
