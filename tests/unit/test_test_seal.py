import pytest

from rhs_rcmtr.utils.test_seal import TestSealViolation, assert_test_access_allowed


@pytest.mark.parametrize("scale", ["smoke", "baseline", "innovation_screening_seed0", "strengthening_seed0"])
def test_test_is_rejected_before_final(scale: str) -> None:
    with pytest.raises(TestSealViolation):
        assert_test_access_allowed({"execution_scale": scale, "test_seal_status": "sealed"}, "test")


def test_final_test_is_allowed_only_when_explicit() -> None:
    assert_test_access_allowed({"execution_scale": "final_test", "test_seal_status": "final_test"}, "test")
