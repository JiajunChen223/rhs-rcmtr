from scripts.audit_conditional_route_elasticity import _monotonic_fraction, _trapz


def test_conditional_route_elasticity_helpers_are_directional() -> None:
    assert _monotonic_fraction([[0.8, 0.6, 0.4], [0.7, 0.7, 0.5]]) == 1.0
    assert _monotonic_fraction([[0.8, 0.9, 0.4]]) == 0.0
    assert _trapz([1.0, 0.5, 0.0], (0.0, 0.5, 1.0)) == 0.5
