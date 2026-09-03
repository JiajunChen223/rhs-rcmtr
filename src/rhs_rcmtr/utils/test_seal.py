"""Runtime test-set seal shared by training and evaluation entry points."""


class TestSealViolation(RuntimeError):
    __test__ = False

    pass


def assert_test_access_allowed(run_manifest: dict[str, object], requested_split: str) -> None:
    split = requested_split.strip().casefold()
    if split not in {"train", "validation", "val", "test"}:
        raise TestSealViolation(f"unknown split: {requested_split}")
    if split == "test" and (
        run_manifest.get("execution_scale") != "final_test"
        or run_manifest.get("test_seal_status") != "final_test"
    ):
        raise TestSealViolation("test split is sealed until the FINAL_TEST gate")
