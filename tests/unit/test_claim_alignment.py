"""Claim-alignment CI guard for the R6 SAR-DE claim registry (CLAM).

Iterates ``CLAIM_ALIGNMENT_TABLE`` and fails unless each entry's
``impl_anchor`` resolves to a real symbol in the codebase and its ``unit_test``
name exists in the test tree. This is the machine-enforced half of the
spec-vs-impl alignment contract; the receipt half is
``scripts/audit_claim_impl_alignment.py``.
"""

from __future__ import annotations

import importlib
import pathlib
import re

import pytest

from rhs_rcmtr.utils.contracts import CLAIM_ALIGNMENT_TABLE

CODE_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _resolve_anchor(anchor: str) -> bool:
    # Format: "module.path:Attribute.chain" — the module part must import and
    # every attribute after the colon must resolve.
    module_name, _, attrs = anchor.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False
    target: object = module
    for part in attrs.split("."):
        if not part:
            return False
        if not hasattr(target, part):
            return False
        target = getattr(target, part)
    return True


def _collect_test_names() -> set[str]:
    names: set[str] = set()
    for path in (CODE_ROOT / "tests").rglob("test_*.py"):
        text = path.read_text(encoding="utf-8")
        names.update(re.findall(r"^def (test_\w+)", text, re.MULTILINE))
    return names


def test_claim_alignment_impl_anchors_resolve() -> None:
    unresolved = [
        row["claim_id"] for row in CLAIM_ALIGNMENT_TABLE
        if not _resolve_anchor(row["impl_anchor"])
    ]
    assert not unresolved, f"CLAM impl anchors do not resolve: {unresolved}"


def test_claim_alignment_unit_tests_exist() -> None:
    test_names = _collect_test_names()
    missing = [
        row["unit_test"] for row in CLAIM_ALIGNMENT_TABLE
        if row["unit_test"] not in test_names
    ]
    assert not missing, f"CLAM unit tests missing from the test tree: {missing}"


def test_claim_alignment_fields_complete() -> None:
    for row in CLAIM_ALIGNMENT_TABLE:
        assert set(row) == {"claim_id", "claim", "impl_anchor", "unit_test", "audit_field"}
        assert row["claim_id"] and row["claim"] and row["audit_field"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))