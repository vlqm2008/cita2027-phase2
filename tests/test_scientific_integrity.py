"""Unit test covering the Scientific Integrity Audit Suite."""
from __future__ import annotations

from pathlib import Path
import pytest

from scripts.verify_scientific_integrity import ScientificIntegrityAuditor

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_scientific_integrity_suite():
    auditor = ScientificIntegrityAuditor(root=PROJECT_ROOT)
    df = auditor.run_full_audit()
    assert not df.empty
    assert len(df) == 5
    # All 5 audits must pass with 100% compliance
    assert df["passed"].all(), f"Failed audits: {df[~df['passed']]}"
