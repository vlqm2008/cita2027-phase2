"""Unit test covering the Scientific Integrity Audit Suite."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_scientific_integrity import ScientificIntegrityAuditor


def test_scientific_integrity_suite():
    auditor = ScientificIntegrityAuditor(root=PROJECT_ROOT)
    df = auditor.run_full_audit()
    assert not df.empty
    assert len(df) == 5
    # All 5 audits must pass with 100% compliance
    assert df["passed"].all(), f"Failed audits: {df[~df['passed']]}"
