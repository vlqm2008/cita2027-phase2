# TASK-009 — BTC Evidence Freeze & Cross-Market Synthesis

Additive repository patch. It adds `scripts/task009_freeze_synthesis.py`, `tests/test_task009_freeze.py`, `docs/TASK-009_RESULTS_DISCUSSION.md`, and three `paper_tables/*.csv` exports. No accepted source, calibration engine, model adapter, SPY artifacts, or forecast cache is overwritten.

Windows execution (from `C:\AI-Workspace\projects\cita2027-phase2`):

```powershell
& {
    $ErrorActionPreference = "Stop"
    Expand-Archive -Path "$HOME\Downloads\TASK-009-btc-evidence-freeze-synthesis-patch.zip" -DestinationPath "." -Force -ErrorAction Stop
    $env:PYTHONPATH = @((Resolve-Path .\src).Path, (Resolve-Path .\scripts).Path) -join [IO.Path]::PathSeparator
    & .\.venv\Scripts\python.exe -m pytest -q .\tests\test_task009_freeze.py
    if ($LASTEXITCODE -ne 0) { throw "TASK-009 focused tests FAIL" }
    & .\.venv\Scripts\python.exe .\scripts\task009_freeze_synthesis.py --root .
    if ($LASTEXITCODE -ne 0) { throw "TASK-009 evidence freeze FAIL — preserve all accepted artifacts" }
}
```

The command only hashes and reads existing accepted data and forecasts, validates identities with the existing TASK-008B/008E verifiers, verifies every one of 4,384 BTC forecast JSON records, and publishes write-once index and CSV exports under `artifacts/evidence/task009/`. It does **not** run inference, compute new calibrations, access Final Test, or rewrite SPY artifacts. A complete source cache is required: fail-closed if any file is absent or changed. `artifacts/evidence/task007/23aac3be...json` must already exist and have the accepted digest.

Expected success status: `BTC_EVIDENCE_INDEX_VERIFIED_NOT_SCIENTIFIC_CLEARANCE`, with `forecast_records_verified:4384` and a content-addressed manifest path. Until that Windows log is received, deep BTC evidence freeze has **not** been executed or accepted.

CIs in the exported contrast table are sourced from accepted source report JSON, not generated. SPY Development only reported selected paired normalized-Rolling CI at one bootstrap setting; unreported B10/B40 and variable-gamma paired CIs are blank (never imputed). Every within-period contrast uses the exact same origin set for paired methods; cross-market and cross-period comparisons remain descriptive.
