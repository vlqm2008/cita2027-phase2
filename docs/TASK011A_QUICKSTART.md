# TASK-011A — fail-closed checkpoint identity schema hotfix

**Reason:** TASK-011 `accepted_anchors()` directly compared accepted TASK-009
`checkpoint_id` with TASK-011 probe's `id`, falsely reporting drift.
Both original local frozen checkpoint SHA probes passed on Windows.

## Changed surface

Only `src/cita2027/finaltest/lock.py` (a new TASK-011 module, not a previously
accepted adapter/calibrator/model) is modified, *after* checking the exact
original SHA `94dbac2dc60656c055150cc7ca78e5a106465f4d31eb17ab9554cb7b00fba91f`.
The new SHA must equal `b3271354fa853e372b04c205ffa097bbc54e284997147c48cd5bb8f6b37ce00b`.
The script refuses to modify an unrecognized or already pre-test-locked source.
Run it idempotently; do not re-extract the original TASK-011 ZIP afterward.

## Windows PowerShell (ONE block, repo root)

```powershell
& {
    $ErrorActionPreference = "Stop"
    Expand-Archive -Path "$HOME\Downloads\TASK-011A-checkpoint-identity-schema-hotfix.zip" -DestinationPath "." -Force -ErrorAction Stop
    & .\.venv\Scripts\python.exe .\scripts\task011_fix_checkpoint_schema.py --root .
    if ($LASTEXITCODE -ne 0) { throw "TASK011A HOTFIX FAILED: STOP" }
    $env:PYTHONPATH = @((Resolve-Path .\src).Path,(Resolve-Path .\scripts).Path) -join [IO.Path]::PathSeparator
    & .\.venv\Scripts\python.exe -m pytest -q .\tests\test_task011_identity_schema_hotfix.py
    if ($LASTEXITCODE -ne 0) { throw "TASK011A REGRESSION TESTS FAILED: STOP" }
    & .\.venv\Scripts\python.exe .\scripts\task011_final_test.py run --root . --allow-final-test
    if ($LASTEXITCODE -ne 0) { throw "TASK011 PHASE A/B BLOCKED: PRESERVE LOG; DO NOT BYPASS" }
}
```

The same original TASK-011 orchestrator is used; it may still stop at other
legitimate Phase-A integrity/source gates. It must not proceed to Phase B
before the pre-test manifest has been published and hash verified. No
Development/Validation inference is run again. Provenance flags remain false.
