# TASK-011 — Final Test protocol lock + frozen execution (additive)

New files only; extract into the existing `cita2027-phase2` root. Do not change
any accepted artifacts, adapters, calibrators, plans or source snapshots.

**Before running:** This command is an explicit irreversible permission to
inspect 2025–2026 Final Test outcomes **after** metadata-only protocol freeze.
If Phase A fails, no Test target is decoded by this runner. Preserve console
error, do not bypass gates. The existing SPY snapshot must already contain the
2025–2026 sessions; BTC needs access to official Binance 2025/2026 archives.

```powershell
& {
  $ErrorActionPreference = "Stop"
  Expand-Archive -Path "$HOME\Downloads\TASK-011-final-test-lock-execution-patch.zip" -DestinationPath "." -Force -ErrorAction Stop
  $env:PYTHONPATH = @((Resolve-Path .\src).Path,(Resolve-Path .\scripts).Path) -join [IO.Path]::PathSeparator
  & .\.venv\Scripts\python.exe .\scripts\task011_final_test.py run --root . --allow-final-test
  if ($LASTEXITCODE -ne 0) { throw "TASK-011 BLOCKED/FAILED: do not alter accepted artifacts; preserve logs" }
}
```

- Only focused TASK-011 tests run before source preflight.
- Phase A verifies source coverage and publishes immutable SHA-256 lock before
  Phase B reads any realized Final Test target.
- Phase B runs Chronos/Moirai sequentially in distinct frozen venvs, resumes
  without deleting published records and audits continuous calibration.
- The final report is scientific evidence, not a certificate of contamination
  freedom or point-in-time provenance.

See `docs/TASK-011_FINAL_TEST_PROTOCOL.md` for the lock specification.
