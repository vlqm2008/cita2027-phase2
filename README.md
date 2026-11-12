# TASK-008E — additive patch

Files added (no existing file overwritten):

- `src/cita2027/btc/validation.py` — preflight, read-only accepted-state audit, resumable validation forecast cache and CPU calibration.
- `scripts/task008e_btc_validation.py` — single CLI, runs 10 focused tests, preflight, two separate virtual-env workers sequentially, CPU report and audit.
- `tests/test_task008e_btc_validation.py` — scoped CPU tests.
- `docs/task008e_btc_validation_lock.md` — protocol and limitations.

Use only in the **existing** `cita2027-phase2` repository with TASK-008B and TASK-008D files already integrated, and existing `.venv` / `.venv-moirai`. No changes to frozen adapters, calibrators or SPY artifacts.

Development/Validation report hashes are not precomputed here; they are minted once the user's Windows execution succeeds. This build was checked only in a Linux container using **synthetic** forecast records, not actual Windows/GPU inference.
