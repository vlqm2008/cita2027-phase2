# TASK-004B.1 — Full development pre-run audit (planning ONLY)

No full inference is performed by `plan_full_development.py`. Existing Phase 2,
Pre-Phase 3, and TASK-004A files stay untouched.

## Data boundary

Use `load_processed`, `make_window`, and the frozen `configs/data.yaml` split.
Eligible `i` satisfies `i >= 256`, `i < len(df)-1`, and **both** `split[i]`
and `split[i+1]` are development. Thus a development observation requiring a
warm-up *origin* is NOT included. The first valid development origin **does**
use warm-up *context*, which is causal. If the data audit still reports 1,259
development rows starting at row 3272, expect 1,258 pairs, origin indices
3272..4529 and target indices 3273..4530. The immutable plan and exact SHA can
only be certified using the local processed snapshot, not this documentation.

Never use test, validation or revised post-origin observations as inference inputs.
A daily return ending at `i` is only available after session `i` closes.

## Locked protocol and identities

- Context=256, horizon=1, 100 samples, empirical linear quantiles 0.1/0.5/0.9.
- Origin seed = 2027 + row index; first TWO origins replayed with tolerance 1e-5.
- Same frozen adapter and weight files, same source and checkpoint SHA audits.
- All full-plan rows carry a context SHA, dates, indices and seed; realized
  targets are intentionally excluded from the plan file.
- Pilot forecasts must not be imported or linked into the full run. The **full**
  plan uses `kind=full_development_not_pilot`, distinct version, path and SHA.
- Contamination-free and point-in-time adjusted-price flags remain FALSE.

## Run steps (Windows PowerShell, no GPU inference)

```powershell
$C = '.\.venv\Scripts\python.exe'
$M = '.\.venv-moirai\Scripts\python.exe'
$env:PYTHONPATH = (Resolve-Path .\src).Path
& $C -m pytest -q
& $C .\scripts\plan_full_development.py --mode plan --root . --config .\configs\smoke.yaml --checkpoint-audit .\artifacts\checkpoints\hf_audit.json
& $C .\scripts\plan_full_development.py --mode preflight --model chronos --root . --config .\configs\smoke.yaml --checkpoint-audit .\artifacts\checkpoints\hf_audit.json
& $M .\scripts\plan_full_development.py --mode preflight --model moirai --root . --config .\configs\smoke.yaml --checkpoint-audit .\artifacts\checkpoints\hf_audit.json
& $C .\scripts\plan_full_development.py --mode verify-plan --root . --config .\configs\smoke.yaml --checkpoint-audit .\artifacts\checkpoints\hf_audit.json
```

Capture all outputs and SHA. Preflight is **planning only**: it checks the
actual local checkpoint bytes via the locked HF audit, runs a small CUDA matrix
operation, checks installed backend dependency and available disk (>1 GiB), and
refuses nonempty full-run destination paths. Disk threshold is policy, not a
measured model/cache requirement. This is NOT final environment clearance or
proof of contamination-free evaluation.

## TASK-004B.2 execution contract (NOT YET IMPLEMENTED)

B.2 must implement a separate `--mode run` and `--mode audit` using the SAME
plan SHA and same `make_window`/existing adapters, without changing TASK-004A.
Each model uses an exclusive per-run lock, a sequence of batches (default 100
origins); write a record per origin with fsync and same-volume no-overwrite
atomic publish. A crash before a record publication leaves only an ignored
staging file; a crash after publication is verified before resume. No combined
CSV or completed manifest until ALL records pass validation. On restart:
check expected plan SHA/checkpoint/data hashes/seed for existing records;
reject duplicates/unexpected records; never re-run or modify validated records.
If CSV exists but manifest does not, reconstruct CSV bytes from the full set of
records and only then finalize. A broken record is an ERROR, not a cache miss.
The completed-manifest state must be distinguishable from `PILOT_COMPLETE_UNAUDITED`.

Until B.2 code and tests exist, do NOT issue GPU run commands: a planning-only
CLI cannot safely execute/resume full inference. In B.2, the per-env commands
should have the following shape, but remain **disabled until implemented**:

```text
& $C .\scripts\full_development_cache.py --mode run --model chronos --plan-sha <ACTUAL_SHA> --batch-size 100 ...
& $M .\scripts\full_development_cache.py --mode run --model moirai  --plan-sha <ACTUAL_SHA> --batch-size 100 ...
& $C .\scripts\full_development_cache.py --mode audit --plan-sha <ACTUAL_SHA> ...
```

The same run commands will be the resume commands, but must first be reviewed
and tested in B.2. NEVER use the pilot `--count 24` runner to fill full cache.

## Resource planning

TASK-004A audit measured for 24 origins: Chronos 1.3070653 seconds and Moirai
6.1683519 seconds, EXCLUDING loading and replay. Linear estimates for 1258
origins: Chronos ~68.5 s, Moirai ~323.3 s **if including each pilot's first
origin initialization overhead in the mean**. Better steady-state estimates
assuming 23 warm origins and one cold origin, using approximate logs:
Chronos 0.03596 s/steady origin => ~45.7 s inference,
Moirai 0.2283 s/steady origin => ~288 s inference.
These are throughput *estimates*, not actual full-cache runtimes; Python IO,
initial model loading, replay and integrity scans add overhead. Reserve 1 GiB
free as conservative disk policy. One JSON record and one combined CSV row per
origin should generally be a few MiB total, but measured byte sizes from the
actual pilot are necessary for a defensible precise storage estimate. Neither
75.3 MiB nor 940.5 MiB peak PyTorch allocation means total process VRAM.

## Acceptance before B.2

- [ ] Exact processed source hash and checkpoint audit hash verified on Windows
- [ ] Full plan published with immutable canonical JSON SHA
- [ ] Count/first/last/date indices and zero validation/final targets confirmed
- [ ] Both per-environment CUDA/checkpoint preflights PASS
- [ ] No full-run path conflicts, adequate free disk
- [ ] CPU tests PASS on Windows
- [ ] Actual pilot file sizes measured; risk flags remain FALSE
- [ ] No model inference performed in B.1
