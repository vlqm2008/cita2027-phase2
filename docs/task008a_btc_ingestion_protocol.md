# TASK-008A — BTCUSDT SPOT 1d UTC ingestion / causal plan (locked v1)

## Scope

An **additive** module inside the accepted `cita2027-phase2` repository. No reading/modification of SPY, SPY Final Test, BTC Final Test, checkpoint weights, or calibration outputs. Only network requests to Binance official monthly **Spot** `BTCUSDT/1d` archives from **2017-08 through 2024-12** and their `.CHECKSUM` URLs. This is **not** point-in-time-vintage proof; the historical source may be revised retrospectively.

Official documentation: https://github.com/binance/binance-public-data/blob/master/README.md; terms https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md. Official docs disclose retrospective replacements, list an August 2022 historical kline correction, and indicate SPOT timestamps switch to microseconds from 2025-01-01 (outside this task). **2025+ access explicitly prohibited in TASK-008A**.

## Fixed dates and targets

* Source period (inclusive UTC): 2017-08-17 through 2024-12-31. Expected **2694** daily rows, assuming an exactly complete series. Unobserved/missing earlier trading days are fatal; do not shift start.
* Warm-up: through 2018-12-31; Development *targets*: 2019-01-01 through 2022-12-31 (1461 days); Validation *targets*: 2023-01-01 through 2024-12-31 (731 days).
* Development bridge: origin `2018-12-31` → target `2019-01-01`. Validation bridge: origin `2022-12-31` → target `2023-01-01`. No target from Jan 2025 or later may appear in plans, snapshots, archive URLs, or parser.
* Row t records a **completed** UTC calendar-day bar. Origin t becomes usable **after** that daily bar closes. The archive's open_time at 00:00 UTC is *not* the information availability time. `log_return_t=log(C_t/C_{t-1})` is first defined from 2017-08-18 onward. A forecast at the close of t predicts `log(C_{t+1}/C_t)`. Context is previous 256 complete return observations ending at t; target `t+1` never enters context.
* Fixed seed `2027 + immutable global origin_index`. Both TSFMs would reuse the same origin plans, 256 context, 1 horizon, 100 samples, q(0.1,0.5,0.9). No TSFM is run in this task.
* BTC independent calibration: 252 Development-origin predictions as burn-in, 1209 scored Development targets; validation is 731 targets with **continuous carry-forward** of score buffer/EWMA/ACI alpha from Development. No reset at Development→Validation. SPY state must **never** be transferred to BTC.
* All 9 original factorial cells/model; alpha 0.2, W 252, gamma0 0.01, EWMA 0.94, shared return-EWMA gamma driver in [0.5,2], expansion-only CQR primary; signed CQR ablation; blocks 10/20/40; 1000 resamples seed 2027. No tuning/selection in BTC Validation.

## Integrity requirements and publication

* Use official ZIP + associated `*.zip.CHECKSUM` for **every month**; check SHA-256 before opening ZIP. On resume, retrieve current official CHECKSUM, compare to immutable saved checksum bytes, and compare cached ZIP SHA. Any change aborts (`UPSTREAM_CHECKSUM_REVISION`); do not overwrite or silently accept provider edits. Source URL and both byte-level SHAs are preserved in manifest. URL paths are fixed; no alias/fallback from daily to futures.
* ZIP contains one `.csv` with exact expected filename; 12 expected Binance fields; optional CSV header is accepted. Before 2025, timestamps must be 13-digit milliseconds. Every UTC open exactly 00:00, UTC close exactly `open+86,400,000−1 ms`; day must match archive month.
* Reject price ≤0, non-finite amounts, invalid high/low ordering, negative volume, noninteger trade count, duplicate, out-of-order, or missing dates. Require all **2694 calendar days** including weekends. No synthetic candles, interpolation, or split change.
* Source snapshot is a deterministic content-addressed CSV (`artifacts/btcusdt_data/snapshots/<sha>.csv`), with immutable plans, quality and inventory manifests in `artifacts/btcusdt_data/{plans,quality,manifests}`. Source ZIP/cache under `data/btcusdt/binance_spot_monthly_1d`. Write-once creation; no artifact overwritten. Raw archive download uses atomic temp+link; unmodified cached ZIPs can resume. A checksum revision is a provenance **FAIL** requiring manual investigation, not a reason to alter the locked protocol.
* The manifest contains all 89 upstream ZIPs and checksums. Since historical `.CHECKSUM` is mutable externally, successful immutable local SHA verification is evidence of **reproducibility of ingested bytes**, not historical point-in-time availability. Keep `btc_historical_point_in_time_verified=false`, `contamination_free_verified=false`, `scientific_clearance=false`. SPY's `point_in_time_adjusted_prices_verified=false` remains unchanged in all existing evidence.

## GO / NO-GO

* Offline tests PASS means code-level boundary invariants pass on synthetic test fixtures only, not that Binance archives pass.
* `BTC_DATA_INTEGRITY_PASS_NOT_SCIENTIFIC_CLEARANCE` from Windows CLI (with output SHAs) is required for **technical GO** to prepare frozen BTC inference, after independent review of quality and manifests.
* Any mismatch/absent 2017-08 source, gap, non-midnight initial candle, invalid CHECKSUM or unpublished/missing month: **NO-GO**. Do not backfill with unverified sources or modify dates.
* Scientific clearance remains **NO-GO** until training-corpus contamination and PIT/revision risks are resolved. No SPY Final Test or BTC Final Test access during TASK-008A.
