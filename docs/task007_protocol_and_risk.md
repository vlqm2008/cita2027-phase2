# TASK-007 — Evidence Freeze, Provenance, BTCUSDT Protocol v1

## Evidence and audit boundary

- SPY Development and Validation calibration protocols are immutable. Do not alter any existing forecast, trace, report, configuration, or hyperparameter.
- Execute `scripts/task007_freeze.py` on the **same Windows repository** as TASK-006B, from `.venv`. It reads the accepted manifests, the 3,520 per-origin forecast records, the locked plans, protocol implementation files and configs; SHA-verifies each and writes only a **new** content-addressed evidence index under `artifacts/evidence/task007/<sha256>.json`.
- Any missing file, altered SHA, changed checkpoint, missing record, or flag drift aborts **before publication**. Existing evidence is never repaired, rewritten, or recalculated. Rerun is idempotent if inputs are identical.
- The index verifies byte integrity and declared identifiers; it cannot prove a model's training corpus excludes test data or that downloaded adjusted prices are true point-in-time snapshots.
- `no_final_test_read=true` is a claim about this script's explicit access pattern, not an attestation about external processes. No BTC market data is downloaded.

## Checkpoint training provenance and epistemic limits

| Backbone | Frozen identity | Verified upstream training-family disclosure | Exact checkpoint-specific training cutoff / SPY/BTCUSDT disjointness |
| --- | --- | --- | --- |
| Chronos | `amazon/chronos-t5-small`, commit `a971ba21945c4f1796b17a91fe69214b5f4ad472`, weight SHA `9c8b6fde5300f72b01c173153bf9288fa0a200614275bf0585071ad71a6a3d43` | Public time-series corpus plus synthetic Gaussian-process series; original Chronos T5. | **UNVERIFIED** |
| Moirai | `Salesforce/moirai-1.0-R-small`, commit `f5bd5d01f0d67de856107d43abdd637380aae0a3`, weight SHA `eaafeb166a3dfcf5728ab52a9ccc5392a23333b42a41a14e5528089ece3ef377` | LOTSA v1 time-series archive. | **UNVERIFIED** |

Do not infer a training cutoff from Hugging Face commit/upload date, conference publication date, repo launch, training script, or latest version. Need source-level manifest of *exact series*, min/max timestamps used in training, dataset extraction window and checkpoints' training job provenance to claim exclusion. At present `contamination_free_verified=false` for both SPY and proposed BTCUSDT. Use "frozen foundation models", not "strictly out-of-training zero shot" unless independently demonstrated.

Authoritative references:
- https://huggingface.co/amazon/chronos-t5-small
- https://www.amazon.science/publications/chronos-learning-the-language-of-time-series
- https://huggingface.co/Salesforce/moirai-1.0-R-small
- https://github.com/SalesforceAIResearch/uni2ts/blob/main/project/moirai-1/README.md

## SPY adjusted-price provenance

The accepted SPY source is `yahoo_yfinance`, daily `adj_close`, exchange calendar `XNYS`, `America/New_York`. Yahoo's adjusted close adjusts prior closes for subsequent corporate actions (splits and dividends); this says **nothing about historical point-in-time availability**. The accepted yfinance historical request parameters and processed snapshot hash must remain unchanged. Potential future dividend/revision back-adjustment could alter historic price levels; daily adjusted **log returns** may be affected near corporate action boundaries or revisions. Do not assert leakage absent without comparing against a point-in-time-vintaged data source / documented dividend and split event files.

**Non-destructive remediation:** obtain an independent event-sourced price + cash dividend + split archive with vintage/as-of timestamps, freeze its license and checksums, build a separate research *sensitivity* series based only on events known by each observation's publication time. Report divergences and causal implications; do NOT replace current SPY processed series or change accepted evidence. If historical as-of reconstruction cannot be established, retain the flag as false and explicitly label the SPY findings retrospective.

References: https://in.help.yahoo.com/kb/adjusted-close-sln28256.html , https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html

## Proposed BTC/USDT source-and-time protocol (pre-data lock, v1)

**Instrument:** Binance **Spot** BTCUSDT (not futures; no synthetic aggregated market), 1d klines via official `https://data.binance.vision/` monthly ZIP archive; use `close` in quote USDT/BTC. One bar per UTC calendar day with open 00:00:00 UTC; close at the end of the day. The origin is the **completed UTC daily bar t**, not its opening timestamp; the target is close-to-close log return of next completed UTC bar, `r_{t+1}=log(C_{t+1}/C_t)`. Units are dimensionless log return in decimal. Data timestamp is open time; **availability timestamp is after the bar closes**. Do not use a bar still open at origin. The target is not Bitcoin/USD and is not a stock total-return index. Spot exchange-specific trading risk is distinct from SPY.

**Period proposal locked before download:**
- Warm-up: 2017-08-17 through 2018-12-31 UTC, source coverage permitting; source starts later or has gaps -> STOP and audit (do not silently shift dates).
- Development targets: 2019-01-01 through 2022-12-31 inclusive.
- Validation targets: 2023-01-01 through 2024-12-31 inclusive.
- Untouched Final Test targets: from 2025-01-01 onward; TASK-007 must not download, inspect, or run it. A later final-test end date must be prespecified before access.
- 24/7 UTC calendar: no weekends/holiday drops. Verify unique dates, daily continuity and no missing/duplicate candles. Final available source data and 2017 launch history must pass the pre-download/ingestion source-coverage gate.
- Development first target `2019-01-01` uses origin `2018-12-31`, Validation first target `2023-01-01` uses origin `2022-12-31`. A split is defined by **target date**, not origin date. The final prediction within each split cannot target the next split's targets.
- Return feature `r_t=log(C_t/C_{t-1})`; context is last 256 **completed returns** through t. Forecast horizon=1, samples=100, quantiles=0.1/0.5/0.9, seed=2027+origin_index (index assigned after snapshot freeze; never globally normalized using future stats).
- Independent BTC state starts from the *earliest eligible BTC forecast origin*, with the pre-specified **252 forecast-origin burn-in** to initialize score buffers and ACI. No SPY state transfer. Within BTC, carry all causal states continuously over Development → Validation, including the boundary bridge, without resetting alpha, score buffers, or return/residual EWMA states. First eligible Development evaluation origin is number 253 among Development-origin forecasts; do not count burn-in as evaluated. Validation must not tune hyperparameters.
- Factorial unchanged: scales `{none,residual_ewma,return_ewma}` × controllers `{rolling_cqr,fixed_gamma_aci,variable_gamma_aci}`; alpha 0.2, W=252, gamma0 0.01, EWMA decay 0.94, positive floor 1e-6, shared return-EWMA driver clipped [0.5,2], expansion-only CQR main, signed-CQR ablation, native q10/q90 comparator. Do not copy any BTC hyperparameters chosen on outcomes.
- Volatility regime: high iff causally available `sigma_t > quantile_0.75(prior 252 origin-time sigmas)`, equality other. Threshold chosen without future scores/returns; no regime rebalance. Rolling empirical coverage window 50, not conditional coverage.
- Primary Winkler at alpha=.2; supplementary marginal coverage, mean width, lower/upper misses and *separate* tail penalties, rolling coverage, regime-specific diagnostics, adaptation, runtime. Paired circular moving-block bootstrap with lengths 10/20/40, 1000 replicates, seed 2027. All CIs exploratory without multiplicity adjustment. No model selection in BTC Validation (0 selection operations). Any later confirmatory claims require prespecified familywise procedure and independent untouched testing.
- Do not automatically assert 1d bar availability at exact 00:00 UTC. Record the archival publication delay and use bars only after close; offline backtest assumes end-of-day close is observable once the bar completes.
- Data provenance: save each ZIP and upstream `.CHECKSUM` as immutable bytes, file SHA-256, capture timestamp, provider URL, licensing terms, parsing version and source-revision log. Binance notes that archive files may be corrected later. Also validate **milliseconds versus microseconds** for SPOT timestamps from 2025 onward, before any future final-test work.

**Scientific comparability:** equal sampling frequency (daily) but different market microstructure, calendar (24/7 vs trading sessions), underlying return definition (crypto close-close vs dividend-adjusted SPY), and potentially pretraining exposure. Report as *second-dataset external robustness*, not like-for-like controlled market test. These are locked choices rather than data-driven optimizations.

References: https://github.com/binance/binance-public-data/blob/master/README.md ; https://data.binance.vision/

## GO / NO-GO

- GO: run read-only SPY evidence freeze and archive manifest; prepare BTC downloader/parser *without downloading data in TASK-007*; retain 2 environments, 2 small frozen checkpoints, same codebase.
- Conditional GO: subsequent BTC ingestion only after provider/source coverage, terms, checksum and timestamp checks; keep scientific risk flags false. Crypto evaluation is explicitly subject to the known contamination limitations.
- NO-GO: claiming point-in-time Yahoo series verified, claiming known training cutoff, changing SPY evidence, selecting BTC hyperparameters using BTC results, running SPY Final Test, promoting exploratory CIs to confirmatory results.

### Scope caveat for processed SPY snapshot

The accepted processed SPY CSV spans warm-up, Development, Validation and an untouched reserved Final Test. TASK-007 deliberately **does not open or rehash that CSV**, because doing so would access reserved Final Test bytes. Its already-accepted processed SHA-256 is recorded as an anchor from predecessor reports/plans, **not independently reverified here**. Any requirement for rehashing the entire processed dataset should be deferred to a separately authorized integrity protocol with explicit accounting for held-out data access. No numerical Final Test values are inspected by this freeze script.
