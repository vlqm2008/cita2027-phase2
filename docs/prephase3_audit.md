# Pre-Phase 3: methodological decisions and audit status

## Phase 2 acceptance

The user's Windows runs reported 21/21 tests passing, a successful SPY processing run,
5,492 observed and expected XNYS sessions, 5,491 valid log returns, and status PASS.
**Independent read-only file/hash verification remains pending on the user's PC.**
The provided ZIP cannot contain the user's real snapshot or its machine-specific artifacts.
Run `scripts/verify_phase2.py` against the user's existing `artifacts/` files.

## Historical adjusted prices

Yahoo's retrospectively adjusted daily close series may depend on future corporate actions.
Even if multiplicative adjustments cancel in historical adjacent log differences, the
cancellation and dividend conventions have not been proven for this vendor or period.
Source data are retrospective and not point-in-time certified. No inference or leakage
claim should imply that a 2026 snapshot was available as-is at 2005 forecast origins.

## Forecast target and information set

Let `r_i = log(adj_close_i) - log(adj_close_(i-1))` for chronological XNYS row `i`.
At the end of session row `t`, assume the adjusted return `r_t` is known.
The frozen model receives exactly `[r_(t-255), ..., r_t]`, length 256.
It outputs a distribution for `r_(t+1)`. No transformations from price forecast
are permitted, no future return/target is passed into the adapter.
`origin_index=t`, `target_index=t+1`; target session date may skip weekends/holidays.
Potential daily vendor publication lag or adjustment revision is a separate as-of issue.

Both adapters are **sample-based**: empirical q10/50/90 are extracted via
`numpy.quantile(..., method='linear')` from 100 simulated 1-step return samples.
This convention is **not** a conformal order statistic and must not be confused
with finite-sample CQR quantile definitions. Monte Carlo error matters, especially
at 0.1/0.9; 100 samples is a smoke value, not a benchmark-locked budget.

## Checkpoint matrix (documentation review only)

| Item | Moirai-1.0-R-small | Chronos-T5-small |
|---|---|---|
| Official ID | `Salesforce/moirai-1.0-R-small` | `amazon/chronos-t5-small` |
| Model family | Moirai 1.0-R, 14M params | Tokenized T5 sampling model (not Bolt) |
| Official inference interface | `MoiraiModule.from_pretrained`, `MoiraiForecast.create_predictor` (GluonTS) | `ChronosPipeline.from_pretrained`, `predict` |
| Context 256 / horizon 1 | API permits; **runtime unverified** | API permits; **runtime unverified** |
| Predictive output | Forecast samples; empirical quantiles | Forecast samples; empirical quantiles |
| Public release family | March 2024 Moirai-1.0-R | March 2024 Chronos research |
| Training description | LOTSA, broad domains | Public time series and synthetic Gaussian-process series |
| SPY training overlap exclusion | **UNKNOWN** | **UNKNOWN** |
| Training cutoff per source series | **UNKNOWN** | **UNKNOWN** |
| Exact HF commit and weights | Resolve on local machine with checkpoint audit | Resolve on local machine with checkpoint audit |
| License | CC-BY-NC-4.0 (check allowed paper/redistribution use) | Apache-2.0 |
| VCPI relationship | Legacy baseline, not published Moirai-2.0 comparator | Legacy baseline, not published Chronos-Bolt comparator |

Release date is not evidence that a SPY test interval is out-of-training. Since the
model family was available around 2024, a test beginning 2025 reduces obvious
chronological training-window overlap *for those original weights*, but neither
checkpoint dataset composition, synthetic data sources, nor precise training cutoff
has been fully traced. This inference is a hypothesis, not verified independence.

For scientific positioning, keep both legacy checkpoints as predeclared controlled
baselines, but do not label the results a faithful replication of VCPI's published
backbone setup. Dedicated VCPI comparisons need aligned checkpoints or explicitly
labeled adaptations in a later phase.

## GPU and dependency gates

The user's prior local log showed **RTX 5060 Ti, ~16 GiB, sm_120, driver 610.88,
Python 3.11.1**; this supersedes the initial 8 GiB assumption.
PyTorch was NOT_INSTALLED when that log was taken. We have no local CUDA execution
logs or proof of a compatible installed build. `audit_gpu.py` must return `PASS`
with actual matrix and elementwise execution before any model download/inference.

Install one model family at a time. `uni2ts` has a large, version-sensitive stack;
prefer a fresh dedicated Python 3.11 environment for the Moirai smoke test if the
existing Chronos/PyTorch setup produces solver conflicts. Never rewrite the
current project `.venv` blindly or alter the data dependency lock without review.

## Remaining NO-GO conditions

* Independent processed/manifest verification log unavailable.
* GPU actual CUDA tensor execution unavailable.
* Exact pinned model revisions and local weight hashes unavailable.
* Chronos and Moirai inference, replay, latency and VRAM untested on user's hardware.
* Contamination exclusion unproven; split dates remain provisional.
* Moirai 1.0-R/`uni2ts` versions on Windows require dedicated integration test.

**Full forecast cache: NO-GO.** This does not block running development-only smoke tests
once each preceding machine-local gate is satisfied. Do not begin ACI, DtACI, VCPI,
or full benchmarking here.
