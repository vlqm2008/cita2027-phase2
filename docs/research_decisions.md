# Phase 2 decision register (2026-11-04)

| ID | Decision | State | Required action |
|---|---|---|---|
| D01 | SPY daily log return on Yahoo adjusted close | PROVISIONAL | Audit corporate-action treatment; point-in-time claim blocked |
| D02 | Calendar XNYS; timestamp is exchange session date | PHASE-2 DEFINED | Validate against actual NYSE session calendar |
| D03 | Warmup < 2018; dev 2018–22; validation 2023–24; test 2025–26 | PROVISIONAL | Reassess vs training corpus and checkpoint release dates |
| D04 | 80% primary; 90% secondary nominal coverage | OPEN | Decide based on financial tail relevance, finite-sample resolution and literature, before Phase 3 |
| D05 | Signed CQR primary, expansion-only ablation | OPEN MATHEMATICAL AUDIT | Confirm empty set / negative-score rules with independent tests |
| D06 | ACI unprojected vs bounded implementation | OPEN MATHEMATICAL AUDIT | Specify out-of-range quantiles and projection/theory mismatch |
| D07 | Moirai 1.0-R Small and Chronos-T5 Small | PROVISIONAL | Pin revisions; training-data provenance and dependency compatibility |
| D08 | Primary local RTX 5060 8GB, Colab not primary | ACCEPTED | Measure device/driver/runtime via scripts/inspect_environment.py |
| D09 | Keep torch out of Phase 2 data dependency group | ACCEPTED | Install separate Blackwell-compatible backend after GPU audit |

## Primary coverage: 80% vs 90%

There is no scientific basis to commit to 80% as primary without considering the
intended financial use. 80% has more tail misses and better local diagnostics on a
short final test, but less relevance to severe downside risk. 90% is more aligned
with common interval reporting but still does not establish VaR exception validity
or tail risk quality. Select one **before** running final test, report both and
report upper/lower misses. Do not choose based on which gives a better result.

## Theoretical and research limitations

Signed CQR may produce an empty set for excessive negative correction. `alpha_t`
outside (0,1) requires an explicit quantile convention. Variable-gamma ACI does
not automatically inherit the fixed-gamma unweighted error-rate identity. No
pretraining contamination-free claim has been established.
