# Research Handoff — Collaborator Project C+

**Project title:** Asynchronous Cross-Horizon Probabilistic Adaptation for Frozen Time-Series Foundation Models under Distribution Shift

**Status:** Research ideation and method specification. No new experiments are authorized yet.

**Hardware target:** NVIDIA RTX 5060 Ti 16 GB, Windows, Python/PyTorch.

## 1. Project Background

This research builds upon the completed CITA 2027 project:

*When Does Volatility-Aware Calibration Help? A Cross-Market Evaluation of Frozen Time-Series Foundation Models.*

The previous project used:

- Frozen Chronos-T5-small and Moirai-1.0-R-small.
- SPY and BTCUSDT daily log returns.
- Context length 256, forecasting horizon 1.
- 100 predictive samples per forecast.
- Rolling conformal calibration and adaptive conformal controllers.
- Chronological Development, Validation, and locked Final Test partitions.
- Winkler scores, coverage, width, and paired block-bootstrap evaluations.

**Locked findings:** No primary Holm-corrected Final Test rejections (0/8); all 24 primary confidence intervals across three bootstrap block lengths included zero. The experiments did not establish consistent incremental benefits from volatility normalization.

These findings must not be altered or retrospectively reinterpreted.

### Main scientific lesson

Post-hoc interval calibration is not equivalent to improving the predictive adaptability of a time-series foundation model.

The new research must focus on forecast quality and predictive distribution adaptation, not merely conformal coverage.

## 2. Your Assigned Research Problem

**Research question:**

Can a frozen probabilistic TSFM exploit asynchronously arriving forecast feedback across different horizons to adapt more effectively under distribution shift than independent horizon-wise online correction?

The key question is not simply how to correct errors after they are observed.

Instead, investigate whether newly observed feedback from some forecasting horizons contains useful information for correcting other horizons whose corresponding outcomes are not yet available.

### Core scenario

At forecast origin `t`, the model predicts:

`y[t+1], y[t+2], ..., y[t+H]`

Forecast errors become observable at different times.

For example, at time `t+4`, the outcome for horizon 4 becomes available, but the outcomes for horizons 8 and 16 are still unknown.

Can the horizon-4 feedback improve the horizon-8 or horizon-16 predictive distributions before their own feedback arrives?

The answer may depend on whether the drift affects multiple horizons through a shared latent structure.

This is the scientific focus of the project.

## 3. Research Hypotheses

### H1 — Cross-Horizon Transferability of Forecast Mismatch

**Hypothesis:**

Some distribution shifts induce forecast errors that share a low-dimensional structure across horizons. A shared correction state may therefore improve probabilistic forecasts at horizons with limited or delayed supervision.

**Proposed mechanism:**

Decompose the correction for horizon `h` into:

`bias[t,h] = shared_bias[t,h] + local_bias[t,h]`

The shared component is generated from a low-dimensional state, while the local component captures horizon-specific deviations.

An analogous decomposition may be investigated for log-scale corrections.

**Scientific tests:**

- Shared state versus independent horizon-specific corrections.
- Shared state versus fully pooled correction.
- Different latent-state dimensions.
- Persistent versus transient distribution shifts.
- Regimes with shared versus independent horizon-level forecast errors.

**Potential contribution:**

Identify whether and when cross-horizon feedback transfer produces measurable predictive value for frozen probabilistic TSFMs.

**Important:** Shared low-rank adaptation is not automatically novel. The literature audit must establish whether its application to asynchronous probabilistic correction provides a meaningful distinction from existing multi-horizon online learners.

### H2 — Stale-Feedback-Induced Negative Transfer

**Hypothesis:**

Under abrupt distribution shifts, incorporating delayed residuals without accounting for their age or regime relevance can damage adaptation.

**Proposed mechanism:**

Associate every feedback event with its original forecast timestamp, target timestamp, horizon, and availability timestamp.

Investigate whether the adaptation update should depend on feedback age and independently available evidence about current regime conditions.

**Required comparisons:**

- Uniform update weight.
- Fixed age-decay weighting.
- Horizon-specific correction without age adjustment.
- Drift-consistency-aware weighting, if justified.

**Novelty warning:**

PROCEED and PADRE already address delayed supervision and drift-informed updating. Therefore, feedback-age weighting alone is insufficient as the main contribution.

H2 should initially be treated as a supporting hypothesis.

### H3 — Overlap-Aware Update Stability

**Hypothesis:**

Overlapping multi-horizon forecast errors can contain strongly dependent information. Treating related feedback events as independent may amplify noise and cause unstable online adaptation.

**Proposed mechanism:**

Assess whether a dependence-aware update strength or effective-information adjustment improves the stability of shared probabilistic correction states.

**Required comparisons:**

- Naive feedback accumulation.
- Conservative fixed updates.
- Horizon-aware update scaling.
- Overlap-adjusted update scaling.
- CALM-inspired learning-rate normalization where applicable.

**Novelty warning:**

Stay CALM (October 2026 preprint) already examines overlap-driven instability in online forecasting aggregation.

The proposed contribution must concern the estimation or correction of probabilistic forecast distributions, not merely a different expert-weighting learning rate.

### Hypothesis Priority

**Primary hypothesis:** H1 — Cross-horizon transfer of probabilistic forecast mismatch.

**Secondary hypotheses:** H2 and H3.

Do not combine all three into a complex model before establishing whether H1 is scientifically supported.

## 4. Proposed MVP Architecture

Use the frozen FM to produce predictive samples for multiple forecast horizons.

For each origin and horizon, estimate the predictive mean or robust location and a forecast dispersion statistic.

Apply a horizon-dependent output transformation:

`adapted_sample = base_location + bias_correction + scale_multiplier * (base_sample - base_location)`

The scale multiplier must remain positive.

A possible shared-state model is:

`bias[t,h] = horizon_embedding[h] @ shared_state[t] + local_bias[t,h]`

Start with a minimal shared correction structure and a small latent dimension.

Avoid large neural networks, backbone fine-tuning, sophisticated gating networks, or additional conformal controllers in the first MVP.

### Essential modules

| Proposed module | Responsibility |
|---|---|
| `multi_horizon_forecast.py` | Generate multi-horizon probabilistic forecasts |
| `forecast_cache.py` | Store samples with full forecast provenance |
| `feedback_queue.py` | Release observations only at valid availability times |
| `independent_corrector.py` | Horizon-specific online bias/scale correction |
| `shared_state_corrector.py` | Cross-horizon probabilistic adaptation |
| `prequential_replay.py` | Execute the causal forecast-update sequence |
| `probabilistic_metrics.py` | Compute horizon-wise proper scoring rules |
| `shift_diagnostics.py` | Analyze adaptation under controlled distribution shifts |

These are suggested responsibilities and filenames, not verified descriptions of existing repository files.

## 5. Reuse of the CITA 2027 Repository

You will receive a ZIP snapshot of the previous project.

Treat it as a reference implementation whose reusable components must be audited before modification.

### Components expected to be reusable

**Data pipeline**

Inspect ingestion, timestamps, preprocessing, target construction, and chronological splitting.

Do not assume that all datasets have the same normalization or time-frequency conventions.

**Frozen foundation-model adapters**

Inspect the existing Chronos and Moirai wrappers.

Check whether each checkpoint supports the required multi-horizon forecasts and whether returned samples preserve meaningful cross-horizon dependence.

**Forecast cache**

Reuse the existing serialization, manifest, and hashing conventions where appropriate.

The previous horizon-1 caches cannot substitute for new multi-horizon predictions.

**Online states**

Inspect EWMA residual and volatility-state implementations.

They may provide useful causal state-management infrastructure, but the old scale-normalization algorithm is not the proposed new method.

**Evaluation**

Reuse verified implementations of metrics, block bootstrap, and statistical reporting when their assumptions remain valid.

Introduce CRPS as a primary distributional metric.

**Conformal prediction**

Reuse existing conformal calibration only as a secondary reliability layer.

Do not make adaptive CP the primary new algorithm.

### Repository boundaries

Do not modify the original locked CITA results, tests, or Final Test interpretation.

Create an independent research branch or project copy.

Maintain separate output directories and experiment identifiers.

Preserve source-file and artifact provenance.

Do not assume that the old SPY/BTCUSDT Final Test is an untouched evaluation set for the new method.

## 6. Strict Causal Evaluation Protocol

At forecast origin `t`:

1. Read only observations available by time `t`.
2. Process feedback events whose labels have become available.
3. Construct the shared and horizon-specific correction states.
4. Generate frozen and adapted predictive distributions.
5. Save all forecasts before observing their future targets.
6. Release future outcomes only when their target timestamps become observable.
7. Update online states using only matured feedback events.

The cache should track at least:

- Dataset and series identifier.
- Forecast origin timestamp.
- Forecast target timestamp.
- Forecast horizon.
- Ground-truth availability timestamp.
- Model/checkpoint identifier.
- Forecast samples or sufficient distribution representation.
- Transform and preprocessing metadata.
- State/configuration version.

Multiple forecasts may correspond to the same eventual target observation. These are distinct valid predictions, but their feedback must not automatically be treated as independent evidence.

**Critical invariant:**

A prediction for `y[t+h]` may not depend on `y[t+h]` or on any later observation unavailable at origin `t`.

## 7. Experimental Design

### MVP Phase 1 — Controlled Synthetic Mechanisms

Construct simple processes with:

- Stationary baseline.
- Mean-only structural shift.
- Variance-only shift.
- Joint mean–variance shift.
- Shared cross-horizon drift structure.
- Horizon-specific drift without transferable structure.
- Abrupt regime reversal.
- Short-lived outliers.

The most important comparison is between shared and nonshared drift structures.

If cross-horizon transfer helps only when the synthetic generator explicitly imposes strong shared structure, report that limitation rather than claiming broad adaptability.

### MVP Phase 2 — Real Time Series

After mechanism validation, select real datasets from at least two domains, such as electricity, weather, traffic, or solar.

Finance may be included for exploratory stress testing, but the previous locked SPY/BTCUSDT test cannot serve as a fresh confirmatory holdout.

Candidate horizons can initially be `[1, 4, 8, 16]`, subject to dataset frequency, checkpoint capability, and computational budget.

Forecasting horizons should be fixed before confirmatory evaluation.

### Primary Metric

Mean CRPS, reported separately by horizon.

Also report an explicitly pre-specified aggregate across horizons if an overall primary comparison is required.

### Secondary Metrics

- MAE and MASE.
- RMSE where relevant.
- Forecast bias.
- Prediction interval Winkler score, width, and coverage.
- Recovery after distribution shift.
- Negative-transfer severity.
- Adaptation-state stability.
- Computational overhead.

Use joint multivariate proper scoring rules only if the model produces valid joint predictive trajectories across horizons.

Do not construct artificial joint samples by independently combining marginal predictions and then claim to have evaluated the original model's joint distribution.

## 8. Mandatory Baselines

**Baseline B0:** Frozen TSFM without adaptation.

**Baseline B1:** Independent horizon-specific EWMA bias correction.

**Baseline B2:** Independent horizon-specific location-and-scale correction.

**Baseline B3:** Fully pooled correction shared across horizons.

**Baseline B4:** A partially shared correction model without the proposed asynchronous mechanism.

**Baseline B5:** Existing delayed-feedback or online aggregation methods, reproduced faithfully where task and architecture permit.

Relevant competitors include PROCEED, PADRE, and Stay CALM.

Match information availability, update frequency, adaptation-state capacity, and tuning budget wherever possible.

Do not label a simplified approximation as an official reproduction.

## 9. Required Ablation Studies

Investigate the contribution of each component independently:

| Ablation | Scientific purpose |
|---|---|
| Shared versus independent correction | Test cross-horizon transfer |
| Fully shared versus partially shared | Test need for horizon specificity |
| Bias-only versus scale-only versus joint | Identify correction mechanism |
| Different latent-state dimensions | Test representation capacity |
| With versus without age-aware weighting | Quantify stale-feedback effects |
| With versus without overlap adjustment | Quantify dependence effects |
| Different feedback delays | Examine sensitivity to supervision latency |
| No-shift versus shift regimes | Identify negative transfer |

Do not run every possible combination by default.

Use Development and Validation results to identify the smallest set necessary to distinguish competing explanations, before locking final experiments.

## 10. Novelty Gate and Stop Conditions

Before implementing the proposed shared-state method, prepare a prior-art comparison addressing:

1. What does PROCEED already solve?
2. What does PADRE already solve?
3. What does Stay CALM already solve about overlapping feedback?
4. What is already established about multi-horizon conformal calibration?
5. What is genuinely different about cross-horizon probabilistic adaptation?
6. Why is the mechanism more than a low-rank adapter attached to a frozen model?

**Stop or revise the algorithmic claim if:**

- The mechanism is already established in directly comparable literature.
- Independent horizon-specific EWMA provides equivalent performance.
- Fully shared correction is sufficient.
- Improvements depend on information unavailable at prediction time.
- Gains exist only in synthetic settings with artificially strong shared drift.
- Improvements are explained solely by post-hoc interval widening.
- The method fails under regimes where no cross-horizon transfer should exist.

Negative findings are valid scientific results, but they must not be repackaged as successful method development.

## 11. Required Deliverables and Stage Gates

### Gate 1 — Repository Audit

Produce:

- Actual repository tree and key module map.
- Environment and checkpoint inventory.
- Existing cache-schema report.
- List of reusable components.
- Multi-horizon capability and compatibility assessment.
- Identified leakage and reproducibility risks.

No GPU experiment is required for this gate.

### Gate 2 — Literature and Novelty Review

Produce:

- A table of closest related methods.
- Exact distinctions between proposed and existing mechanisms.
- Three testable hypotheses.
- One selected primary contribution claim.
- Baselines and failure conditions.

Do not write the new method code before this gate is reviewed.

### Gate 3 — MVP Specification

Produce:

- Exact probabilistic correction equations.
- Shared-state update procedure.
- Information-availability assumptions.
- Feedback-event ordering.
- Cache contract.
- Unit-test plan.
- Validation protocol.

### Gate 4 — Technical Validation

Required checks:

- Deterministic replay.
- Correct forecast/target alignment.
- Correct feedback release times.
- No future-label leakage.
- Exact frozen-forecast identity when correction is disabled.
- Resume/restart consistency.
- Proper metric computation.

### Gate 5 — Scientific Evaluation

Only after technical validation:

- Mechanism experiments.
- Fair baseline comparisons.
- Ablation studies.
- Cross-domain validation.
- New locked confirmatory evaluation.
- Appropriate paired statistical inference.

Report both successes and failures.

## 12. Literature Starting Points

Read the following primary works before formalizing the method:

1. **PROCEED:** Zhao & Shen, KDD 2025 — *Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting.* https://arxiv.org/abs/2412.08435

2. **PADRE:** Li et al., IJCAI 2026 — *Beyond Uniform Updates: Drift Pattern Aware Online Time Series Forecasting Under Delayed Feedback.* https://ijcai.org/proceedings/2026/503

3. **Stay CALM:** Prakash, October 2026 preprint — *Calibrated Online Aggregation for Multi-Horizon Forecasting Under Delayed, Overlapping Feedback.* https://github.com/Shivesh777/CALM

4. **AcMCP:** Wang & Hyndman — *Online Conformal Inference for Multi-Step Time Series Forecasting.* https://arxiv.org/abs/2410.13115

Review the original methodologies and their latest versions, including any directly competing methods published later.

## 13. Scope and Coordination

The companion research project investigates skill-gated location/scale adaptation for primarily one-step probabilistic forecasting.

Your project investigates the information structure of asynchronous multi-horizon feedback.

Both projects may reuse the same frozen FM adapters, datasets, evaluation utilities, and reproducibility infrastructure.

However, the two projects must maintain independent hypotheses, adaptation algorithms, development experiments, and confirmatory protocols.

**Your first assignment is not to build a large model.**

Your first assignment is to establish whether sharing asynchronously observed probabilistic forecast errors across horizons provides a scientifically meaningful and testable advantage over independent or fully pooled online correction.

The immediate deliverable is the repository audit, literature comparison, and minimal method specification — not experimental results.