---
license: apache-2.0
model_card_spec: "1.2"
pipeline_tag: time-series-forecasting
task: "Others - Time-Series Forecasting"
tags:
  - time-series-forecasting
  - time-series-foundation-model
  - zero-shot
base_model: google/timesfm-2.5-200m-pytorch
date_published: "2025-10-02"
date_published_source: "The checkpoint's Hugging Face model card records that on October 2, 2025 the model structure was changed to fuse the QKV matrices; the current files date from that revision. The TimesFM 2.5 family itself was announced on 2025-09-15 (google-research/timesfm README)."
---

# TimesFM 2.5 200M (PyTorch) — Time-Series Foundation Model (Zero-Shot Forecasting)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Ftimesfm--2.5--200m--pytorch-ffcc4d?style=flat)](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-google--research%2Ftimesfm-181717?style=flat&logo=github&logoColor=white)](https://github.com/google-research/timesfm)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2310.10688-b31b1b.svg)](https://arxiv.org/abs/2310.10688)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are downloaded unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebook are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebook downloads third-party weights and data governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides one ready-to-run Google Colab notebook that exercises the package end to end in an isolated hash-locked environment: stage and check the pinned checkpoint, validate a series into an input manifest, hold out the future, score baselines, run the zero-shot forecast, evaluate, and export a result bundle:

- **Task Inference Tutorial** (`TASK-INFERENCE` / `GUIDED`):  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/timesfm-forecasting-pipeline/blob/main/tutorials/timesfm_forecasting_colab.ipynb) [`timesfm_forecasting_colab.ipynb`](https://github.com/kurtvalcorza/timesfm-forecasting-pipeline/blob/main/tutorials/timesfm_forecasting_colab.ipynb)  
  *Forecast hourly temperature for three Philippine cities (or your own CSV) one day ahead with median and quantile output, scored beside naive, seasonal-naive and exponential-smoothing baselines; no training occurs. No hosted run is recorded yet.*

---

#### Description

TimesFM 2.5 is a pretrained time-series foundation model from Google Research. This repository packages its 200M-parameter PyTorch checkpoint, `google/timesfm-2.5-200m-pytorch` on the Hugging Face Hub, loaded through the upstream `timesfm` package pinned at `3.0.2` on PyPI (the `timesfm` package version 3.0.2 carries both the 2.5 implementation used here and a separate `timesfm3` package that this repository does not import). The architecture is a decoder-only transformer over patches of 32 time steps: 20 layers, hidden size 1280, 16 attention heads, a 128-step output patch, a context limit of 16,384 points, and a continuous quantile head that emits nine deciles (0.1 to 0.9) plus a separate mean output (values read from the checkpoint's `config.json`; the parameter count of 231.3M is the Hub's figure and the loader records the count it observes). At inference the model reads the most recent points of a series as plain numbers, with no frequency label, and decodes the future autoregressively in output patches; adaptation happens neither through training nor through in-context examples. This repository performs no training.

The DIMER wrapper adds, as code a reader runs: a pinned checkpoint identity with a committed file manifest and load-time size and digest checks; refusal of any other model source and of pickle-format weights; a data loader and validator for long-format CSV with frequency inference, gap and missing-value policies and named refusals; a forecasting wrapper that exports the median as the point forecast and the deciles as named columns; chronological holdout evaluation with naive, seasonal-naive and exponential-smoothing baselines and the metrics `mase`, `smape` and `quantile_loss`; and a result bundle with provenance and a reload-parity check. The pin's immutable commit and the weight digest are to be confirmed on the first hosted run; see *Checkpoint and runtime provenance* below.

#### Intended Use and Limitations

The pipeline is a zero-shot forecasting component and teaching unit; the subsections state its task, its users and its boundaries.

###### Primary Intended Uses

The task is zero-shot univariate time-series forecasting. The input is a long-format table with a timestamp column, a finite numeric value column and an optional series-id column, sampled at one fixed interval (hourly, daily, every 15 minutes, weekly) or one calendar interval (monthly, quarterly, yearly). The output is one row per series and future step carrying the future timestamp, `prediction` (the model's median, quantile 0.5), `mean` (the model's mean head) and the requested deciles as `q0.1` … `q0.9`. Application domains envisioned during development are regularly sampled environmental and operational series: hourly weather variables (the shipped sample is hourly air temperature for Manila, Cebu and Davao), daily or monthly demand and consumption counts, and telemetry with a stable sampling clock. The intended role is a zero-configuration forecasting baseline that a reader evaluates against naive and seasonal-naive rules on their own chronological holdout before relying on it, and a component a reader's own application can embed through the `forecast` function. Series with several independent ids are forecast independently in one call.

###### Primary Intended Users

Intended users are data scientists, analysts and application developers who hold regularly sampled numerical series, and learners in a self-paced setting who want to see a foundation model compared with simple baselines. The envisioned deployment settings are research, teaching and in-house or self-hosted use on infrastructure the user controls, within a tutorial and developer-preview boundary; no stable serving contract is claimed. Users are assumed to understand what a forecast horizon and a chronological holdout are, that a point forecast here is a median and not a mean, that a model quantile is not a calibrated interval until its coverage has been measured on their own data, and that a seasonal rule can be the stronger forecaster on short, strongly periodic series. A user who would read `q0.1`–`q0.9` as a guaranteed 80 % interval, or who cannot run a holdout on their own history, is outside the assumed competency.

###### Out-of-scope use cases

1. **Capability boundaries:** covariates (past-only or known-future regressors; the upstream XReg path is not exposed), multivariate joint forecasting, fine-tuning or any training, anomaly detection, imputation as a product feature (the validator's interpolation is a reported data policy, not a model), classification, and representation extraction are outside this repository. TimesFM 3.0 is not packaged: its weights are distributed under a separate non-commercial licence.
2. **Input boundaries:** a series needs at least `horizon + 16` rows and at least 3 rows to infer its interval; files with duplicate timestamps within a series, irregular spacing that is neither one fixed interval nor a calendar interval, non-numeric or infinite values, more than 1,000 series or 2,000,000 rows are refused; missing values and skipped grid points are refused unless `nan_policy="interpolate"` / `gap_policy="fill"` are set, and at most 20 % of a series may be interpolated; histories longer than 16,384 points are truncated to their most recent points with a report; the horizon is capped at 512 steps and the context at 16,384; quantile levels outside the nine trained deciles are refused rather than interpolated.
3. **Decision boundaries:** autonomous or high-consequence decisions taken from a forecast without independent domain validation on the operator's own history, monitoring and human review are out of scope.

---

#### Factors

###### Groups

The pipeline is not human-centric: it consumes numerical series without demographic attributes, and the shipped sample is gridded weather. The upstream pretraining corpus (GiftEvalPretrain, Wikimedia pageviews, Google Trends queries and synthetic data, per the checkpoint's model card) is not group-audited by this repository, and the upstream authors do not publish a group-level analysis. A deployment whose series describe people — footfall, demand by neighbourhood, service queues — can inherit structural bias from that data and from the pretraining mixture. The operator of such a deployment is expected to perform a subgroup error analysis on their own holdout, comparing `mase` and band coverage across the groups their decision affects, before acting on the forecasts.

###### Instrumentation

Evaluation data in this repository is produced by the Open-Meteo historical weather API, which serves reanalysis estimates (ECMWF ERA5 among its sources) for model grid cells several kilometres across; the values are hourly, in °C with one decimal, and describe a grid cell, not a weather station. The pretraining data was produced by the systems the upstream authors name (web pageview counters, search-trend aggregates, public benchmark collections) and by synthetic generators; their sampling rates and calibration are not disclosed beyond that. Instrument error reaches the model as value error: a sensor drift or a changed collection procedure arrives as a level shift the model forecasts forward as if real, and a clock fault arrives as a gap or a duplicate. The validator detects the structural defects it can see — duplicate timestamps, skipped grid points, irregular spacing, missing and non-finite values — and refuses or reports them; it cannot detect drift or recalibration in the values themselves.

###### Environment

Operating environment: Python `3.12` (the notebook builds a managed CPython `3.12.12`), `timesfm==3.0.2`, `torch==2.14.0`, `numpy==2.5.3`, `pandas==3.0.5`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, locked with hashes in `tutorials/requirements-colab.lock.txt` and `uv.lock`. CPU in float32 is the default; CUDA is used automatically when available; the upstream `torch.compile` option is off by default. The notebook supports Linux x86_64 runtimes only because its lock holds manylinux wheels. Data environment: the model assumes the deployment series is regularly sampled and that its recent history is informative about its near future; a structural break, a regime change, a new series shorter than the model has effectively seen, or a horizon far beyond the context's cycle degrades the forecast without any signal of its own unreliability. The shipped sample's two-week window starts after the checkpoint's release, so its exact values cannot be in the pretraining data; weather from the same sources for earlier periods may be.

---

#### Metrics

The subsections state what the evaluation code reports, the decision rule, and how the numbers were estimated.

###### Performance Measures

`evaluate_forecast` reports, per series and pooled: `mae`; **MASE** (`mase`) — mean absolute error divided by the in-sample mean absolute error of the seasonal-naive forecast at `season_length` (or 1), which is unit-free and lets series of different scales be pooled and read against the seasonal rule; **sMAPE** (`smape`) — symmetric mean absolute percentage error in percent, scale-free but undefined near zero and so reported beside MASE; **quantile loss** (`quantile_loss`, the pinball loss) at each exported quantile with its mean (`mean_quantile_loss`), the proper score for a quantile forecast, whose `q0.5` term equals half the MAE; and **interval coverage** (`interval_coverage`) — the share of held-out points inside the outermost exported quantile band, against its nominal width. `evaluation_report` sets these beside the same `mae`, `mase` and `smape` of the `naive`, `seasonal_naive` and `ses` baselines and records which baselines the model beats on `mase`; it asserts nothing. Reading MASE alone would hide the calibration of the band; reading coverage alone would hide how far the median is from the truth. RMSE, CRPS and WAPE are not implemented. All reported values are tutorial metrics on one holdout, not benchmark results; the upstream authors' benchmark figures are not reproduced here.

###### Decision thresholds

The point forecast is the model's median (quantile 0.5), exported as `prediction`; the wrapper asserts at every call that the upstream point output equals the median slot and raises `ModelIntegrityError` otherwise. The mean head is exported as `mean` and is never used as the point forecast. No acceptance threshold was set during development: the evaluation report records whether the model's `mase` is below each baseline's and does not stop the run on the answer, because a seasonal rule winning on a short periodic sample is a legitimate result. No operational threshold — an alert level, a reorder point, a staffing cut-off — is shipped; the operator sets it on their own holdout from the asymmetric cost of a forecast that is too high against one that is too low, using the quantile columns rather than the median where the costs differ.

###### Approaches to uncertainty and variability

Every metric comes from a single chronological tail holdout of `horizon` points per series (24 by default), with no dispersion estimate: no repeated origins, folds or bootstrap are run, so one value is one observation and the report labels it `sample-sanity`. Predictive uncertainty is represented by the model's quantiles from the trained decile grid; they are uncalibrated model quantiles, not prediction intervals, and `interval_coverage` on the holdout is the only calibration evidence the pipeline produces. A caller who needs calibrated intervals must measure coverage over a backtest with many origins on their own data and recalibrate (for example conformally). Run-to-run variability: the pipeline draws no samples and uses no seed; CPU float32 inference is deterministic for fixed inputs and pins, CUDA kernels may differ in the last digits, and the hash-locked dependency set removes version drift.

---

#### Ethical considerations and biases

No external board reviewed this pipeline and no clearance testing with a specific group took place; the considerations below are the developers' own.

###### Data

The checkpoint was pretrained, per its model card, on GiftEvalPretrain, Wikimedia pageviews (cut-off November 2023), Google Trends top queries (cut-off end of 2022) and synthetic and augmented data; the disclosure ends at those names, and the mixture cannot be reconstructed from this repository. Whether any of it contains personal or proprietary series is not ruled out: pageview and query aggregates are derived from human activity. This repository distributes code, a 31 KB weather sample (Open-Meteo, CC BY 4.0, no personal data) and a file manifest; it does not distribute the weights, which the notebook downloads from the Hub at run time, and it commits no caches. The operator is responsible for auditing the series they supply at inference for personal, confidential or regulated content; the pipeline inspects schema and values, not meaning.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or emergency response, and this repository does not certify TimesFM for any of them. Its validation is limited to offline unit and contract tests, a stand-in execution of the notebook's cells, and — pending — a hosted run of the notebook; no clinical, regulatory or independent domain validation has been carried out by the developers or by any external body. Where a sensitive-domain use is foreseeable — forecasting patient arrivals to set a ward roster, for example — it would be admissible only with independent validation on that operator's own history, a human decision-maker between the forecast and the action, monitoring for drift, and any clearance the domain requires.

###### Mitigations

1. **Supply-chain integrity:** `load_pinned_model` accepts only the committed manifest's identity (`google/timesfm-2.5-200m-pytorch`); `check_model_source` refuses other repositories, local paths, URIs and other refs; `verify_snapshot` checks every manifest entry's byte size, checks `config.json` against its pinned SHA-256 and against the pinned architecture values, hashes `model.safetensors` and reports its digest as unverified until the maintainer pins it, and refuses any pickle-format file in the snapshot. The notebook asserts the inline manifest equals the carried module's identity before fetching.
2. **Input integrity:** `load_series_csv` refuses a missing file, an empty file, duplicate header names and missing columns; `validate_series` refuses unparseable or duplicate timestamps, irregular spacing, gaps and missing values (unless the named policies are set, in which case every fill is reported in the input manifest), non-numeric and infinite values, mixed intervals across series, and lengths below `horizon + 16`; every refusal is a `ValidationError` naming the file and the rule.
3. **Output integrity:** the forecasting wrapper checks the upstream output shapes, finiteness and the median contract on every call and maps each quantile to its slot by level; `evaluate_forecast` refuses forecast and truth rows that do not align on `(series_id, timestamp)` instead of dropping them.
4. **Reproducibility:** hash-locked dependencies (`--require-hashes --only-binary :all:`), a notebook generated from the package with a parity check in CI, and provenance on every bundle recording identity, revision status, observed digests, versions, device and the input manifest.
5. **Refusals:** no random split helper exists; quantile levels off the trained grid are refused; horizons above 512 are refused; a different model source is refused; remote model code is never requested.

###### Risks and harms

Model-intrinsic: a regime change, a structural break or a series unlike the pretraining distribution produces a confident-looking forecast with no signal of its unreliability, and the operator — and whoever the operator's decision affects — bears the cost; the quantile band is uncalibrated, so an operator who provisions to `q0.9` under-provisions at a rate the pipeline does not measure (its one-day coverage figure is one observation); the model cannot see drivers, so a forecast across a holiday, an outage or an intervention is wrong in a direction a covariate-aware method would have caught. Use-context: automation bias, where a numerically precise median displaces the operator's judgement; a wrong `season_length` that silently mis-scales MASE and makes a weak seasonal baseline look beaten; instrument drift forecast forward as a real trend; and leakage, where a holdout built outside this package exposes future values to the context. Likelihood under normal use is highest for the calibration and `season_length` risks because neither needs a fault to occur; magnitude scales with what the forecast gates, from an over-ordered inventory to a mis-staffed shift.

###### Use cases

Distinct from the boundaries above, the developers consider the following uses prohibited even where the model would produce a plausible forecast: surveillance or profiling of individuals from telemetry that traces to a person; forecasting a person's behaviour, creditworthiness, employment, housing or insurance outcome as an input to a decision about that person; social scoring; autonomous control of physical systems where a forecast error can injure someone; presenting a model quantile as a certified prediction interval; and any use that violates the Apache-2.0 terms of the upstream weights and package or the terms of the runtime that executes the pipeline. The repository is not authorisation for any of these, and organisational controls remain the operator's obligation.

---

## Checkpoint and runtime provenance

| Item | Value | Evidence level |
|---|---|---|
| Model id | `google/timesfm-2.5-200m-pytorch` | verified: Hub listing |
| Revision | `main` — the immutable commit is **to be confirmed on the first hosted run**; `MODEL_REVISION_STATUS` and the manifest's `revisionStatus` say so, and `stage_missing_files` writes the commit the Hub served to `resolved-revision.json` | deviation from an immutable pin, recorded in `STATUS.md` |
| Files | `config.json` (475 bytes), `model.safetensors` (925,181,104 bytes) | verified: Hub listing |
| `config.json` SHA-256 | `cd3315b760d5cc7e278d7afdf41b897031ced888fc6c115bd9b3ac0ea2c47408` | computed from the Hub-served text |
| `model.safetensors` SHA-256 | pending; `verify_snapshot` reports the observed digest with `verified: false` and the notebook exports it | not yet established |
| Weight licence | Apache-2.0 | verified: model card front matter; upstream README states weights up to 2.5 are Apache-2.0 and 3.0 weights are non-commercial |
| Package | `timesfm==3.0.2` (PyPI, wheel SHA-256 `cfa3d22ddc3f019f0d9d95f19055428c4a4713526dafd7e5d1a7769782e7613f`), Apache-2.0 | verified: PyPI |
| Runtime pins | `torch==2.14.0`, `numpy==2.5.3`, `pandas==3.0.5`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `matplotlib==3.10.8` | locked with hashes |
| Decode settings | `max_context` rounded up to a multiple of 32, `max_horizon` to a multiple of 128, `normalize_inputs`, `use_continuous_quantile_head`, `force_flip_invariance`, `infer_is_positive`, `fix_quantile_crossing` all on | code (`model.py`) |

## Verification records

No execution of the notebook on a hosted runtime has been recorded. The offline checks below are static or stand-in evidence, not executions of the pretrained model:

- **Date:** 2026-10-05
- **Subject:** `tests/` and `tools/validate_release_assets.py` at the first pull-request revision
- **Runtime:** CPU-only Linux host, Python `3.12.12`, `numpy==2.5.3`, `pandas==3.0.5`; no `torch`, no `timesfm`
- **Procedure:** `pytest` over the unit, contract, BYOD-matrix and stand-in tests, the notebook's own cells executed with stub `torch`/`timesfm` modules and a sparse placeholder weight file; `ruff`; generator `--check`; a dry-run install of the hash lock
- **Observed result:** all checks passed; the stand-in produced the full result bundle with reload parity
- **Caveats:** stand-in evidence only — no pretrained inference occurred, no hosted runtime was used, and the weight digest and commit remain unverified

## References

- Das, A., Kong, W., Sen, R., & Zhou, Y. (2024). A decoder-only foundation model for time-series forecasting. *ICML 2024*. https://arxiv.org/abs/2310.10688
- Google Research. TimesFM 2.5 200M PyTorch checkpoint. https://huggingface.co/google/timesfm-2.5-200m-pytorch
- google-research/timesfm. https://github.com/google-research/timesfm
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting, 22*(4), 679–688.
