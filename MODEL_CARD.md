---
license: other
license_name: timesfm-non-commercial-license-v1.0
license_link: https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE
model_card_spec: "1.2"
pipeline_tag: time-series-forecasting
task: "Others - Time-Series Forecasting"
tags:
  - time-series-forecasting
  - time-series-foundation-model
  - zero-shot
base_model: google/timesfm-3.0-pytorch
date_published: "2026-09-02"
date_published_source: "The last-updated date of google/timesfm-3.0-pytorch on the Hugging Face Hub as read through the Hub connector on 2026-10-06 (the date of the repository's current revision, which is the pinned checkpoint). The Hub commit history could not be read from the build environment, so an earlier first-publication date of the same files cannot be excluded; the maintainer confirms the date from the commit history. The timesfm 3.0.2 wheel that loads it was uploaded to PyPI on 2026-09-09."
---

# TimesFM 3.0 (PyTorch) — Time-Series Foundation Model (Zero-Shot Forecasting)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Ftimesfm--3.0--pytorch-ffcc4d?style=flat)](https://huggingface.co/google/timesfm-3.0-pytorch)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-google--research%2Ftimesfm-181717?style=flat&logo=github&logoColor=white)](https://github.com/google-research/timesfm)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2310.10688-b31b1b.svg)](https://arxiv.org/abs/2310.10688)
[![Weights: TimesFM Non-Commercial v1.0](https://img.shields.io/badge/Weights-TimesFM%20Non--Commercial%20v1.0-red.svg)](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are downloaded unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebook are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebook downloads third-party weights and data governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

> [!IMPORTANT]
> **The TimesFM 3.0 weights are non-commercial and non-production.** `google/timesfm-3.0-pytorch` is distributed by Google LLC under the *TimesFM Non-Commercial License v1.0* (Hub front matter `license: other`, `license_name: timesfm-non-commercial-license-v1.0`). In the licence's own words a Non-Commercial Purpose is "testing, evaluation, or research not tied to commercial gain, production deployment, or revenue generation"; use "in direct or indirect interactions with end users or production systems", in "client deliverables, or paid products/services", or to train other models for commercial use is excluded, and the model and any derivative may not be distributed. A commercial licence from Google is required for anything else. The pipeline code is Apache-2.0; that licence does not extend to the weights, which the notebook downloads at run time and this repository never commits. If the restriction does not fit a use, the repository's TimesFM 2.5 build (Apache-2.0 weights, commit `e47259ae75b93e7611c67e13a14d6607997c38f3`) is the fallback.

---

## Interactive Colab Tutorials

This pipeline provides one ready-to-run Google Colab notebook that exercises the package end to end in an isolated hash-locked environment: state the weights licence, stage and check the pinned checkpoint, validate a series into an input manifest, hold out the future, score baselines, run the zero-shot forecast, evaluate, and export a result bundle:

- **Task Inference Tutorial** (`TASK-INFERENCE` / `GUIDED`):  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/timesfm-forecasting-pipeline/blob/main/tutorials/timesfm_forecasting_colab.ipynb) [`timesfm_forecasting_colab.ipynb`](https://github.com/kurtvalcorza/timesfm-forecasting-pipeline/blob/main/tutorials/timesfm_forecasting_colab.ipynb)  
  *Forecast hourly temperature for three Philippine cities (or your own CSV) one day ahead with median and quantile output, scored beside naive, seasonal-naive and exponential-smoothing baselines; no training occurs. No hosted run is recorded yet.*

---

#### Description

TimesFM 3.0 is a pretrained time-series foundation model from Google Research. This repository packages its PyTorch checkpoint, `google/timesfm-3.0-pytorch` on the Hugging Face Hub, loaded through the `timesfm3` package of the upstream `timesfm` distribution pinned at `3.0.2` on PyPI (`timesfm3.TimesFM3Forecaster`; the same distribution's `timesfm` package, which implements TimesFM 2.5, is not imported). The architecture, read from the checkpoint's `config.json`, is a stacked mixing transformer with variate attention and iterative CPM RevIN: 20 transformer layers, model dimension 1280, 16 attention heads, RMS normalisation, rotary position embedding over the sequence (`use_rope_seq: true`, `use_rope_var: false`), a 32-point input patch, a 64-point output patch, linear detrending, stitching, and an output head that emits nine quantiles (the deciles 0.1 to 0.9) per horizon point. There is no separate mean head. The upstream forecaster caps the context at 15,360 points (`_MAX_CONTEXT_LENGTH`), pads each batch's context up to a multiple of 32 and decodes the horizon autoregressively in 64-point patches; the parameter count of 330.7M is the Hub's figure, and the loader records the count it observes. At inference the model reads the most recent points of a series as plain numbers with no frequency label; adaptation happens neither through training nor through in-context examples. This repository performs no training.

The DIMER wrapper adds, as code a reader runs: a pinned checkpoint identity with a committed file manifest and load-time size, digest and architecture checks; refusal of any other model source and of pickle-format weights; loading from the verified snapshot directory with `local_files_only=True`; a data loader and validator for long-format CSV with frequency inference, gap and missing-value policies and named refusals; a forecasting wrapper that exports the median (quantile slot 4) as the point forecast and the requested deciles as named columns; chronological holdout evaluation with naive, seasonal-naive and exponential-smoothing baselines and the metrics `mase`, `smape` and `quantile_loss`; and a result bundle with provenance, including the weights licence terms, and a reload-parity check. The pin's immutable commit and the weight digest are the values observed on the maintainer's hosted run of 2026-10-06; see *Checkpoint and runtime provenance* below.

#### Intended Use and Limitations

The pipeline is a zero-shot forecasting component and teaching unit whose weights are licensed for non-commercial, non-production use only; the subsections state its task, its users and its boundaries.

###### Primary Intended Uses

The task is zero-shot univariate time-series forecasting for testing, evaluation, research and teaching — the uses the weights licence permits. The input is a long-format table with a timestamp column, a finite numeric value column and an optional series-id column, sampled at one fixed interval (hourly, daily, every 15 minutes, weekly) or one calendar interval (monthly, quarterly, yearly). The output is one row per series and future step carrying the future timestamp, `prediction` (the model's median, quantile 0.5, slot 4 of the nine quantile slots) and the requested deciles as `q0.1` … `q0.9`; no `mean` column exists because the model has no mean head. Application domains envisioned during development are regularly sampled environmental and operational series studied offline: hourly weather variables (the shipped sample is hourly air temperature for Manila, Cebu and Davao), daily or monthly demand and consumption counts, and telemetry with a stable sampling clock. The intended role is a zero-configuration forecasting baseline that a reader evaluates against naive and seasonal-naive rules on their own chronological holdout, and a research component a reader's own non-commercial, non-production experiment can embed through the `forecast` function. Series with several independent ids are forecast independently in one call.

###### Primary Intended Users

Intended users are researchers, students, data scientists and analysts who hold regularly sampled numerical series and want to evaluate a foundation model against simple baselines, and learners in a self-paced setting. The envisioned deployment settings are research, teaching, internal benchmarking and experimentation on infrastructure the user controls — the settings the weights licence calls a Non-Commercial Purpose — within a tutorial and developer-preview boundary; no production or serving contract is claimed, and the licence excludes one. Users are assumed to understand what a forecast horizon and a chronological holdout are, that a point forecast here is a median, that a model quantile is not a calibrated interval until its coverage has been measured on their own data, that a seasonal rule can be the stronger forecaster on short, strongly periodic series, and that results may not be used in commercial decision-making, client deliverables or paid products. A user who would read `q0.1`–`q0.9` as a guaranteed 80 % interval, who cannot run a holdout on their own history, or whose use is commercial or production is outside the assumed competency and the licence.

###### Out-of-scope use cases

1. **Licence boundaries:** any commercial or production use of the weights — revenue-generating activity, direct or indirect interaction with end users or production systems, client deliverables, paid products or services, and training, fine-tuning or distilling other models for commercial use — is outside the TimesFM Non-Commercial License v1.0, as is redistributing the weights or a derivative. The repository's TimesFM 2.5 build (Apache-2.0 weights) exists for uses the restriction does not fit.
2. **Capability boundaries:** covariates (past-only or known-future regressors; the upstream `past_only_covariates` / `past_future_covariates` inputs are not exposed), multivariate joint forecasting through variate attention across series (each series is forecast independently), fine-tuning or any training, anomaly detection, imputation as a product feature (the validator's interpolation is a reported data policy, not a model), classification, and representation extraction are outside this repository.
3. **Input boundaries:** a series needs at least `horizon + 16` rows and at least 3 rows to infer its interval; files with duplicate timestamps within a series, irregular spacing that is neither one fixed interval nor a calendar interval, non-numeric or infinite values, more than 1,000 series or 2,000,000 rows are refused; missing values and skipped grid points are refused unless `nan_policy="interpolate"` / `gap_policy="fill"` are set, and at most 20 % of a series may be interpolated; histories longer than 15,360 points are truncated to their most recent points with a report; the horizon is capped at 512 steps and the context at 15,360; quantile levels outside the nine trained deciles are refused rather than interpolated.
4. **Decision boundaries:** autonomous or high-consequence decisions taken from a forecast without independent domain validation on the operator's own history, monitoring and human review are out of scope.

---

#### Factors

###### Groups

The pipeline is not human-centric: it consumes numerical series without demographic attributes, and the shipped sample is gridded weather. The upstream pretraining corpus (GiftEvalPretrain excluding the datasets that overlap fev-bench, Wikipedia pageviews with a November 2023 cut-off, Google Trends top queries with an end-of-2022 cut-off, and synthetic and augmented data, per the checkpoint's model card) is not group-audited by this repository, and the upstream authors do not publish a group-level analysis. A study whose series describe people — footfall, demand by neighbourhood, service queues — can inherit structural bias from that data and from the pretraining mixture. The operator of such a study is expected to perform a subgroup error analysis on their own holdout, comparing `mase` and band coverage across the groups their analysis concerns, before drawing conclusions from the forecasts.

###### Instrumentation

Evaluation data in this repository is produced by the Open-Meteo historical weather API, which serves reanalysis estimates (ECMWF ERA5 among its sources) for model grid cells several kilometres across; the values are hourly, in °C with one decimal, and describe a grid cell, not a weather station. The pretraining data was produced by the systems the upstream authors name (web pageview counters, search-trend aggregates, public benchmark collections) and by synthetic generators; their sampling rates and calibration are not disclosed beyond that. Instrument error reaches the model as value error: a sensor drift or a changed collection procedure arrives as a level shift the model forecasts forward as if real, and a clock fault arrives as a gap or a duplicate. The validator detects the structural defects it can see — duplicate timestamps, skipped grid points, irregular spacing, missing and non-finite values — and refuses or reports them; it cannot detect drift or recalibration in the values themselves.

###### Environment

Operating environment: Python `3.12` (the notebook builds a managed CPython `3.12.12`), `timesfm==3.0.2`, `torch==2.14.0`, `numpy==2.5.3`, `pandas==3.0.5`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, locked with hashes in `tutorials/requirements-colab.lock.txt` and `uv.lock`. CPU in float32 is the default; CUDA is used automatically when available; the upstream forecaster's batch size is its default of 4 series. The notebook supports Linux x86_64 runtimes only because its lock holds manylinux wheels. Data environment: the model assumes the deployment series is regularly sampled and that its recent history is informative about its near future; a structural break, a regime change, a new series shorter than the model has effectively seen, or a horizon far beyond the context's cycle degrades the forecast without any signal of its own unreliability. The shipped sample's two-week window (2026-08-26 to 2026-09-08) lies after every pretraining cut-off the checkpoint's model card states; the checkpoint's Hub revision date (2026-09-02) overlaps the window, so the sample's exact values are unlikely, not impossible, to be in the pretraining data, and weather from the same sources for earlier periods may be.

---

#### Metrics

The subsections state what the evaluation code reports, the decision rule, and how the numbers were estimated.

###### Performance Measures

`evaluate_forecast` reports, per series and pooled: `mae`; **MASE** (`mase`) — mean absolute error divided by the in-sample mean absolute error of the seasonal-naive forecast at `season_length` (or 1), which is unit-free and lets series of different scales be pooled and read against the seasonal rule; **sMAPE** (`smape`) — symmetric mean absolute percentage error in percent, scale-free but undefined near zero and so reported beside MASE; **quantile loss** (`quantile_loss`, the pinball loss) at each exported quantile with its mean (`mean_quantile_loss`), the proper score for a quantile forecast, whose `q0.5` term equals half the MAE; and **interval coverage** (`interval_coverage`) — the share of held-out points inside the outermost exported quantile band, against its nominal width. `evaluation_report` sets these beside the same `mae`, `mase` and `smape` of the `naive`, `seasonal_naive` and `ses` baselines and records which baselines the model beats on `mase`; it asserts nothing. Reading MASE alone would hide the calibration of the band; reading coverage alone would hide how far the median is from the truth. RMSE, CRPS and WAPE are not implemented. All reported values are tutorial metrics on one holdout, not benchmark results; the upstream authors' benchmark figures are not reproduced here.

###### Decision thresholds

The point forecast is the model's median (quantile 0.5, slot 4 of the nine quantile slots), exported as `prediction`; the wrapper asserts at every call that the upstream point output equals the median slot and raises `ModelIntegrityError` otherwise. No mean is exported because TimesFM 3.0 has no mean head, and the pipeline does not derive one from the quantiles. No acceptance threshold was set during development: the evaluation report records whether the model's `mase` is below each baseline's and does not stop the run on the answer, because a seasonal rule winning on a short periodic sample is a legitimate result. No operational threshold — an alert level, a reorder point, a staffing cut-off — is shipped; an operator who sets one does so on their own holdout from the asymmetric cost of a forecast that is too high against one that is too low, using the quantile columns rather than the median where the costs differ, and within the non-commercial, non-production licence.

###### Approaches to uncertainty and variability

Every metric comes from a single chronological tail holdout of `horizon` points per series (24 by default), with no dispersion estimate: no repeated origins, folds or bootstrap are run, so one value is one observation and the report labels it `sample-sanity`. Predictive uncertainty is represented by the model's quantiles from the trained decile grid, sorted monotone by the upstream forecaster (`sort_quantiles=True`); they are uncalibrated model quantiles, not prediction intervals, and `interval_coverage` on the holdout is the only calibration evidence the pipeline produces. A caller who needs calibrated intervals must measure coverage over a backtest with many origins on their own data and recalibrate (for example conformally). Run-to-run variability: the pipeline draws no samples and uses no seed; CPU float32 inference is deterministic for fixed inputs and pins, CUDA kernels may differ in the last digits, and the hash-locked dependency set removes version drift.

---

#### Ethical considerations and biases

No external board reviewed this pipeline and no clearance testing with a specific group took place; the considerations below are the developers' own.

###### Data

The checkpoint was pretrained, per its model card, on GiftEvalPretrain (excluding the datasets that overlap fev-bench), Wikipedia pageviews (cut-off November 2023), Google Trends top queries (cut-off end of 2022) and synthetic and augmented data; the disclosure ends at those names, and the mixture cannot be reconstructed from this repository. Whether any of it contains personal or proprietary series is not ruled out: pageview and query aggregates are derived from human activity. This repository distributes code, a 31 KB weather sample (Open-Meteo, CC BY 4.0, no personal data) and a file manifest; it does not distribute the weights, which the notebook downloads from the Hub at run time under their non-commercial licence, and it commits no caches. The operator is responsible for auditing the series they supply at inference for personal, confidential or regulated content; the pipeline inspects schema and values, not meaning.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or emergency response, and this repository does not certify TimesFM for any of them; the weights licence additionally excludes production deployment of any kind. Its validation is limited to offline unit and contract tests, a stand-in execution of the notebook's cells, and — pending — a hosted run of the notebook; no clinical, regulatory or independent domain validation has been carried out by the developers or by any external body. Where a sensitive-domain study is foreseeable — forecasting patient arrivals in a research setting, for example — it would be admissible only as non-production research with independent validation on that operator's own history, a human decision-maker between any forecast and any action, and any clearance the domain requires.

###### Mitigations

1. **Supply-chain integrity:** `load_pinned_model` accepts only the committed manifest's identity (`google/timesfm-3.0-pytorch`); `check_model_source` refuses other repositories, local paths, URIs and other refs; `verify_snapshot` checks every manifest entry's byte size, checks `config.json` against its pinned SHA-256 and against the pinned architecture values (`input_patch_len` 32, `output_patch_len` 64, the nine quantiles, 20 layers, model dimension 1280, 16 heads, `use_rope_var: false`, variate attention, iterative CPM RevIN, linear detrending, stitching), hashes `model.safetensors` and reports its digest as unverified until the maintainer pins it, and refuses any pickle-format file in the snapshot. The forecaster is given the verified snapshot directory with `local_files_only=True`, so it builds the model from the staged `config.json` and cannot download at load time; the loader then checks that the forecaster reports the pinned patch lengths, quantiles and median index. The notebook asserts the inline manifest equals the carried module's identity before fetching.
2. **Licence transparency:** `MODEL_LICENSE_TERMS` is carried in the module, written to `resolved-revision.json` by `stage_missing_files`, exported in every provenance record, and stated in the notebook's header, prerequisites and the note before the download cell, so a run cannot stage the weights without the terms being recorded and shown.
3. **Input integrity:** `load_series_csv` refuses a missing file, an empty file, duplicate header names and missing columns; `validate_series` refuses unparseable or duplicate timestamps, irregular spacing, gaps and missing values (unless the named policies are set, in which case every fill is reported in the input manifest), non-numeric and infinite values, mixed intervals across series, and lengths below `horizon + 16`; every refusal is a `ValidationError` naming the file and the rule.
4. **Output integrity:** the forecasting wrapper checks the upstream output shapes (`(n, horizon)` and `(n, horizon, 9)`), finiteness and the median contract (slot 4) on every call and maps each quantile to its slot by level; `evaluate_forecast` refuses forecast and truth rows that do not align on `(series_id, timestamp)` instead of dropping them.
5. **Reproducibility:** hash-locked dependencies (`--require-hashes --only-binary :all:`), a notebook generated from the package with a parity check in CI, and provenance on every bundle recording identity, licence, revision status, observed digests, versions, device, decode settings and the input manifest.
6. **Refusals:** no random split helper exists; quantile levels off the trained grid are refused; horizons above 512 are refused; a different model source is refused; remote model code is never requested.

###### Risks and harms

Model-intrinsic: a regime change, a structural break or a series unlike the pretraining distribution produces a confident-looking forecast with no signal of its unreliability, and the operator — and whoever the operator's decision affects — bears the cost; the quantile band is uncalibrated, so an operator who provisions to `q0.9` under-provisions at a rate the pipeline does not measure (its one-day coverage figure is one observation); the model cannot see drivers on this path, so a forecast across a holiday, an outage or an intervention is wrong in a direction a covariate-aware method would have caught. Use-context: automation bias, where a numerically precise median displaces the operator's judgement; a wrong `season_length` that silently mis-scales MASE and makes a weak seasonal baseline look beaten; instrument drift forecast forward as a real trend; leakage, where a holdout built outside this package exposes future values to the context; and licence drift, where a research forecast is carried into a commercial decision, a client deliverable or a production system in breach of the weights licence. Likelihood under normal use is highest for the calibration, `season_length` and licence-drift risks because none needs a fault to occur; magnitude scales with what the forecast gates, from an over-ordered inventory to a mis-staffed shift, and, for licence drift, with the legal exposure of the operator.

###### Use cases

Distinct from the boundaries above, the developers consider the following uses prohibited even where the model would produce a plausible forecast: any commercial or production use of the weights, which the TimesFM Non-Commercial License v1.0 does not grant; surveillance or profiling of individuals from telemetry that traces to a person; forecasting a person's behaviour, creditworthiness, employment, housing or insurance outcome as an input to a decision about that person; social scoring; autonomous control of physical systems where a forecast error can injure someone; presenting a model quantile as a certified prediction interval; redistributing the weights or a fine-tuned derivative; and any use that violates the weights licence, the Apache-2.0 terms of the upstream inference code or the terms of the runtime that executes the pipeline. The repository is not authorisation for any of these, and organisational controls remain the operator's obligation.

---

## Checkpoint and runtime provenance

| Item | Value | Evidence level |
|---|---|---|
| Model id | `google/timesfm-3.0-pytorch` | verified: Hub listing (2026-10-06) |
| Revision | `43046b85ec22d584a13f8098c2ed39c889e129c2` — the immutable commit pinned in `MODEL_REVISION` and the manifest; `stage_missing_files` requests every file at it, writes the commit the Hub reports to `resolved-revision.json` and refuses a different one | observed: the commit the Hub served for ref `main` on the maintainer's hosted Colab run of 2026-10-06 (`resolved_revision` in `docs/execution-evidence/2026-10-06/timesfm_forecasting_colab_86f2213_run2_passed.ipynb`) |
| Files in the checkpoint repository | `.gitattributes` (1,519 bytes), `LICENSE` (7,270 bytes), `README.md` (1,436 bytes), `config.json` (1,273 bytes), `model.safetensors` (1,322,898,824 bytes, LFS); the manifest stages `config.json` and `model.safetensors` | verified: Hub listing |
| `config.json` SHA-256 | `ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6` | computed from the Hub-served text (1,273 bytes, equal to the listed size) |
| `model.safetensors` SHA-256 | `a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da` — pinned in the manifest; `verify_snapshot` refuses the snapshot on a mismatch (size and digest, every file) | observed: computed in-run over the 1,322,898,824-byte staged file on the 2026-10-06 hosted run (`weights_sha256` in `docs/execution-evidence/2026-10-06/timesfm_forecasting_colab_86f2213_run2_passed.ipynb`; that run reported `verified_against_manifest: False` because the manifest then held a pending sentinel). The run predates the pin; the next hosted run on the pinned head re-verifies it |
| Weights licence | `timesfm-non-commercial-license-v1.0` (TimesFM Non-Commercial License v1.0): non-commercial and non-production use only; no distribution of the model or derivatives | verified: model card front matter (`license: other`, `license_name`, `license_link: LICENSE`) and the full `LICENSE` text |
| Architecture | input patch 32, output patch 64, quantiles 0.1–0.9 (median at slot 4), 20 layers, model dimension 1280, 16 heads, `use_rope_seq: true`, `use_rope_var: false`, variate attention, iterative CPM RevIN, linear detrending (threshold 0.5), stitching; 330.7M parameters | verified: `config.json` text; the parameter count is the Hub's figure |
| Checkpoint date | `date_published: 2026-09-02` | the Hub's last-updated date through the connector; the commit history was not readable, so the maintainer confirms it |
| Package | `timesfm==3.0.2` (PyPI, wheel SHA-256 `cfa3d22ddc3f019f0d9d95f19055428c4a4713526dafd7e5d1a7769782e7613f`, uploaded 2026-09-09), Apache-2.0; loader `timesfm3.TimesFM3Forecaster` | verified: PyPI metadata and wheel contents |
| Runtime pins | `torch==2.14.0`, `numpy==2.5.3`, `pandas==3.0.5`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `matplotlib==3.10.8` | locked with hashes |
| Decode settings | `predict_batch` with `return_quantiles=True`, `sort_quantiles=True`, `use_symmetric_averaging=False`, `make_positive=False`, `use_znorm=False`, `padding_mode="none"`, `per_core_batch_size=4`; context padded to a multiple of 32 up to 15,360, horizon decoded in 64-point patches | code (`model.py`, `DECODE_SETTINGS`) |
| Fallback build | TimesFM 2.5 (`google/timesfm-2.5-200m-pytorch`, Apache-2.0 weights) at commit `e47259ae75b93e7611c67e13a14d6607997c38f3` | verified: repository history |

## Verification records

Two maintainer runs on Google Colab are recorded in `docs/release-verification.md` (2026-10-06). The first (`c7bf4a6`) failed at the first forecast and is not evidence of anything beyond the staging and loading cells. The second is recorded here; `clean_runtime_evidence` stays `false` because BYOD, an export-cell re-run and a repeated Run all were not exercised.

- **Date:** 2026-10-06
- **Subject:** `tutorials/timesfm_forecasting_colab.ipynb` at repository commit `86f2213` (notebook blob `c6a9263a`), **default path** (`USE_BYOD = False`, every form field at its default)
- **Runtime:** Google Colab, CUDA (`cuda:0`), isolated environment Python `3.12.12`, `torch 2.14.0+cu130`, `timesfm 3.0.2`
- **Procedure:** maintainer **Run all** on a fresh runtime, one pass; the executed notebook is kept as `docs/execution-evidence/2026-10-06/timesfm_forecasting_colab_86f2213_run2_passed.ipynb`
- **Observed result:** 19/19 code cells, no errors. The Hub served commit `43046b85ec22d584a13f8098c2ed39c889e129c2` for ref `main` (`resolved_revision`), and `verify_snapshot` computed `model.safetensors` SHA-256 `a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da` over the 1,322,898,824-byte staged file. Held-out 24 h × 3 series: TimesFM MASE 0.6943 / sMAPE 1.698 against seasonal-naive 0.7417 / 1.837, naive 3.2201, SES 3.2398; q0.1–q0.9 coverage 0.764 (nominal 0.8); reload parity ok
- **Caveats:** at that commit `MODEL_REVISION` was still the ref `main` and the manifest held a pending sentinel, so the run reported `weights_sha256_verified_against_manifest: False`; the commit and the digest it observed are now pinned in `model.py` and the manifest, which means **the run predates the pin** and the next hosted run on the pinned head is the one that re-verifies the pinned digest and commit. One seeded holdout, not a benchmark; BYOD (REL12), an export-cell re-run and a repeated Run all were not exercised

The offline checks below are static or stand-in evidence, not executions of the pretrained model:

- **Date:** 2026-10-06
- **Subject:** `tests/` and `tools/validate_release_assets.py` at the TimesFM 3.0 port revision of the pull-request branch
- **Runtime:** CPU-only Linux host, Python `3.12.12`, `numpy==2.5.3`, `pandas==3.0.5`; no `torch`, no `timesfm`
- **Procedure:** `pytest` over the unit, contract, BYOD-matrix and stand-in tests, the notebook's own cells executed with stub `torch`/`timesfm3` modules and a sparse placeholder weight file; `ruff`; generator `--check`; a dry-run install of the hash lock
- **Observed result:** all checks passed; the stand-in produced the full result bundle with reload parity and the seven-column forecast (`prediction`, `q0.1`, `q0.5`, `q0.9`, no `mean`)
- **Caveats:** stand-in evidence only — no pretrained inference occurred and no hosted runtime was used; the stand-in substitutes the sparse placeholder's digest, so it does not verify the checkpoint

## References

- Das, A., Kong, W., Sen, R., & Zhou, Y. (2024). A decoder-only foundation model for time-series forecasting. *ICML 2024*. https://arxiv.org/abs/2310.10688
- Google Research. TimesFM 3.0 PyTorch checkpoint (TimesFM Non-Commercial License v1.0). https://huggingface.co/google/timesfm-3.0-pytorch
- google-research/timesfm. https://github.com/google-research/timesfm
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting, 22*(4), 679–688.
