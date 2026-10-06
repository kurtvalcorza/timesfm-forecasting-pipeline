# TimesFM Forecasting — DIMER Pipeline

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/timesfm-forecasting-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/timesfm-forecasting-pipeline/blob/main/tutorials/timesfm_forecasting_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Ftimesfm--3.0--pytorch-ffcc4d?style=flat)](https://huggingface.co/google/timesfm-3.0-pytorch)
[![Upstream](https://img.shields.io/badge/Upstream-google--research%2Ftimesfm-181717?style=flat&logo=github&logoColor=white)](https://github.com/google-research/timesfm)
[![Code: Apache-2.0](https://img.shields.io/badge/Code-Apache_2.0-blue.svg)](LICENSE)
[![Weights: non-commercial](https://img.shields.io/badge/Weights-TimesFM%20Non--Commercial%20v1.0-red.svg)](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)

A DIMER pipeline that runs **zero-shot time-series forecasting** with Google's
[TimesFM 3.0](https://huggingface.co/google/timesfm-3.0-pytorch), a pretrained forecasting
foundation model, and teaches it in a guided Colab notebook. You supply a long-format CSV (timestamp,
value, optional series id); the pipeline validates it, stages and checks one pinned checkpoint, and
returns a median point forecast with model quantiles, scored chronologically beside naive, seasonal-naive
and exponential-smoothing baselines. Nothing is trained or fine-tuned.

> **Weights licence.** The TimesFM 3.0 weights are distributed under the *TimesFM Non-Commercial License
> v1.0*: **non-commercial and non-production use only**, no redistribution of the model or of derivatives.
> The code here is Apache-2.0. See [Why TimesFM 3.0](#why-timesfm-30) and [NOTICE](NOTICE).

See [MODEL_CARD.md](MODEL_CARD.md) for capabilities, limits, supply-chain pins and licence.

## Status

**Candidate** — verification pending; no hosted run recorded; `clean_runtime_evidence: false`. See
[STATUS.md](STATUS.md) and [docs/release-verification.md](docs/release-verification.md). The checkpoint
is pinned by repository id, file names, byte sizes and the digest of its `config.json`; its immutable
commit and the digest of `model.safetensors` are **to be confirmed on the first hosted run**, which prints
and exports what the Hub served so the maintainer can pin them.

## Why TimesFM 3.0

The maintainer chose the current TimesFM generation. The `timesfm` package on PyPI (pinned at `3.0.2`) ships
the 3.0 implementation as its `timesfm3` package, and `google/timesfm-3.0-pytorch` is the checkpoint it
loads: a stacked mixing transformer (20 layers, model dimension 1280, 16 heads; 330.7M parameters per the
Hub) with variate attention and iterative CPM RevIN, reading 32-point input patches, decoding 64-point
output patches, with a 15,360-point context limit and nine quantile slots (deciles 0.1–0.9; the median at
slot 4 is the point forecast; there is no mean head). The pipeline passes plain numbers with no frequency
label and exposes no covariates.

**Licence restriction.** The 3.0 weights are distributed by Google LLC under the
[TimesFM Non-Commercial License v1.0](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)
(Hub front matter `license: other`, `license_name: timesfm-non-commercial-license-v1.0`). In the licence's
own terms that is **non-commercial and non-production use only**: testing, evaluation and research not tied
to commercial gain, production deployment or revenue generation; no use in end-user-facing or production
systems; no redistribution of the model or of derivatives; a commercial licence from Google for anything
else. The pipeline code, tests, tooling and notebook are Apache-2.0 and the sample is CC BY 4.0; the
Apache-2.0 licence does not extend to the weights, which the notebook downloads at run time and this
repository never commits. The restriction is stated in [NOTICE](NOTICE), [MODEL_CARD.md](MODEL_CARD.md),
[STATUS.md](STATUS.md) and in the notebook's header, prerequisites and the note before its download cell.

**Fallback.** If the restriction does not fit your use, the repository's TimesFM 2.5 build — pinned to
`google/timesfm-2.5-200m-pytorch`, whose weights are Apache-2.0 — is the fallback. It is the state of this
branch at commit `e47259ae75b93e7611c67e13a14d6607997c38f3` (same package layout and notebook contract;
2.5 exports an additional `mean` column and uses a 16,384-point context limit).

## Live tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/timesfm-forecasting-pipeline/blob/main/tutorials/timesfm_forecasting_colab.ipynb)

[`tutorials/timesfm_forecasting_colab.ipynb`](tutorials/timesfm_forecasting_colab.ipynb) is declared
`TASK-INFERENCE` / `GUIDED` under DIMER Notebook Specification 2.2 and is **standalone**: generated by
`tools/build_notebook.py` (`build_notebook.py/2.2`) from `tools/notebook_template.py`, it carries the
package's seven modules, the model identity and manifest, and a hash-locked runtime, so it runs without
this repository. Never edit the `.ipynb` by hand; edit the package or the template and regenerate
(`python tools/build_notebook.py`; `--check` is enforced in CI). The default path:

1. builds an isolated `uv` environment (managed CPython 3.12.12, 52 hash-locked manylinux wheels) and routes
   every later cell to it — nothing is installed into the kernel and no restart is needed;
2. carries the package verbatim, states the weights licence, then stages `google/timesfm-3.0-pytorch`
   (1.32 GB) and checks sizes, the `config.json` digest and the pinned architecture (the weight digest is
   reported, pending its pin);
3. fetches the digest-pinned sample — hourly 2 m air temperature for Manila, Cebu and Davao, 14 days,
   Open-Meteo, CC BY 4.0, this repository's own committed copy at a pinned commit — or your own CSV via
   `BYOD_PATH` / the guarded Colab upload;
4. validates into an input manifest with named refusals, holds out the last 24 hours chronologically,
   scores three history-only baselines, runs the zero-shot forecast with `prediction` (median),
   `q0.1`, `q0.5`, `q0.9`, and records MASE, sMAPE, quantile loss and band coverage beside the baselines
   with **no quality assertion**;
5. runs one Predict → Change one thing → Run → Observe → Explain activity (context length), forecasts
   beyond the data (`not-measurable`), and exports a result bundle (`result.json`, forecast CSV,
   evaluation and provenance JSON) that is reloaded with a parity check.

The guided layer (audience, how to use, roadmap, task contract, predictions, checkpoints, troubleshooting,
glossary, conclusion scaffold) follows NOTEBOOK_SPEC 2.2 §3.5; infrastructure cells are labelled and
collapsed. Checkpoint answers are qualitative because no run is recorded yet.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The core package needs only numpy and pandas;
the model stack is the `model` extra.

```bash
uv sync --locked --extra dev                 # what CI installs: no torch
uv run ruff check src tests tools
uv run pytest -m "not integration"           # unit, contract, BYOD matrix, stand-in notebook exec
uv run python tools/validate_release_assets.py
uv run python tools/build_notebook.py --check

uv sync --locked --extra dev --extra model   # real weights (downloads ~1.32 GB on first use; non-commercial licence)
uv run pytest -m integration
```

A minimal forecast:

```python
from timesfm_forecasting import ForecastConfig, forecast, load_pinned_model, load_series_csv, validate_series

config = ForecastConfig(horizon=24, context_length=512, series_id_column="series_id", season_length=24)
frame = load_series_csv("my_series.csv", config)              # timestamp, value, series_id
validation = validate_series(frame, config, source="my_series.csv")
model = load_pinned_model(device="auto")                       # only the pinned checkpoint is accepted
result = forecast(validation, config, model)

result.forecast      # series_id, timestamp, step, prediction, q0.1, q0.5, q0.9
result.provenance    # horizon, effective context per series, decode settings, semantics
```

`prediction` is the **median (q0.5)**, slot 4 of the model's nine quantile slots; TimesFM 3.0 has no mean
head and the pipeline invents none. The nine slots are sorted monotone by the upstream forecaster.

## Chronological evaluation

```python
from timesfm_forecasting import (
    chronological_holdout, evaluate_forecast, evaluation_report,
    naive_baseline, seasonal_naive_baseline, ses_baseline,
)

split = chronological_holdout(validation.frame, config.horizon)      # last `horizon` rows per series
result = forecast(validation.with_frame(split.history), config, model)
model_eval = evaluate_forecast(result.forecast, split.truth, split.history, config)
baselines = {
    "naive": naive_baseline(split.history, validation.frequency, config.horizon),
    "seasonal_naive": seasonal_naive_baseline(split.history, validation.frequency, config.horizon, 24),
    "ses": ses_baseline(split.history, validation.frequency, config.horizon),
}
report = evaluation_report(
    model_eval, {k: evaluate_forecast(v, split.truth, split.history, config) for k, v in baselines.items()},
    horizon=config.horizon, sample_kind="my series",
)
```

There is no random split helper. Forecast and truth must align exactly on `(series_id, timestamp)`.
Metrics: `mase` (scaled by the in-sample seasonal-naive MAE at `season_length`), `smape`, `quantile_loss`
per level, `interval_coverage`. The report records which baselines the model beats; it asserts nothing.

## Data contract

One CSV, long format. Rules (every refusal reads `[RULE] file: reason`):

| Rule | Refuses |
|---|---|
| `REQUIRED_COLUMNS`, `DUPLICATE_HEADER`, `FILE_EMPTY`, `FILE_NOT_FOUND` | header problems, before parsing |
| `TIMESTAMP_PARSE`, `TIMESTAMP_DUPLICATE` | unparseable or repeated timestamps within a series |
| `FREQUENCY_IRREGULAR`, `FREQUENCY_GAP`, `FREQUENCY_MIXED` | spacing that is neither fixed nor calendar; skipped grid points (unless `gap_policy="fill"`); different intervals across series |
| `VALUE_NUMERIC`, `VALUE_INFINITE`, `VALUE_MISSING`, `VALUE_MISSING_FRACTION` | non-numeric, infinite or missing values (unless `nan_policy="interpolate"`, at most 20 % of a series) |
| `MIN_OBSERVATIONS`, `MIN_HISTORY`, `MAX_SERIES`, `MAX_ROWS` | fewer than 3 rows; fewer than `horizon + 16` rows; more than 1,000 series or 2,000,000 rows |

Histories above 15,360 points (the model's context limit) are truncated to their most recent points and the
truncation is reported.
Monthly, quarterly and yearly series are accepted as calendar frequencies.

## Repository layout

```text
src/timesfm_forecasting/
  errors.py        refusal codes
  config.py        ForecastConfig and the named ceilings
  data.py          CSV loading, frequency inference, gap/NaN policies, input manifest
  model.py         pinned identity, staging, integrity verification, lazy loading
  forecasting.py   median + quantile forecast over a validated table
  evaluation.py    chronological holdout, baselines, MASE / sMAPE / quantile loss, recorded verdicts
  provenance.py    runtime versions, result bundle, reload parity
tools/
  build_notebook.py          fleet generator (/2.2) + the `revision_pending` and `weights_licence_note` keys
  notebook_template.py       the notebook's prose and stage cells
  validate_release_assets.py static release-asset validator (not execution evidence)
  run_notebook.py            sequential cell executor (CI / stand-in)
tutorials/                   notebook, requirements-colab.in (pins), requirements-colab.lock.txt (hashes)
weights/timesfm-3.0-pytorch/dimer-base-manifest.json   file manifest and licence record (weights never committed)
examples/sample-data/        the Open-Meteo sample, its SHA256SUMS and dataset card
tests/                       unit, contract, BYOD matrix, stand-in notebook execution
docs/release-verification.md release gate and the (empty) recorded-executions table
```

## Licence

Pipeline code: [Apache-2.0](LICENSE), Copyright 2026 Kurt Valcorza. The `timesfm` package (inference code)
is Apache-2.0 (Google LLC). The **TimesFM 3.0 weights are not Apache-2.0**: they are under the TimesFM
Non-Commercial License v1.0 — non-commercial and non-production use only, no redistribution — and are
downloaded at run time, never committed. The sample is CC BY 4.0 (Open-Meteo). See [NOTICE](NOTICE),
[MODEL_CARD.md](MODEL_CARD.md) and [Why TimesFM 3.0](#why-timesfm-30).

## AI Assistance Disclosure

This repository's code and accompanying documentation were developed with generative AI assistance for
code development and technical writing under maintainer direction. The maintainer remains responsible for
reviewing the implementation, validating results, and making release decisions. AI assistance does not
constitute independent verification, provider endorsement, or release approval.
