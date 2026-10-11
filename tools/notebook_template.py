"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier, /2.2).

Only the task-specific prose and stage cells live here. The isolated-runtime install cell, the carried
package modules (seven, in dependency order) and the model pin/stage/verify cell are produced by the
generator from repository sources so they cannot drift from the package.

This template configures a GUIDED TASK-INFERENCE unit ("Can a foundation model beat a seasonal rule of
thumb?"): the pinned TimesFM 3.0 checkpoint (non-commercial weights) is staged and size/digest-checked; a digest-pinned real series
(Open-Meteo hourly temperature for three Philippine cities) is fetched, validated into an input manifest and
split chronologically; three history-only baselines are scored; the zero-shot forecast is run with point
(median) and quantile output; MASE, sMAPE and quantile loss are recorded beside the baselines with no quality
assertion; a bounded Predict → Change one thing → Run → Observe → Explain activity varies the context length;
a forecast beyond the end of the data is produced and marked not measurable; and a result bundle is exported
and reloaded with a parity check.

Stage-cell ``md``/``code`` strings are written as plain text and escaped by ``_cell`` (the generator runs
``str.format`` on them); ``__STEM__`` is the only placeholder, replaced by the notebook stem.
"""

# ruff: noqa: E501  -- learner-facing prose is kept on single lines so the rendered markdown stays readable
from __future__ import annotations

REPO = "timesfm-forecasting-pipeline"
STEM = "timesfm_forecasting"
# The committed copy under examples/sample-data/ of this repository, at the commit that added it.
SAMPLE_COMMIT = "e47259ae75b93e7611c67e13a14d6607997c38f3"
SAMPLE_URL = (
    f"https://raw.githubusercontent.com/kurtvalcorza/{REPO}/{SAMPLE_COMMIT}/"
    "examples/sample-data/openmeteo_ph_hourly_temperature.csv"
)
SAMPLE_SHA256 = "74163ee609cda87869b7c13f4c2aa59343f94b4ba42d2e57331034902fe04f1a"
SAMPLE_BYTES = 31275


def _cell(md: str, code: str | None = None) -> dict[str, str]:
    """Escape literal braces for the generator's ``str.format`` and substitute the stem placeholder."""

    def esc(text: str) -> str:
        return text.replace("{", "{{").replace("}", "}}").replace("__STEM__", "{stem}")

    out = {"md": esc(md)}
    if code is not None:
        out["code"] = esc(code)
    return out


# ---------------------------------------------------------------------------------------------------------
# Stage cells (Sections 4-12). Sections 1-3 are generator-owned infrastructure.
# ---------------------------------------------------------------------------------------------------------

CELL_4 = _cell(
    """## 4. Obtain and inspect the sample (or bring your own CSV)

*Core concept.* **Input:** one CSV, long format: a timestamp column, a numeric value column and, optionally, a series-id column. **System:** a pinned download with a size and SHA-256 check, or your own file. **Output:** a raw table the next section validates.

The default sample is **real weather**: hourly air temperature at 2 m for Manila, Cebu and Davao over 14 days (2026-08-26 to 2026-09-08 UTC), 3 series × 336 hours, from the Open-Meteo historical API (CC BY 4.0, *weather data by Open-Meteo.com*), fetched from a commit-pinned GitHub URL (this repository's own committed copy) and refused if its size or digest differs. The checkpoint's model card names its pretraining sources (GiftEvalPretrain, Wikipedia pageviews to November 2023, Google Trends to the end of 2022, synthetic and augmented data), and the observation window lies after every stated cut-off; the Hub revision of the checkpoint (2026-09-02) does overlap the window, so these exact values are unlikely, not impossible, to be in its pretraining data, and weather from the same sources for earlier periods may be. Three grid cells over two weeks are **tutorial data, not a benchmark**.

**Form fields.** `USE_BYOD`, `BYOD_PATH` and the three column fields (`TIMESTAMP_FIELD`, `VALUE_FIELD`, `SERIES_ID_FIELD`; leave the last empty for a single-series file) are the only values meant to be edited. With `USE_BYOD = True`, the cell reads `BYOD_PATH` when it is set and otherwise opens the Colab upload dialog (only then is `google.colab` imported). `HORIZON` is the number of steps held out and forecast; `CONTEXT_LENGTH` the most recent points shown to the model; `SEASON_LENGTH` the period of the seasonal-naive baseline and of MASE scaling (24 for hourly data with a daily cycle; set it to your data's period, or 1 to disable the seasonal baseline). `NAN_POLICY` and `GAP_POLICY` decide whether missing values and skipped timestamps are refused or filled with a report.

**Before you bring your own data**, read the data contract in the Prerequisites: your file needs at least `HORIZON + 16` rows per series, one sampling interval (or a calendar interval such as monthly), no duplicate timestamps, finite numeric values, and no more than 1,000 series or 2,000,000 rows. The next section names the file and the rule in every refusal. Do not upload data you are not authorised to process in a hosted runtime; the file stays in this runtime, and the notebook publishes nothing.

**What to notice.** The printed source record: the sample's URL, byte size and SHA-256, or your file's path; then the first rows and the per-series row counts.""",
    """# Learner-facing: data source and request settings (form fields are the interface for executors too)
USE_BYOD = False  # @param {type:"boolean"}
BYOD_PATH = ''  # @param {type:"string"}
TIMESTAMP_FIELD = 'timestamp'  # @param {type:"string"}
VALUE_FIELD = 'target'  # @param {type:"string"}
SERIES_ID_FIELD = 'series_id'  # @param {type:"string"}
HORIZON = 24  # @param {type:"integer"}
CONTEXT_LENGTH = 312  # @param {type:"integer"}
SEASON_LENGTH = 24  # @param {type:"integer"}
NAN_POLICY = 'refuse'  # @param ["refuse", "interpolate"]
GAP_POLICY = 'refuse'  # @param ["refuse", "fill"]

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_URL = '__SAMPLE_URL__'
SAMPLE_SHA256 = '__SAMPLE_SHA256__'
SAMPLE_BYTES = __SAMPLE_BYTES__
SAMPLE_LICENSE = 'CC BY 4.0 (weather data by Open-Meteo.com); the committed copy under examples/sample-data/ of kurtvalcorza/timesfm-forecasting-pipeline at a pinned commit'
Path('outputs').mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)

config = ForecastConfig(
    horizon=int(HORIZON),
    context_length=int(CONTEXT_LENGTH),
    quantile_levels=(0.1, 0.5, 0.9),
    timestamp_column=TIMESTAMP_FIELD.strip(),
    value_column=VALUE_FIELD.strip(),
    series_id_column=SERIES_ID_FIELD.strip() or None,
    season_length=int(SEASON_LENGTH) if int(SEASON_LENGTH) > 1 else None,
    nan_policy=NAN_POLICY,
    gap_policy=GAP_POLICY,
)


def fetch_sample(url, sha256, n_bytes, cache_dir='data'):
    \"\"\"Download the pinned sample once; refuse a size or digest mismatch before anything reads it.\"\"\"
    target = Path(cache_dir) / url.rsplit('/', 1)[-1]
    if target.is_file() and target.stat().st_size == n_bytes and hashlib.sha256(target.read_bytes()).hexdigest() == sha256:
        return target, False
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read(n_bytes + 1)
    if len(payload) != n_bytes or hashlib.sha256(payload).hexdigest() != sha256:
        raise RuntimeError(
            f'The sample download does not match its pinned size/SHA-256 ({len(payload)} bytes); refusing to use it. '
            'Run the cell again; if it repeats, the file at the pinned URL was altered.'
        )
    target.write_bytes(payload)
    return target, True


if USE_BYOD:
    if BYOD_PATH.strip():
        byod_path = Path(BYOD_PATH.strip())
        if not byod_path.is_file():
            raise FileNotFoundError(f'BYOD_PATH {byod_path} is not a file; fix the path or clear it to use the upload dialog.')
    else:
        try:
            from google.colab import files  # imported only on this branch
        except ImportError as exc:
            raise RuntimeError('No upload dialog outside Colab: set BYOD_PATH to a CSV on this runtime.') from exc
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise RuntimeError(f'Upload exactly one CSV (got {len(uploaded)}), or set BYOD_PATH.')
        name, payload = next(iter(uploaded.items()))
        byod_path = Path('data') / Path(name).name
        byod_path.write_bytes(payload)
    data_source = {'kind': 'byod', 'path': str(byod_path), 'bytes': byod_path.stat().st_size, 'sha256': hashlib.sha256(byod_path.read_bytes()).hexdigest(), 'license': 'user-supplied'}
else:
    sample_path, downloaded = fetch_sample(SAMPLE_URL, SAMPLE_SHA256, SAMPLE_BYTES)
    data_source = {'kind': 'sample', 'name': 'Open-Meteo hourly 2 m temperature, Manila/Cebu/Davao, 2026-08-26..2026-09-08 UTC', 'path': str(sample_path), 'url': SAMPLE_URL, 'bytes': SAMPLE_BYTES, 'sha256': SAMPLE_SHA256, 'downloaded_this_run': downloaded, 'license': SAMPLE_LICENSE}

frame = load_series_csv(data_source['path'], config)
print({k: v for k, v in data_source.items() if k != 'url'})
print(frame.head(5).to_string(index=False))
print({'rows': len(frame), 'rows_per_series': frame[config.series_id_column].value_counts().to_dict() if config.series_id_column else {'series': len(frame)}})""",
)

CELL_5 = _cell(
    """## 5. Validate before the model runs

*Evaluation / engineering practice.* Validation is a first-class stage, not a side effect: `validate_series` parses the timestamps, infers the sampling interval (one fixed step, or a calendar interval such as monthly), finds skipped grid points and missing values, applies `NAN_POLICY` / `GAP_POLICY` while **reporting every change**, enforces the length ceilings, and returns a normalised table (`series_id`, `timestamp`, `value`) plus an **input manifest** that is written to `outputs/`. The model has not run yet.

Every refusal reads `[RULE] file: reason` and names the series, row or column, so a wrong file fails here and not inside the model. To show that, the second half of the cell builds a deliberately broken copy of the first series (one duplicated timestamp) and asks validation to refuse it; the printed line is what a BYOD mistake looks like. The check that the refusal *is* a `ValidationError` is a contract check: the notebook stops if validation ever lets a duplicate timestamp through.

**Prediction.** Before running: which rule do you expect to be the most common reason a real-world CSV is refused — a missing column, a gap, or too few rows?

**What to notice.** The inferred frequency (`h` for the sample), the per-series `n_rows_out`, an empty `changes` list for the sample (nothing was imputed or truncated), and the `[TIMESTAMP_DUPLICATE]` refusal for the probe.""",
    """validation = validate_series(frame, config, source=data_source['path'])
input_manifest = validation.manifest()
with open('outputs/__STEM___input_manifest.json', 'w', encoding='utf-8') as handle:
    json.dump(input_manifest, handle, indent=2)
print({'frequency': input_manifest['frequency'], 'n_series': input_manifest['n_series'], 'n_rows': input_manifest['n_rows'], 'changes': input_manifest['changes']})
for report in input_manifest['series']:
    print({k: report[k] for k in ('series_id', 'n_rows_in', 'n_rows_out', 'start', 'end', 'n_interpolated', 'n_gap_points_filled', 'truncated_from')})
print({'ceilings': input_manifest['ceilings']})
print({'rules': list(RULES)})

# Refusal probe: a copy of the first series with one duplicated timestamp must be refused by name.
first_id = validation.series_ids[0]
probe = validation.frame[validation.frame['series_id'] == first_id].copy()
probe = pd.concat([probe, probe.iloc[[5]]], ignore_index=True)
probe_config = ForecastConfig(horizon=config.horizon, context_length=config.context_length, timestamp_column='timestamp', value_column='value', series_id_column='series_id', season_length=config.season_length)
try:
    validate_series(probe, probe_config, source='probe-duplicate-timestamp.csv')
except ValidationError as exc:
    refusal = {'code': exc.code, 'source': exc.source, 'message': str(exc)}
else:
    raise RuntimeError('validation accepted a duplicated timestamp; the data contract is broken')
print({'refusal_probe': refusal})""",
)

CELL_6 = _cell(
    """## 6. Split chronologically: the future is held out

*Core concept.* A forecast is only evidence if the model never saw the period it is scored on. `chronological_holdout` removes the **last `HORIZON` points of every series** as truth; everything before them is the history the baselines and the model may use. There is no random split in this package: shuffling a time series would let future values leak into the context.

**Prediction.** For the sample, the held-out day is 2026-09-08 UTC. Do you expect the three cities' temperature cycles on that day to look like the previous days (a daily cycle), or to contain something a history-only forecaster cannot know?

**What to notice.** The leakage check prints `True` per series — the last history timestamp is strictly before the first truth timestamp — and the notebook stops if it does not (a contract check, not a quality check).""",
    """split = chronological_holdout(validation.frame, config.horizon)
leakage = split.leakage_check()
history_validation = validation.with_frame(split.history)
print({'history_rows': len(split.history), 'truth_rows': len(split.truth), 'horizon': split.horizon, 'leakage_check': leakage})
for series_id, g in split.truth.groupby('series_id', sort=False):
    print({'series_id': series_id, 'truth_from': str(g['timestamp'].min()), 'truth_to': str(g['timestamp'].max()), 'history_to': str(split.history[split.history['series_id'] == series_id]['timestamp'].max())})
if not all(leakage.values()):
    raise RuntimeError(f'holdout leakage check failed: {leakage}; the split is not chronological')""",
)

CELL_7 = _cell(
    """## 7. Baselines first: what does "good" have to beat?

*Evaluation practice.* Three history-only forecasters set the bar before the model runs:

- **naive** — repeat the last observed value;
- **seasonal naive** — repeat the last full season (`SEASON_LENGTH` steps), the usual rule of thumb for data with a daily cycle;
- **SES** — simple exponential smoothing: a flat forecast at a level that weights recent points more, with the smoothing weight chosen on the history alone.

Each is scored with the same three metrics the model will get: **MASE** (mean absolute error divided by the in-sample seasonal-naive error; below 1 beats that rule on the history's own scale, and it is unit-free so series can be pooled), **sMAPE** (symmetric percentage error) and, for the model only, the **quantile loss** (pinball loss) at each exported quantile.

**Prediction.** Rank the three baselines before you run the cell. For hourly temperature with a clear daily cycle, which one should have the lowest MASE, and why can the naive forecast be badly wrong at a 24-step horizon?

**What to notice.** The table's `mase` column. A seasonal-naive MASE well below the naive one says the daily cycle is strong; a value near or below 1 says the last day was a good template for the next.""",
    """frequency = validation.frequency
baselines = {'naive': naive_baseline(split.history, frequency, config.horizon), 'ses': ses_baseline(split.history, frequency, config.horizon)}
if config.season_length is not None:
    baselines['seasonal_naive'] = seasonal_naive_baseline(split.history, frequency, config.horizon, config.season_length)
baseline_evals = {name: evaluate_forecast(frame_, split.truth, split.history, config) for name, frame_ in baselines.items()}
baseline_table = pd.DataFrame([{'method': name, **{k: round(ev['aggregate'][k], 4) for k in ('mae', 'mase', 'smape')}} for name, ev in baseline_evals.items()])
print(baseline_table.to_string(index=False))
print({'mase_scale': f"in-sample seasonal-naive MAE with season_length={config.season_length or 1}", 'holdout': 'last %d steps per series' % config.horizon})""",
)

CELL_8 = _cell(
    """## 8. Zero-shot forecast with TimesFM

*Core concept.* **Input:** the most recent `CONTEXT_LENGTH` points of each series, as plain numbers (TimesFM 3.0 needs no frequency label). **Model:** a stacked mixing transformer (20 layers, model dimension 1280, 16 heads; the Hub lists 330.7M parameters) pretrained on a large corpus of time series, reading 32-point input patches and decoding 64-point output patches; nothing is trained or tuned here, which is what *zero-shot* means. **Output:** for every series and future step, the **median** forecast (`prediction`, the model's quantile 0.5, slot 4 of its nine quantile slots) and the requested **quantiles** (`q0.1`, `q0.5`, `q0.9`). TimesFM 3.0 has no separate mean head, so none is exported.

**Quantiles are model quantiles.** The band between `q0.1` and `q0.9` is what the model's quantile head outputs for the nominal 80 % interval (sorted monotone by the upstream forecaster); it is **not** a calibrated prediction interval for this data until Section 9 measures how often the truth actually falls inside it. The point forecast is the median: with a skewed quantile spread it is not the centre of the band.

**Prediction.** Will the model's first few steps hug the last observed value (like the naive forecast) or follow the daily cycle (like the seasonal baseline)? Will the band widen with the horizon?

**What to notice.** The plot: history tail, truth (dashed), median forecast and the shaded q0.1–q0.9 band for the first series. Look at where the truth leaves the band.""",
    """import matplotlib.pyplot as plt

result = forecast(history_validation, config, pipe)
forecast_frame = result.forecast
print(forecast_frame.head(6).to_string(index=False))
print({'effective_context': result.provenance['effective_context_length'], 'decode_settings': result.provenance['decode_settings'], 'point_semantics': result.provenance['point_forecast_semantics']})

plot_id = validation.series_ids[0]
hist_tail = split.history[split.history['series_id'] == plot_id].tail(3 * config.horizon)
truth_part = split.truth[split.truth['series_id'] == plot_id]
fc_part = forecast_frame[forecast_frame['series_id'] == plot_id]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hist_tail['timestamp'], hist_tail['value'], color='0.3', label='history (tail)')
ax.plot(truth_part['timestamp'], truth_part['value'], color='black', linestyle='--', label='held-out truth')
ax.plot(fc_part['timestamp'], fc_part['prediction'], color='tab:blue', label='TimesFM median')
ax.fill_between(fc_part['timestamp'], fc_part['q0.1'], fc_part['q0.9'], color='tab:blue', alpha=0.2, label='q0.1-q0.9 (model quantiles)')
if 'seasonal_naive' in baselines:
    sn_part = baselines['seasonal_naive'][baselines['seasonal_naive']['series_id'] == plot_id]
    ax.plot(sn_part['timestamp'], sn_part['prediction'], color='tab:orange', linestyle=':', label='seasonal naive')
ax.set_title(f'{plot_id}: {config.horizon}-step zero-shot forecast vs held-out truth')
ax.legend(loc='upper left', fontsize=8)
fig.autofmt_xdate()
plt.show()""",
)

CELL_9 = _cell(
    """## 9. Evaluate beside the baselines (verdicts are recorded, not asserted)

*Evaluation practice.* The model's forecast is scored on the **same held-out rows with the same metrics** as the baselines, and the comparison is written to `outputs/` as an evaluation report. The report's verdict is `sample-sanity`: one chronological holdout on one small sample shows that the pipeline works end to end and how the model compares *on this sample*. It is not a benchmark, and the notebook does not stop if the model loses to a baseline — a foundation model losing to a seasonal rule on a strongly periodic two-week series is a legitimate, informative result.

Two more numbers belong to the model alone: the mean **quantile loss** over q0.1 / q0.5 / q0.9 (lower is better; the q0.5 term is half the MAE), and the empirical **coverage** of the q0.1–q0.9 band — the share of truth points inside it, to compare with the nominal 0.8.

**What to notice.** Whether `timesfm` has a lower MASE than `seasonal_naive`, and the coverage: well below 0.8 means the band is too narrow for this data; well above means too wide.

<details><summary>Check your reasoning</summary>

The seasonal-naive baseline encodes exactly the structure hourly temperature has — a daily cycle — so it is the one to watch. If the model beats it on MASE, the model has used more than the last day (several days of cycle, plus the trend between them). If it does not, remember what the holdout is: one day per city. A single unusual day (rain, a front) moves every method, and the ranking can flip on another day. Coverage below the nominal 0.8 is the common outcome for a zero-shot band on a new domain: read it as *the band is a model statement, not a guarantee*. Only a longer backtest over many origins could say whether the ranking is stable.

</details>""",
    """model_eval = evaluate_forecast(forecast_frame, split.truth, split.history, config)
sample_kind = data_source.get('name', 'user-supplied data') if data_source['kind'] == 'sample' else 'user-supplied data (BYOD)'
report = evaluation_report(model_eval, baseline_evals, horizon=config.horizon, sample_kind=sample_kind)
with open('outputs/__STEM___evaluation_report.json', 'w', encoding='utf-8') as handle:
    json.dump(report, handle, indent=2)
comparison_table = pd.DataFrame(report['table'])
print(comparison_table.round(4).to_string(index=False))
print({'verdict': report['verdict'], 'reason': report['reason']})
print({'model_beats_on_mase': report['model_beats_on_mase'], 'quantile_loss': {k: round(v, 4) for k, v in report['model_quantile_loss'].items()}, 'interval': report['interval'], 'interval_coverage': round(report['model_interval_coverage'], 3) if report['model_interval_coverage'] is not None else None})
for series_id, entry in model_eval['per_series'].items():
    print({'series_id': series_id, 'mase': round(entry['mase'], 4), 'smape': round(entry['smape'], 3), 'coverage': round(entry.get('interval_coverage', float('nan')), 3)})""",
)

CELL_10 = _cell(
    """## 10. Activity: Predict → Change one thing → Run → Observe → Explain

*Evaluation practice (optional; bounded; it does not change the canonical results above).* The one thing that changes is the **context length**: `ACTIVITY_CONTEXT_LENGTH` points of history instead of `CONTEXT_LENGTH`. Everything else — the split, the truth, the metrics, the baselines — stays fixed, so any difference in the table is caused by how much history the model sees.

1. **Predict.** With a short context (the default below is two days of hourly data), will the model's MASE get worse, better, or stay about the same? Will the q0.1–q0.9 band get wider or narrower?
2. **Change one thing.** Edit `ACTIVITY_CONTEXT_LENGTH` if you like (any value from 16 up to the series length); the default is a bounded, safe choice.
3. **Run** the cell.
4. **Observe** the two rows of the printed table.
5. **Explain** the difference in one sentence, in terms of what a shorter context removes (earlier days of the cycle, the week-scale trend) and what it keeps (the most recent day).

<details><summary>Check your reasoning</summary>

With a daily cycle, two days of context still contain the pattern, so the median often stays close; what usually changes is the spread, because the model has fewer repetitions of the cycle to judge its regularity from. A larger loss of accuracy with a short context points to information beyond the last two days — a slow drift between days — that the longer context carried. There is no universally right context length; this is why the pipeline reports the *effective* context in provenance rather than assuming one.

</details>""",
    """ACTIVITY_CONTEXT_LENGTH = 48  # @param {type:"integer"}

activity_config = ForecastConfig(
    horizon=config.horizon, context_length=int(ACTIVITY_CONTEXT_LENGTH), quantile_levels=config.quantile_levels,
    timestamp_column=config.timestamp_column, value_column=config.value_column, series_id_column=config.series_id_column,
    season_length=config.season_length, nan_policy=config.nan_policy, gap_policy=config.gap_policy,
)
activity_result = forecast(history_validation, activity_config, pipe)
activity_eval = evaluate_forecast(activity_result.forecast, split.truth, split.history, activity_config)
width = lambda f: float((f['q0.9'] - f['q0.1']).mean())  # noqa: E731 - one-line helper for the printed table
activity_table = pd.DataFrame([
    {'context_length': config.context_length, 'effective': result.provenance['effective_context_length'], 'mase': round(model_eval['aggregate']['mase'], 4), 'smape': round(model_eval['aggregate']['smape'], 3), 'mean_band_width': round(width(forecast_frame), 3), 'coverage': model_eval['aggregate'].get('interval_coverage')},
    {'context_length': activity_config.context_length, 'effective': activity_result.provenance['effective_context_length'], 'mase': round(activity_eval['aggregate']['mase'], 4), 'smape': round(activity_eval['aggregate']['smape'], 3), 'mean_band_width': round(width(activity_result.forecast), 3), 'coverage': activity_eval['aggregate'].get('interval_coverage')},
])
print(activity_table.to_string(index=False))
activity_record = {'changed': 'context_length', 'values': [config.context_length, activity_config.context_length], 'mase': [model_eval['aggregate']['mase'], activity_eval['aggregate']['mase']], 'mean_band_width': [width(forecast_frame), width(activity_result.forecast)], 'verdict': 'recorded; no quality assertion'}
print(activity_record)""",
)

CELL_11 = _cell(
    """## 11. Forecast beyond the end of the data

*Core concept.* The evaluation above forecast a period whose truth we had. The operational use is the other one: give the model **all** the history and forecast the next `HORIZON` steps, for which no truth exists yet. The output has the same columns as before and is written to `outputs/`; its evaluation verdict is `not-measurable`, and the notebook says so rather than inventing a score.

**What to notice.** The timestamps continue the series' grid (the next hours after 2026-09-08 23:00 UTC for the sample), and the first forecast values start near the last observed value.""",
    """future_result = forecast(validation, config, pipe)
future_frame = future_result.forecast
future_frame.to_csv('outputs/__STEM___future_forecast.csv', index=False, lineterminator='\\n')
future_report = {'verdict': 'not-measurable', 'reason': 'no truth exists for these timestamps yet; score them with evaluate_forecast once it does', 'horizon': config.horizon, 'from': str(future_frame['timestamp'].min()), 'to': str(future_frame['timestamp'].max())}
print(future_frame.groupby('series_id', sort=False).head(2).to_string(index=False))
print(future_report)""",
)

CELL_12 = _cell(
    """## 12. Export the result bundle and reload it

*Engineering / reproducibility.* `write_result_bundle` writes `outputs/__STEM___forecast.csv` (the held-out forecast with its quantiles), `__STEM___evaluation.json`, `__STEM___provenance.json` (model identity, licence terms and revision status, the observed weight digest, the runtime versions, the configuration, the input manifest, the data source) and `__STEM___result.json`, which indexes the other files with their byte sizes and SHA-256 digests and carries the evaluation report. `reload_result_bundle` then reads the forecast **back from the files** after checking those digests, and `check_reload_parity` compares it with the in-memory forecast. Parity is a contract check — it proves the exported bundle reproduces what was evaluated, not that the forecast is good — so a mismatch stops the notebook.

**What to notice.** `reload_parity.ok` is `True`, and `sorted(os.listdir('outputs'))` lists every file this notebook produced.""",
    """provenance = build_provenance(pipe, config, history_validation, result, data_source=data_source, notebook_source=NOTEBOOK_SOURCE)
provenance['holdout'] = {'estimation': 'chronological tail holdout', 'horizon': config.horizon, 'leakage_check': leakage}
provenance['future_forecast'] = future_report
bundle_paths = write_result_bundle(
    'outputs', '__STEM__', forecast=forecast_frame, evaluation=model_eval, report=report, provenance=provenance,
    extra={'baselines': {name: ev['aggregate'] for name, ev in baseline_evals.items()}, 'activity': activity_record, 'input_manifest_file': '__STEM___input_manifest.json', 'future_forecast_file': '__STEM___future_forecast.csv'},
)
reloaded_forecast, reloaded_result = reload_result_bundle('outputs', '__STEM__')
reload_parity = check_reload_parity(forecast_frame, reloaded_forecast)
print({'bundle': bundle_paths, 'reload_parity': {k: reload_parity[k] for k in ('rows', 'ok', 'tolerance', 'boundary')}})
print({'model': {k: provenance['model'].get(k) for k in ('model_id', 'revision', 'resolved_revision', 'weights_sha256', 'weights_sha256_verified_against_manifest', 'license', 'license_terms')}, 'device': provenance['device'], 'runtime': {k: provenance['runtime'].get(k) for k in ('python', 'timesfm', 'torch', 'cuda_available')}})
print(sorted(os.listdir('outputs')))""",
)

CLOSING = """## Interpretation and limits

**Conclude with evidence.** Write a short conclusion from your own run using this scaffold (an optional personal note, not a required submission):

> On [data, series, holdout], the zero-shot TimesFM median forecast reached MASE [value] and sMAPE [value], against [value] for the seasonal-naive baseline and [value] for the naive forecast; the q0.1–q0.9 band covered [share] of the held-out points against a nominal 0.8. Shortening the context to [value] points changed MASE by [amount]. This suggests [a bounded conclusion about this sample], but does not establish [a broader claim about the model or about other series].

**What this evidence supports.** On the demonstrated sample the pipeline validated a real series into an input manifest, held out the future chronologically, scored three history-only baselines, produced a zero-shot median-and-quantile forecast from the pinned checkpoint, scored it with the same metrics, varied one thing (the context) in a controlled way, forecast beyond the data, and exported a bundle that reloads with parity. Whatever the ranking in your run, that ranking is the evidence — recorded, not asserted.

**What it does not establish.** One holdout day per series on three grid cells over two weeks cannot rank forecasting models or show that a ranking is stable; the quantile band is the model's statement and its measured coverage on one day is not a calibration result; MASE and sMAPE do not characterise every error mode (a forecast that misses the timing of the daily peak can score well on sMAPE); the pipeline passes no covariates to TimesFM 3.0 (the upstream covariate inputs are not exposed on this path), so weather drivers, holidays or interventions are invisible to it; and nothing here shows how the model behaves on series with structural breaks, on counts near zero, or at horizons far beyond the sample's daily cycle. The checkpoint's pretraining mixture is described upstream and cannot be reconstructed here; weather from the same sources for earlier dates may be in it.

**Carry to real data.** Read the seasonal-naive and naive rows on *your* series before the model's; if the seasonal rule wins, say so. Use a backtest over many forecast origins (rolling holdouts) before trusting a ranking, measure the band's coverage on that backtest before using it to size anything, and keep the horizon within what your decision needs. Set `SEASON_LENGTH` to your data's real period (7 for daily data with a weekly cycle, 12 for monthly data with a yearly cycle) or the MASE scale and the seasonal baseline will be wrong for it.

**Next experiments (optional; they do not affect the default path).** Note the numbers you want to compare before changing anything; in Colab, **Runtime → Run after** runs the selected cell and every cell below it.

- **Another horizon:** set `HORIZON = 48` in Section 4 and choose **Run after**: the holdout becomes two days, the seasonal baseline repeats one day twice, and the band should widen with the step.
- **Fill instead of refuse:** delete a few rows from a copy of the sample, set `GAP_POLICY = 'fill'` and `NAN_POLICY = 'interpolate'` with `USE_BYOD = True` and `BYOD_PATH` pointing to the copy, and read the `changes` list in Section 5 before the metrics.
- **Monthly data:** bring a monthly series (`SEASON_LENGTH = 12`); validation reports the calendar interval `MS`, and the baselines and the model use it for the future timestamps.
- **Your own data:** BYOD (Section 4, then **Run after**); read the baselines before the model's number.

**Bring your own data.** Supply one CSV with a timestamp column, a numeric value column and optionally a series-id column; set the three column fields in Section 4. The same validation, split, baselines, forecast, evaluation and export run on it; a refusal names the file and the rule. Only upload data you are permitted to process in a hosted runtime; files stay in this runtime and are not published by the notebook.

Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can build its isolated environment, stage the pinned checkpoint at its immutable commit and check its sizes and digests, fetch and digest-verify a real public series, validate the data contract, split without leakage, run the zero-shot forecast with point and quantile output, score it beside history-only baselines, and emit the shown machine-readable bundle with reload parity — without the repository being reachable. It does **not** establish benchmark superiority, calibration, forecast quality on other series or horizons, or production fitness.

**AI Assistance Disclosure:** This notebook's code and explanations were developed with generative AI assistance under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results and making release decisions. AI assistance does not constitute independent verification, provider endorsement or release approval.

## Troubleshooting

- **Section 1 stops with "needs a Linux x86_64 runtime".** The locked environment is built from manylinux wheels. Use Google Colab, Kaggle or a Linux Jupyter server; Windows and macOS kernels are not supported.
- **Section 1 fails while downloading** (`uv` wheel, Python build or packages). The runtime needs PyPI and the python-build-standalone download; run the cell again once the network is back. A "size/SHA-256" or "does not match its digest" error means a file was altered: do not edit the cell, regenerate the notebook from the repository.
- **"The isolated environment's Python process exited".** Usually the runtime ran out of memory. Restart the session and choose **Run all**; on CPU, close other notebooks first. The checkpoint needs about 1.3 GB of RAM in float32 plus the context.
- **Out of disk or the checkpoint download stops.** The locked install (PyTorch with its CUDA libraries) and the 1.32 GB checkpoint need several GB of free space. Start a fresh runtime, or delete `dimer_isolated_env_*/` and `weights/` from an earlier attempt.
- **A size or digest mismatch in Section 3 or 4.** A download was incomplete or altered; the notebook refuses it rather than continuing. Run the cell again; never edit the manifest or the sample digest. A `config.json` mismatch means the Hub served a different checkpoint than the pinned one — stop and report it.
- **Section 3 reports a `model.safetensors` SHA-256 mismatch, or a commit other than the pinned one in `resolved-revision.json`.** The Hub served bytes or a commit that differ from what the maintainer's hosted run of 2026-10-06 observed and pinned; the notebook refuses the snapshot. Run the cell again on a fresh runtime; if it repeats, stop and report it with the printed digest and commit.
- **BYOD is refused in Section 5.** The message names the file and the rule: `[REQUIRED_COLUMNS]` (set the column fields), `[TIMESTAMP_DUPLICATE]`, `[FREQUENCY_GAP]` / `[FREQUENCY_IRREGULAR]`, `[VALUE_MISSING]` / `[VALUE_NUMERIC]`, `[MIN_HISTORY]` (fewer than `HORIZON + 16` rows), `[FREQUENCY_MIXED]`. Fix the file or the policy fields and choose **Run after** from Section 4. An empty or cancelled upload asks you to choose a file or set `BYOD_PATH`.
- **`SEASON_TOO_LONG` in Section 7.** A series' history is shorter than `SEASON_LENGTH`; lower it (or set it to 1 to disable the seasonal baseline).
- **Reload parity fails in Section 12.** The exported bundle does not reproduce the evaluated forecast; this is a contract failure, not a data problem. Restart the session and choose **Run all**; if it repeats, report it with the printed problems.
- **Numbers differ slightly between CPU and GPU runs.** CUDA kernels are not bit-for-bit identical to CPU float32; differences in the last digits are expected. Larger differences on the default settings mean a form field was changed.
- **You re-ran the Section 1 cell.** It reuses the environment and keeps the live worker, so later cells keep working; only a restarted session starts over.

## Glossary

- **Zero-shot forecasting:** forecasting a series the model was not trained or tuned on, from its history alone.
- **Horizon:** the number of future steps forecast (`HORIZON`); also the length of the holdout.
- **Context (length):** the most recent history points the model reads; the *effective* context is what it actually received (shorter series are passed whole).
- **Chronological holdout:** removing the last `HORIZON` points of each series as truth so that the model never sees the period it is scored on.
- **Leakage:** any way future information reaches the forecaster; a random split would be one.
- **Naive / seasonal naive / SES:** repeat the last value; repeat the last season; a flat forecast at an exponentially-smoothed level.
- **Season length:** the period of the cycle the seasonal baseline repeats and MASE scales by (24 for hourly data with a daily cycle).
- **MASE:** mean absolute scaled error — MAE divided by the in-sample seasonal-naive MAE; unit-free; below 1 beats that rule on the history's scale.
- **sMAPE:** symmetric mean absolute percentage error, in percent.
- **Quantile (pinball) loss:** the loss that a correct quantile forecast minimises; lower is better.
- **Model quantiles / coverage:** the quantile head's outputs (q0.1 … q0.9) and the measured share of truth points inside a band; a model band is not a calibrated interval until coverage is measured on enough data.
- **Median as the point forecast:** the point forecast here is the median (q0.5, slot 4 of the model's nine quantile slots); TimesFM 3.0 has no mean head, and with a skewed spread the median is not the centre of the band.
- **Input manifest:** the machine-readable record of what was validated, under which rules, ceilings and policies, and what changed.
- **Result bundle / reload parity:** the exported CSV and JSON files indexed by digest in `result.json`, and the check that the bundle reproduces the in-memory forecast.

## References

- Das, A., Kong, W., Sen, R., & Zhou, Y. (2024). A decoder-only foundation model for time-series forecasting. *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*. https://arxiv.org/abs/2310.10688
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting, 22*(4), 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
- Google Research. (2026). TimesFM 3.0 PyTorch checkpoint (TimesFM Non-Commercial License v1.0). Hugging Face Hub. https://huggingface.co/google/timesfm-3.0-pytorch
- Open-Meteo. (2026). Historical weather API. https://open-meteo.com/ (data: CC BY 4.0)
"""

GUIDED_OPENING = [
    _cell(
        """## Who this notebook is for, and how to use it

**Learner.** Someone who can open a hosted notebook, run cells and read basic Python, and who is new to time-series forecasting or to foundation models for it. No prior forecasting or machine-learning course is assumed: horizon, context, holdout, baseline, MASE and quantile are explained where they first matter and collected in the **Glossary** at the end. No accelerator is required; the default path runs on CPU and uses a GPU automatically when one is present.

**How to use this notebook.** In Colab, choose any runtime (CPU is fine), then **Runtime → Run all**. Sections 1–3 are **infrastructure** — the isolated environment, the carried package code, and the checkpoint staging — labelled as such and collapsed; you may run them without studying their implementation. The learning path starts in Section 4. The form fields (`# @param`) in Sections 4 and 10 are the only values meant to be edited, and the defaults reproduce the canonical path. Each stage states its question, asks you to **predict** before it runs, and ends with **What to notice**; collapsible **Check your reasoning** blocks give worked guidance after you have answered. **Troubleshooting** and the **Glossary** are at the end. Nothing you write in the conclusion scaffold is a required submission.

**By the end you should be able to** (1) explain the task as *history → pretrained model → median and quantile forecast*; (2) identify why the future must be held out chronologically; (3) compare a foundation model with naive, seasonal-naive and exponential-smoothing baselines on the same held-out rows; (4) interpret MASE, sMAPE, quantile loss and band coverage without reading a model quantile as a guarantee; (5) predict and then measure the effect of one change (the context length); and (6) apply the same validation, forecast and export contract to your own CSV.""",
    ),
    _cell(
        """## Roadmap and the task contract

**Roadmap.** 4 obtain the sample (or your CSV) → 5 validate into an input manifest, see a refusal → 6 split chronologically → 7 three baselines → 8 **zero-shot forecast** with point and quantile output → 9 evaluate beside the baselines (verdicts recorded, never asserted) → 10 **change one thing: the context length** → 11 forecast beyond the data → 12 export and reload → conclude. A fast path is the default path: nothing is optional until Section 10, and Section 10 is bounded.

**Input → Model → Output.** *Input:* a long-format table — timestamp, numeric value, optional series id — sampled at one interval. *Model:* TimesFM 3.0 (a stacked mixing transformer pretrained on time series; the Hub lists 330.7M parameters; weights under a non-commercial, non-production licence) reading the most recent `CONTEXT_LENGTH` points of each series, with nothing trained here. *Output:* for each series and each of `HORIZON` future steps, a median point forecast and the model quantiles q0.1 / q0.5 / q0.9; beside it, the same metrics for three history-only baselines, an input manifest, an evaluation report and a result bundle with provenance.

**Three ideas to hold on to.** *Zero-shot* describes this notebook's task — no training on these series — not the model's ignorance of weather in general. *Baselines come first*: a seasonal rule of thumb is the bar, and the notebook records whether the model clears it rather than assuming it. *Quantiles are the model's statements*: the band's measured coverage on the holdout is what tells you how to read it.""",
    ),
]

TEMPLATE = {
    "package": "timesfm_forecasting",
    "repo_name": REPO,
    "stem": STEM,
    "notebook_name": f"{STEM}_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "infrastructure_labels": True,
    "isolated_runtime": True,
    # The fleet's uv isolated-environment mechanism: managed CPython, a size- and SHA-256-verified uv wheel (the
    # digest below was recomputed locally from the wheel), and a lock compiled from tutorials/requirements-colab.in.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pins_file": "tutorials/requirements-colab.in",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds the isolated hash-locked environment (nothing is "
        "installed into the kernel, no restart), stages the pinned TimesFM 3.0 checkpoint (1.32 GB safetensors, non-commercial weights licence) and "
        "checks its sizes and SHA-256 digests, fetches the digest-pinned Open-Meteo sample (31 KB, no credential), "
        "validates it into an input manifest and shows one named refusal, holds out the last 24 hours of each series "
        "chronologically, scores naive, seasonal-naive and exponential-smoothing baselines, runs the zero-shot forecast "
        "with median and quantile output, records MASE, sMAPE, quantile loss and band coverage beside the baselines "
        "without asserting a winner, varies the context length once, forecasts beyond the data, and exports a result "
        "bundle that is reloaded with a parity check. The default path needs no repository clone, no DIMER worker or "
        "service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.2 §5)."
    ),
    "byod": (
        "After the sample workflow completes, set `USE_BYOD = True` in Section 4, set `BYOD_PATH` to a CSV on the runtime "
        "(or leave it empty for the Colab upload dialog), set the three column fields to your file's names, and choose "
        "**Run after**. Your data passes through the same validation (every refusal names the file and the rule), split, "
        "baselines, forecast, evaluation and export as the sample; the expected schema, the ceilings and the privacy "
        "guidance are stated in the Prerequisites and in Section 4, and the file stays inside this runtime. BYOD is "
        "optional and never part of the default path."
    ),
    "pipeline_class": "LoadedModel",
    "model_load": "load_pinned_model(weights_dir=WEIGHTS_DIR)",
    "weights_key": "timesfm-3.0-pytorch",
    "modules": ["errors.py", "config.py", "data.py", "model.py", "forecasting.py", "evaluation.py", "provenance.py"],
    "entry_module": "model.py",
    "identity_names": {},
    "runtime_imports": ["torch", "numpy", "pandas"],
    "title": "DIMER Notebook: Zero-Shot Time-Series Forecasting with TimesFM — Can a Foundation Model Beat a Seasonal Rule of Thumb?",
    "badges": [
        ("GitHub", "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white", f"https://github.com/kurtvalcorza/{REPO}"),
        ("Open In Colab", "https://colab.research.google.com/assets/colab-badge.svg", f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/{STEM}_colab.ipynb"),
        ("Hugging Face", "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Ftimesfm--3.0--pytorch-ffcc4d?style=flat", "https://huggingface.co/google/timesfm-3.0-pytorch"),
        ("Upstream", "https://img.shields.io/badge/Upstream-google--research%2Ftimesfm-181717?style=flat&logo=github&logoColor=white", "https://github.com/google-research/timesfm"),
        ("arXiv", "https://img.shields.io/badge/arXiv-2310.10688-b31b1b.svg", "https://arxiv.org/abs/2310.10688"),
        ("Code: Apache-2.0", "https://img.shields.io/badge/Code-Apache--2.0-blue.svg", f"https://github.com/kurtvalcorza/{REPO}/blob/main/LICENSE"),
        ("Weights: non-commercial", "https://img.shields.io/badge/Weights-TimesFM%20Non--Commercial%20v1.0-red.svg", "https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE"),
    ],
    "capability": (
        "zero-shot univariate time-series forecasting over one or many independent series with the pinned TimesFM 3.0 "
        "PyTorch checkpoint (non-commercial weights): a median point forecast and model quantiles from the trained decile grid, "
        "evaluated chronologically against naive, seasonal-naive and exponential-smoothing baselines with MASE, sMAPE "
        "and quantile loss"
    ),
    "intro": (
        "**The question this notebook answers on one real series:** can a pretrained time-series foundation model, given "
        "nothing but the recent history, forecast the next day of hourly temperature in three Philippine cities better "
        "than repeating yesterday? The answer is measured, recorded and interpreted; it is not assumed either way.\n\n"
        "**Zero-shot does not mean untrained.** TimesFM 3.0 was pretrained by Google Research on a large corpus of time "
        "series. *Zero-shot* describes this notebook's task: the model forecasts these series without any training on "
        "them. Everything the model knows about daily cycles it learned elsewhere.\n\n"
        "**The point forecast is the median, and the band is a model statement.** The pipeline exports the model's "
        "quantile 0.5 (slot 4 of its nine quantile slots) as `prediction` and the requested deciles as `q0.1` / `q0.5` / "
        "`q0.9`; TimesFM 3.0 has no mean head. The q0.1–q0.9 band is what the model's quantile head emits for a nominal "
        "80 % interval; Section 9 measures how often the truth falls inside it, and only that measurement says how to "
        "read it. No decision threshold is shipped.\n\n"
        "**Weights licence.** The TimesFM 3.0 weights that Section 3 downloads are distributed by Google LLC under the "
        "*TimesFM Non-Commercial License v1.0* (`license: other` on the Hub): **non-commercial and non-production use "
        "only** — testing, evaluation and research not tied to commercial gain, production deployment or revenue "
        "generation — with no redistribution of the model or of derivatives, and a commercial licence required for "
        "anything else. Running this notebook for a client deliverable, a paid product, or any end-user-facing or "
        "production system is outside that licence. The notebook's code and the pipeline package are Apache-2.0; the "
        "sample is CC BY 4.0. If the restriction does not fit your use, the repository's TimesFM 2.5 build (Apache-2.0 "
        "weights) is the fallback; see the repository README.\n\n"
        "**Model revision status.** The checkpoint is pinned by repository id, immutable commit, file names, byte sizes "
        "and the SHA-256 of `config.json` and `model.safetensors`. The commit and the weight digest are the values the "
        "Hub served and the pipeline computed on the maintainer's hosted Colab run of 2026-10-06 (recorded in the "
        "repository's `docs/release-verification.md`); Section 3 refuses the snapshot on any size or digest mismatch, "
        "and Section 12 exports the commit and the digest it verified."
    ),
    "learning_objectives": (
        "by the end you should be able to (1) explain *history → pretrained model → median and quantile forecast*; "
        "(2) identify why the future is held out chronologically and what leakage would look like; (3) compare the "
        "model with naive, seasonal-naive and exponential-smoothing baselines on the same held-out rows; (4) interpret "
        "MASE, sMAPE, quantile loss and band coverage, and distinguish a model quantile from a calibrated interval; "
        "(5) predict and measure the effect of the context length in a controlled activity; and (6) apply the same "
        "validation, forecast and export contract to your own CSV and read its refusals."
    ),
    "exclusions": (
        "covariates (past-only or known-future regressors; TimesFM 3.0's covariate inputs are not exposed), multivariate joint "
        "forecasting, fine-tuning or any training, anomaly detection, imputation as a product feature (the pipeline's "
        "interpolation is a reported validation policy, not a model), probability calibration or conformal intervals, "
        "rolling-origin backtesting, multivariate variate attention across series (each series is forecast "
        "independently), and any claim that one holdout day on three series stands in for your data. The repository "
        "exposes none of these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and Colab familiarity; no prior forecasting experience. New terms (horizon, context, holdout, baseline, MASE, quantile) are explained where they are first used and collected in the Glossary.",
        "- **Weights licence (read before Section 3):** the pinned `google/timesfm-3.0-pytorch` weights are under the *TimesFM Non-Commercial License v1.0* — **non-commercial and non-production use only**, no redistribution of the model or of derivatives, a commercial licence from Google LLC required for any other use. Downloading them in Section 3 means accepting those terms. The pipeline code is Apache-2.0 and the sample is CC BY 4.0; the licence text is in the checkpoint repository's `LICENSE` file.",
        "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab, Kaggle or Linux Jupyter. Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter, and a Windows or macOS kernel is not supported (Section 1 stops with that message). The default path runs on CPU in float32 and uses CUDA automatically when available. The locked install (PyTorch 2.14.0 with its CUDA libraries) and the 1.32 GB checkpoint are the large downloads. No hosted run is recorded yet, so no runtime figure is given; the model stages on this sample are three series of 312 points at a 24-step horizon, which is small for a 330M-parameter model on CPU (an expectation, not a measurement).",
        "- **Data contract:** one CSV, long format — a timestamp column (ISO-8601 or any pandas-parseable format), a finite numeric value column, and optionally a series-id column; one sampling interval per file (fixed, such as hourly or daily, or a calendar interval such as monthly); no duplicate timestamps within a series; at least `HORIZON + 16` rows per series (40 with the default horizon) and at least 3 rows to infer the interval; at most 1,000 series and 2,000,000 rows; histories longer than 15,360 points (the model's context limit) are truncated to their most recent points with a report. Missing values and skipped grid points are refused by default and may instead be interpolated / filled with a report (`NAN_POLICY`, `GAP_POLICY`; at most 20 % of a series may be interpolated). Section 5 checks all of this before any model call and names the file and the rule it refuses.",
        "- **Validation is structural, not semantic:** nothing checks that the values mean what you think, that the series is stationary, or that the season length you set is the data's real period; a wrong `SEASON_LENGTH` makes the seasonal baseline and the MASE scale wrong without any error.",
        "- **Privacy:** do not upload confidential or restricted data to a hosted runtime unless you are authorised to process it there — operational telemetry or sales figures may be exactly that. The default path uploads nothing.",
    ],
    "external_access": (
        "the Hugging Face Hub, to fetch the pinned `{MODEL_ID}` files (~{total_mb:.0f} MB in total; non-commercial weights "
        "licence, see above) at the immutable revision `{MODEL_REVISION}` (the commit the Hub served on the maintainer's hosted "
        "run of 2026-10-06; the weight digest observed on that run is pinned in the manifest), and `raw.githubusercontent.com` for the pinned "
        f"sample (`openmeteo_ph_hourly_temperature.csv`, {SAMPLE_BYTES:,} bytes, SHA-256 `{SAMPLE_SHA256[:8]}…`, the copy "
        f"committed under `examples/sample-data/` of this repository at commit `{SAMPLE_COMMIT[:12]}…`; Open-Meteo, CC BY 4.0), "
        "refused on any mismatch before it is read. The sample is data, not code: nothing is installed from this repository "
        "and no credentials are required"
    ),
    "weights_licence_note": (
        "**Weights licence — read before running the next cell.** The cell downloads `google/timesfm-3.0-pytorch`, whose "
        "weights Google LLC distributes under the *TimesFM Non-Commercial License v1.0*: **non-commercial and non-production "
        "use only** (testing, evaluation and research not tied to commercial gain, production deployment or revenue "
        "generation), no use in end-user-facing or production systems, no redistribution of the model or of derivatives, "
        "and a commercial licence required for anything else. Downloading the weights means accepting those terms; the "
        "run records them in `resolved-revision.json` and in the exported provenance. The code in this notebook is "
        "Apache-2.0 and the sample is CC BY 4.0."
    ),
    "guided": {"opening": [cell["md"] for cell in GUIDED_OPENING]},
    "cells": [CELL_4, CELL_5, CELL_6, CELL_7, CELL_8, CELL_9, CELL_10, CELL_11, CELL_12],
    "closing": CLOSING.replace("{", "{{").replace("}", "}}").replace("__STEM__", "{stem}"),
}

# The sample constants are substituted after escaping so that the URL's characters are carried verbatim.
for _c in TEMPLATE["cells"]:
    if "code" in _c:
        _c["code"] = (
            _c["code"]
            .replace("__SAMPLE_URL__", SAMPLE_URL)
            .replace("__SAMPLE_SHA256__", SAMPLE_SHA256)
            .replace("__SAMPLE_BYTES__", str(SAMPLE_BYTES))
        )
