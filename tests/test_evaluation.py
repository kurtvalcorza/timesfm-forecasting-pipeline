"""Chronological holdout, baselines and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from timesfm_forecasting import (
    ForecastConfig,
    ValidationError,
    chronological_holdout,
    evaluate_forecast,
    evaluation_report,
    forecast,
    mase,
    naive_baseline,
    quantile_loss,
    seasonal_naive_baseline,
    ses_baseline,
    smape,
    validate_series,
)

from .conftest import FakeModel, make_series


def _validated(n: int = 96, horizon: int = 12):
    frame = pd.concat([make_series("A", n=n, seed=1), make_series("B", n=n, seed=2)], ignore_index=True)
    return validate_series(frame, ForecastConfig(horizon=horizon, series_id_column="series_id", season_length=24), source="s.csv")


def test_holdout_is_chronological_and_leak_free() -> None:
    v = _validated()
    split = chronological_holdout(v.frame, 12)
    assert split.horizon == 12
    assert split.truth.groupby("series_id").size().to_dict() == {"A": 12, "B": 12}
    assert split.history.groupby("series_id").size().to_dict() == {"A": 84, "B": 84}
    assert split.leakage_check() == {"A": True, "B": True}
    assert set(split.history["timestamp"]).isdisjoint(set(split.truth["timestamp"]))


def test_holdout_refuses_a_series_it_would_empty() -> None:
    v = _validated(n=40)
    with pytest.raises(ValidationError, match=r"\[MIN_HISTORY\]"):
        chronological_holdout(v.frame, 40)


def test_naive_and_seasonal_naive_values() -> None:
    v = _validated()
    split = chronological_holdout(v.frame, 12)
    naive = naive_baseline(split.history, v.frequency, 12)
    a_last = split.history[split.history["series_id"] == "A"]["value"].iloc[-1]
    assert np.allclose(naive[naive["series_id"] == "A"]["prediction"], a_last)
    assert list(naive.columns) == ["series_id", "timestamp", "step", "prediction", "method"]
    seasonal = seasonal_naive_baseline(split.history, v.frequency, 12, 24)
    a_hist = split.history[split.history["series_id"] == "A"]["value"].to_numpy()
    assert np.allclose(seasonal[seasonal["series_id"] == "A"]["prediction"], a_hist[-24:][:12])
    # timestamps continue the grid
    first_truth = split.truth[split.truth["series_id"] == "A"]["timestamp"].iloc[0]
    assert naive[naive["series_id"] == "A"]["timestamp"].iloc[0] == first_truth


def test_seasonal_naive_refuses_a_short_series() -> None:
    v = _validated(n=40)
    split = chronological_holdout(v.frame, 4)
    with pytest.raises(ValidationError, match=r"\[SEASON_TOO_LONG\]"):
        seasonal_naive_baseline(split.history, v.frequency, 4, 100)
    with pytest.raises(ValueError):
        seasonal_naive_baseline(split.history, v.frequency, 4, 0)


def test_ses_is_flat_and_between_min_and_max() -> None:
    v = _validated()
    split = chronological_holdout(v.frame, 12)
    ses = ses_baseline(split.history, v.frequency, 12)
    a = ses[ses["series_id"] == "A"]["prediction"].to_numpy()
    assert np.allclose(a, a[0])
    hist = split.history[split.history["series_id"] == "A"]["value"]
    assert hist.min() <= a[0] <= hist.max()


def test_metric_definitions() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert smape(y, y) == 0.0
    assert smape(np.array([100.0]), np.array([50.0])) == pytest.approx(100 * 50 / 75)
    assert smape(np.array([0.0]), np.array([0.0])) == 0.0
    insample = np.array([1.0, 2.0, 4.0, 7.0])  # one-step naive MAE = 2
    assert mase(y, y + 1, insample, 1) == pytest.approx(0.5)
    assert np.isnan(mase(y, y, np.array([1.0]), 1))
    assert np.isnan(mase(y, y, np.array([3.0, 3.0, 3.0]), 1))
    assert quantile_loss(np.array([1.0]), np.array([0.0]), 0.9) == pytest.approx(0.9)
    assert quantile_loss(np.array([0.0]), np.array([1.0]), 0.9) == pytest.approx(0.1)
    assert quantile_loss(y, y, 0.5) == 0.0


def test_evaluate_forecast_reports_per_series_quantile_terms_and_coverage() -> None:
    v = _validated()
    config = v.config
    split = chronological_holdout(v.frame, 12)
    result = forecast(v.with_frame(split.history), config, FakeModel(spread=4.0))
    ev = evaluate_forecast(result.forecast, split.truth, split.history, config)
    assert set(ev["per_series"]) == {"A", "B"}
    agg = ev["aggregate"]
    assert agg["mase_season_length"] == 24 and agg["n_points"] == 24
    assert set(agg["quantile_loss"]) == {"q0.1", "q0.5", "q0.9"}
    assert agg["interval"] == {"lower": "q0.1", "upper": "q0.9", "nominal": 0.8}
    assert 0.0 <= agg["interval_coverage"] <= 1.0
    assert agg["quantile_loss"]["q0.5"] == pytest.approx(agg["mae"] / 2)


def test_evaluate_refuses_misaligned_rows() -> None:
    v = _validated()
    split = chronological_holdout(v.frame, 12)
    naive = naive_baseline(split.history, v.frequency, 12)
    with pytest.raises(ValidationError, match=r"\[EVALUATION_MISALIGNED\]"):
        evaluate_forecast(naive.iloc[:-1], split.truth, split.history, v.config)


def test_report_records_verdicts_without_asserting() -> None:
    v = _validated()
    split = chronological_holdout(v.frame, 12)
    config = v.config
    # A deliberately bad "model": far from the truth, so it loses to every baseline.
    bad = naive_baseline(split.history, v.frequency, 12).assign(prediction=lambda f: f["prediction"] + 50)
    model_eval = evaluate_forecast(bad, split.truth, split.history, config)
    baselines = {
        "naive": evaluate_forecast(naive_baseline(split.history, v.frequency, 12), split.truth, split.history, config),
        "ses": evaluate_forecast(ses_baseline(split.history, v.frequency, 12), split.truth, split.history, config),
    }
    report = evaluation_report(model_eval, baselines, horizon=12, sample_kind="synthetic")
    assert report["verdict"] == "sample-sanity"
    assert report["model_beats_on_mase"] == []
    assert {r["method"] for r in report["table"]} == {"timesfm", "naive", "ses"}
    assert report["comparison"]["naive"]["model_mase_lower"] is False
