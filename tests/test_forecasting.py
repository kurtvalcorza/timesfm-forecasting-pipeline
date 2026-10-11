"""Forecast frame contract with a stand-in model (no pretrained inference)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from timesfm_forecasting import ForecastConfig, ModelIntegrityError, forecast, validate_series
from timesfm_forecasting.forecasting import POINT_COLUMN, STEP_COLUMN

from .conftest import FakeModel, make_series


def _validated(**overrides):
    config = ForecastConfig(**{"horizon": 12, "context_length": 48, "series_id_column": "series_id", **overrides})
    frame = pd.concat([make_series("A", n=96), make_series("B", n=70)], ignore_index=True)
    return validate_series(frame, config, source="s.csv"), config


def test_forecast_rows_columns_and_median_contract() -> None:
    v, config = _validated(quantile_levels=(0.1, 0.5, 0.9))
    model = FakeModel()
    result = forecast(v, config, model)
    f = result.forecast
    assert list(f.columns) == ["series_id", "timestamp", STEP_COLUMN, POINT_COLUMN, "q0.1", "q0.5", "q0.9"]
    assert len(f) == 24 and f.groupby("series_id").size().to_dict() == {"A": 12, "B": 12}
    assert (f[POINT_COLUMN] == f["q0.5"]).all()
    assert (f["q0.1"] <= f["q0.5"]).all() and (f["q0.5"] <= f["q0.9"]).all()
    assert "mean" not in f.columns  # TimesFM 3.0 has no mean head; nothing is invented for one
    assert f[STEP_COLUMN].tolist() == list(range(1, 13)) * 2
    assert result.provenance["effective_context_length"] == {"A": 48, "B": 48}
    assert result.provenance["point_forecast_semantics"].startswith("median")
    assert result.provenance["decode_settings"]["stand_in"] is True
    assert model.calls[0]["lengths"] == [48, 48]


def test_short_series_is_passed_whole_and_timestamps_continue_the_grid() -> None:
    v, config = _validated(context_length=512)
    result = forecast(v, config, FakeModel())
    assert result.provenance["effective_context_length"] == {"A": 96, "B": 70}
    last_b = v.frame[v.frame["series_id"] == "B"]["timestamp"].max()
    first = result.forecast[result.forecast["series_id"] == "B"]["timestamp"].iloc[0]
    assert first == last_b + pd.Timedelta(hours=1)


def test_all_nine_quantiles_map_to_their_slots() -> None:
    v, config = _validated(quantile_levels=tuple(round(0.1 * k, 1) for k in range(1, 10)))
    result = forecast(v, config, FakeModel(spread=4.0))
    f = result.forecast
    for k in range(1, 10):
        assert np.allclose(f[f"q{k / 10:g}"] - f["q0.5"], (k - 5) * 1.0)


class _WrongShape(FakeModel):
    def predict(self, inputs, horizon, config):
        point, q = super().predict(inputs, horizon, config)
        return point[:, :-1], q


class _NotMedian(FakeModel):
    def predict(self, inputs, horizon, config):
        point, q = super().predict(inputs, horizon, config)
        return point + 1.0, q


@pytest.mark.parametrize("model", [_WrongShape(), _NotMedian()])
def test_row_layout_oracle_refuses_a_changed_upstream_contract(model) -> None:
    v, config = _validated()
    with pytest.raises(ModelIntegrityError):
        forecast(v, config, model)
