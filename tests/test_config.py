from __future__ import annotations

import pytest

from timesfm_forecasting import MAX_HORIZON, TRAINED_QUANTILES, ForecastConfig


def test_defaults_carry_the_median() -> None:
    config = ForecastConfig()
    assert config.horizon == 24
    assert 0.5 in config.quantile_levels
    assert config.quantile_levels == (0.1, 0.5, 0.9)


def test_median_is_always_added() -> None:
    assert ForecastConfig(quantile_levels=(0.1, 0.9)).quantile_levels == (0.1, 0.5, 0.9)


@pytest.mark.parametrize("levels", [(0.05,), (0.95,), (0.33,), (0.5, 0.123)])
def test_off_grid_quantiles_are_refused_not_interpolated(levels) -> None:
    with pytest.raises(ValueError, match="trained grid"):
        ForecastConfig(quantile_levels=levels)


def test_every_trained_quantile_is_accepted() -> None:
    assert ForecastConfig(quantile_levels=TRAINED_QUANTILES).quantile_levels == TRAINED_QUANTILES


@pytest.mark.parametrize("horizon", [0, -1, MAX_HORIZON + 1, True, 1.5])
def test_horizon_bounds(horizon) -> None:
    with pytest.raises(ValueError):
        ForecastConfig(horizon=horizon)


@pytest.mark.parametrize("context", [15, 16_385, "32"])
def test_context_bounds(context) -> None:
    with pytest.raises(ValueError):
        ForecastConfig(context_length=context)


def test_policies_and_columns_are_checked() -> None:
    with pytest.raises(ValueError, match="nan_policy"):
        ForecastConfig(nan_policy="drop")
    with pytest.raises(ValueError, match="gap_policy"):
        ForecastConfig(gap_policy="interpolate")
    with pytest.raises(ValueError, match="distinct"):
        ForecastConfig(timestamp_column="x", value_column="x")
    with pytest.raises(ValueError, match="season_length"):
        ForecastConfig(season_length=0)


def test_to_dict_is_json_friendly() -> None:
    d = ForecastConfig(series_id_column="id").to_dict()
    assert d["quantile_levels"] == [0.1, 0.5, 0.9]
    assert d["series_id_column"] == "id"
