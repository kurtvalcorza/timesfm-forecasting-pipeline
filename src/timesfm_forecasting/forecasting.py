"""Zero-shot forecasting over a validated table: point (median) and quantile output.

``forecast`` takes a :class:`ValidationResult`, passes each series' most recent
``context_length`` points to the model, and returns one tidy row per ``(series_id, step)`` with
the future timestamp, the median point forecast, the mean head and the requested quantiles.
The row layout is asserted against the model's output shape so a forecast can never be
attached to the wrong series or step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import TRAINED_QUANTILES, ForecastConfig
from .data import ID_COLUMN, TIMESTAMP_COLUMN, VALUE_COLUMN, ValidationResult, future_timestamps
from .errors import ModelIntegrityError

__all__ = ["ForecastResult", "forecast", "quantile_column", "POINT_COLUMN", "MEAN_COLUMN", "STEP_COLUMN"]

POINT_COLUMN = "prediction"
MEAN_COLUMN = "mean"
STEP_COLUMN = "step"


def quantile_column(level: float) -> str:
    return f"q{level:g}"


@dataclass(frozen=True)
class ForecastResult:
    """A forecast frame plus the inference provenance that produced it."""

    forecast: pd.DataFrame
    provenance: dict[str, Any]

    @property
    def quantile_columns(self) -> list[str]:
        return [c for c in self.forecast.columns if c.startswith("q")]


def _series_arrays(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {str(k): g for k, g in frame.groupby(ID_COLUMN, sort=False)}


def forecast(validation: ValidationResult, config: ForecastConfig, model: Any) -> ForecastResult:
    """Forecast every series in ``validation.frame`` ``config.horizon`` steps ahead."""
    groups = _series_arrays(validation.frame)
    ids = list(groups)
    inputs: list[np.ndarray] = []
    effective_context: dict[str, int] = {}
    for series_id in ids:
        values = groups[series_id][VALUE_COLUMN].to_numpy(dtype=float)
        window = values[-config.context_length :]
        effective_context[series_id] = int(len(window))
        inputs.append(window)
    point, quantiles = model.predict(inputs, config.horizon, config)
    if point.shape != (len(ids), config.horizon) or quantiles.shape[:2] != (len(ids), config.horizon):
        raise ModelIntegrityError(f"model returned {point.shape} / {quantiles.shape} for {len(ids)} series")
    if quantiles.shape[2] != len(TRAINED_QUANTILES) + 1:
        raise ModelIntegrityError(f"expected {len(TRAINED_QUANTILES) + 1} quantile slots, got {quantiles.shape[2]}")
    # Row-layout oracle: the point forecast must equal the median slot for every series/step.
    if not np.allclose(point, quantiles[:, :, 5], rtol=0, atol=1e-6):
        raise ModelIntegrityError("point forecast is not the median slot; the upstream contract changed")
    rows = []
    for i, series_id in enumerate(ids):
        last = pd.Timestamp(groups[series_id][TIMESTAMP_COLUMN].iloc[-1])
        stamps = future_timestamps(last, validation.frequency, config.horizon)
        for step in range(config.horizon):
            row = {
                ID_COLUMN: series_id,
                TIMESTAMP_COLUMN: stamps[step],
                STEP_COLUMN: step + 1,
                POINT_COLUMN: float(point[i, step]),
                MEAN_COLUMN: float(quantiles[i, step, 0]),
            }
            for level in config.quantile_levels:
                slot = 1 + int(round(level * 10)) - 1
                row[quantile_column(level)] = float(quantiles[i, step, slot])
            rows.append(row)
    out = pd.DataFrame(rows)
    provenance = {
        "task": "zero-shot time-series forecasting",
        "point_forecast_semantics": "median (model quantile 0.5); `mean` is the separate mean head",
        "quantile_semantics": "model quantiles from the trained decile grid; not calibrated intervals",
        "horizon": config.horizon,
        "requested_context_length": config.context_length,
        "effective_context_length": effective_context,
        "quantile_levels": list(config.quantile_levels),
        "n_series": len(ids),
        "frequency": validation.frequency.to_dict(),
        "compiled": model.compile_for(config) if hasattr(model, "compile_for") else None,
    }
    return ForecastResult(forecast=out, provenance=provenance)
