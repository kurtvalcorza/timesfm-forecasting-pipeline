"""Forecast request configuration.

The configuration is a frozen dataclass so that a request can be hashed, serialised into
provenance and compared across the sample and BYOD paths. Every ceiling is named here once and
re-used by validation, the model wrapper and the notebook's ceiling print.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

__all__ = [
    "ForecastConfig",
    "TRAINED_QUANTILES",
    "MEDIAN_SLOT",
    "MAX_HORIZON",
    "MAX_CONTEXT_POINTS",
    "MIN_CONTEXT_POINTS",
    "DEFAULT_QUANTILE_LEVELS",
]

#: The nine deciles TimesFM 3.0 was trained to emit (``config.json`` ``quantiles``). The model's
#: quantile array has exactly these nine slots, in this order; slot ``MEDIAN_SLOT`` (index 4) is the
#: median and is the point forecast. There is no separate mean head. Any other level is refused
#: rather than interpolated.
TRAINED_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MEDIAN_SLOT = TRAINED_QUANTILES.index(0.5)
DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.5, 0.9)

#: Longest forecast one request may ask for. TimesFM 3.0 decodes the horizon autoregressively in
#: 64-point output patches with no fixed upper bound; the pipeline keeps a ceiling so a tutorial
#: request stays cheap and a horizon far beyond the context's cycle is never silently accepted.
MAX_HORIZON = 512
#: TimesFM 3.0 context limit (``timesfm3`` ``_MAX_CONTEXT_LENGTH = 15360``; the forecaster truncates
#: longer contexts to their most recent points). Longer histories are truncated here first and the
#: truncation is reported, never silent.
MAX_CONTEXT_POINTS = 15_360
#: Fewest history points the pipeline will pass to the model per series. The model pads shorter
#: inputs with zeros; below this many real points the forecast is dominated by padding.
MIN_CONTEXT_POINTS = 16

NanPolicy = Literal["refuse", "interpolate"]
GapPolicy = Literal["refuse", "fill"]


def _is_trained_quantile(level: float) -> bool:
    return any(abs(level - q) < 1e-9 for q in TRAINED_QUANTILES)


@dataclass(frozen=True)
class ForecastConfig:
    """One forecasting request.

    Attributes
    ----------
    horizon
        Number of future steps to forecast (1..``MAX_HORIZON``).
    context_length
        Most recent history points passed to the model per series
        (``MIN_CONTEXT_POINTS``..``MAX_CONTEXT_POINTS``). Shorter series are passed whole.
    quantile_levels
        Quantiles to export, each one of ``TRAINED_QUANTILES``; ``0.5`` is always carried
        because the point forecast *is* the median.
    timestamp_column, value_column, series_id_column
        Column names of the long-format input. ``series_id_column=None`` means the file holds
        one series, labelled ``"series"``.
    season_length
        Period in steps for the seasonal-naive baseline and MASE scaling; ``None`` disables the
        seasonal baseline and scales MASE by the one-step naive error.
    nan_policy
        ``"refuse"`` rejects any missing value; ``"interpolate"`` fills interior gaps linearly
        and drops leading missing values, reporting every change.
    gap_policy
        ``"refuse"`` rejects a series whose timestamps skip grid points; ``"fill"`` inserts the
        missing timestamps (then ``nan_policy`` decides what happens to their values).
    """

    horizon: int = 24
    context_length: int = 512
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILE_LEVELS
    timestamp_column: str = "timestamp"
    value_column: str = "value"
    series_id_column: str | None = None
    season_length: int | None = None
    nan_policy: NanPolicy = "refuse"
    gap_policy: GapPolicy = "refuse"

    def __post_init__(self) -> None:
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise ValueError(f"horizon must be an int, got {self.horizon!r}")
        if not 1 <= self.horizon <= MAX_HORIZON:
            raise ValueError(f"horizon must be in 1..{MAX_HORIZON}, got {self.horizon}")
        if isinstance(self.context_length, bool) or not isinstance(self.context_length, int):
            raise ValueError(f"context_length must be an int, got {self.context_length!r}")
        if not MIN_CONTEXT_POINTS <= self.context_length <= MAX_CONTEXT_POINTS:
            raise ValueError(
                f"context_length must be in {MIN_CONTEXT_POINTS}..{MAX_CONTEXT_POINTS}, "
                f"got {self.context_length}"
            )
        levels = tuple(float(q) for q in self.quantile_levels)
        bad = [q for q in levels if not _is_trained_quantile(q)]
        if bad:
            raise ValueError(
                f"quantile_levels {bad} are not in the trained grid {TRAINED_QUANTILES}; "
                "the pipeline does not interpolate quantiles"
            )
        if 0.5 not in levels:
            levels = tuple(sorted({*levels, 0.5}))
        object.__setattr__(self, "quantile_levels", tuple(sorted(set(levels))))
        if self.season_length is not None and (
            isinstance(self.season_length, bool)
            or not isinstance(self.season_length, int)
            or self.season_length < 1
        ):
            raise ValueError(f"season_length must be a positive int or None, got {self.season_length!r}")
        if self.nan_policy not in ("refuse", "interpolate"):
            raise ValueError(f"nan_policy must be 'refuse' or 'interpolate', got {self.nan_policy!r}")
        if self.gap_policy not in ("refuse", "fill"):
            raise ValueError(f"gap_policy must be 'refuse' or 'fill', got {self.gap_policy!r}")
        for name in ("timestamp_column", "value_column"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        names = [self.timestamp_column, self.value_column] + (
            [self.series_id_column] if self.series_id_column else []
        )
        if len(set(names)) != len(names):
            raise ValueError(f"column names must be distinct, got {names}")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["quantile_levels"] = list(self.quantile_levels)
        return out
