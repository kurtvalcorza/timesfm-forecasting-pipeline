"""Chronological evaluation, naive and statistical baselines, and forecasting metrics.

The only split helper is chronological: the last ``horizon`` points of every series are held
out as truth and removed from the history the model sees. Baselines use history only. Metrics
are MASE (scaled by the in-sample seasonal-naive error), sMAPE and the quantile (pinball) loss.
``evaluation_report`` records verdicts; it never asserts that the model beats a baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ForecastConfig
from .data import ID_COLUMN, TIMESTAMP_COLUMN, VALUE_COLUMN, FrequencyInfo, future_timestamps
from .errors import ValidationError
from .forecasting import POINT_COLUMN, STEP_COLUMN

__all__ = [
    "HoldoutSplit",
    "chronological_holdout",
    "naive_baseline",
    "seasonal_naive_baseline",
    "ses_baseline",
    "mase",
    "smape",
    "quantile_loss",
    "evaluate_forecast",
    "evaluation_report",
    "BASELINE_NAMES",
]

BASELINE_NAMES = ("naive", "seasonal_naive", "ses")


@dataclass(frozen=True)
class HoldoutSplit:
    history: pd.DataFrame
    truth: pd.DataFrame
    horizon: int

    def leakage_check(self) -> dict[str, Any]:
        """Per series: the last history timestamp precedes the first truth timestamp."""
        out = {}
        for series_id, g in self.truth.groupby(ID_COLUMN, sort=False):
            h = self.history[self.history[ID_COLUMN] == series_id]
            out[str(series_id)] = bool(h[TIMESTAMP_COLUMN].max() < g[TIMESTAMP_COLUMN].min())
        return out


def chronological_holdout(frame: pd.DataFrame, horizon: int) -> HoldoutSplit:
    """Hold out the final ``horizon`` rows of every series as truth."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    histories, truths = [], []
    for series_id, g in frame.groupby(ID_COLUMN, sort=False):
        g = g.sort_values(TIMESTAMP_COLUMN, kind="mergesort")
        if len(g) <= horizon:
            raise ValidationError(
                "MIN_HISTORY",
                "<holdout>",
                f"series {series_id!r} has {len(g)} rows; a {horizon}-step holdout leaves no history.",
                {"series_id": series_id, "n_rows": len(g), "horizon": horizon},
            )
        histories.append(g.iloc[:-horizon])
        truths.append(g.iloc[-horizon:])
    return HoldoutSplit(
        history=pd.concat(histories, ignore_index=True),
        truth=pd.concat(truths, ignore_index=True),
        horizon=horizon,
    )


# ---------------------------------------------------------------------------
# Baselines (history only)
# ---------------------------------------------------------------------------


def _baseline_frame(
    history: pd.DataFrame, frequency: FrequencyInfo, horizon: int, fn, name: str
) -> pd.DataFrame:
    rows = []
    for series_id, g in history.groupby(ID_COLUMN, sort=False):
        g = g.sort_values(TIMESTAMP_COLUMN, kind="mergesort")
        values = g[VALUE_COLUMN].to_numpy(dtype=float)
        preds = np.asarray(fn(values), dtype=float)
        if preds.shape != (horizon,):
            raise RuntimeError(f"{name}: expected {horizon} predictions, got {preds.shape}")
        stamps = future_timestamps(pd.Timestamp(g[TIMESTAMP_COLUMN].iloc[-1]), frequency, horizon)
        for step in range(horizon):
            rows.append(
                {
                    ID_COLUMN: series_id,
                    TIMESTAMP_COLUMN: stamps[step],
                    STEP_COLUMN: step + 1,
                    POINT_COLUMN: float(preds[step]),
                    "method": name,
                }
            )
    return pd.DataFrame(rows)


def naive_baseline(history: pd.DataFrame, frequency: FrequencyInfo, horizon: int) -> pd.DataFrame:
    """Repeat the last observed value."""
    return _baseline_frame(history, frequency, horizon, lambda v: np.repeat(v[-1], horizon), "naive")


def seasonal_naive_baseline(
    history: pd.DataFrame, frequency: FrequencyInfo, horizon: int, season_length: int
) -> pd.DataFrame:
    """Repeat the last full season; refuses a series shorter than one season."""
    if isinstance(season_length, bool) or not isinstance(season_length, int) or season_length < 1:
        raise ValueError(f"season_length must be a positive int, got {season_length!r}")

    def fn(v: np.ndarray) -> np.ndarray:
        if len(v) < season_length:
            raise ValidationError(
                "SEASON_TOO_LONG",
                "<history>",
                f"a series has {len(v)} rows, fewer than season_length={season_length}.",
                {"n_rows": len(v), "season_length": season_length},
            )
        season = v[-season_length:]
        return np.array([season[k % season_length] for k in range(horizon)])

    return _baseline_frame(history, frequency, horizon, fn, "seasonal_naive")


def _ses_fit(values: np.ndarray, alphas: np.ndarray) -> tuple[float, float]:
    """Simple exponential smoothing: choose alpha by in-sample one-step SSE; return (alpha, level)."""
    best_alpha, best_sse, best_level = 0.5, np.inf, float(values[-1])
    for alpha in alphas:
        level = float(values[0])
        sse = 0.0
        for x in values[1:]:
            sse += (x - level) ** 2
            level = alpha * x + (1 - alpha) * level
        if sse < best_sse:
            best_alpha, best_sse, best_level = float(alpha), sse, level
    return best_alpha, best_level


def ses_baseline(history: pd.DataFrame, frequency: FrequencyInfo, horizon: int) -> pd.DataFrame:
    """Simple exponential smoothing (flat forecast at the fitted level), alpha chosen in-sample."""
    alphas = np.linspace(0.05, 0.95, 19)
    return _baseline_frame(history, frequency, horizon, lambda v: np.repeat(_ses_fit(v, alphas)[1], horizon), "ses")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mase(y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray, season_length: int = 1) -> float:
    """Mean absolute scaled error: MAE over the in-sample seasonal-naive MAE (Hyndman & Koehler)."""
    y_true, y_pred, insample = (np.asarray(a, dtype=float) for a in (y_true, y_pred, insample))
    if len(insample) <= season_length:
        return float("nan")
    scale = np.mean(np.abs(insample[season_length:] - insample[:-season_length]))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in percent; a term with zero denominator counts as zero error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / denom)
    return float(100 * np.mean(terms))


def quantile_loss(y_true: np.ndarray, y_q: np.ndarray, level: float) -> float:
    """Pinball loss at ``level``: mean of ``max(level*(y-q), (level-1)*(y-q))``."""
    y_true, y_q = np.asarray(y_true, dtype=float), np.asarray(y_q, dtype=float)
    diff = y_true - y_q
    return float(np.mean(np.maximum(level * diff, (level - 1) * diff)))


def _aligned(forecast_df: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    keys = [ID_COLUMN, TIMESTAMP_COLUMN]
    f = forecast_df.copy()
    t = truth[[*keys, VALUE_COLUMN]].copy()
    f[TIMESTAMP_COLUMN] = pd.to_datetime(f[TIMESTAMP_COLUMN])
    t[TIMESTAMP_COLUMN] = pd.to_datetime(t[TIMESTAMP_COLUMN])
    merged = f.merge(t, on=keys, how="outer", indicator=True)
    bad = merged[merged["_merge"] != "both"]
    if len(bad):
        raise ValidationError(
            "EVALUATION_MISALIGNED",
            "<evaluation>",
            f"{len(bad)} forecast/truth rows do not line up on (series_id, timestamp); first: "
            f"{bad.iloc[0][ID_COLUMN]!r} at {bad.iloc[0][TIMESTAMP_COLUMN]} ({bad.iloc[0]['_merge']}).",
            {"n_misaligned": int(len(bad))},
        )
    return merged.drop(columns="_merge")


def evaluate_forecast(
    forecast_df: pd.DataFrame,
    truth: pd.DataFrame,
    history: pd.DataFrame,
    config: ForecastConfig,
) -> dict[str, Any]:
    """Per-series and pooled metrics of ``forecast_df`` against ``truth``.

    Rows must match exactly on ``(series_id, timestamp)``; a mismatch is refused rather than
    dropped. MASE is scaled by ``config.season_length`` (or 1).
    """
    aligned = _aligned(forecast_df, truth)
    m = config.season_length or 1
    q_cols = [c for c in forecast_df.columns if c.startswith("q") and c[1:].replace(".", "", 1).isdigit()]
    per_series = {}
    for series_id, g in aligned.groupby(ID_COLUMN, sort=False):
        g = g.sort_values(TIMESTAMP_COLUMN, kind="mergesort")
        insample = history[history[ID_COLUMN] == series_id].sort_values(TIMESTAMP_COLUMN)[VALUE_COLUMN].to_numpy(float)
        y, yhat = g[VALUE_COLUMN].to_numpy(float), g[POINT_COLUMN].to_numpy(float)
        entry: dict[str, Any] = {
            "n": int(len(g)),
            "mae": float(np.mean(np.abs(y - yhat))),
            "mase": mase(y, yhat, insample, m),
            "smape": smape(y, yhat),
        }
        if q_cols:
            entry["quantile_loss"] = {c: quantile_loss(y, g[c].to_numpy(float), float(c[1:])) for c in q_cols}
            entry["mean_quantile_loss"] = float(np.mean(list(entry["quantile_loss"].values())))
            lo, hi = min(q_cols, key=lambda c: float(c[1:])), max(q_cols, key=lambda c: float(c[1:]))
            if lo != hi:
                inside = (y >= g[lo].to_numpy(float)) & (y <= g[hi].to_numpy(float))
                entry["interval"] = {"lower": lo, "upper": hi, "nominal": round(float(hi[1:]) - float(lo[1:]), 2)}
                entry["interval_coverage"] = float(np.mean(inside))
        per_series[str(series_id)] = entry
    pooled: dict[str, Any] = {
        "n_series": len(per_series),
        "n_points": int(sum(e["n"] for e in per_series.values())),
        "mae": float(np.mean([e["mae"] for e in per_series.values()])),
        "mase": float(np.nanmean([e["mase"] for e in per_series.values()])),
        "smape": float(np.mean([e["smape"] for e in per_series.values()])),
        "mase_season_length": m,
    }
    if q_cols:
        pooled["mean_quantile_loss"] = float(np.mean([e["mean_quantile_loss"] for e in per_series.values()]))
        pooled["quantile_loss"] = {
            c: float(np.mean([e["quantile_loss"][c] for e in per_series.values()])) for c in q_cols
        }
        cov = [e["interval_coverage"] for e in per_series.values() if "interval_coverage" in e]
        if cov:
            pooled["interval_coverage"] = float(np.mean(cov))
            pooled["interval"] = next(iter(per_series.values()))["interval"]
    return {"estimation": "chronological tail holdout", "per_series": per_series, "aggregate": pooled}


def evaluation_report(
    model_eval: dict[str, Any],
    baseline_evals: dict[str, dict[str, Any]],
    *,
    horizon: int,
    sample_kind: str,
) -> dict[str, Any]:
    """Compare the model with each baseline and *record* verdicts (no assertion).

    ``verdict`` is ``sample-sanity``: one tail holdout on one small sample shows the pipeline
    works end to end and how the model compares on that sample; it is not benchmark evidence.
    """
    rows = [{"method": "timesfm", **{k: model_eval["aggregate"].get(k) for k in ("mae", "mase", "smape")}}]
    comparison = {}
    for name, ev in baseline_evals.items():
        agg = ev["aggregate"]
        rows.append({"method": name, **{k: agg.get(k) for k in ("mae", "mase", "smape")}})
        comparison[name] = {
            "model_mase_lower": bool(model_eval["aggregate"]["mase"] < agg["mase"])
            if np.isfinite(model_eval["aggregate"]["mase"]) and np.isfinite(agg["mase"])
            else None,
            "model_smape_lower": bool(model_eval["aggregate"]["smape"] < agg["smape"]),
        }
    beaten = [n for n, c in comparison.items() if c["model_mase_lower"]]
    return {
        "verdict": "sample-sanity",
        "reason": (
            f"one chronological holdout of {horizon} steps per series on {sample_kind}; metrics are tutorial "
            "evidence, not a benchmark"
        ),
        "horizon": horizon,
        "metrics": {"mase": "MAE / in-sample seasonal-naive MAE", "smape": "percent", "quantile_loss": "pinball"},
        "table": rows,
        "comparison": comparison,
        "model_beats_on_mase": beaten,
        "model_quantile_loss": model_eval["aggregate"].get("quantile_loss"),
        "model_interval_coverage": model_eval["aggregate"].get("interval_coverage"),
        "interval": model_eval["aggregate"].get("interval"),
    }
