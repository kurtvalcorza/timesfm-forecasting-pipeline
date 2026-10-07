"""Data loading and validation: every refusal names the file and the rule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timesfm_forecasting import (
    MAX_CONTEXT_POINTS,
    RULES,
    ForecastConfig,
    ValidationError,
    future_timestamps,
    infer_frequency,
    load_series_csv,
    min_history_required,
    validate_series,
)

from .conftest import make_series


def _refusal(code: str, expected_source: str, fn, *args, **kwargs) -> ValidationError:
    with pytest.raises(ValidationError) as info:
        fn(*args, **kwargs)
    exc = info.value
    assert exc.code == code, str(exc)
    assert exc.source == expected_source
    assert str(exc).startswith(f"[{code}] {expected_source}: ")
    assert code in RULES
    return exc


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_sample_loads_and_validates(sample_path: Path, sample_config: ForecastConfig) -> None:
    frame = load_series_csv(sample_path, sample_config)
    result = validate_series(frame, sample_config, source=str(sample_path))
    assert result.series_ids == ["manila", "cebu", "davao"]
    assert result.frequency.kind == "fixed" and result.frequency.alias == "h"
    assert all(r.n_rows_out == 336 for r in result.series.values())
    assert result.changes == []
    manifest = result.manifest()
    assert manifest["verdict"] == "accepted"
    assert manifest["ceilings"]["min_history_required"] == 40
    assert list(result.frame.columns) == ["series_id", "timestamp", "value"]


def test_missing_file_names_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.csv"
    _refusal("FILE_NOT_FOUND", str(missing), load_series_csv, missing, ForecastConfig())


def test_duplicate_header_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "dup.csv"
    path.write_text("timestamp,value,value\n2024-01-01,1,2\n", encoding="utf-8")
    exc = _refusal("DUPLICATE_HEADER", str(path), load_series_csv, path, ForecastConfig())
    assert exc.details["duplicates"] == ["value"]


def test_missing_required_column_names_it(tmp_path: Path) -> None:
    path = tmp_path / "cols.csv"
    path.write_text("date,y\n2024-01-01,1\n", encoding="utf-8")
    exc = _refusal("REQUIRED_COLUMNS", str(path), load_series_csv, path, ForecastConfig())
    assert exc.details["missing"] == ["timestamp", "value"]


def test_empty_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("timestamp,value\n", encoding="utf-8")
    _refusal("FILE_EMPTY", str(path), load_series_csv, path, ForecastConfig())
    path.write_text("", encoding="utf-8")
    _refusal("FILE_EMPTY", str(path), load_series_csv, path, ForecastConfig())


def test_bom_and_whitespace_headers_are_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    stamps = pd.date_range("2024-01-01", periods=48, freq="h")
    body = "\n".join(f"{t.isoformat()}, {i}" for i, t in enumerate(stamps))
    path.write_text("﻿timestamp, value\n" + body + "\n", encoding="utf-8")
    frame = load_series_csv(path, ForecastConfig(horizon=12))
    result = validate_series(frame, ForecastConfig(horizon=12), source=str(path))
    assert result.series["series"].n_rows_out == 48


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def test_duplicate_timestamp() -> None:
    frame = make_series(n=48)
    frame = pd.concat([frame, frame.iloc[[3]]], ignore_index=True)
    exc = _refusal("TIMESTAMP_DUPLICATE", "d.csv", validate_series, frame, ForecastConfig(horizon=12, series_id_column="series_id"), source="d.csv")
    assert "'A'" in str(exc) and exc.details["n_duplicates"] == 1


def test_unparseable_timestamp() -> None:
    frame = make_series(n=48).astype({"timestamp": str})
    frame.loc[5, "timestamp"] = "not-a-date"
    _refusal("TIMESTAMP_PARSE", "t.csv", validate_series, frame, ForecastConfig(horizon=12), source="t.csv")


def test_non_numeric_and_infinite_values() -> None:
    frame = make_series(n=48).astype({"value": object})
    frame.loc[7, "value"] = "seven"
    exc = _refusal("VALUE_NUMERIC", "v.csv", validate_series, frame, ForecastConfig(horizon=12), source="v.csv")
    assert "row 7" in str(exc)
    frame = make_series(n=48)
    frame.loc[7, "value"] = np.inf
    _refusal("VALUE_INFINITE", "v.csv", validate_series, frame, ForecastConfig(horizon=12), source="v.csv")


def test_missing_values_refused_by_default_and_interpolated_on_request() -> None:
    frame = make_series(n=48)
    frame.loc[[0, 10, 11], "value"] = np.nan
    exc = _refusal("VALUE_MISSING", "m.csv", validate_series, frame, ForecastConfig(horizon=12), source="m.csv")
    assert exc.details["n_missing"] == 3
    result = validate_series(frame, ForecastConfig(horizon=12, nan_policy="interpolate"), source="m.csv")
    report = result.series["series"]
    assert report.n_leading_missing_dropped == 1 and report.n_interpolated == 2 and report.n_rows_out == 47
    assert any("interpolated 2" in c for c in result.changes) and any("dropped 1 leading" in c for c in result.changes)
    assert np.isfinite(result.frame["value"]).all()
    assert result.manifest()["changes"] == result.changes


def test_trailing_missing_cannot_be_interpolated() -> None:
    frame = make_series(n=48)
    frame.loc[47, "value"] = np.nan
    _refusal("VALUE_MISSING", "m.csv", validate_series, frame, ForecastConfig(horizon=12, nan_policy="interpolate"), source="m.csv")


def test_missing_fraction_ceiling() -> None:
    frame = make_series(n=48)
    frame.loc[1:20, "value"] = np.nan
    _refusal("VALUE_MISSING_FRACTION", "m.csv", validate_series, frame, ForecastConfig(horizon=12, nan_policy="interpolate"), source="m.csv")


def test_too_few_observations_and_min_history() -> None:
    frame = make_series(n=2)
    _refusal("MIN_OBSERVATIONS", "s.csv", validate_series, frame, ForecastConfig(horizon=12), source="s.csv")
    frame = make_series(n=20)
    exc = _refusal("MIN_HISTORY", "s.csv", validate_series, frame, ForecastConfig(horizon=12), source="s.csv")
    assert exc.details["required"] == min_history_required(ForecastConfig(horizon=12)) == 28
    assert "requires at least 28" in str(exc)


def test_gaps_refused_by_default_and_filled_on_request() -> None:
    frame = make_series(n=60).drop(index=[5, 6]).reset_index(drop=True)
    exc = _refusal("FREQUENCY_GAP", "g.csv", validate_series, frame, ForecastConfig(horizon=12), source="g.csv")
    assert exc.details["n_gaps"] == 2 and "2024" not in str(exc) and "2026-01-01 04:00:00" in str(exc)
    # fill needs interpolate for the inserted values
    _refusal("VALUE_MISSING", "g.csv", validate_series, frame, ForecastConfig(horizon=12, gap_policy="fill"), source="g.csv")
    result = validate_series(frame, ForecastConfig(horizon=12, gap_policy="fill", nan_policy="interpolate"), source="g.csv")
    assert result.series["series"].n_gap_points_filled == 2 and result.series["series"].n_interpolated == 2
    assert result.frequency.kind == "fixed" and result.series["series"].n_rows_out == 60


def test_irregular_timestamps_are_refused() -> None:
    stamps = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-09", "2024-01-10"] * 1)
    stamps = pd.DatetimeIndex(stamps).append(pd.date_range("2024-02-01", periods=40, freq="7h"))
    frame = pd.DataFrame({"timestamp": stamps, "value": np.arange(len(stamps), dtype=float)})
    _refusal("FREQUENCY_IRREGULAR", "i.csv", validate_series, frame, ForecastConfig(horizon=12), source="i.csv")


def test_calendar_frequency_monthly_is_accepted() -> None:
    frame = pd.DataFrame({"timestamp": pd.date_range("2015-01-01", periods=60, freq="MS"), "value": np.arange(60.0)})
    result = validate_series(frame, ForecastConfig(horizon=12), source="monthly.csv")
    assert result.frequency.kind == "calendar" and result.frequency.alias == "MS"
    nxt = future_timestamps(pd.Timestamp(result.frame["timestamp"].iloc[-1]), result.frequency, 2)
    assert list(nxt) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]


def test_mixed_frequencies_across_series_are_refused() -> None:
    frame = pd.concat([make_series("A", n=48), make_series("B", n=48, freq="D")], ignore_index=True)
    _refusal("FREQUENCY_MIXED", "mix.csv", validate_series, frame, ForecastConfig(horizon=12, series_id_column="series_id"), source="mix.csv")


def test_blank_series_id_is_refused() -> None:
    frame = make_series("A", n=48)
    frame.loc[3, "series_id"] = ""
    _refusal("SERIES_ID_EMPTY", "id.csv", validate_series, frame, ForecastConfig(horizon=12, series_id_column="series_id"), source="id.csv")


def test_too_many_series_is_refused() -> None:
    from timesfm_forecasting import data as data_mod

    frame = pd.concat([make_series(str(i), n=3) for i in range(data_mod.MAX_SERIES + 1)], ignore_index=True)
    _refusal("MAX_SERIES", "many.csv", validate_series, frame, ForecastConfig(horizon=1, series_id_column="series_id"), source="many.csv")


def test_long_history_is_truncated_with_a_report() -> None:
    n = MAX_CONTEXT_POINTS + 10
    frame = pd.DataFrame({"timestamp": pd.date_range("2000-01-01", periods=n, freq="h"), "value": np.arange(n, dtype=float)})
    result = validate_series(frame, ForecastConfig(horizon=12), source="long.csv")
    assert result.series["series"].truncated_from == n and result.series["series"].n_rows_out == MAX_CONTEXT_POINTS
    assert any("most recent 15360" in c for c in result.changes)


def test_unsorted_rows_are_sorted_and_tz_dropped() -> None:
    frame = make_series(n=48).iloc[::-1].reset_index(drop=True)
    frame["timestamp"] = frame["timestamp"].dt.tz_localize("Asia/Manila")
    result = validate_series(frame, ForecastConfig(horizon=12), source="s.csv")
    stamps = result.frame["timestamp"]
    assert stamps.is_monotonic_increasing and stamps.dt.tz is None
    assert stamps.iloc[0] == pd.Timestamp("2025-12-31 16:00:00")  # converted to UTC


def test_single_series_without_id_column_is_labelled() -> None:
    result = validate_series(make_series(n=48).drop(columns="series_id"), ForecastConfig(horizon=12), source="one.csv")
    assert result.series_ids == ["series"]


def test_infer_frequency_kinds() -> None:
    assert infer_frequency(pd.date_range("2024-01-01", periods=5, freq="D")).alias == "1D"
    assert infer_frequency(pd.date_range("2024-01-01", periods=5, freq="15min")).alias == "15min"
    gapped = infer_frequency(pd.date_range("2024-01-01", periods=10, freq="h").delete([4]))
    assert gapped.kind == "gapped" and gapped.n_gaps == 1 and str(gapped.first_gap_after) == "2024-01-01 03:00:00"
    assert infer_frequency(pd.DatetimeIndex(["2024-01-01"])).kind == "irregular"


def test_with_frame_keeps_the_contract() -> None:
    result = validate_series(make_series(n=48), ForecastConfig(horizon=12), source="s.csv")
    sub = result.with_frame(result.frame.iloc[:40])
    assert len(sub.frame) == 40 and sub.frequency == result.frequency and sub.source == "s.csv"
