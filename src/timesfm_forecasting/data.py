"""Loading and validation of long-format time-series tables (sample path and BYOD alike).

The contract is deliberately small: a CSV with a timestamp column and a numeric value column,
optionally a series-id column. Validation establishes the sampling frequency, finds gaps and
missing values, applies the configured policies while *reporting* every change, enforces the
length ceilings, and returns a normalised frame plus an input manifest. Every refusal is a
:class:`ValidationError` whose text reads ``[RULE] source: reason``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MAX_CONTEXT_POINTS, MIN_CONTEXT_POINTS, ForecastConfig
from .errors import ValidationError

__all__ = [
    "MIN_OBSERVATIONS",
    "MAX_SERIES",
    "MAX_ROWS",
    "MAX_NAN_FRACTION",
    "INPUT_SCHEMA",
    "RULES",
    "FrequencyInfo",
    "SeriesReport",
    "ValidationResult",
    "infer_frequency",
    "future_timestamps",
    "load_series_csv",
    "validate_series",
    "min_history_required",
]

#: Normalised column names of every frame the package passes around after validation.
ID_COLUMN = "series_id"
TIMESTAMP_COLUMN = "timestamp"
VALUE_COLUMN = "value"
SINGLE_SERIES_ID = "series"

#: Fewest rows for which a sampling interval can be inferred and checked.
MIN_OBSERVATIONS = 3
MAX_SERIES = 1_000
MAX_ROWS = 2_000_000
#: Largest share of a series that ``nan_policy="interpolate"`` may fill.
MAX_NAN_FRACTION = 0.2

INPUT_SCHEMA = {
    "format": "CSV, long format (one row per observation)",
    "timestamp_column": "ISO-8601 or any pandas-parseable timestamp; one per row",
    "value_column": "finite numeric target",
    "series_id_column": "optional; distinct values become independent series",
    "frequency": "inferred: one fixed interval, or a calendar interval (month/quarter/year)",
    "min_rows_per_series": f"horizon + {MIN_CONTEXT_POINTS} (see min_history_required)",
    "max_series": MAX_SERIES,
    "max_rows": MAX_ROWS,
    "max_context_points": MAX_CONTEXT_POINTS,
}

#: Rule identifiers and what each refuses. Kept in one place so the notebook can print them.
RULES = {
    "FILE_NOT_FOUND": "the CSV path does not exist or is not a file",
    "FILE_EMPTY": "the CSV has no data rows",
    "DUPLICATE_HEADER": "two header cells share a name; the table is ambiguous",
    "REQUIRED_COLUMNS": "a configured column is missing from the header",
    "SERIES_ID_EMPTY": "a series id is blank or null",
    "TIMESTAMP_PARSE": "a timestamp cannot be parsed",
    "TIMESTAMP_DUPLICATE": "one series has two rows for the same timestamp",
    "VALUE_NUMERIC": "a value is not numeric",
    "VALUE_INFINITE": "a value is +/-inf",
    "VALUE_MISSING": "a value is missing and nan_policy is 'refuse', or cannot be interpolated",
    "VALUE_MISSING_FRACTION": f"more than {MAX_NAN_FRACTION:.0%} of a series would be interpolated",
    "MIN_OBSERVATIONS": f"a series has fewer than {MIN_OBSERVATIONS} rows",
    "FREQUENCY_IRREGULAR": "timestamps are neither one fixed interval nor a calendar interval",
    "FREQUENCY_GAP": "a series skips grid points and gap_policy is 'refuse'",
    "FREQUENCY_MIXED": "series in one file have different sampling intervals",
    "MIN_HISTORY": "a series is shorter than horizon + the minimum context",
    "MAX_SERIES": f"more than {MAX_SERIES} series",
    "MAX_ROWS": f"more than {MAX_ROWS} rows",
}

_CALENDAR_BASES = {
    "MS", "ME", "BMS", "BME", "SMS", "SME", "QS", "QE", "BQS", "BQE", "YS", "YE", "BYS", "BYE",
    "B", "C",
}


def _fail(code: str, source: str, message: str, **details: Any) -> None:
    raise ValidationError(code, source, message, details)


def min_history_required(config: ForecastConfig) -> int:
    """Rows a series needs so that a holdout of ``horizon`` leaves a usable context."""
    return config.horizon + MIN_CONTEXT_POINTS


@dataclass(frozen=True)
class FrequencyInfo:
    """How timestamps are spaced.

    ``kind`` is ``"fixed"`` (one constant interval, ``step`` set), ``"calendar"`` (a pandas
    calendar alias such as ``MS``, ``step`` None), ``"gapped"`` (fixed grid with skipped points)
    or ``"irregular"``.
    """

    kind: str
    alias: str | None
    step: pd.Timedelta | None
    n_gaps: int = 0
    first_gap_after: pd.Timestamp | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "alias": self.alias,
            "step_seconds": None if self.step is None else float(self.step.total_seconds()),
            "n_gaps": int(self.n_gaps),
            "first_gap_after": None if self.first_gap_after is None else str(self.first_gap_after),
        }


def _calendar_alias(stamps: pd.DatetimeIndex) -> str | None:
    if len(stamps) < 3:
        return None
    try:
        alias = pd.infer_freq(stamps)
    except (TypeError, ValueError):
        return None
    if not alias:
        return None
    base = alias.split("-", 1)[0].lstrip("0123456789")
    return alias if base in _CALENDAR_BASES else None


def _step_alias(step: pd.Timedelta) -> str:
    """A readable alias for a fixed step: whole days as ``<n>D``, otherwise pandas' own spelling."""
    day = pd.Timedelta(days=1)
    if step >= day and step % day == pd.Timedelta(0):
        return f"{step // day}D"
    return pd.tseries.frequencies.to_offset(step).freqstr


def infer_frequency(stamps: pd.DatetimeIndex) -> FrequencyInfo:
    """Classify the spacing of a sorted, duplicate-free timestamp index."""
    if len(stamps) < 2:
        return FrequencyInfo("irregular", None, None)
    # pandas >= 2 stores an index in its parsed unit (s, ms, us or ns); compare in nanoseconds.
    diffs = np.diff(stamps.as_unit("ns").asi8)
    unique = np.unique(diffs)
    if len(unique) == 1:
        step = pd.Timedelta(int(unique[0]), unit="ns")
        return FrequencyInfo("fixed", _step_alias(step), step)
    alias = _calendar_alias(stamps)
    if alias is not None:
        return FrequencyInfo("calendar", alias, None)
    step_ns = int(unique[0])
    if step_ns > 0 and np.all(diffs % step_ns == 0):
        multiples = diffs // step_ns
        n_gaps = int(np.sum(multiples - 1))
        first = int(np.argmax(multiples > 1))
        step = pd.Timedelta(step_ns, unit="ns")
        return FrequencyInfo("gapped", _step_alias(step), step, n_gaps, stamps[first])
    return FrequencyInfo("irregular", None, None)


def future_timestamps(last: pd.Timestamp, frequency: FrequencyInfo, horizon: int) -> pd.DatetimeIndex:
    """The ``horizon`` grid points after ``last`` for a fixed or calendar frequency."""
    if frequency.kind == "fixed" and frequency.step is not None:
        return pd.DatetimeIndex([last + frequency.step * (k + 1) for k in range(horizon)])
    if frequency.alias is None:
        raise ValueError("future timestamps need a fixed or calendar frequency")
    return pd.date_range(start=last, periods=horizon + 1, freq=frequency.alias)[1:]


@dataclass(frozen=True)
class SeriesReport:
    """What validation observed and changed for one series."""

    series_id: str
    n_rows_in: int
    n_rows_out: int
    start: str
    end: str
    n_leading_missing_dropped: int = 0
    n_interpolated: int = 0
    n_gap_points_filled: int = 0
    truncated_from: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "start": self.start,
            "end": self.end,
            "n_leading_missing_dropped": self.n_leading_missing_dropped,
            "n_interpolated": self.n_interpolated,
            "n_gap_points_filled": self.n_gap_points_filled,
            "truncated_from": self.truncated_from,
        }


@dataclass(frozen=True)
class ValidationResult:
    """The normalised table plus the evidence that it passed every rule."""

    frame: pd.DataFrame
    frequency: FrequencyInfo
    series: dict[str, SeriesReport]
    source: str
    config: ForecastConfig
    changes: list[str] = field(default_factory=list)

    @property
    def series_ids(self) -> list[str]:
        return list(self.series)

    def with_frame(self, frame: pd.DataFrame) -> ValidationResult:
        """The same validated contract over a sub-table (e.g. the history half of a holdout)."""
        return replace(self, frame=frame)

    def manifest(self) -> dict[str, Any]:
        """Machine-readable input manifest (what was validated, under which rules and ceilings)."""
        return {
            "source": self.source,
            "schema": INPUT_SCHEMA,
            "columns": {
                "timestamp": self.config.timestamp_column,
                "value": self.config.value_column,
                "series_id": self.config.series_id_column,
            },
            "frequency": self.frequency.to_dict(),
            "n_series": len(self.series),
            "n_rows": int(len(self.frame)),
            "series": [report.to_dict() for report in self.series.values()],
            "ceilings": {
                "min_observations": MIN_OBSERVATIONS,
                "min_history_required": min_history_required(self.config),
                "max_series": MAX_SERIES,
                "max_rows": MAX_ROWS,
                "max_context_points": MAX_CONTEXT_POINTS,
                "context_length": self.config.context_length,
                "horizon": self.config.horizon,
            },
            "policies": {"nan_policy": self.config.nan_policy, "gap_policy": self.config.gap_policy},
            "changes": list(self.changes),
            "verdict": "accepted",
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _header_cells(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return []


def load_series_csv(path: str | Path, config: ForecastConfig) -> pd.DataFrame:
    """Read a CSV and check its header before any parsing of values.

    Returns the raw frame restricted to the configured columns; :func:`validate_series` does the
    rest. Refuses a missing file, an empty file, duplicate header names and missing columns,
    naming the file in every case.
    """
    path = Path(path)
    source = str(path)
    if not path.is_file():
        _fail("FILE_NOT_FOUND", source, "no such file.", path=source)
    header = _header_cells(path)
    if not header:
        _fail("FILE_EMPTY", source, "the file has no header row.")
    names = [cell.strip() for cell in header]
    seen: dict[str, int] = {}
    for name in names:
        seen[name] = seen.get(name, 0) + 1
    duplicates = sorted(name for name, count in seen.items() if count > 1)
    if duplicates:
        _fail(
            "DUPLICATE_HEADER",
            source,
            f"header names {duplicates} appear more than once; rename them so each column is unambiguous.",
            duplicates=duplicates,
        )
    required = [config.timestamp_column, config.value_column] + (
        [config.series_id_column] if config.series_id_column else []
    )
    missing = [name for name in required if name not in names]
    if missing:
        _fail(
            "REQUIRED_COLUMNS",
            source,
            f"required column(s) {missing} not in header {names}; set the column fields to match your file.",
            missing=missing,
            present=names,
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame[required]
    if len(frame) == 0:
        _fail("FILE_EMPTY", source, "the file has a header but no data rows.")
    return frame


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _parse_values(raw: pd.Series, source: str, series_id: str) -> np.ndarray:
    # pandas >= 3 keeps a float NaN as a missing string under astype(str); normalise explicitly.
    text = pd.Series(["" if pd.isna(x) else str(x).strip() for x in raw], index=raw.index, dtype=object)
    blank = text.isin(["", "nan", "NaN", "NA", "null", "None"])
    values = pd.to_numeric(text.where(~blank, None), errors="coerce")
    unparsable = values.isna() & ~blank
    if unparsable.any():
        bad = text[unparsable].iloc[0]
        _fail(
            "VALUE_NUMERIC",
            source,
            f"series {series_id!r} has a non-numeric value {bad!r} at row {int(unparsable.idxmax())}.",
            series_id=series_id,
            example=bad,
        )
    out = values.to_numpy(dtype=float)
    if np.isinf(out).any():
        _fail(
            "VALUE_INFINITE",
            source,
            f"series {series_id!r} contains an infinite value at row {int(np.argmax(np.isinf(out)))}.",
            series_id=series_id,
        )
    return out


def _apply_missing_policy(
    values: np.ndarray, config: ForecastConfig, source: str, series_id: str
) -> tuple[np.ndarray, int, int]:
    """Return (values, n_leading_dropped, n_interpolated) under ``config.nan_policy``."""
    missing = np.isnan(values)
    if not missing.any():
        return values, 0, 0
    if config.nan_policy == "refuse":
        first = int(np.argmax(missing))
        _fail(
            "VALUE_MISSING",
            source,
            f"series {series_id!r} has {int(missing.sum())} missing value(s), first at position {first}; "
            "remove them or set nan_policy='interpolate'.",
            series_id=series_id,
            n_missing=int(missing.sum()),
            first_missing=first,
        )
    first_valid = int(np.argmax(~missing))
    if missing.all():
        _fail("VALUE_MISSING", source, f"series {series_id!r} has no non-missing value.", series_id=series_id)
    values = values[first_valid:]
    missing = missing[first_valid:]
    if missing[-1]:
        _fail(
            "VALUE_MISSING",
            source,
            f"series {series_id!r} ends with a missing value; interpolation needs a later observation.",
            series_id=series_id,
        )
    n_interp = int(missing.sum())
    if n_interp / len(values) > MAX_NAN_FRACTION:
        _fail(
            "VALUE_MISSING_FRACTION",
            source,
            f"series {series_id!r} would have {n_interp}/{len(values)} values interpolated, above the "
            f"{MAX_NAN_FRACTION:.0%} ceiling.",
            series_id=series_id,
            n_missing=n_interp,
            n_rows=len(values),
        )
    if n_interp:
        idx = np.arange(len(values))
        values = values.copy()
        values[missing] = np.interp(idx[missing], idx[~missing], values[~missing])
    return values, first_valid, n_interp


def validate_series(
    frame: pd.DataFrame, config: ForecastConfig, *, source: str = "<frame>"
) -> ValidationResult:
    """Apply every rule in :data:`RULES` and return the normalised table.

    The returned frame has columns ``series_id``, ``timestamp`` (datetime64), ``value`` (float),
    sorted by series then time. Nothing here touches the model.
    """
    required = [config.timestamp_column, config.value_column] + (
        [config.series_id_column] if config.series_id_column else []
    )
    missing_cols = [c for c in required if c not in frame.columns]
    if missing_cols:
        _fail(
            "REQUIRED_COLUMNS",
            source,
            f"required column(s) {missing_cols} not in {list(frame.columns)}.",
            missing=missing_cols,
        )
    if len(frame) == 0:
        _fail("FILE_EMPTY", source, "the table has no rows.")
    if len(frame) > MAX_ROWS:
        _fail("MAX_ROWS", source, f"{len(frame)} rows exceed the {MAX_ROWS}-row ceiling.", n_rows=len(frame))

    if config.series_id_column:
        ids = frame[config.series_id_column].astype(str).str.strip()
        blank = ids.isin(["", "nan", "None"]) | frame[config.series_id_column].isna()
        if blank.any():
            _fail(
                "SERIES_ID_EMPTY",
                source,
                f"{int(blank.sum())} row(s) have a blank series id (first at row {int(blank.idxmax())}).",
                n_blank=int(blank.sum()),
            )
    else:
        ids = pd.Series([SINGLE_SERIES_ID] * len(frame), index=frame.index)
    series_ids = list(dict.fromkeys(ids.tolist()))
    if len(series_ids) > MAX_SERIES:
        _fail(
            "MAX_SERIES", source, f"{len(series_ids)} series exceed the {MAX_SERIES}-series ceiling.",
            n_series=len(series_ids),
        )

    try:
        stamps = pd.to_datetime(frame[config.timestamp_column], errors="raise")
    except (ValueError, TypeError, pd.errors.ParserError) as exc:
        _fail("TIMESTAMP_PARSE", source, f"column {config.timestamp_column!r} could not be parsed: {exc}")
    if stamps.isna().any():
        row = int(stamps.isna().idxmax())
        _fail("TIMESTAMP_PARSE", source, f"column {config.timestamp_column!r} is empty at row {row}.", row=row)
    if getattr(stamps.dt, "tz", None) is not None:
        stamps = stamps.dt.tz_convert("UTC").dt.tz_localize(None)

    changes: list[str] = []
    reports: dict[str, SeriesReport] = {}
    parts: list[pd.DataFrame] = []
    frequencies: dict[str, FrequencyInfo] = {}
    needed = min_history_required(config)
    for series_id in series_ids:
        mask = (ids == series_id).to_numpy()
        part = pd.DataFrame(
            {TIMESTAMP_COLUMN: stamps[mask].to_numpy(), "_raw": frame.loc[mask, config.value_column].to_numpy()}
        )
        n_in = len(part)
        part = part.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
        dup = part[TIMESTAMP_COLUMN].duplicated()
        if dup.any():
            when = part.loc[dup.idxmax(), TIMESTAMP_COLUMN]
            _fail(
                "TIMESTAMP_DUPLICATE",
                source,
                f"series {series_id!r} has {int(dup.sum())} duplicate timestamp(s), first at {when}.",
                series_id=series_id,
                n_duplicates=int(dup.sum()),
            )
        if n_in < MIN_OBSERVATIONS:
            _fail(
                "MIN_OBSERVATIONS",
                source,
                f"series {series_id!r} has {n_in} row(s); at least {MIN_OBSERVATIONS} are needed to infer a frequency.",
                series_id=series_id,
                n_rows=n_in,
            )
        values = _parse_values(part["_raw"], source, series_id)
        index = pd.DatetimeIndex(part[TIMESTAMP_COLUMN])
        freq = infer_frequency(index)
        n_gap_filled = 0
        if freq.kind == "irregular":
            _fail(
                "FREQUENCY_IRREGULAR",
                source,
                f"series {series_id!r} is not regularly sampled (no single interval and no calendar interval).",
                series_id=series_id,
            )
        if freq.kind == "gapped":
            if config.gap_policy == "refuse":
                _fail(
                    "FREQUENCY_GAP",
                    source,
                    f"series {series_id!r} skips {freq.n_gaps} grid point(s) at interval {freq.alias}, first after "
                    f"{freq.first_gap_after}; fill them or set gap_policy='fill'.",
                    series_id=series_id,
                    n_gaps=freq.n_gaps,
                    first_gap_after=str(freq.first_gap_after),
                )
            full = pd.date_range(index[0], index[-1], freq=freq.step)
            filled = pd.Series(values, index=index).reindex(full)
            n_gap_filled = int(len(full) - len(index))
            values = filled.to_numpy(dtype=float)
            index = full
            changes.append(f"{series_id}: inserted {n_gap_filled} missing grid point(s) (gap_policy='fill')")
            freq = FrequencyInfo("fixed", freq.alias, freq.step)
        values, n_lead, n_interp = _apply_missing_policy(values, config, source, series_id)
        if n_lead:
            index = index[n_lead:]
            changes.append(f"{series_id}: dropped {n_lead} leading missing value(s)")
        if n_interp:
            changes.append(f"{series_id}: linearly interpolated {n_interp} missing value(s)")
        if len(values) < needed:
            _fail(
                "MIN_HISTORY",
                source,
                f"series {series_id!r} has {len(values)} usable row(s); horizon {config.horizon} + minimum context "
                f"{MIN_CONTEXT_POINTS} requires at least {needed}.",
                series_id=series_id,
                n_rows=len(values),
                required=needed,
            )
        truncated_from: int | None = None
        if len(values) > MAX_CONTEXT_POINTS:
            truncated_from = len(values)
            values = values[-MAX_CONTEXT_POINTS:]
            index = index[-MAX_CONTEXT_POINTS:]
            changes.append(
                f"{series_id}: kept the most recent {MAX_CONTEXT_POINTS} of {truncated_from} rows (model context limit)"
            )
        frequencies[series_id] = freq
        reports[series_id] = SeriesReport(
            series_id=series_id,
            n_rows_in=n_in,
            n_rows_out=len(values),
            start=str(index[0]),
            end=str(index[-1]),
            n_leading_missing_dropped=n_lead,
            n_interpolated=n_interp,
            n_gap_points_filled=n_gap_filled,
            truncated_from=truncated_from,
        )
        parts.append(
            pd.DataFrame({ID_COLUMN: series_id, TIMESTAMP_COLUMN: index, VALUE_COLUMN: values.astype(float)})
        )

    aliases = {f.alias for f in frequencies.values()}
    if len(aliases) > 1:
        _fail(
            "FREQUENCY_MIXED",
            source,
            f"series have different sampling intervals: { {k: v.alias for k, v in frequencies.items()} }.",
            aliases=sorted(a for a in aliases if a),
        )
    out = pd.concat(parts, ignore_index=True)
    return ValidationResult(
        frame=out,
        frequency=next(iter(frequencies.values())),
        series=reports,
        source=source,
        config=config,
        changes=changes,
    )
