"""BYOD matrix: the notebook's own Section 4 and 5 cell source, executed with fields set the way an
executor sets them (EXE1/EXE6), over representative good and bad files. Every refusal must name the
file and the rule. Stand-in evidence (no model call is made in these two cells)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import timesfm_forecasting as pkg

from .conftest import ROOT, make_series

NOTEBOOK = ROOT / "tutorials" / "timesfm_forecasting_colab.ipynb"


def _cells() -> tuple[str, str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    src = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in code]
    section_4 = next(s for s in src if s.startswith("# Learner-facing: data source"))
    section_5 = next(s for s in src if s.startswith("validation = validate_series("))
    return section_4, section_5


def set_field(source: str, name: str, value) -> str:
    """Replace a `NAME = literal  # @param` line; fail if the field is not found (EXE6)."""
    pattern = re.compile(rf"^{name} = .*?  # @param.*$", re.M)
    assert pattern.search(source), f"field {name} not found"
    return pattern.sub(f"{name} = {value!r}  # @param", source, count=1)


def run_byod(tmp_path: Path, csv: Path, monkeypatch, **fields) -> dict:
    s4, s5 = _cells()
    s4 = set_field(s4, "USE_BYOD", True)
    s4 = set_field(s4, "BYOD_PATH", str(csv))
    for name, value in fields.items():
        s4 = set_field(s4, name, value)
    monkeypatch.chdir(tmp_path)
    ns = {"__name__": "__main__", **{k: getattr(pkg, k) for k in pkg.__all__}}
    exec(compile(s4, "section-4", "exec"), ns, ns)
    exec(compile(s5, "section-5", "exec"), ns, ns)
    return ns


def _write(tmp_path: Path, name: str, frame: pd.DataFrame) -> Path:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_good_single_series_file(tmp_path: Path, monkeypatch) -> None:
    csv = _write(tmp_path, "good.csv", make_series(n=96).drop(columns="series_id"))
    ns = run_byod(tmp_path, csv, monkeypatch, VALUE_FIELD="value", SERIES_ID_FIELD="")
    assert ns["data_source"]["kind"] == "byod" and ns["data_source"]["path"] == str(csv)
    assert ns["validation"].series_ids == ["series"] and ns["refusal"]["code"] == "TIMESTAMP_DUPLICATE"
    assert (tmp_path / "outputs" / "timesfm_forecasting_input_manifest.json").is_file()


def test_good_multi_series_file_with_own_column_names(tmp_path: Path, monkeypatch) -> None:
    frame = pd.concat([make_series("x", n=60), make_series("y", n=60)], ignore_index=True)
    frame = frame.rename(columns={"timestamp": "when", "value": "y", "series_id": "site"})
    csv = _write(tmp_path, "multi.csv", frame)
    ns = run_byod(tmp_path, csv, monkeypatch, TIMESTAMP_FIELD="when", VALUE_FIELD="y", SERIES_ID_FIELD="site", HORIZON=12)
    assert ns["validation"].series_ids == ["x", "y"] and ns["config"].horizon == 12


def test_monthly_file_is_accepted_with_calendar_frequency(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame({"timestamp": pd.date_range("2016-01-01", periods=72, freq="MS"), "value": np.arange(72.0)})
    csv = _write(tmp_path, "monthly.csv", frame)
    ns = run_byod(tmp_path, csv, monkeypatch, VALUE_FIELD="value", SERIES_ID_FIELD="", HORIZON=12, SEASON_LENGTH=12)
    assert ns["validation"].frequency.kind == "calendar"


def test_gap_and_nan_policies_reach_the_file(tmp_path: Path, monkeypatch) -> None:
    frame = make_series(n=96).drop(columns="series_id").drop(index=[10, 11]).reset_index(drop=True)
    frame.loc[20, "value"] = np.nan
    csv = _write(tmp_path, "gappy.csv", frame)
    with pytest.raises(pkg.ValidationError, match=r"\[FREQUENCY_GAP\] " + re.escape(str(csv))):
        run_byod(tmp_path, csv, monkeypatch, VALUE_FIELD="value", SERIES_ID_FIELD="")
    ns = run_byod(tmp_path, csv, monkeypatch, VALUE_FIELD="value", SERIES_ID_FIELD="", GAP_POLICY="fill", NAN_POLICY="interpolate")
    assert ns["validation"].series["series"].n_gap_points_filled == 2
    assert ns["validation"].series["series"].n_interpolated == 3


@pytest.mark.parametrize(
    ("name", "frame_fn", "fields", "code"),
    [
        ("missing-column.csv", lambda: make_series(n=96).rename(columns={"value": "temperature"}), {"VALUE_FIELD": "value"}, "REQUIRED_COLUMNS"),
        ("short.csv", lambda: make_series(n=30).drop(columns="series_id"), {"VALUE_FIELD": "value", "SERIES_ID_FIELD": ""}, "MIN_HISTORY"),
        ("duplicate.csv", lambda: pd.concat([make_series(n=96)] * 2, ignore_index=True), {"VALUE_FIELD": "value"}, "TIMESTAMP_DUPLICATE"),
        ("text.csv", lambda: make_series(n=96).astype({"value": object}).assign(value=lambda f: f["value"].mask(f.index == 5, "n/a")), {"VALUE_FIELD": "value"}, "VALUE_NUMERIC"),
        ("nan.csv", lambda: make_series(n=96).assign(value=lambda f: f["value"].mask(f.index == 5, np.nan)), {"VALUE_FIELD": "value"}, "VALUE_MISSING"),
        ("irregular.csv", lambda: make_series(n=96).assign(timestamp=lambda f: f["timestamp"] + pd.to_timedelta(np.random.default_rng(0).integers(0, 1000, len(f)), unit="s")), {"VALUE_FIELD": "value"}, "FREQUENCY_IRREGULAR"),
    ],
)
def test_incompatible_files_are_refused_by_name(tmp_path: Path, monkeypatch, name, frame_fn, fields, code) -> None:
    csv = _write(tmp_path, name, frame_fn())
    with pytest.raises(pkg.ValidationError) as info:
        run_byod(tmp_path, csv, monkeypatch, **fields)
    assert info.value.code == code
    assert str(info.value).startswith(f"[{code}] {csv}: ")


def test_duplicate_header_is_refused_before_parsing(tmp_path: Path, monkeypatch) -> None:
    csv = tmp_path / "dup-header.csv"
    csv.write_text("timestamp,value,value\n" + "\n".join(f"2024-01-01T{h:02d}:00:00,{h},{h}" for h in range(24)) + "\n")
    with pytest.raises(pkg.ValidationError, match=r"\[DUPLICATE_HEADER\] " + re.escape(str(csv))):
        run_byod(tmp_path, csv, monkeypatch, VALUE_FIELD="value", SERIES_ID_FIELD="")


def test_missing_byod_path_and_no_upload_dialog_are_named(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(FileNotFoundError, match="BYOD_PATH"):
        run_byod(tmp_path, tmp_path / "absent.csv", monkeypatch)
    s4, _ = _cells()
    s4 = set_field(set_field(s4, "USE_BYOD", True), "BYOD_PATH", "")
    monkeypatch.chdir(tmp_path)
    ns = {"__name__": "__main__", **{k: getattr(pkg, k) for k in pkg.__all__}}
    with pytest.raises(RuntimeError, match="set BYOD_PATH"):
        exec(compile(s4, "section-4", "exec"), ns, ns)


def test_default_path_uses_the_cached_sample_without_network(tmp_path: Path, monkeypatch) -> None:
    s4, s5 = _cells()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "openmeteo_ph_hourly_temperature.csv").write_bytes((ROOT / "examples" / "sample-data" / "openmeteo_ph_hourly_temperature.csv").read_bytes())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    ns = {"__name__": "__main__", **{k: getattr(pkg, k) for k in pkg.__all__}}
    exec(compile(s4, "section-4", "exec"), ns, ns)
    exec(compile(s5, "section-5", "exec"), ns, ns)
    assert ns["data_source"]["kind"] == "sample" and ns["data_source"]["downloaded_this_run"] is False
    assert ns["validation"].series_ids == ["manila", "cebu", "davao"]
