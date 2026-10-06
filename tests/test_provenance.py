"""Result bundle export, reload and parity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from timesfm_forecasting import (
    ForecastConfig,
    build_provenance,
    check_reload_parity,
    chronological_holdout,
    evaluate_forecast,
    forecast,
    reload_result_bundle,
    runtime_versions,
    validate_series,
    write_result_bundle,
)

from .conftest import FakeModel, make_series


def _bundle(tmp_path: Path):
    config = ForecastConfig(horizon=12, context_length=48, season_length=24)
    v = validate_series(make_series(n=96), config, source="s.csv")
    split = chronological_holdout(v.frame, 12)
    result = forecast(v.with_frame(split.history), config, FakeModel())
    ev = evaluate_forecast(result.forecast, split.truth, split.history, config)
    prov = build_provenance(FakeModel(), config, v, result, data_source={"kind": "test"}, notebook_source={"x": 1})
    paths = write_result_bundle(tmp_path, "t", forecast=result.forecast, evaluation=ev, report={"verdict": "sample-sanity"}, provenance=prov)
    return result, paths


def test_runtime_versions_work_without_torch() -> None:
    versions = runtime_versions()
    assert versions["python"] and versions["numpy"] and versions["pandas"]
    assert "cuda_available" in versions


def test_bundle_round_trip_and_parity(tmp_path: Path) -> None:
    result, paths = _bundle(tmp_path)
    assert set(paths) == {"forecast_csv", "evaluation_json", "provenance_json", "result_json"}
    res = json.loads(Path(paths["result_json"]).read_text(encoding="utf-8"))
    assert res["format"] == "dimer-forecast-bundle" and res["n_rows"] == 12
    assert res["model"]["model_id"] == "google/timesfm-3.0-pytorch"
    assert res["model"]["license"] == "timesfm-non-commercial-license-v1.0"
    assert set(res["files"]) == {"forecast_csv", "evaluation_json", "provenance_json"}
    reloaded, res2 = reload_result_bundle(tmp_path, "t")
    assert res2 == res
    parity = check_reload_parity(result.forecast, reloaded)
    assert parity["ok"] and parity["rows"] == 12 and parity["problems"] == []
    prov = json.loads(Path(paths["provenance_json"]).read_text(encoding="utf-8"))
    assert prov["model"]["revision_status"].startswith("pinned")
    assert prov["notebook_source"] == {"x": 1}


def test_tampered_bundle_file_is_refused(tmp_path: Path) -> None:
    _, paths = _bundle(tmp_path)
    Path(paths["forecast_csv"]).write_text("series_id,timestamp\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match its digest"):
        reload_result_bundle(tmp_path, "t")


def test_parity_failure_raises_with_named_problems(tmp_path: Path) -> None:
    result, _ = _bundle(tmp_path)
    changed = result.forecast.copy()
    changed.loc[0, "prediction"] += 1e-3
    with pytest.raises(RuntimeError, match="prediction: max abs difference"):
        check_reload_parity(result.forecast, changed)
    with pytest.raises(RuntimeError, match="columns differ"):
        check_reload_parity(result.forecast, changed.drop(columns="q0.1"))
