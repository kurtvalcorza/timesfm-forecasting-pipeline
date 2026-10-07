"""Execute the notebook's own code cells sequentially against stand-in modules.

STAND-IN EVIDENCE ONLY. ``torch`` and ``timesfm`` are replaced by deterministic stubs and the weight file
is a sparse placeholder of the pinned size whose digest is substituted by ``standin_sha256_file`` after
the carried module cell defines the real one, so this proves that every cell runs in order, that the data
contract, evaluation, activity, export and reload-parity stages work, and that the outputs have the
declared shape. It is not pretrained-model inference, not a digest verification of the checkpoint, and not
clean-runtime execution evidence (docs/release-verification.md). The real digest check is covered by
``tests/test_model.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from .conftest import ROOT, SAMPLE, install_standin_modules, stage_standin_weights, standin_sha256_file

NOTEBOOK = ROOT / "tutorials" / "timesfm_forecasting_colab.ipynb"


def _runner():
    spec = importlib.util.spec_from_file_location("run_notebook", ROOT / "tools" / "run_notebook.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    (work / "data").mkdir(parents=True)
    shutil.copy(SAMPLE, work / "data" / SAMPLE.name)  # the cell's digest check accepts the cached copy; no network
    stage_standin_weights(work / "weights" / "timesfm-3.0-pytorch")
    return work


def test_notebook_cells_run_end_to_end_against_standins(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIMER_NOTEBOOK_CI_PREINSTALLED", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    install_standin_modules(monkeypatch)

    def substitute_digest(_index: int, namespace: dict) -> None:
        # The carried model.py cell defines sha256_file in the shared namespace; verify_snapshot looks it up
        # there at call time. Only the sparse placeholder's digest is substituted (see conftest).
        if "sha256_file" in namespace and namespace["sha256_file"] is not standin_sha256_file:
            namespace["sha256_file"] = standin_sha256_file

    _runner().execute_notebook(NOTEBOOK, workdir=workdir, after_cell=substitute_digest)

    outputs = workdir / "outputs"
    names = sorted(os.listdir(outputs))
    assert names == [
        "timesfm_forecasting_evaluation.json",
        "timesfm_forecasting_evaluation_report.json",
        "timesfm_forecasting_forecast.csv",
        "timesfm_forecasting_future_forecast.csv",
        "timesfm_forecasting_input_manifest.json",
        "timesfm_forecasting_provenance.json",
        "timesfm_forecasting_result.json",
    ]
    result = json.loads((outputs / "timesfm_forecasting_result.json").read_text(encoding="utf-8"))
    assert result["format"] == "dimer-forecast-bundle"
    assert result["evaluation_report"]["verdict"] == "sample-sanity"
    assert {r["method"] for r in result["evaluation_report"]["table"]} == {"timesfm", "naive", "seasonal_naive", "ses"}
    assert result["activity"]["changed"] == "context_length" and result["activity"]["values"] == [312, 48]
    assert result["model"]["model_id"] == "google/timesfm-3.0-pytorch"
    assert result["model"]["license"] == "timesfm-non-commercial-license-v1.0"
    assert "non-production" in result["model"]["license_terms"]
    # stand-in: the placeholder's substituted digest equals the pinned one, so the flag is True here; it says
    # nothing about the checkpoint (tests/test_model.py exercises the real check)
    assert result["model"]["weights_sha256_verified_against_manifest"] is True
    assert result["model"]["revision"] == "43046b85ec22d584a13f8098c2ed39c889e129c2"
    assert result["provenance_summary"]["runtime"]["torch"] is None  # stand-in: torch is not an installed distribution
    forecast = pd.read_csv(outputs / "timesfm_forecasting_forecast.csv")
    assert list(forecast.columns) == ["series_id", "timestamp", "step", "prediction", "q0.1", "q0.5", "q0.9"]
    assert len(forecast) == 72 and (forecast["prediction"] == forecast["q0.5"]).all()
    future = pd.read_csv(outputs / "timesfm_forecasting_future_forecast.csv")
    assert future["timestamp"].min() == "2026-09-09 00:00:00"
    manifest = json.loads((outputs / "timesfm_forecasting_input_manifest.json").read_text(encoding="utf-8"))
    assert manifest["frequency"]["alias"] == "h" and manifest["n_series"] == 3 and manifest["changes"] == []
    assert (workdir / "weights" / "timesfm-3.0-pytorch" / "config.json").is_file()
    assert not (workdir / "weights" / "timesfm-3.0-pytorch" / "resolved-revision.json").exists()  # nothing fetched
