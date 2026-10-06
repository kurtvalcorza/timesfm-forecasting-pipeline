"""Pinned identity, staging and integrity. Real-weight loading is an integration test and skips without torch."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from timesfm_forecasting import (
    ForecastConfig,
    ModelIntegrityError,
    ModelSourceError,
    check_model_source,
    read_manifest,
    stage_missing_files,
    verify_snapshot,
)
from timesfm_forecasting import model as model_mod

from .conftest import CONFIG_JSON_TEXT, MANIFEST, install_standin_modules, stage_standin_weights

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("timesfm") is not None


def test_committed_manifest_matches_the_module_identity() -> None:
    manifest = read_manifest()
    assert manifest["modelId"] == model_mod.MODEL_ID == "google/timesfm-2.5-200m-pytorch"
    assert manifest["revision"] == model_mod.MODEL_REVISION
    files = {f["path"]: f for f in manifest["files"]}
    assert set(files) == {"config.json", "model.safetensors"}
    assert files["model.safetensors"]["bytes"] == model_mod.PINNED_WEIGHTS_BYTES == 925_181_104
    assert files["model.safetensors"]["sha256"].startswith("pending:"), "the weight digest is pending; never invent one"
    assert files["config.json"]["sha256"] == model_mod.PINNED_CONFIG_SHA256
    assert "to be confirmed on the first hosted run" in manifest["revisionStatus"]
    assert "to be confirmed on the first hosted run" in model_mod.MODEL_REVISION_STATUS


def test_config_json_digest_was_computed_from_the_hub_text() -> None:
    import hashlib

    data = CONFIG_JSON_TEXT.encode("utf-8")
    assert len(data) == 475
    assert hashlib.sha256(data).hexdigest() == model_mod.PINNED_CONFIG_SHA256
    assert json.loads(CONFIG_JSON_TEXT)["quantiles"] == list(model_mod.TRAINED_QUANTILES)


@pytest.mark.parametrize(
    ("model_id", "revision"),
    [
        ("google/timesfm-3.0-pytorch", "main"),
        ("/tmp/weights", "main"),
        ("hf://google/timesfm-2.5-200m-pytorch", "main"),
        ("google/timesfm-2.5-200m-pytorch", "v1"),
    ],
)
def test_other_sources_and_revisions_are_refused(model_id, revision) -> None:
    with pytest.raises(ModelSourceError):
        check_model_source(model_id, revision)
    check_model_source(model_mod.MODEL_ID, model_mod.MODEL_REVISION)


def test_stage_missing_files_uses_the_injected_downloader_and_records_the_revision(tmp_path: Path) -> None:
    root = tmp_path / "w"
    root.mkdir()
    (root / model_mod.MANIFEST_NAME).write_bytes(MANIFEST.read_bytes())
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    fetched = []

    def downloader(relative: str, target_root: Path):
        fetched.append(relative)
        (target_root / relative).write_bytes(b"x")
        return {"path": relative, "resolved_revision": "0" * 40}

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == ["config.json", "model.safetensors"]
    assert fetched == ["config.json", "model.safetensors"]
    record = json.loads((root / "resolved-revision.json").read_text(encoding="utf-8"))
    assert record["resolved_revision"] == "0" * 40 and record["requested_revision"] == "main"
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []


def test_verify_snapshot_reports_pending_digest_and_checks_sizes(standin_weights: Path) -> None:
    snapshot = verify_snapshot(standin_weights)
    assert snapshot["digests_pending"] == ["model.safetensors"]
    by_path = {f["path"]: f for f in snapshot["files"]}
    assert by_path["config.json"]["verified"] is True
    assert by_path["model.safetensors"]["verified"] is False and len(by_path["model.safetensors"]["sha256"]) == 64
    assert snapshot["totalBytes"] == 925_181_579
    assert snapshot["resolved_revision"] is None


def test_verify_snapshot_refuses_size_digest_architecture_and_pickle(tmp_path: Path) -> None:
    root = stage_standin_weights(tmp_path / "w")
    (root / "config.json").write_bytes(CONFIG_JSON_TEXT.replace('"patch_length": 32', '"patch_length": 16').encode())
    with pytest.raises(ModelIntegrityError, match="SHA-256"):
        verify_snapshot(root)
    (root / "config.json").write_bytes(CONFIG_JSON_TEXT.encode())
    with (root / "model.safetensors").open("wb") as handle:
        handle.truncate(10)
    with pytest.raises(ModelIntegrityError, match="bytes"):
        verify_snapshot(root)
    root = stage_standin_weights(tmp_path / "w2")
    (root / "pytorch_model.bin").write_bytes(b"\x80")
    with pytest.raises(ModelIntegrityError, match="pickle"):
        verify_snapshot(root)
    root = stage_standin_weights(tmp_path / "w3")
    manifest = json.loads((root / model_mod.MANIFEST_NAME).read_text())
    manifest["revision"] = "other"
    (root / model_mod.MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ModelIntegrityError, match="not the pinned"):
        verify_snapshot(root)


def test_load_pinned_model_through_standin_modules(monkeypatch: pytest.MonkeyPatch, standin_weights: Path) -> None:
    """Stand-in evidence: the loader's control flow with fake torch/timesfm modules, not pretrained inference."""
    install_standin_modules(monkeypatch)
    loaded = model_mod.load_pinned_model(weights_dir=standin_weights)
    assert loaded.identity["model_id"] == model_mod.MODEL_ID
    assert loaded.identity["weights_sha256_verified_against_manifest"] is False
    assert loaded.model.loaded_from == str(standin_weights / "model.safetensors")
    config = ForecastConfig(horizon=30, context_length=100)
    compiled = loaded.compile_for(config)
    assert compiled["max_context"] == 128 and compiled["max_horizon"] == 128  # rounded to patch sizes
    import numpy as np

    point, q = loaded.predict([np.arange(50.0)], 30, config)
    assert point.shape == (1, 30) and q.shape == (1, 30, 10)
    with pytest.raises(ValueError, match="MAX_HORIZON"):
        loaded.predict([np.arange(50.0)], 10_000, config)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_TORCH, reason="torch and timesfm are not installed (CI runs without the model stack)")
def test_real_pinned_weights_load_and_forecast(tmp_path: Path) -> None:
    import numpy as np

    root = tmp_path / "weights"
    root.mkdir()
    (root / model_mod.MANIFEST_NAME).write_bytes(MANIFEST.read_bytes())
    stage_missing_files(root, allow_download=True)
    loaded = model_mod.load_pinned_model(weights_dir=root, device="cpu")
    point, q = loaded.predict([np.sin(np.linspace(0, 20, 200))], 12, ForecastConfig(horizon=12, context_length=200))
    assert point.shape == (1, 12) and q.shape == (1, 12, 10)
    assert np.allclose(point, q[:, :, 5])
