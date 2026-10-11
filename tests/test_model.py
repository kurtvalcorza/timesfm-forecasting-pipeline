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

from .conftest import (
    CONFIG_JSON_TEXT,
    MANIFEST,
    MEDIAN_SLOT,
    N_SLOTS,
    RECORDED_REVISION,
    RECORDED_WEIGHTS_BYTES,
    RECORDED_WEIGHTS_SHA256,
    ROOT,
    install_standin_digest,
    install_standin_modules,
    stage_standin_weights,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("timesfm3") is not None


def test_committed_manifest_matches_the_module_identity() -> None:
    manifest = read_manifest()
    assert manifest["modelId"] == model_mod.MODEL_ID == "google/timesfm-3.0-pytorch"
    assert manifest["revision"] == model_mod.MODEL_REVISION
    files = {f["path"]: f for f in manifest["files"]}
    assert set(files) == {"config.json", "model.safetensors"}
    assert files["model.safetensors"]["bytes"] == model_mod.PINNED_WEIGHTS_BYTES == 1_322_898_824
    assert files["model.safetensors"]["sha256"] == model_mod.PINNED_WEIGHTS_SHA256
    assert files["config.json"]["sha256"] == model_mod.PINNED_CONFIG_SHA256
    assert manifest["revisionStatus"].startswith("pinned:") and "2026-10-06" in manifest["revisionStatus"]
    assert model_mod.MODEL_REVISION_STATUS.startswith("pinned:") and "2026-10-06" in model_mod.MODEL_REVISION_STATUS
    assert {"revision", "config.json", "model.safetensors"} <= set(manifest["provenance"])
    assert all(model_mod._is_digest(f["sha256"]) for f in manifest["files"]), "no sentinel digest survives the pin"
    assert manifest["license"]["name"] == "TimesFM Non-Commercial License v1.0"
    assert model_mod.MODEL_LICENSE == "timesfm-non-commercial-license-v1.0"
    assert "non-commercial" in model_mod.MODEL_LICENSE_TERMS and "non-production" in model_mod.MODEL_LICENSE_TERMS


def test_pin_equals_the_values_recorded_on_the_hosted_run() -> None:
    """The commit and weight digest are the ones the 2026-10-06 hosted run observed; nothing else is accepted."""
    evidence = ROOT / "docs" / "execution-evidence" / "2026-10-06" / "timesfm_forecasting_colab_86f2213_run2_passed.ipynb"
    notebook = json.loads(evidence.read_text(encoding="utf-8"))
    last = [c for c in notebook["cells"] if c["cell_type"] == "code"][-1]
    printed = "".join("".join(o.get("text", [])) for o in last.get("outputs", []))
    assert f"'resolved_revision': '{RECORDED_REVISION}'" in printed
    assert f"'weights_sha256': '{RECORDED_WEIGHTS_SHA256}'" in printed
    assert "'weights_sha256_verified_against_manifest': False" in printed  # the run predates the pin
    assert model_mod.MODEL_REVISION == RECORDED_REVISION
    assert model_mod.PINNED_WEIGHTS_SHA256 == RECORDED_WEIGHTS_SHA256
    assert model_mod.PINNED_WEIGHTS_BYTES == RECORDED_WEIGHTS_BYTES
    manifest = read_manifest()
    assert manifest["revision"] == RECORDED_REVISION
    weights = next(f for f in manifest["files"] if f["path"] == "model.safetensors")
    assert (weights["sha256"], weights["bytes"]) == (RECORDED_WEIGHTS_SHA256, RECORDED_WEIGHTS_BYTES)


def test_config_json_digest_was_computed_from_the_hub_text() -> None:
    import hashlib

    data = CONFIG_JSON_TEXT.encode("utf-8")
    assert len(data) == 1273
    assert hashlib.sha256(data).hexdigest() == model_mod.PINNED_CONFIG_SHA256
    config = json.loads(CONFIG_JSON_TEXT)
    assert config["quantiles"] == list(model_mod.TRAINED_QUANTILES) and config["quantiles"][MEDIAN_SLOT] == 0.5
    assert config["input_patch_len"] == 32 and config["output_patch_len"] == 64
    assert config["transformer_config"]["num_layers"] == 20
    assert config["transformer_config"]["transformer"]["use_rope_var"] is False  # differs from the forecaster's bare-file default
    assert model_mod._config_mismatches(config) == []


@pytest.mark.parametrize(
    ("model_id", "revision"),
    [
        ("google/timesfm-2.5-200m-pytorch", "main"),
        ("/tmp/weights", "main"),
        ("hf://google/timesfm-3.0-pytorch", "main"),
        ("google/timesfm-3.0-pytorch", "v1"),
        ("google/timesfm-3.0-pytorch", "main"),
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
        return {"path": relative, "resolved_revision": model_mod.MODEL_REVISION}

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == ["config.json", "model.safetensors"]
    assert fetched == ["config.json", "model.safetensors"]
    record = json.loads((root / "resolved-revision.json").read_text(encoding="utf-8"))
    assert record["resolved_revision"] == record["requested_revision"] == RECORDED_REVISION
    assert record["resolved_revision_matches_pin"] is True
    assert record["license"] == "timesfm-non-commercial-license-v1.0" and "non-production" in record["license_terms"]
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []


def test_stage_missing_files_flags_a_commit_other_than_the_pinned_one(tmp_path: Path) -> None:
    root = tmp_path / "w"
    root.mkdir()
    (root / model_mod.MANIFEST_NAME).write_bytes(MANIFEST.read_bytes())

    def downloader(relative: str, target_root: Path):
        (target_root / relative).write_bytes(b"x")
        return {"path": relative, "resolved_revision": "0" * 40}

    with pytest.raises(ModelIntegrityError, match="pinned revision"):
        stage_missing_files(root, allow_download=True, downloader=downloader)
    record = json.loads((root / "resolved-revision.json").read_text(encoding="utf-8"))
    assert record["resolved_revision"] == "0" * 40 and record["resolved_revision_matches_pin"] is False


def test_verify_snapshot_checks_sizes_and_digests(monkeypatch: pytest.MonkeyPatch, standin_weights: Path) -> None:
    with pytest.raises(ModelIntegrityError, match="model.safetensors SHA-256"):
        verify_snapshot(standin_weights)  # the sparse placeholder does not carry the pinned digest
    install_standin_digest(monkeypatch)  # stand-in evidence: the placeholder's digest is substituted
    snapshot = verify_snapshot(standin_weights)
    assert "digests_pending" not in snapshot
    by_path = {f["path"]: f for f in snapshot["files"]}
    assert by_path["config.json"]["verified"] is True
    assert by_path["model.safetensors"]["verified"] is True
    assert by_path["model.safetensors"]["sha256"] == RECORDED_WEIGHTS_SHA256
    assert snapshot["revision"] == RECORDED_REVISION
    assert snapshot["totalBytes"] == 1_322_900_097
    assert snapshot["license"] == "timesfm-non-commercial-license-v1.0"
    assert snapshot["resolved_revision"] is None


def _stage_small_weights(root: Path, payload: bytes, digest: str | None) -> Path:
    """A real (small) weight file with the manifest's size and digest set to it; ``digest`` overrides the digest."""
    import hashlib

    stage_standin_weights(root)
    (root / "model.safetensors").write_bytes(payload)
    manifest = json.loads((root / model_mod.MANIFEST_NAME).read_text(encoding="utf-8"))
    entry = next(f for f in manifest["files"] if f["path"] == "model.safetensors")
    entry["bytes"] = len(payload)
    entry["sha256"] = digest if digest is not None else hashlib.sha256(payload).hexdigest()
    manifest["totalBytes"] = sum(int(f["bytes"]) for f in manifest["files"])
    (root / model_mod.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root


def test_verify_snapshot_rejects_a_wrong_weight_digest_and_accepts_the_right_one(tmp_path: Path) -> None:
    """The real check, with no stand-in: a small weight file hashed for real against the manifest."""
    payload = b"not the checkpoint, but a real file with a real digest\n" * 64
    wrong = "0" * 63 + "1"
    root = _stage_small_weights(tmp_path / "wrong", payload, wrong)
    with pytest.raises(ModelIntegrityError, match=f"model.safetensors SHA-256 .* does not match the manifest digest {wrong}"):
        verify_snapshot(root)
    root = _stage_small_weights(tmp_path / "pending", payload, "pending: a sentinel is no longer accepted")
    with pytest.raises(ModelIntegrityError, match="carries no SHA-256 digest for model.safetensors"):
        verify_snapshot(root)
    root = _stage_small_weights(tmp_path / "right", payload, None)
    snapshot = verify_snapshot(root)
    weights = next(f for f in snapshot["files"] if f["path"] == "model.safetensors")
    assert weights["verified"] is True and weights["bytes"] == len(payload)
    (root / "model.safetensors").write_bytes(payload[:-1] + b"?")  # same size, one byte changed
    with pytest.raises(ModelIntegrityError, match="model.safetensors SHA-256"):
        verify_snapshot(root)


def test_verify_snapshot_refuses_size_digest_architecture_and_pickle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import hashlib

    install_standin_digest(monkeypatch)  # the placeholder weight passes; everything below is about config.json
    root = stage_standin_weights(tmp_path / "w")
    (root / "config.json").write_bytes(CONFIG_JSON_TEXT.replace('"output_patch_len": 64', '"output_patch_len": 32').encode())
    with pytest.raises(ModelIntegrityError, match="config.json SHA-256"):
        verify_snapshot(root)
    manifest = json.loads((root / model_mod.MANIFEST_NAME).read_text())
    changed = CONFIG_JSON_TEXT.replace('"use_rope_var": false', '"use_rope_var": true ')
    manifest["files"][0]["sha256"] = hashlib.sha256(changed.encode()).hexdigest()  # digest passes; architecture must not
    (root / model_mod.MANIFEST_NAME).write_text(json.dumps(manifest))
    (root / "config.json").write_bytes(changed.encode())
    with pytest.raises(ModelIntegrityError, match="transformer_config.use_rope_var"):
        verify_snapshot(root)  # architecture is asserted independently of the digest
    (root / model_mod.MANIFEST_NAME).write_bytes(MANIFEST.read_bytes())
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
    install_standin_digest(monkeypatch)
    loaded = model_mod.load_pinned_model(weights_dir=standin_weights)
    assert loaded.identity["model_id"] == model_mod.MODEL_ID
    assert loaded.identity["revision"] == RECORDED_REVISION
    assert loaded.identity["license"] == "timesfm-non-commercial-license-v1.0"
    assert "non-commercial" in loaded.identity["license_terms"]
    assert loaded.identity["median_quantile_index"] == MEDIAN_SLOT
    assert loaded.identity["weights_sha256_verified_against_manifest"] is True  # stand-in digest, see conftest
    # the forecaster is given the snapshot directory (so it reads the staged config.json), never the Hub
    assert loaded.model.loaded_from == str(standin_weights)
    assert loaded.model.kwargs["local_files_only"] is True and loaded.model.kwargs["device"] == "cpu"
    config = ForecastConfig(horizon=30, context_length=100)
    settings = loaded.decode_settings(config)
    assert settings["context_patch_envelope"] == 128 and settings["horizon_patch_envelope"] == 64  # patch sizes 32 / 64
    assert settings["sort_quantiles"] is True and settings["use_znorm"] is False and settings["median_quantile_index"] == 4
    import numpy as np

    point, q = loaded.predict([np.arange(50.0)], 30, config)
    assert point.shape == (1, 30) and q.shape == (1, 30, N_SLOTS)
    assert np.allclose(point, q[:, :, MEDIAN_SLOT])
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
    assert point.shape == (1, 12) and q.shape == (1, 12, N_SLOTS)
    assert np.allclose(point, q[:, :, MEDIAN_SLOT])
    assert np.all(np.diff(q, axis=-1) >= 0)  # sort_quantiles=True makes the slots monotone


def test_settings_match_the_upstream_call_signatures():
    """Hosted run 2026-10-06 failed: ``per_core_batch_size`` was passed to ``predict_batch``."""
    from timesfm_forecasting.model import DECODE_SETTINGS, LOAD_SETTINGS

    from .conftest import UPSTREAM_CONFIG_FIELDS, UPSTREAM_PREDICT_BATCH_PARAMS

    assert set(DECODE_SETTINGS) <= UPSTREAM_PREDICT_BATCH_PARAMS - {"contexts", "horizon"}
    assert set(LOAD_SETTINGS) <= UPSTREAM_CONFIG_FIELDS
