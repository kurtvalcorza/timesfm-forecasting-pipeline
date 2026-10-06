"""Shared fixtures. Nothing here needs torch, timesfm3 or the network.

The stand-in stubs (``install_standin_modules``, ``stage_standin_weights``) let the notebook's own
cells execute end to end in CI. Every result they produce is **stand-in evidence**: it proves the
cells, the contract and the exports work, and says nothing about pretrained-model inference.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "sample-data" / "openmeteo_ph_hourly_temperature.csv"
MANIFEST = ROOT / "weights" / "timesfm-3.0-pytorch" / "dimer-base-manifest.json"
#: The Hub-served config.json text (1,273 bytes, no trailing newline); its SHA-256 is pinned in model.py.
CONFIG_JSON_TEXT = (
    '{\n  "input_patch_len": 32,\n  "input_transform": "identity",\n  "linear_detrending_threshold": 0.5,\n'
    '  "output_patch_len": 64,\n  "quantiles": [\n    0.1,\n    0.2,\n    0.3,\n    0.4,\n    0.5,\n    0.6,\n'
    '    0.7,\n    0.8,\n    0.9\n  ],\n  "residual_block_config": {\n    "activation": "relu",\n'
    '    "dropout": 0.0,\n    "hidden_dims": 1280,\n    "identity_skip": false,\n    "output_dims": 1280,\n'
    '    "prenorm": "none",\n    "use_bias": false\n  },\n  "transformer_config": {\n    "num_layers": 20,\n'
    '    "transformer": {\n      "attention_norm": "rms",\n      "causal_attention": true,\n'
    '      "debug_no_masking": false,\n      "deterministic": true,\n      "feedforward_norm": "rms",\n'
    '      "ff_activation": "relu",\n      "hidden_dims": 1280,\n      "max_variates": 32,\n'
    '      "model_dims": 1280,\n      "num_heads": 16,\n      "paired_token_skip_second": false,\n'
    '      "qk_norm": "rms",\n      "training": true,\n      "use_bias": false,\n'
    '      "use_memory_efficient_attention": true,\n      "use_rope_seq": true,\n      "use_rope_var": false,\n'
    '      "use_sdpa": true,\n      "v_norm": "none"\n    },\n    "use_remat": true\n  },\n'
    '  "use_frozen_running_stats": false,\n  "use_iterative_cpm_revin": true,\n  "use_linear_detrending": true,\n'
    '  "use_stitching": true,\n  "use_variate_attention": true,\n  "value_clip": 1e+20\n}'
)
MEDIAN_SLOT = 4
N_SLOTS = 9


def make_series(
    series_id: str = "A", n: int = 96, start: str = "2026-01-01", freq: str = "h", seed: int = 0, period: int = 24
) -> pd.DataFrame:
    """One regular series with a seasonal shape plus small noise (long format, package column names)."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range(start=start, periods=n, freq=freq)
    t = np.arange(n)
    values = 25.0 + 3.0 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.1, n)
    return pd.DataFrame({"series_id": series_id, "timestamp": stamps, "value": values})


class FakeModel:
    """Deterministic stand-in for ``LoadedModel``: seasonal-naive median with a symmetric spread.

    Stand-in evidence only. ``predict`` honours the real contract: ``(n, horizon)`` point equal to the
    median slot and ``(n, horizon, 9)`` quantiles, slot ``k`` being the decile ``0.1 * (k + 1)``.
    """

    identity = {
        "model_id": "google/timesfm-3.0-pytorch",
        "revision": "main",
        "license": "timesfm-non-commercial-license-v1.0",
    }
    device = "cpu"
    dtype = "torch.float32"
    source = "stand-in (no pretrained weights)"

    def __init__(self, period: int = 24, spread: float = 0.5) -> None:
        self.period = period
        self.spread = spread
        self.calls: list[dict[str, Any]] = []

    def decode_settings(self, config: Any) -> dict[str, Any]:
        return {"context_patch_envelope": config.context_length, "horizon_patch_envelope": config.horizon, "stand_in": True}

    def predict(self, inputs: list[np.ndarray], horizon: int, config: Any) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append({"n": len(inputs), "horizon": horizon, "lengths": [len(x) for x in inputs]})
        n = len(inputs)
        q = np.zeros((n, horizon, N_SLOTS))
        for i, x in enumerate(inputs):
            x = np.asarray(x, dtype=float)
            m = self.period if len(x) >= self.period else 1
            season = x[-m:]
            base = np.array([season[k % m] for k in range(horizon)])
            for slot in range(N_SLOTS):
                q[i, :, slot] = base + (slot - MEDIAN_SLOT) * self.spread / 4
        return q[:, :, MEDIAN_SLOT].copy(), q


@pytest.fixture
def sample_path() -> Path:
    return SAMPLE


@pytest.fixture
def sample_config():
    from timesfm_forecasting import ForecastConfig

    return ForecastConfig(
        horizon=24, context_length=312, series_id_column="series_id", value_column="target", season_length=24
    )


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


# ---------------------------------------------------------------------------
# Stand-in modules for the notebook's own cells (no torch, no timesfm3, no network)
# ---------------------------------------------------------------------------


class _FakeParam:
    device = "cpu"
    dtype = "torch.float32"

    def numel(self) -> int:
        return 7


class _FakeInner:
    def __init__(self) -> None:
        self.device = "cpu"

    def parameters(self):
        return iter([_FakeParam()])

    def to(self, device: str) -> None:
        self.device = device


class _FakeForecasterConfig:
    """What ``TimesFM3Forecaster.config`` reports after loading from a directory with the pinned config.json."""

    input_patch_length = 32
    output_patch_length = 64
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    median_quantile_index = MEDIAN_SLOT


class _ForecastOutput:
    def __init__(self, forecast: np.ndarray, quantiles: np.ndarray | None) -> None:
        self.ts_id = None
        self.forecast = forecast
        self.quantiles = quantiles


class _FakeTimesFM3Forecaster:
    """Stand-in for ``timesfm3.TimesFM3Forecaster``: same constructor and ``predict_batch`` surface."""

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.loaded_from: str | None = None
        path = kwargs.get("checkpoint_path")
        if path is None or not Path(path).is_dir() or not (Path(path) / "model.safetensors").is_file():
            raise FileNotFoundError(f"stand-in forecaster needs a snapshot directory with model.safetensors: {path}")
        self.loaded_from = str(path)
        self.model = _FakeInner()
        self.device = kwargs.get("device") or "cpu"
        self.model.to(self.device)
        self.config = _FakeForecasterConfig()

    def predict_batch(self, contexts: list[np.ndarray], horizon: int, return_quantiles: bool = False, **kwargs: Any):
        point, q = FakeModel().predict(list(contexts), horizon, None)
        for i in range(len(contexts)):
            yield _ForecastOutput(point[i], q[i] if return_quantiles else None)


def install_standin_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake ``torch`` and ``timesfm3`` modules (stand-in evidence only)."""
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0.0+standin"
    cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    torch.cuda = cuda
    torch.device = lambda name: name
    monkeypatch.setitem(sys.modules, "torch", torch)
    timesfm3 = types.ModuleType("timesfm3")
    timesfm3.__version__ = "0.0.0+standin"
    timesfm3.TimesFM3Forecaster = _FakeTimesFM3Forecaster
    monkeypatch.setitem(sys.modules, "timesfm3", timesfm3)


def stage_standin_weights(root: Path) -> Path:
    """Write the manifest, the exact config.json and a sparse model.safetensors of the pinned size."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "config.json").write_bytes(CONFIG_JSON_TEXT.encode("utf-8"))
    weights = root / "model.safetensors"
    size = next(f["bytes"] for f in manifest["files"] if f["path"] == "model.safetensors")
    with weights.open("wb") as handle:
        handle.truncate(size)  # sparse: the digest is pending in the manifest, so any bytes of this size pass
    return root


@pytest.fixture
def standin_weights(tmp_path: Path) -> Path:
    return stage_standin_weights(tmp_path / "weights" / "timesfm-3.0-pytorch")
