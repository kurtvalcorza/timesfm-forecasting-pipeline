"""Shared fixtures. Nothing here needs torch, timesfm or the network.

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
MANIFEST = ROOT / "weights" / "timesfm-2.5-200m-pytorch" / "dimer-base-manifest.json"
#: The Hub-served config.json text (475 bytes, no trailing newline); its SHA-256 is pinned in model.py.
CONFIG_JSON_TEXT = (
    '{\n  "architectures": [\n    "TimesFmModelForPrediction"\n  ],\n  "context_length": 16384,\n  "head_dim": 80,\n'
    '  "hidden_size": 1280,\n  "horizon_length": 128,\n  "intermediate_size": 1280,\n  "model_type": "timesfm",\n'
    '  "num_attention_heads": 16,\n  "num_hidden_layers": 20,\n  "patch_length": 32,\n  "quantile_horizon_length": 1024,\n'
    '  "quantiles": [\n    0.1,\n    0.2,\n    0.3,\n    0.4,\n    0.5,\n    0.6,\n    0.7,\n    0.8,\n    0.9\n  ],\n'
    '  "rms_norm_eps": 1e-06,\n  "torch_compile": false\n}'
)


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
    median slot and ``(n, horizon, 10)`` quantiles with the mean head in slot 0.
    """

    identity = {"model_id": "google/timesfm-2.5-200m-pytorch", "revision": "main", "license": "apache-2.0"}
    device = "cpu"
    dtype = "torch.float32"
    source = "stand-in (no pretrained weights)"

    def __init__(self, period: int = 24, spread: float = 0.5) -> None:
        self.period = period
        self.spread = spread
        self.calls: list[dict[str, Any]] = []

    def compile_for(self, config: Any) -> dict[str, Any]:
        return {"max_context": config.context_length, "max_horizon": config.horizon, "stand_in": True}

    def predict(self, inputs: list[np.ndarray], horizon: int, config: Any) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append({"n": len(inputs), "horizon": horizon, "lengths": [len(x) for x in inputs]})
        n = len(inputs)
        q = np.zeros((n, horizon, 10))
        for i, x in enumerate(inputs):
            x = np.asarray(x, dtype=float)
            m = self.period if len(x) >= self.period else 1
            season = x[-m:]
            base = np.array([season[k % m] for k in range(horizon)])
            q[i, :, 0] = base + 0.01  # mean head differs from the median by construction
            for slot in range(1, 10):
                q[i, :, slot] = base + (slot - 5) * self.spread / 4
        return q[:, :, 5].copy(), q


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
# Stand-in modules for the notebook's own cells (no torch, no timesfm, no network)
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


class _FakeTimesFM:
    """Stand-in for ``timesfm.TimesFM_2p5_200M_torch``: same call surface, deterministic numbers."""

    def __init__(self, torch_compile: bool = False, **kwargs: Any) -> None:
        self.model = _FakeInner()
        self.torch_compile = torch_compile
        self.forecast_config = None
        self.loaded_from: str | None = None

    def load_checkpoint(self, path: str, **kwargs: Any) -> None:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        self.loaded_from = path

    def compile(self, forecast_config: Any, **kwargs: Any) -> None:
        self.forecast_config = forecast_config

    def forecast(self, horizon: int, inputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if self.forecast_config is None:
            raise RuntimeError("Model is not compiled. Please call compile() first.")
        return FakeModel().predict(inputs, horizon, None)


def install_standin_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake ``torch`` and ``timesfm`` modules (stand-in evidence only)."""
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0.0+standin"
    cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    torch.cuda = cuda
    torch.device = lambda name: name
    monkeypatch.setitem(sys.modules, "torch", torch)
    timesfm = types.ModuleType("timesfm")
    timesfm.__version__ = "0.0.0+standin"

    class ForecastConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    timesfm.ForecastConfig = ForecastConfig
    timesfm.TimesFM_2p5_200M_torch = _FakeTimesFM
    monkeypatch.setitem(sys.modules, "timesfm", timesfm)


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
    return stage_standin_weights(tmp_path / "weights" / "timesfm-2.5-200m-pytorch")
