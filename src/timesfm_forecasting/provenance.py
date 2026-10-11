"""Runtime provenance, the exported result bundle, and the reload-parity check.

The bundle written by :func:`write_result_bundle` is what a downstream reader consumes: a
forecast CSV, an evaluation JSON and ``result.json`` carrying every identity, digest, version
and configuration needed to interpret the forecast. :func:`check_reload_parity` reloads the
bundle from disk and compares it with the in-memory forecast; a mismatch is a contract failure
and raises.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ForecastConfig
from .data import ValidationResult
from .forecasting import ForecastResult
from .model import MODEL_ID, MODEL_LICENSE, MODEL_REVISION, MODEL_REVISION_STATUS

__all__ = [
    "runtime_versions",
    "build_provenance",
    "write_result_bundle",
    "reload_result_bundle",
    "check_reload_parity",
    "sha256_bytes",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_versions() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "timesfm": _version("timesfm"),
        "torch": _version("torch"),
        "huggingface_hub": _version("huggingface-hub"),
        "safetensors": _version("safetensors"),
    }
    try:
        import torch

        out["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        out["cuda_available"] = None
    return out


def build_provenance(
    model: Any,
    config: ForecastConfig,
    validation: ValidationResult,
    result: ForecastResult,
    *,
    data_source: dict[str, Any],
    notebook_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = dict(getattr(model, "identity", {}) or {})
    identity.setdefault("model_id", MODEL_ID)
    identity.setdefault("revision", MODEL_REVISION)
    identity.setdefault("revision_status", MODEL_REVISION_STATUS)
    identity.setdefault("license", MODEL_LICENSE)
    return {
        "model": identity,
        "device": getattr(model, "device", None),
        "dtype": getattr(model, "dtype", None),
        "weights_source": getattr(model, "source", None),
        "config": config.to_dict(),
        "inference": result.provenance,
        "input_manifest": validation.manifest(),
        "data_source": data_source,
        "runtime": runtime_versions(),
        "notebook_source": notebook_source,
        "determinism": (
            "CPU float32 inference is deterministic for fixed inputs and pins; CUDA kernels may differ "
            "in the last bits"
        ),
    }


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_result_bundle(
    out_dir: str | Path,
    stem: str,
    *,
    forecast: pd.DataFrame,
    evaluation: dict[str, Any] | None,
    report: dict[str, Any] | None,
    provenance: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write ``<stem>_forecast.csv``, ``<stem>_evaluation.json``, ``<stem>_provenance.json`` and
    ``<stem>_result.json`` (which indexes the others with their digests) under ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    csv_path = out / f"{stem}_forecast.csv"
    forecast.to_csv(csv_path, index=False, lineterminator="\n", float_format="%.10g")
    paths["forecast_csv"] = str(csv_path)
    files = {"forecast_csv": csv_path}
    if evaluation is not None:
        p = out / f"{stem}_evaluation.json"
        p.write_text(json.dumps(_json_ready(evaluation), indent=2) + "\n", encoding="utf-8")
        paths["evaluation_json"] = str(p)
        files["evaluation_json"] = p
    p = out / f"{stem}_provenance.json"
    p.write_text(json.dumps(_json_ready(provenance), indent=2) + "\n", encoding="utf-8")
    paths["provenance_json"] = str(p)
    files["provenance_json"] = p
    result = {
        "format": "dimer-forecast-bundle",
        "format_version": 1,
        "stem": stem,
        "model": provenance.get("model"),
        "files": {
            k: {"path": v.name, "bytes": v.stat().st_size, "sha256": sha256_bytes(v.read_bytes())}
            for k, v in files.items()
        },
        "forecast_columns": list(forecast.columns),
        "n_rows": int(len(forecast)),
        "evaluation_report": report,
        "provenance_summary": {
            "horizon": provenance.get("config", {}).get("horizon"),
            "n_series": provenance.get("inference", {}).get("n_series"),
            "device": provenance.get("device"),
            "runtime": provenance.get("runtime"),
        },
        **(extra or {}),
    }
    rp = out / f"{stem}_result.json"
    rp.write_text(json.dumps(_json_ready(result), indent=2) + "\n", encoding="utf-8")
    paths["result_json"] = str(rp)
    return paths


def reload_result_bundle(out_dir: str | Path, stem: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the bundle back from files, checking each indexed file's digest first."""
    out = Path(out_dir)
    result = json.loads((out / f"{stem}_result.json").read_text(encoding="utf-8"))
    for entry in result["files"].values():
        path = out / entry["path"]
        data = path.read_bytes()
        if len(data) != entry["bytes"] or sha256_bytes(data) != entry["sha256"]:
            raise RuntimeError(f"bundle file {path.name} does not match its digest in result.json")
    forecast = pd.read_csv(out / result["files"]["forecast_csv"]["path"], parse_dates=["timestamp"])
    return forecast, result


def check_reload_parity(original: pd.DataFrame, reloaded: pd.DataFrame, *, tolerance: float = 1e-8) -> dict[str, Any]:
    """Compare the reloaded forecast with the in-memory one; raise on any mismatch (contract integrity)."""
    numeric = [c for c in original.columns if pd.api.types.is_numeric_dtype(original[c])]
    problems = []
    if list(original.columns) != list(reloaded.columns):
        problems.append(f"columns differ: {list(original.columns)} vs {list(reloaded.columns)}")
    elif len(original) != len(reloaded):
        problems.append(f"row counts differ: {len(original)} vs {len(reloaded)}")
    else:
        for c in original.columns:
            if c in numeric:
                delta = np.max(np.abs(original[c].to_numpy(float) - reloaded[c].to_numpy(float)))
                if delta > tolerance:
                    problems.append(f"{c}: max abs difference {delta:.3g} > {tolerance}")
            else:
                a = original[c].astype(str).to_numpy()
                b = reloaded[c].astype(str).to_numpy()
                if c == "timestamp":
                    a = pd.to_datetime(original[c]).astype(str).to_numpy()
                    b = pd.to_datetime(reloaded[c]).astype(str).to_numpy()
                if not np.array_equal(a, b):
                    problems.append(f"{c}: values differ after reload")
    parity = {
        "rows": int(len(original)),
        "columns": list(original.columns),
        "tolerance": tolerance,
        "ok": not problems,
        "problems": problems,
        "boundary": "forecast re-read from the CSV named in result.json, after a digest check",
    }
    if problems:
        raise RuntimeError("reload parity failed: " + "; ".join(problems))
    return parity
