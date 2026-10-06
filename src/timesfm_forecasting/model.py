"""Pinned TimesFM 3.0 checkpoint: identity, staging, integrity and loading.

Only ``google/timesfm-3.0-pytorch`` is accepted. Its files are listed in the committed manifest
(``weights/timesfm-3.0-pytorch/dimer-base-manifest.json``) with byte sizes and, where established,
SHA-256 digests. ``stage_missing_files`` fetches absent entries from the Hugging Face Hub,
``verify_snapshot`` re-checks every file against the manifest, and ``load_pinned_model`` loads the
verified snapshot directory through the ``timesfm3`` package of the pinned ``timesfm`` distribution
(``TimesFM3Forecaster``, which reads the staged ``config.json`` and ``model.safetensors`` and never
consults the Hub). Pickle-format weights are refused.

Weights licence
---------------
The TimesFM 3.0 weights are distributed under the *TimesFM Non-Commercial License v1.0*
(``license: other`` on the Hub, ``LICENSE`` in the checkpoint repository): non-commercial and
non-production use only, no redistribution of the model or of derivatives. The pipeline code is
Apache-2.0; the weights are downloaded at run time and never committed. ``MODEL_LICENSE_TERMS``
carries the restriction into every provenance record.

Revision status
---------------
The Hub's commit API was not reachable from the build environment, so the immutable commit of the
checkpoint and the SHA-256 of ``model.safetensors`` could not be resolved. ``MODEL_REVISION``
therefore names the ref ``main`` and ``MODEL_REVISION_STATUS`` says so; the first hosted run records
the commit the Hub served (``resolved_revision``) and the observed weight digest, and the maintainer
pins both in this module and the manifest. The file *names* and *byte sizes* and the digest of
``config.json`` were verified against the Hub's file listing and file contents (see the manifest's
``verification`` block). ``torch`` and ``timesfm3`` are imported lazily so that the rest of the
package works without them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import MAX_CONTEXT_POINTS, MAX_HORIZON, MEDIAN_SLOT, TRAINED_QUANTILES, ForecastConfig
from .errors import HubUnavailableError, ModelIntegrityError, ModelSourceError

__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "MODEL_REVISION_STATUS",
    "MODEL_LICENSE",
    "MODEL_LICENSE_SOURCE",
    "MODEL_LICENSE_TERMS",
    "MODEL_LICENSE_URL",
    "MODEL_KEY",
    "MODEL_URL",
    "MANIFEST_NAME",
    "DEFAULT_WEIGHTS_DIR",
    "WEIGHTS_FILENAME",
    "CONFIG_FILENAME",
    "PINNED_WEIGHTS_BYTES",
    "PINNED_CONFIG_SHA256",
    "REFUSED_WEIGHT_SUFFIXES",
    "EXPECTED_CONFIG",
    "EXPECTED_TRANSFORMER",
    "DECODE_SETTINGS",
    "LoadedModel",
    "check_model_source",
    "read_manifest",
    "stage_missing_files",
    "verify_snapshot",
    "load_pinned_model",
    "sha256_file",
]

MODEL_ID = "google/timesfm-3.0-pytorch"
MODEL_REVISION = "main"
MODEL_REVISION_STATUS = (
    "unresolved: revision digest to be confirmed on the first hosted run (the Hub's commit API was not "
    "reachable when this pin was written; the run records the commit served and the model.safetensors SHA-256)"
)
#: The Hub front matter says ``license: other`` with ``license_name: timesfm-non-commercial-license-v1.0``.
MODEL_LICENSE = "timesfm-non-commercial-license-v1.0"
MODEL_LICENSE_SOURCE = (
    "Hugging Face model card front matter of google/timesfm-3.0-pytorch: `license: other`, "
    "`license_name: timesfm-non-commercial-license-v1.0`, `license_link: LICENSE` (7,270-byte LICENSE file in "
    "the checkpoint repository, read in full on 2026-10-06)"
)
MODEL_LICENSE_TERMS = (
    "non-commercial and non-production use only (testing, evaluation and research not tied to commercial gain, "
    "production deployment or revenue generation); no use in end-user-facing or production systems; no "
    "distribution of the model or of derivatives; a commercial licence from Google LLC is required for any "
    "other use"
)
MODEL_LICENSE_URL = "https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE"
MODEL_KEY = "timesfm-3.0-pytorch"
MODEL_URL = f"https://huggingface.co/{MODEL_ID}"
MANIFEST_NAME = "dimer-base-manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
WEIGHTS_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"
#: Byte size of model.safetensors as listed by the Hub (verified); its SHA-256 is pending.
PINNED_WEIGHTS_BYTES = 1_322_898_824
#: SHA-256 of the 1,273-byte config.json, computed from the Hub-served text (verified).
PINNED_CONFIG_SHA256 = "ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6"
REFUSED_WEIGHT_SUFFIXES = frozenset({".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"})
#: Architecture facts the loader asserts after reading config.json (top-level keys).
EXPECTED_CONFIG = {
    "input_patch_len": 32,
    "output_patch_len": 64,
    "quantiles": list(TRAINED_QUANTILES),
    "use_variate_attention": True,
    "use_iterative_cpm_revin": True,
    "use_linear_detrending": True,
    "use_stitching": True,
}
#: Architecture facts asserted under ``transformer_config`` (``num_layers``) and ``transformer_config.transformer``.
EXPECTED_TRANSFORMER = {"num_layers": 20, "model_dims": 1280, "num_heads": 16, "use_rope_var": False}
#: Upstream ``TimesFM3Forecaster.predict_batch`` options the pipeline passes on every call. ``sort_quantiles``
#: makes the nine slots monotone (the upstream default); the three transforms are off so the output is the
#: model's own quantile head, and the batch size is the upstream default.
DECODE_SETTINGS = {
    "per_core_batch_size": 4,
    "return_quantiles": True,
    "sort_quantiles": True,
    "use_symmetric_averaging": False,
    "make_positive": False,
    "use_znorm": False,
    "padding_mode": "none",
}

_INPUT_PATCH = 32
_OUTPUT_PATCH = 64
_N_SLOTS = len(TRAINED_QUANTILES)


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model_source(model_id: Any, revision: Any) -> None:
    """Refuse any identity other than the pinned one (local paths, URIs, other repos, other refs)."""
    if model_id != MODEL_ID:
        raise ModelSourceError(
            f"Only the pinned checkpoint {MODEL_ID!r} is accepted; got {model_id!r}. Local paths, "
            "URIs and other repositories are refused."
        )
    if revision != MODEL_REVISION:
        raise ModelSourceError(
            f"Only the pinned revision {MODEL_REVISION!r} ({MODEL_REVISION_STATUS.split(':')[0]}) is "
            f"accepted; got {revision!r}."
        )


def read_manifest(root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    root_path = Path(root) if root is not None else DEFAULT_WEIGHTS_DIR
    path = root_path / MANIFEST_NAME
    if not path.is_file():
        raise ModelIntegrityError(f"manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ModelIntegrityError(
            f"manifest at {path} names {manifest.get('modelId')!r}@{manifest.get('revision')!r}, "
            f"not the pinned {MODEL_ID!r}@{MODEL_REVISION!r}"
        )
    return manifest


def _hub_download(relative_path: str, root: Path) -> dict[str, Any]:
    """Fetch one manifest entry from the Hub into ``root``; returns what the Hub said it served."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise HubUnavailableError("huggingface_hub is not installed") from exc
    try:
        local = hf_hub_download(
            repo_id=MODEL_ID, filename=relative_path, revision=MODEL_REVISION, local_dir=str(root)
        )
    except Exception as exc:  # noqa: BLE001 - every hub failure is reported the same way
        raise HubUnavailableError(f"could not fetch {relative_path} from {MODEL_URL}: {exc}") from exc
    resolved: str | None = None
    try:
        from huggingface_hub import HfApi

        resolved = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION).sha
    except Exception:  # noqa: BLE001 - best effort provenance only
        resolved = None
    return {"path": relative_path, "local": str(local), "resolved_revision": resolved}


def stage_missing_files(
    path: str | os.PathLike[str] | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], Any] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent from the snapshot directory.

    Returns the relative paths fetched. Also writes ``resolved-revision.json`` next to the
    manifest with whatever the downloader reported, so a hosted run leaves the commit it
    actually served on disk for the maintainer to pin. Downloading the weights accepts the
    non-commercial licence recorded in ``MODEL_LICENSE_TERMS``; the record repeats it.
    """
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = read_manifest(root)
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them from "
            f"{MODEL_URL} at {MODEL_REVISION!r} (weights licence: {MODEL_LICENSE}, {MODEL_LICENSE_TERMS})"
        )
    fetch = downloader or _hub_download
    records = []
    for relative_path in missing:
        records.append(fetch(relative_path, root))
    resolved = [r.get("resolved_revision") for r in records if isinstance(r, dict)]
    (root / "resolved-revision.json").write_text(
        json.dumps(
            {
                "modelId": MODEL_ID,
                "requested_revision": MODEL_REVISION,
                "revision_status": MODEL_REVISION_STATUS,
                "resolved_revision": next((r for r in resolved if r), None),
                "fetched": missing,
                "license": MODEL_LICENSE,
                "license_terms": MODEL_LICENSE_TERMS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return missing


def _config_mismatches(config: dict[str, Any]) -> list[str]:
    """Pinned architecture values that the staged ``config.json`` does not carry."""
    out = []
    for key, value in EXPECTED_CONFIG.items():
        if config.get(key) != value:
            out.append(f"{key}={config.get(key)!r} (pinned {value!r})")
    transformer = config.get("transformer_config") or {}
    inner = transformer.get("transformer") or {} if isinstance(transformer, dict) else {}
    for key, value in EXPECTED_TRANSFORMER.items():
        got = transformer.get(key) if key == "num_layers" else inner.get(key)
        if got != value:
            out.append(f"transformer_config.{key}={got!r} (pinned {value!r})")
    return out


def verify_snapshot(snapshot_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Check every manifest entry by size and, where the manifest carries one, by SHA-256.

    Entries whose manifest digest is not a 64-hex value (the ``"pending: ..."`` sentinel) are hashed
    and reported with ``verified: false``; the pipeline never claims a digest it did not pin.
    Pickle-format weight files anywhere in the directory are refused.
    """
    root = Path(snapshot_path)
    if not root.is_dir():
        raise ModelIntegrityError(f"snapshot path is not a directory: {root}")
    manifest = read_manifest(root)
    refused = sorted(p.name for p in root.rglob("*") if p.is_file() and p.suffix.lower() in REFUSED_WEIGHT_SUFFIXES)
    if refused:
        raise ModelIntegrityError(
            f"refusing to load: pickle-format weight files {refused} are present in {root}; only "
            f"{WEIGHTS_FILENAME} is acceptable"
        )
    files = []
    pending = []
    for entry in manifest["files"]:
        target = root / entry["path"]
        if not target.is_file():
            raise ModelIntegrityError(f"missing {entry['path']} in snapshot {root}")
        actual_bytes = target.stat().st_size
        if actual_bytes != int(entry["bytes"]):
            raise ModelIntegrityError(
                f"{entry['path']} is {actual_bytes} bytes, expected {entry['bytes']} (manifest)"
            )
        observed = sha256_file(target)
        expected = entry.get("sha256")
        if not _is_digest(expected):
            # The manifest marks a digest it does not know with a non-hex sentinel ("pending: ..."), never a
            # made-up value; the observed digest is reported with verified=False so the maintainer can pin it.
            pending.append(entry["path"])
            files.append({"path": entry["path"], "bytes": actual_bytes, "sha256": observed, "verified": False})
            continue
        if observed != expected:
            raise ModelIntegrityError(
                f"{entry['path']} SHA-256 {observed} does not match the manifest digest {expected}"
            )
        files.append({"path": entry["path"], "bytes": actual_bytes, "sha256": observed, "verified": True})
    if (root / CONFIG_FILENAME).is_file():
        config = json.loads((root / CONFIG_FILENAME).read_text(encoding="utf-8"))
        mismatches = _config_mismatches(config)
        if mismatches:
            raise ModelIntegrityError(
                f"{CONFIG_FILENAME} differs from the pinned architecture: {'; '.join(mismatches)}; this is not "
                "the TimesFM 3.0 PyTorch checkpoint"
            )
    resolved_file = root / "resolved-revision.json"
    resolved = None
    if resolved_file.is_file():
        resolved = json.loads(resolved_file.read_text(encoding="utf-8")).get("resolved_revision")
    return {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "revision_status": MODEL_REVISION_STATUS,
        "resolved_revision": resolved,
        "license": MODEL_LICENSE,
        "files": files,
        "digests_pending": pending,
        "totalBytes": sum(int(f["bytes"]) for f in files),
    }


@dataclass
class LoadedModel:
    """The pinned TimesFM 3.0 forecaster: one upstream ``TimesFM3Forecaster`` and its identity record."""

    model: Any
    identity: dict[str, Any]
    device: str
    dtype: str
    source: str
    snapshot: dict[str, Any]

    @property
    def trained_quantiles(self) -> tuple[float, ...]:
        return TRAINED_QUANTILES

    def decode_settings(self, config: ForecastConfig) -> dict[str, Any]:
        """The upstream options one ``predict`` call uses, plus the patch envelope the request rounds to.

        TimesFM 3.0 needs no compile step: the forecaster pads each batch's context up to a multiple of
        the 32-point input patch (capped at the 15,360-point context limit) and decodes the horizon in
        64-point output patches, so these values are recorded for provenance, not applied by the pipeline.
        """
        return {
            **DECODE_SETTINGS,
            "context_patch_envelope": min(-(-config.context_length // _INPUT_PATCH) * _INPUT_PATCH, MAX_CONTEXT_POINTS),
            "horizon_patch_envelope": -(-config.horizon // _OUTPUT_PATCH) * _OUTPUT_PATCH,
            "median_quantile_index": MEDIAN_SLOT,
            "input_patch_len": _INPUT_PATCH,
            "output_patch_len": _OUTPUT_PATCH,
            "context_limit": MAX_CONTEXT_POINTS,
        }

    def predict(self, inputs: list[np.ndarray], horizon: int, config: ForecastConfig) -> tuple[np.ndarray, np.ndarray]:
        """Zero-shot forecast: ``(point, quantiles)`` of shapes ``(n, horizon)`` and ``(n, horizon, 9)``.

        ``quantiles[..., k]`` is the model's quantile ``TRAINED_QUANTILES[k]`` (0.1 .. 0.9) and ``point``
        is its median, ``quantiles[..., MEDIAN_SLOT]`` (slot 4). There is no separate mean head in
        TimesFM 3.0.
        """
        if horizon > MAX_HORIZON:
            raise ValueError(f"horizon {horizon} exceeds MAX_HORIZON={MAX_HORIZON}")
        arrays = [np.asarray(x, dtype=np.float32) for x in inputs]
        outputs = list(self.model.predict_batch(contexts=arrays, horizon=horizon, **DECODE_SETTINGS))
        if len(outputs) != len(arrays):
            raise ModelIntegrityError(f"upstream returned {len(outputs)} forecasts for {len(arrays)} series")
        point = np.asarray(np.stack([np.asarray(o.forecast, dtype=float) for o in outputs]), dtype=float)
        quantiles = np.asarray(np.stack([np.asarray(o.quantiles, dtype=float) for o in outputs]), dtype=float)
        if point.shape != (len(arrays), horizon) or quantiles.shape != (len(arrays), horizon, _N_SLOTS):
            raise ModelIntegrityError(
                f"upstream returned shapes {point.shape} / {quantiles.shape}; expected "
                f"({len(arrays)}, {horizon}) / ({len(arrays)}, {horizon}, {_N_SLOTS})"
            )
        if not (np.isfinite(point).all() and np.isfinite(quantiles).all()):
            raise ModelIntegrityError("upstream returned a non-finite forecast value")
        return point, quantiles


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pinned_model(
    weights_dir: str | os.PathLike[str] | None = None,
    *,
    device: str = "auto",
) -> LoadedModel:
    """Verify the staged snapshot, then load it through ``timesfm3.TimesFM3Forecaster``.

    The loader never consults the Hub for the identity: the committed manifest is the anchor. The
    forecaster is given the snapshot *directory*, so it builds the model from the staged
    ``config.json`` (the same path upstream's ``from_pretrained`` takes) rather than from the
    package's built-in defaults, and ``local_files_only=True`` forbids any download at load time.
    """
    root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
    snapshot = verify_snapshot(root)
    import torch
    from timesfm3 import TimesFM3Forecaster

    resolved_device = resolve_device(device)
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    model = TimesFM3Forecaster(
        checkpoint_path=str(root),
        device=resolved_device,
        local_files_only=True,
        per_core_batch_size=DECODE_SETTINGS["per_core_batch_size"],
    )
    forecaster_config = getattr(model, "config", None)
    if forecaster_config is not None:
        observed = {
            "input_patch_length": getattr(forecaster_config, "input_patch_length", None),
            "output_patch_length": getattr(forecaster_config, "output_patch_length", None),
            "quantiles": list(getattr(forecaster_config, "quantiles", []) or []),
            "median_quantile_index": getattr(forecaster_config, "median_quantile_index", None),
        }
        expected = {
            "input_patch_length": _INPUT_PATCH,
            "output_patch_length": _OUTPUT_PATCH,
            "quantiles": list(TRAINED_QUANTILES),
            "median_quantile_index": MEDIAN_SLOT,
        }
        if observed != expected:
            raise ModelIntegrityError(f"the loaded forecaster reports {observed}, not the pinned {expected}")
    param = next(model.model.parameters())
    weight_entry = next(f for f in snapshot["files"] if f["path"] == WEIGHTS_FILENAME)
    identity = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "revision_status": MODEL_REVISION_STATUS,
        "resolved_revision": snapshot.get("resolved_revision"),
        "license": MODEL_LICENSE,
        "license_terms": MODEL_LICENSE_TERMS,
        "license_url": MODEL_LICENSE_URL,
        "license_source": MODEL_LICENSE_SOURCE,
        "source_url": MODEL_URL,
        "weights_file": WEIGHTS_FILENAME,
        "weights_bytes": weight_entry["bytes"],
        "weights_sha256": weight_entry["sha256"],
        "weights_sha256_verified_against_manifest": weight_entry["verified"],
        "config_sha256": PINNED_CONFIG_SHA256,
        "trained_quantiles": list(TRAINED_QUANTILES),
        "median_quantile_index": MEDIAN_SLOT,
        "parameters": int(sum(p.numel() for p in model.model.parameters())),
    }
    return LoadedModel(
        model=model,
        identity=identity,
        device=str(param.device),
        dtype=str(param.dtype),
        source="local-snapshot (manifest-anchored)",
        snapshot=snapshot,
    )
