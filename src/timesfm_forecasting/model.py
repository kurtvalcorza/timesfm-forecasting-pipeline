"""Pinned TimesFM 2.5 checkpoint: identity, staging, integrity and loading.

Only ``google/timesfm-2.5-200m-pytorch`` is accepted. Its files are listed in the committed
manifest (``weights/timesfm-2.5-200m-pytorch/dimer-base-manifest.json``) with byte sizes and,
where established, SHA-256 digests. ``stage_missing_files`` fetches absent entries from the
Hugging Face Hub, ``verify_snapshot`` re-checks every file against the manifest, and
``load_pinned_model`` loads the verified safetensors file through the pinned ``timesfm``
package. Pickle-format weights are refused.

Revision status
---------------
The Hub was not reachable from the build environment, so the immutable commit of the checkpoint
and the SHA-256 of ``model.safetensors`` could not be resolved. ``MODEL_REVISION`` therefore
names the ref ``main`` and ``MODEL_REVISION_STATUS`` says so; the first hosted run records the
commit the Hub served (``resolved_revision``) and the observed weight digest, and the maintainer
pins both in this module and the manifest. The file *names* and *byte sizes* and the digest of
``config.json`` were verified against the Hub's file listing (see the manifest's ``verification``
block). ``torch`` and ``timesfm`` are imported lazily so that the rest of the package works
without them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import MAX_HORIZON, TRAINED_QUANTILES, ForecastConfig
from .errors import HubUnavailableError, ModelIntegrityError, ModelSourceError

__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "MODEL_REVISION_STATUS",
    "MODEL_LICENSE",
    "MODEL_LICENSE_SOURCE",
    "MODEL_KEY",
    "MODEL_URL",
    "MANIFEST_NAME",
    "DEFAULT_WEIGHTS_DIR",
    "WEIGHTS_FILENAME",
    "CONFIG_FILENAME",
    "PINNED_WEIGHTS_BYTES",
    "PINNED_CONFIG_SHA256",
    "REFUSED_WEIGHT_SUFFIXES",
    "LoadedModel",
    "check_model_source",
    "read_manifest",
    "stage_missing_files",
    "verify_snapshot",
    "load_pinned_model",
    "sha256_file",
]

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
MODEL_REVISION = "main"
MODEL_REVISION_STATUS = (
    "unresolved: revision digest to be confirmed on the first hosted run (the Hub was not reachable "
    "when this pin was written; the run records the commit served and the model.safetensors SHA-256)"
)
MODEL_LICENSE = "apache-2.0"
MODEL_LICENSE_SOURCE = (
    "Hugging Face model card front matter `license: apache-2.0` of google/timesfm-2.5-200m-pytorch; "
    "google-research/timesfm README: weights up to version 2.5 are Apache-2.0 (3.0 weights are not)"
)
MODEL_KEY = "timesfm-2.5-200m-pytorch"
MODEL_URL = f"https://huggingface.co/{MODEL_ID}"
MANIFEST_NAME = "dimer-base-manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
WEIGHTS_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"
#: Byte size of model.safetensors as listed by the Hub (verified); its SHA-256 is pending.
PINNED_WEIGHTS_BYTES = 925_181_104
#: SHA-256 of the 475-byte config.json, computed from the Hub-served text (verified).
PINNED_CONFIG_SHA256 = "cd3315b760d5cc7e278d7afdf41b897031ced888fc6c115bd9b3ac0ea2c47408"
REFUSED_WEIGHT_SUFFIXES = frozenset({".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"})
#: Architecture facts the loader asserts after reading config.json (MOD1/MOD3).
EXPECTED_CONFIG = {
    "model_type": "timesfm",
    "context_length": 16384,
    "patch_length": 32,
    "horizon_length": 128,
    "quantile_horizon_length": 1024,
    "quantiles": list(TRAINED_QUANTILES),
}

_PATCH = 32
_OUTPUT_PATCH = 128


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
    actually served on disk for the maintainer to pin.
    """
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = read_manifest(root)
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them from "
            f"{MODEL_URL} at {MODEL_REVISION!r}"
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
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return missing


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
        for key, value in EXPECTED_CONFIG.items():
            if config.get(key) != value:
                raise ModelIntegrityError(
                    f"{CONFIG_FILENAME} {key}={config.get(key)!r} differs from the pinned architecture "
                    f"({value!r}); this is not the TimesFM 2.5 200M checkpoint"
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
        "files": files,
        "digests_pending": pending,
        "totalBytes": sum(int(f["bytes"]) for f in files),
    }


@dataclass
class LoadedModel:
    """The pinned TimesFM model, compiled for one (context, horizon) envelope at a time."""

    model: Any
    identity: dict[str, Any]
    device: str
    dtype: str
    source: str
    snapshot: dict[str, Any]
    compiled_for: tuple[int, int] | None = None
    _config_cache: dict[tuple[int, int], Any] = field(default_factory=dict, repr=False)

    @property
    def trained_quantiles(self) -> tuple[float, ...]:
        return TRAINED_QUANTILES

    def compile_for(self, config: ForecastConfig) -> dict[str, Any]:
        """Compile the decode function for ``config`` (rounded to the model's patch sizes)."""
        import timesfm

        max_context = -(-config.context_length // _PATCH) * _PATCH
        max_horizon = -(-config.horizon // _OUTPUT_PATCH) * _OUTPUT_PATCH
        key = (max_context, max_horizon)
        if self.compiled_for != key:
            self.model.compile(
                timesfm.ForecastConfig(
                    max_context=max_context,
                    max_horizon=max_horizon,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                    per_core_batch_size=32,
                )
            )
            self.compiled_for = key
        return {
            "max_context": max_context,
            "max_horizon": max_horizon,
            "normalize_inputs": True,
            "use_continuous_quantile_head": True,
            "force_flip_invariance": True,
            "infer_is_positive": True,
            "fix_quantile_crossing": True,
        }

    def predict(self, inputs: list[np.ndarray], horizon: int, config: ForecastConfig) -> tuple[np.ndarray, np.ndarray]:
        """Zero-shot forecast: ``(point, quantiles)`` of shapes ``(n, horizon)`` and ``(n, horizon, 10)``.

        ``point`` is the model's median (``quantiles[..., 5]``); ``quantiles[..., 0]`` is the mean
        head and ``quantiles[..., 1:]`` are the deciles 0.1..0.9.
        """
        if horizon > MAX_HORIZON:
            raise ValueError(f"horizon {horizon} exceeds MAX_HORIZON={MAX_HORIZON}")
        self.compile_for(config)
        arrays = [np.asarray(x, dtype=np.float32) for x in inputs]
        point, quantiles = self.model.forecast(horizon=horizon, inputs=arrays)
        point = np.asarray(point, dtype=float)
        quantiles = np.asarray(quantiles, dtype=float)
        if point.shape != (len(arrays), horizon) or quantiles.shape != (len(arrays), horizon, 10):
            raise ModelIntegrityError(
                f"upstream returned shapes {point.shape} / {quantiles.shape}; expected "
                f"({len(arrays)}, {horizon}) / ({len(arrays)}, {horizon}, 10)"
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
    torch_compile: bool = False,
) -> LoadedModel:
    """Verify the staged snapshot, then load it through the pinned ``timesfm`` package.

    The loader never consults the Hub for the identity: the committed manifest is the anchor.
    ``torch_compile`` is off by default because it adds a long first-call compile on CPU
    runtimes and changes nothing in the forecast values the tutorial reports.
    """
    root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
    snapshot = verify_snapshot(root)
    import timesfm
    import torch

    resolved_device = resolve_device(device)
    model = timesfm.TimesFM_2p5_200M_torch(torch_compile=torch_compile)
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    model.load_checkpoint(str(root / WEIGHTS_FILENAME))
    if not resolved_device.startswith("cuda"):
        model.model.to("cpu")
        model.model.device = torch.device("cpu")
    param = next(model.model.parameters())
    weight_entry = next(f for f in snapshot["files"] if f["path"] == WEIGHTS_FILENAME)
    identity = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "revision_status": MODEL_REVISION_STATUS,
        "resolved_revision": snapshot.get("resolved_revision"),
        "license": MODEL_LICENSE,
        "license_source": MODEL_LICENSE_SOURCE,
        "source_url": MODEL_URL,
        "weights_file": WEIGHTS_FILENAME,
        "weights_bytes": weight_entry["bytes"],
        "weights_sha256": weight_entry["sha256"],
        "weights_sha256_verified_against_manifest": weight_entry["verified"],
        "config_sha256": PINNED_CONFIG_SHA256,
        "trained_quantiles": list(TRAINED_QUANTILES),
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
