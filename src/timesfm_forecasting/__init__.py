"""Zero-shot time-series forecasting with a pinned Google TimesFM 3.0 checkpoint.

Modules (carried verbatim into the standalone tutorial notebook, in this dependency order):

- ``errors``: machine-readable refusal codes;
- ``config``: ``ForecastConfig`` and the named ceilings;
- ``data``: CSV loading, frequency inference, gap/NaN policies, length rules, input manifest;
- ``model``: pinned identity, staging, integrity verification, lazy loading through ``timesfm3``;
- ``forecasting``: point (median) and quantile forecast over a validated table;
- ``evaluation``: chronological holdout, naive / seasonal-naive / SES baselines, MASE, sMAPE,
  quantile loss, recorded verdicts;
- ``provenance``: runtime versions, result bundle export, reload-parity check.
"""

from __future__ import annotations

from .config import (
    DEFAULT_QUANTILE_LEVELS,
    MAX_CONTEXT_POINTS,
    MAX_HORIZON,
    MEDIAN_SLOT,
    MIN_CONTEXT_POINTS,
    TRAINED_QUANTILES,
    ForecastConfig,
)
from .data import (
    INPUT_SCHEMA,
    MAX_NAN_FRACTION,
    MAX_ROWS,
    MAX_SERIES,
    MIN_OBSERVATIONS,
    RULES,
    FrequencyInfo,
    SeriesReport,
    ValidationResult,
    future_timestamps,
    infer_frequency,
    load_series_csv,
    min_history_required,
    validate_series,
)
from .errors import (
    HubUnavailableError,
    ModelIntegrityError,
    ModelSourceError,
    TimesFMPipelineError,
    ValidationError,
)
from .evaluation import (
    BASELINE_NAMES,
    HoldoutSplit,
    chronological_holdout,
    evaluate_forecast,
    evaluation_report,
    mase,
    naive_baseline,
    quantile_loss,
    seasonal_naive_baseline,
    ses_baseline,
    smape,
)
from .forecasting import ForecastResult, forecast, quantile_column
from .model import (
    DEFAULT_WEIGHTS_DIR,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_KEY,
    MODEL_LICENSE,
    MODEL_LICENSE_TERMS,
    MODEL_REVISION,
    MODEL_REVISION_STATUS,
    LoadedModel,
    check_model_source,
    load_pinned_model,
    read_manifest,
    stage_missing_files,
    verify_snapshot,
)
from .provenance import (
    build_provenance,
    check_reload_parity,
    reload_result_bundle,
    runtime_versions,
    write_result_bundle,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ForecastConfig",
    "DEFAULT_QUANTILE_LEVELS",
    "TRAINED_QUANTILES",
    "MEDIAN_SLOT",
    "MAX_HORIZON",
    "MAX_CONTEXT_POINTS",
    "MIN_CONTEXT_POINTS",
    "INPUT_SCHEMA",
    "RULES",
    "MIN_OBSERVATIONS",
    "MAX_SERIES",
    "MAX_ROWS",
    "MAX_NAN_FRACTION",
    "FrequencyInfo",
    "SeriesReport",
    "ValidationResult",
    "infer_frequency",
    "future_timestamps",
    "load_series_csv",
    "validate_series",
    "min_history_required",
    "TimesFMPipelineError",
    "ValidationError",
    "ModelSourceError",
    "ModelIntegrityError",
    "HubUnavailableError",
    "HoldoutSplit",
    "chronological_holdout",
    "naive_baseline",
    "seasonal_naive_baseline",
    "ses_baseline",
    "mase",
    "smape",
    "quantile_loss",
    "evaluate_forecast",
    "evaluation_report",
    "BASELINE_NAMES",
    "ForecastResult",
    "forecast",
    "quantile_column",
    "MODEL_ID",
    "MODEL_REVISION",
    "MODEL_REVISION_STATUS",
    "MODEL_LICENSE",
    "MODEL_LICENSE_TERMS",
    "MODEL_KEY",
    "MANIFEST_NAME",
    "DEFAULT_WEIGHTS_DIR",
    "LoadedModel",
    "check_model_source",
    "read_manifest",
    "stage_missing_files",
    "verify_snapshot",
    "load_pinned_model",
    "runtime_versions",
    "build_provenance",
    "write_result_bundle",
    "reload_result_bundle",
    "check_reload_parity",
]
