"""Exception types with stable machine-readable codes.

Every refusal raised by the pipeline names the *rule* that refused the input and, for data
errors, the *source* (file path or table name) it was refusing, so a learner or an executor can
read ``[RULE] source: reason`` and act on it without a traceback.
"""

from __future__ import annotations

from typing import Any


class TimesFMPipelineError(Exception):
    """Base class for every error this package raises deliberately."""


class ValidationError(TimesFMPipelineError, ValueError):
    """An input was refused by a named validation rule.

    Parameters
    ----------
    code
        The rule identifier, e.g. ``MIN_HISTORY``.
    source
        The file path or table name the rule was applied to.
    message
        A plain-language reason, naming the offending series/row/column where known.
    details
        Machine-readable context (counts, names, limits) for executors and tests.
    """

    def __init__(
        self, code: str, source: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        self.code = code
        self.source = source
        self.message = message
        self.details = dict(details or {})
        super().__init__(f"[{code}] {source}: {message}")


class ModelSourceError(TimesFMPipelineError, ValueError):
    """A model identifier or revision other than the pinned one was requested."""


class ModelIntegrityError(TimesFMPipelineError, RuntimeError):
    """A staged snapshot does not match the committed manifest."""


class HubUnavailableError(TimesFMPipelineError, RuntimeError):
    """The Hugging Face Hub could not be reached to stage a missing file."""
