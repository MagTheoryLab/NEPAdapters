"""Stable Python exceptions exposed by NEPAdapters."""

from __future__ import annotations


class NepAdaptersError(RuntimeError):
    """Base class for failures reported through the public Python interface."""

    code = "nep_adapters_error"

    def __init__(
        self,
        message: str,
        *,
        backend: str | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.backend = backend
        self.operation = operation


class InvalidInputError(NepAdaptersError, ValueError):
    code = "invalid_input"


class UnsupportedModelError(NepAdaptersError):
    code = "unsupported_model"


class BackendUnavailableError(NepAdaptersError):
    code = "backend_unavailable"


class ModelLoadError(NepAdaptersError):
    code = "model_load_error"


class BackendRuntimeError(NepAdaptersError):
    code = "backend_runtime_error"


class OutOfMemoryError(BackendRuntimeError):
    code = "out_of_memory"


class CancelledError(NepAdaptersError):
    code = "cancelled"
