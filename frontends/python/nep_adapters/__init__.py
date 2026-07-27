"""NumPy-first Python frontend for NEPAdapters."""

from importlib.metadata import PackageNotFoundError, version

from .runtime import (
    BackendInfo,
    BackendStatus,
    DeviceInfo,
    Model,
    backend_count,
    backend_info,
    backend_status,
    load_model,
    register_cpu,
    register_cuda,
)
from .calculator import (
    ChargePrediction,
    ModelInfo,
    NEPCalculator,
    Prediction,
    SpinPrediction,
    WorkspaceEstimate,
    inspect_model,
)
from .errors import (
    BackendRuntimeError,
    BackendUnavailableError,
    CancelledError,
    InvalidInputError,
    ModelLoadError,
    NepAdaptersError,
    OutOfMemoryError,
    UnsupportedModelError,
)

try:
    __version__ = version("nep-adapters")
except PackageNotFoundError:  # Source-tree imports used by CMake tests.
    from ._version import __version__

__all__ = [
    "BackendInfo",
    "BackendRuntimeError",
    "BackendStatus",
    "BackendUnavailableError",
    "CancelledError",
    "ChargePrediction",
    "DeviceInfo",
    "InvalidInputError",
    "Model",
    "ModelInfo",
    "ModelLoadError",
    "NEPCalculator",
    "NepAdaptersError",
    "OutOfMemoryError",
    "Prediction",
    "SpinPrediction",
    "UnsupportedModelError",
    "WorkspaceEstimate",
    "__version__",
    "backend_count",
    "backend_info",
    "backend_status",
    "inspect_model",
    "load_model",
    "register_cpu",
    "register_cuda",
]
