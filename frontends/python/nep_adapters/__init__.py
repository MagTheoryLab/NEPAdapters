"""NumPy-first Python frontend for NEPAdapters."""

from .runtime import (
    BackendInfo,
    Model,
    backend_count,
    backend_info,
    load_model,
    register_cpu,
    register_cuda,
)
from .calculator import NEPCalculator, Prediction, SpinPrediction

__all__ = [
    "BackendInfo",
    "Model",
    "NEPCalculator",
    "Prediction",
    "SpinPrediction",
    "backend_count",
    "backend_info",
    "load_model",
    "register_cpu",
    "register_cuda",
]
