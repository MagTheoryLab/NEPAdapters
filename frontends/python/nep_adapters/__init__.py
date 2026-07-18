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
from .calculator import ChargePrediction, NEPCalculator, Prediction, SpinPrediction

__all__ = [
    "BackendInfo",
    "ChargePrediction",
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
