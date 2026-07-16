"""Backend dispatch without importing the optional CUDA extension eagerly."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from . import nep_cpu as _cpu


BackendInfo = _cpu.BackendInfo
Model = _cpu.Model

_gpu: ModuleType | None = None


def _load_gpu() -> ModuleType:
    global _gpu
    if _gpu is not None:
        return _gpu
    try:
        _gpu = import_module(".nep_gpu", __package__)
    except ImportError as error:
        raise RuntimeError(
            "CUDA backend is unavailable because nep_gpu could not be loaded: "
            f"{error}"
        ) from error
    return _gpu


def register_cpu() -> None:
    _cpu.register_cpu()


def register_cuda() -> None:
    _load_gpu().register_cuda()


def backend_count() -> int:
    count = int(_cpu.backend_count())
    if _gpu is not None:
        count += int(_gpu.backend_count())
    return count


def backend_info(index: int):
    cpu_count = int(_cpu.backend_count())
    if 0 <= index < cpu_count:
        return _cpu.backend_info(index)
    if _gpu is not None:
        gpu_index = index - cpu_count
        if 0 <= gpu_index < int(_gpu.backend_count()):
            return _gpu.backend_info(gpu_index)
    raise ValueError("backend index out of range")


def load_model(backend_name: str, model_path: str):
    if backend_name == "cpu":
        return _cpu.load_model(backend_name, model_path)
    if backend_name == "cuda":
        return _load_gpu().load_model(backend_name, model_path)
    raise ValueError(f"unknown backend: {backend_name}")
