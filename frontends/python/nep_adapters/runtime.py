"""Backend dispatch, probing, and error translation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, util
from types import ModuleType
from typing import Any

from . import nep_cpu as _cpu
from .errors import (
    BackendRuntimeError,
    BackendUnavailableError,
    CancelledError,
    InvalidInputError,
    ModelLoadError,
    OutOfMemoryError,
    UnsupportedModelError,
)


BackendInfo = _cpu.BackendInfo


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int


@dataclass(frozen=True)
class BackendStatus:
    backend: str
    installed: bool
    available: bool
    reason: str
    detail: str
    driver_version: int | None = None
    runtime_version: int | None = None
    free_memory_bytes: int | None = None
    devices: tuple[DeviceInfo, ...] = ()

_gpu: ModuleType | None = None


def _load_gpu() -> ModuleType:
    global _gpu
    if _gpu is not None:
        return _gpu
    try:
        _gpu = import_module(".nep_gpu", __package__)
    except ImportError as error:
        raise BackendUnavailableError(
            "CUDA backend is unavailable because nep_gpu could not be loaded: "
            f"{error}",
            backend="cuda",
            operation="import_backend",
        ) from error
    return _gpu


def _translate_native_error(
    error: Exception,
    *,
    backend: str,
    operation: str,
) -> Exception:
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return error
    message = str(error)
    lowered = message.lower()
    kwargs = {"backend": backend, "operation": operation}
    if isinstance(error, ValueError) or lowered.startswith("invalid argument"):
        return InvalidInputError(message, **kwargs)
    if lowered.startswith("unsupported"):
        return UnsupportedModelError(message, **kwargs)
    if lowered.startswith("backend unavailable"):
        return BackendUnavailableError(message, **kwargs)
    if lowered.startswith("cancelled"):
        return CancelledError(message, **kwargs)
    if "out of memory" in lowered or "memory allocation" in lowered:
        return OutOfMemoryError(message, **kwargs)
    if operation == "load_model":
        return ModelLoadError(message, **kwargs)
    return BackendRuntimeError(message, **kwargs)


class Model:
    """Stable wrapper around a backend-specific native model."""

    def __init__(self, native: Any, backend: str) -> None:
        self._native = native
        self.backend = backend

    def _call(self, operation: str, *args, **kwargs):
        try:
            return getattr(self._native, operation)(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - translated at this seam.
            translated = _translate_native_error(
                error,
                backend=self.backend,
                operation=operation,
            )
            raise translated from error

    def calculate(self, *args, **kwargs):
        return self._call("calculate", *args, **kwargs)

    def calculate_with_descriptors(self, *args, **kwargs):
        return self._call("calculate_with_descriptors", *args, **kwargs)

    def calculate_charge(self, *args, **kwargs):
        return self._call("calculate_charge", *args, **kwargs)

    def calculate_spin(self, *args, **kwargs):
        return self._call("calculate_spin", *args, **kwargs)

    def find_force(self, *args, **kwargs):
        return self._call("find_force", *args, **kwargs)

    def descriptors(self, *args, **kwargs):
        return self._call("descriptors", *args, **kwargs)

    def descriptors_spin(self, *args, **kwargs):
        return self._call("descriptors_spin", *args, **kwargs)

    def dipoles(self, *args, **kwargs):
        return self._call("dipoles", *args, **kwargs)

    def polarizabilities(self, *args, **kwargs):
        return self._call("polarizabilities", *args, **kwargs)

    def calculate_dftd3(self, *args, **kwargs):
        return self._call("calculate_dftd3", *args, **kwargs)

    def calculate_with_dftd3(self, *args, **kwargs):
        return self._call("calculate_with_dftd3", *args, **kwargs)

    def workspace_estimate(self, atom_capacity: int, structure_capacity: int = 1):
        return self._call("workspace_estimate", atom_capacity, structure_capacity)

    def model_info(self):
        return self._call("model_info")

    def cancel(self) -> None:
        self._call("cancel")

    def reset_cancel(self) -> None:
        self._call("reset_cancel")

    def close(self) -> None:
        self._call("close")

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def register_cpu() -> None:
    try:
        _cpu.register_cpu()
    except Exception as error:  # noqa: BLE001 - translated at this seam.
        raise _translate_native_error(
            error, backend="cpu", operation="register_backend"
        ) from error


def register_cuda() -> None:
    try:
        _load_gpu().register_cuda()
    except BackendUnavailableError:
        raise
    except Exception as error:  # noqa: BLE001 - translated at this seam.
        raise _translate_native_error(
            error, backend="cuda", operation="register_backend"
        ) from error


def backend_status(backend_name: str) -> BackendStatus:
    if backend_name == "cpu":
        return BackendStatus(
            backend="cpu",
            installed=True,
            available=True,
            reason="available",
            detail="CPU backend is available.",
        )
    if backend_name != "cuda":
        raise InvalidInputError(
            f"unknown backend: {backend_name}",
            backend=backend_name,
            operation="backend_status",
        )

    module_name = f"{__package__}.nep_gpu"
    if util.find_spec(module_name) is None:
        return BackendStatus(
            backend="cuda",
            installed=False,
            available=False,
            reason="module_missing",
            detail="This nep-adapters wheel does not contain the CUDA backend.",
        )
    try:
        info = _load_gpu().cuda_runtime_info()
    except Exception as error:  # noqa: BLE001 - status must remain inspectable.
        return BackendStatus(
            backend="cuda",
            installed=True,
            available=False,
            reason="runtime_probe_failed",
            detail=str(error),
        )

    devices = tuple(
        DeviceInfo(
            index=int(item["index"]),
            name=str(item["name"]),
            compute_capability=(int(item["major"]), int(item["minor"])),
            total_memory_bytes=int(item["total_memory_bytes"]),
        )
        for item in info.get("devices", ())
    )
    available = bool(info.get("available", False)) and bool(devices)
    return BackendStatus(
        backend="cuda",
        installed=True,
        available=available,
        reason=str(info.get("reason", "available" if available else "no_device")),
        detail=str(info.get("detail", "CUDA backend is available." if available else "No CUDA device is available.")),
        driver_version=int(info["driver_version"]) if info.get("driver_version") is not None else None,
        runtime_version=int(info["runtime_version"]) if info.get("runtime_version") is not None else None,
        free_memory_bytes=int(info["free_memory_bytes"]) if info.get("free_memory_bytes") is not None else None,
        devices=devices,
    )


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
    raise InvalidInputError(
        "backend index out of range", operation="backend_info"
    )


def load_model(backend_name: str, model_path: str):
    try:
        if backend_name == "cpu":
            native = _cpu.load_model(backend_name, model_path)
        elif backend_name == "cuda":
            native = _load_gpu().load_model(backend_name, model_path)
        else:
            raise InvalidInputError(
                f"unknown backend: {backend_name}",
                backend=backend_name,
                operation="load_model",
            )
    except Exception as error:  # noqa: BLE001 - translated at this seam.
        if isinstance(error, (InvalidInputError, BackendUnavailableError)):
            raise
        translated = _translate_native_error(
            error,
            backend=backend_name,
            operation="load_model",
        )
        raise translated from error
    return Model(native, backend_name)
