"""Smoke-test an installed CPU or GPU-capable wheel."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
import os
from pathlib import Path

import numpy as np

import nep_adapters

from baseline_utils import read_labeled_structure, read_type_map


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    fixture = root / "tests" / "fixtures" / "cpu_baseline"
    model_path = fixture / "nep.txt"
    structure = read_labeled_structure(fixture / "train.xyz")
    type_map = read_type_map(model_path)
    types = np.asarray(
        [type_map[symbol] for symbol in structure.symbols],
        dtype=np.int32,
    )
    box = np.asarray(structure.cell, dtype=np.float64).reshape(9)

    with nep_adapters.load_model("cpu", str(model_path)) as model:
        energy, forces, virial = model.find_force(
            types,
            structure.positions,
            box,
        )

    if not np.isfinite(energy):
        raise AssertionError("installed wheel returned a non-finite energy")
    if forces.shape != (len(types), 3) or virial.shape != (9,):
        raise AssertionError("installed wheel returned invalid output shapes")

    expect_gpu = os.environ.get("NEP_ADAPTERS_EXPECT_GPU_MODULE") == "1"
    gpu_spec = import_module("importlib.util").find_spec("nep_adapters.nep_gpu")
    if expect_gpu:
        if gpu_spec is None:
            raise AssertionError("GPU-capable wheel is missing nep_gpu")
        import_module("nep_adapters.nep_gpu")
    elif gpu_spec is not None:
        raise AssertionError("CPU wheel unexpectedly contains nep_gpu")

    print(
        "installed wheel smoke:",
        f"version={version('nep-adapters')}",
        f"atoms={len(types)}",
        f"gpu_module={int(gpu_spec is not None)}",
    )


if __name__ == "__main__":
    main()
