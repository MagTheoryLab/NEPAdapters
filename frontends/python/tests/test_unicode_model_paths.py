"""Regression coverage for the UTF-8 model-path contract."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import nep_adapters
import numpy as np

from baseline_utils import read_labeled_structure, read_type_map


def _prediction(model_path: Path, types, positions, box):
    with nep_adapters.load_model("cpu", str(model_path)) as model:
        info = model.model_info()
        energy, forces, virial = model.find_force(types, positions, box)
    return info, float(energy), np.asarray(forces), np.asarray(virial)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    fixture = root / "tests" / "fixtures" / "cpu_baseline"
    source_model = fixture / "nep.txt"
    structure = read_labeled_structure(fixture / "train.xyz")
    type_map = read_type_map(source_model)
    types = np.asarray(
        [type_map[symbol] for symbol in structure.symbols], dtype=np.int32
    )
    box = np.asarray(structure.cell, dtype=np.float64).reshape(9)

    relative_paths = (
        Path("ascii") / "model.nep",
        Path("计算文件") / "冰" / "model.nep",
        Path("ascii") / "中文模型.txt",
        Path("folder with spaces") / "model with spaces.nep",
        Path("Unicode-🧊-𠮷") / "model-𝛑-🧪.nep",
    )

    with tempfile.TemporaryDirectory(prefix="nep-adapters-unicode-") as temp:
        temp_root = Path(temp)
        model_paths = []
        for relative_path in relative_paths:
            model_path = temp_root / relative_path
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_model, model_path)
            model_paths.append(model_path)

        baseline_info, baseline_energy, baseline_forces, baseline_virial = _prediction(
            model_paths[0], types, structure.positions, box
        )
        baseline_inspection = nep_adapters.inspect_model(model_paths[0])
        with nep_adapters.NEPCalculator(model_paths[0]) as calculator:
            if calculator.model_info != baseline_inspection:
                raise AssertionError("calculator and inspection metadata differ")

        for model_path in model_paths[1:]:
            info, energy, forces, virial = _prediction(
                model_path, types, structure.positions, box
            )
            inspection = nep_adapters.inspect_model(model_path)
            with nep_adapters.NEPCalculator(model_path) as calculator:
                calculator_info = calculator.model_info
            if info != baseline_info:
                raise AssertionError(f"native metadata differs for {model_path}")
            if inspection != baseline_inspection:
                raise AssertionError(f"inspected metadata differs for {model_path}")
            if calculator_info != baseline_inspection:
                raise AssertionError(f"calculator metadata differs for {model_path}")
            if energy != baseline_energy:
                raise AssertionError(f"energy differs for {model_path}")
            if not np.array_equal(forces, baseline_forces):
                raise AssertionError(f"forces differ for {model_path}")
            if not np.array_equal(virial, baseline_virial):
                raise AssertionError(f"virial differs for {model_path}")

        missing_path = temp_root / "不存在-🧊-𠮷" / "缺失模型-🧪.nep"
        try:
            nep_adapters.load_model("cpu", str(missing_path))
        except Exception as error:  # The public layer translates native errors.
            message = str(error)
            if "failed to open NEP model" not in message:
                raise AssertionError(f"unexpected missing-path error: {message}") from error
            if str(missing_path) not in message or "�" in message:
                raise AssertionError(f"Unicode path was corrupted: {message}") from error
        else:
            raise AssertionError("missing Unicode model path unexpectedly loaded")

    print(
        "unicode model paths:",
        f"variants={len(relative_paths)}",
        f"energy={baseline_energy:.12g}",
    )


if __name__ == "__main__":
    main()
