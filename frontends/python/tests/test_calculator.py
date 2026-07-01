import os

import nep_adapters
import numpy as np

from baseline_utils import read_labeled_structure, read_type_map


def main():
    model_path = os.environ["NEP_ADAPTERS_PYTHON_TEST_MODEL"]
    xyz_path = os.environ["NEP_ADAPTERS_PYTHON_TEST_XYZ"]

    structure = read_labeled_structure(xyz_path)
    calculator = nep_adapters.NEPCalculator(model_path)
    prediction = calculator.predict_structures([structure])
    energy, force_blocks, virial_blocks = calculator.calculate([structure])
    empty_energy, empty_forces, empty_virials = calculator.calculate([])

    type_map = read_type_map(model_path)
    types = np.asarray([type_map[symbol] for symbol in structure.get_chemical_symbols()], dtype=np.int32)
    box = np.asarray(structure.cell, dtype=np.float64).reshape(9)
    with nep_adapters.load_model("cpu_nep3", model_path) as model:
        direct_energy, direct_forces, direct_virial = model.find_force(
            types,
            structure.positions,
            box,
        )

    if "ase" in getattr(nep_adapters, "__dict__", {}):
        raise AssertionError("core nep_adapters import should not import ASE adapter")
    if empty_energy.size != 0 or empty_forces or empty_virials:
        raise AssertionError("empty calculator batch should return empty results")
    if prediction.energy.shape != (1,):
        raise AssertionError("energy shape mismatch")
    if prediction.forces.shape != structure.positions.shape:
        raise AssertionError("force shape mismatch")
    if prediction.virials.shape != (len(structure), 9):
        raise AssertionError("virial shape mismatch")
    if not np.allclose(prediction.energy, energy):
        raise AssertionError("calculate() energy differs from Prediction")
    if not np.allclose(prediction.forces, force_blocks[0]):
        raise AssertionError("calculate() forces differ from Prediction")
    if not np.allclose(prediction.structure_virials[0], virial_blocks[0]):
        raise AssertionError("calculate() virial differs from Prediction")
    if abs(float(prediction.energy[0]) - float(direct_energy)) > 1.0e-10:
        raise AssertionError("calculator energy differs from direct native find_force")
    if not np.allclose(prediction.forces, direct_forces, rtol=0.0, atol=1.0e-10):
        raise AssertionError("calculator forces differ from direct native find_force")
    if not np.allclose(prediction.virials.sum(axis=0), direct_virial, rtol=0.0, atol=1.0e-10):
        raise AssertionError("calculator virial differs from direct native find_force")
    if abs(float(prediction.energy[0]) - structure.energy) > 1.0e-10:
        raise AssertionError("calculator energy differs from fixed baseline label")
    if not np.allclose(prediction.forces, structure.forces, rtol=0.0, atol=1.0e-10):
        raise AssertionError("calculator forces differ from fixed baseline labels")
    if not np.allclose(prediction.virials.sum(axis=0), structure.virial, rtol=0.0, atol=1.0e-10):
        raise AssertionError("calculator virial differs from fixed baseline label")
    print(
        "python calculator smoke:",
        f"atoms={len(structure)}",
        f"energy={float(prediction.energy[0]):.12g}",
    )


if __name__ == "__main__":
    main()
