import os

import nep_adapters
import numpy as np

from baseline_utils import read_descriptor_fixture, read_labeled_structure, read_type_map


def main():
    model_path = os.environ["NEP_ADAPTERS_PYTHON_TEST_MODEL"]
    xyz_path = os.environ["NEP_ADAPTERS_PYTHON_TEST_XYZ"]

    structure = read_labeled_structure(xyz_path)
    expected_descriptors = read_descriptor_fixture(
        os.path.join(os.path.dirname(xyz_path), "descriptor.txt")
    )
    calculator = nep_adapters.NEPCalculator(model_path)
    prediction = calculator.predict_structures([structure])
    descriptors = calculator.predict_descriptors([structure])
    per_atom_descriptor = calculator.get_descriptor(structure)
    mean_descriptor = calculator.get_structures_descriptor([structure])
    energy, force_blocks, virial_blocks = calculator.calculate([structure])
    empty_energy, empty_forces, empty_virials = calculator.calculate([])
    empty_descriptors = calculator.get_structures_descriptor([])

    type_map = read_type_map(model_path)
    types = np.asarray([type_map[symbol] for symbol in structure.get_chemical_symbols()], dtype=np.int32)
    box = np.asarray(structure.cell, dtype=np.float64).reshape(9)
    atom_counts = np.asarray([len(types)], dtype=np.int32)
    with nep_adapters.load_model("cpu", model_path) as model:
        direct_energy, direct_forces, direct_virial = model.find_force(
            types,
            structure.positions,
            box,
        )
        direct_descriptors = model.descriptors(types, box, structure.positions, atom_counts)
        periodic_energy, periodic_forces, periodic_virial = model.find_force(
            types,
            structure.positions,
            box,
            (1, 1, 1),
        )

    if "ase" in getattr(nep_adapters, "__dict__", {}):
        raise AssertionError("core nep_adapters import should not import ASE adapter")
    if empty_energy.size != 0 or empty_forces or empty_virials:
        raise AssertionError("empty calculator batch should return empty results")
    if empty_descriptors.shape != (0, calculator.descriptor_dim):
        raise AssertionError("empty descriptor batch shape mismatch")
    if prediction.energy.shape != (1,):
        raise AssertionError("energy shape mismatch")
    if prediction.forces.shape != structure.positions.shape:
        raise AssertionError("force shape mismatch")
    if prediction.virials.shape != (len(structure), 9):
        raise AssertionError("virial shape mismatch")
    if descriptors.shape != expected_descriptors.shape:
        raise AssertionError("descriptor shape mismatch")
    if not np.allclose(descriptors, direct_descriptors, rtol=0.0, atol=1.0e-10):
        raise AssertionError("calculator descriptors differ from direct native descriptors")
    if not np.allclose(descriptors, expected_descriptors, rtol=0.0, atol=1.0e-10):
        raise AssertionError("calculator descriptors differ from fixed baseline labels")
    if not np.allclose(per_atom_descriptor, expected_descriptors.astype(np.float32), rtol=0.0, atol=1.0e-6):
        raise AssertionError("get_descriptor differs from fixed baseline labels")
    if not np.allclose(mean_descriptor[0], expected_descriptors.mean(axis=0), rtol=0.0, atol=1.0e-6):
        raise AssertionError("mean descriptor differs from fixed baseline labels")
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
    if abs(float(direct_energy) - float(periodic_energy)) > 1.0e-10:
        raise AssertionError("default pbc differs from explicit full periodicity")
    if not np.allclose(direct_forces, periodic_forces, rtol=0.0, atol=1.0e-10):
        raise AssertionError("default pbc forces differ from explicit full periodicity")
    if not np.allclose(direct_virial, periodic_virial, rtol=0.0, atol=1.0e-10):
        raise AssertionError("default pbc virial differs from explicit full periodicity")
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
