import math
import os
import sys

import nep_adapters
import numpy as np

from baseline_utils import read_descriptor_fixture, read_labeled_structure, read_type_map


def main():
    model_path = os.environ["NEP_ADAPTERS_PYTHON_TEST_MODEL"]
    xyz_path = os.environ["NEP_ADAPTERS_PYTHON_TEST_XYZ"]

    nep_adapters.register_cpu()
    infos = [nep_adapters.backend_info(i) for i in range(nep_adapters.backend_count())]
    if "cpu" not in {info.name for info in infos}:
        raise AssertionError("cpu backend is not registered")

    structure = read_labeled_structure(xyz_path)
    expected_descriptors = read_descriptor_fixture(
        os.path.join(os.path.dirname(xyz_path), "descriptor.txt")
    )
    type_map = read_type_map(model_path)
    types = np.asarray([type_map[symbol] for symbol in structure.symbols], dtype=np.int32)
    positions = structure.positions
    box = structure.cell.reshape(9)
    atom_counts = np.asarray([len(types)], dtype=np.int32)
    with nep_adapters.load_model("cpu", model_path) as model:
        info = model.model_info()
        if info["cutoff_max"] <= 0.0 or info["num_types"] <= 0 or info["descriptor_dim"] <= 0:
            raise AssertionError("invalid model metadata")
        potentials, calc_forces, calc_virials = model.calculate(
            types,
            box,
            positions,
            atom_counts,
        )
        descriptors = model.descriptors(types, box, positions, atom_counts)
        explicit_periodic = model.calculate(
            types,
            box,
            positions,
            atom_counts,
            (1, 1, 1),
        )
        try:
            model.calculate_spin(
                types,
                box,
                positions,
                np.zeros((len(types), 3), dtype=np.float64),
                atom_counts,
            )
        except ValueError as error:
            if "spin NEP model" not in str(error):
                raise
        else:
            raise AssertionError("ordinary model must reject the spin API")
        energy, forces, virial = model.find_force(types, positions, box)
        energy_again, forces_again, _ = model.find_force(types, positions, box)
        try:
            model.find_force(
                types,
                positions,
                box,
                np.asarray([1, 1, 0], dtype=np.int32),
            )
        except ValueError as error:
            if "fully periodic" not in str(error):
                raise
        else:
            raise AssertionError("non-periodic input must be rejected")
        try:
            model.find_force(types, positions, box, None)
        except ValueError as error:
            if "omit it instead of passing None" not in str(error):
                raise
        else:
            raise AssertionError("explicit pbc=None must be rejected")

    force_l1 = float(np.abs(forces).sum())
    if not math.isfinite(float(energy)) or force_l1 <= 0:
        raise AssertionError("invalid Python prediction")
    if abs(float(energy) - float(energy_again)) > 1.0e-10:
        raise AssertionError("repeat Python predictions are inconsistent")
    if not np.allclose(forces, forces_again, rtol=0.0, atol=1.0e-10):
        raise AssertionError("repeat Python forces are inconsistent")
    if virial.shape != (9,):
        raise AssertionError("virial shape mismatch")
    if potentials.shape != (len(types),):
        raise AssertionError("potential array shape mismatch")
    if calc_forces.shape != (len(types), 3):
        raise AssertionError("force array shape mismatch")
    if calc_virials.shape != (len(types), 9):
        raise AssertionError("virial array shape mismatch")
    if descriptors.shape != expected_descriptors.shape:
        raise AssertionError("descriptor array shape mismatch")
    if descriptors.shape[1] != info["descriptor_dim"]:
        raise AssertionError("descriptor_dim metadata mismatch")
    if not np.allclose(descriptors, expected_descriptors, rtol=0.0, atol=1.0e-10):
        raise AssertionError("descriptors differ from fixed baseline labels")
    if not np.allclose(calc_forces, forces, rtol=0.0, atol=1.0e-10):
        raise AssertionError("calculate() and find_force() forces differ")
    if not all(
        np.allclose(default, explicit, rtol=0.0, atol=1.0e-10)
        for default, explicit in zip(
            (potentials, calc_forces, calc_virials),
            explicit_periodic,
        )
    ):
        raise AssertionError("default pbc must equal explicit (1, 1, 1)")
    if abs(float(np.sum(potentials)) - float(energy)) > 1.0e-10:
        raise AssertionError("per-atom potentials do not sum to structure energy")
    if abs(float(np.sum(potentials)) - structure.energy) > 1.0e-10:
        raise AssertionError("calculate() energy differs from fixed baseline label")
    if not np.allclose(calc_forces, structure.forces, rtol=0.0, atol=1.0e-10):
        raise AssertionError("calculate() forces differ from fixed baseline labels")
    if not np.allclose(calc_virials.sum(axis=0), structure.virial, rtol=0.0, atol=1.0e-10):
        raise AssertionError("calculate() virial differs from fixed baseline label")
    if abs(float(energy) - structure.energy) > 1.0e-10:
        raise AssertionError("energy differs from fixed baseline label")
    if not np.allclose(forces, structure.forces, rtol=0.0, atol=1.0e-10):
        raise AssertionError("forces differ from fixed baseline labels")
    if not np.allclose(virial, structure.virial, rtol=0.0, atol=1.0e-10):
        raise AssertionError("virial differs from fixed baseline label")

    print(
        "python cpu smoke:",
        f"atoms={len(types)}",
        f"energy={float(energy):.12g}",
        f"force_l1={force_l1:.12g}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
