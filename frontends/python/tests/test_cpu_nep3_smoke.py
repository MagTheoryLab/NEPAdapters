import math
import os
import sys

import nep_adapters
import numpy as np

from baseline_utils import read_labeled_structure, read_type_map


def main():
    model_path = os.environ["NEP_ADAPTERS_PYTHON_TEST_MODEL"]
    xyz_path = os.environ["NEP_ADAPTERS_PYTHON_TEST_XYZ"]

    nep_adapters.register_cpu_nep3()
    infos = [nep_adapters.backend_info(i) for i in range(nep_adapters.backend_count())]
    if "cpu_nep3" not in {info.name for info in infos}:
        raise AssertionError("cpu_nep3 backend is not registered")

    structure = read_labeled_structure(xyz_path)
    type_map = read_type_map(model_path)
    types = np.asarray([type_map[symbol] for symbol in structure.symbols], dtype=np.int32)
    positions = structure.positions
    box = structure.cell.reshape(9)
    atom_counts = np.asarray([len(types)], dtype=np.int32)
    with nep_adapters.load_model("cpu_nep3", model_path) as model:
        info = model.model_info()
        if info["cutoff_max"] <= 0.0 or info["num_types"] <= 0:
            raise AssertionError("invalid model metadata")
        potentials, calc_forces, calc_virials = model.calculate(
            types,
            box,
            positions,
            atom_counts,
        )
        energy, forces, virial = model.find_force(types, positions, box)
        energy_again, forces_again, _ = model.find_force(types, positions, box)

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
    if not np.allclose(calc_forces, forces, rtol=0.0, atol=1.0e-10):
        raise AssertionError("calculate() and find_force() forces differ")
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
        "python cpu_nep3 smoke:",
        f"atoms={len(types)}",
        f"energy={float(energy):.12g}",
        f"force_l1={force_l1:.12g}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
