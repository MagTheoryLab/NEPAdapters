import math
import os
import sys
from pathlib import Path

import nep_adapters
import numpy as np


def read_type_map(model_path):
    header = Path(model_path).read_text(encoding="utf-8").splitlines()[0].split()
    count = int(header[1])
    return {symbol: index for index, symbol in enumerate(header[2:2 + count])}


def parse_lattice(comment):
    key = 'Lattice="'
    begin = comment.index(key) + len(key)
    end = comment.index('"', begin)
    return [float(value) for value in comment[begin:end].split()]


def read_first_structure(xyz_path, type_map):
    with open(xyz_path, encoding="utf-8") as handle:
        atom_count = int(handle.readline())
        comment = handle.readline()
        box = parse_lattice(comment)
        types = []
        positions = []
        for _ in range(atom_count):
            fields = handle.readline().split()
            types.append(type_map[fields[0]])
            positions.append(tuple(float(value) for value in fields[1:4]))
    return (
        np.asarray(types, dtype=np.int32),
        np.asarray(positions, dtype=np.float64),
        np.asarray(box, dtype=np.float64),
    )


def main():
    model_path = os.environ["NEP_ADAPTERS_PYTHON_TEST_MODEL"]
    xyz_path = os.environ["NEP_ADAPTERS_PYTHON_TEST_XYZ"]

    nep_adapters.register_cpu_nep3()
    infos = [nep_adapters.backend_info(i) for i in range(nep_adapters.backend_count())]
    if "cpu_nep3" not in {info.name for info in infos}:
        raise AssertionError("cpu_nep3 backend is not registered")

    types, positions, box = read_first_structure(xyz_path, read_type_map(model_path))
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

    print(
        "python cpu_nep3 smoke:",
        f"atoms={len(types)}",
        f"energy={float(energy):.12g}",
        f"force_l1={force_l1:.12g}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
