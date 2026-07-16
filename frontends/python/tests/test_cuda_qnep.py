import os
from pathlib import Path
import sys

import numpy as np

import nep_adapters

from baseline_utils import read_type_map


def read_xyz_input(path, type_map):
    fields = Path(path).read_text(encoding="utf-8").split()
    atom_count = int(fields[0])
    box_columns = np.asarray(fields[1:10], dtype=np.float64)
    box = box_columns.reshape(3, 3).T.reshape(9)
    types = np.empty(atom_count, dtype=np.int32)
    positions = np.empty((atom_count, 3), dtype=np.float64)
    cursor = 10
    for atom in range(atom_count):
        symbol = fields[cursor]
        types[atom] = type_map[symbol]
        positions[atom] = np.asarray(fields[cursor + 1 : cursor + 4], dtype=np.float64)
        cursor += 4
    return types, positions, box


def max_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def main():
    data_dir = Path(os.environ["NEP_ADAPTERS_QNEP_TEST_DATA_DIR"])
    model_path = data_dir / "nep.txt"
    types, positions, box = read_xyz_input(
        data_dir / "xyz.in",
        read_type_map(model_path),
    )
    atom_counts = np.asarray([len(types)], dtype=np.int32)
    force_reference = np.loadtxt(
        data_dir / "force_analytical_ref.out",
        dtype=np.float64,
    ).reshape(len(types), 3)
    virial_reference = np.loadtxt(
        data_dir / "virial_ref.out",
        dtype=np.float64,
    ).reshape(len(types), 9)
    descriptor_reference = np.loadtxt(
        data_dir / "descriptor_ref.out",
        dtype=np.float64,
    ).reshape(len(types), -1)

    with nep_adapters.load_model("cuda", str(model_path)) as model:
        potentials, forces, virials = model.calculate(
            types,
            box,
            positions,
            atom_counts,
        )
        if not all(np.all(np.isfinite(values)) for values in (potentials, forces, virials)):
            raise AssertionError("qNEP CUDA output contains non-finite values")
        force_error = max_error(forces, force_reference)
        virial_error = max_error(virials, virial_reference)
        descriptors = model.descriptors(types, box, positions, atom_counts)
        descriptor_error = max_error(descriptors, descriptor_reference)
        if force_error > 5.0e-3:
            raise AssertionError(f"qNEP force mismatch: max_abs_error={force_error:.12g}")
        if virial_error > 1.0e-2:
            raise AssertionError(f"qNEP virial mismatch: max_abs_error={virial_error:.12g}")
        if descriptor_error > 5.0e-4:
            raise AssertionError(
                f"qNEP descriptor mismatch: max_abs_error={descriptor_error:.12g}"
            )

        os.environ["NEP_ADAPTERS_QNEP_KSPACE"] = "pppm"
        try:
            model.calculate(types, box, positions, atom_counts)
        except RuntimeError as error:
            if "PPPM support is disabled" not in str(error):
                raise
        else:
            raise AssertionError("a no-PPPM wheel must fail closed for PPPM requests")
        finally:
            os.environ.pop("NEP_ADAPTERS_QNEP_KSPACE", None)

    print(
        "python CUDA qNEP:",
        f"atoms={len(types)}",
        f"force_max_abs_error={force_error:.12g}",
        f"virial_max_abs_error={virial_error:.12g}",
        f"descriptor_max_abs_error={descriptor_error:.12g}",
        "pppm=disabled-fail-closed",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
