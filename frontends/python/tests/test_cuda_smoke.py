import os
import sys

import numpy as np

import nep_adapters

from baseline_utils import read_labeled_structure, read_type_map


def assert_close(name, actual, expected, atol=2.0e-7):
    if not np.allclose(actual, expected, rtol=0.0, atol=atol):
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        raise AssertionError(f"{name} CPU/CUDA mismatch: max_abs_error={error:.12g}")


def evaluate(model_path, types, positions, box, atom_counts, spins=None):
    outputs = {}
    for backend in ("cpu", "cuda"):
        with nep_adapters.load_model(backend, model_path) as model:
            if spins is None:
                outputs[backend] = (
                    *model.calculate(types, box, positions, atom_counts),
                    model.descriptors(types, box, positions, atom_counts),
                )
            else:
                outputs[backend] = (
                    *model.calculate_spin(
                        types,
                        box,
                        positions,
                        spins,
                        atom_counts,
                    ),
                    model.descriptors_spin(
                        types,
                        box,
                        positions,
                        spins,
                        atom_counts,
                    ),
                )
    return outputs


def compare_outputs(label, outputs, tolerances):
    cpu = outputs["cpu"]
    cuda = outputs["cuda"]
    if len(cpu) != len(cuda):
        raise AssertionError(f"{label} output count mismatch")
    if len(cpu) != len(tolerances):
        raise AssertionError(f"{label} tolerance count mismatch")
    for index, (cpu_value, cuda_value, tolerance) in enumerate(
        zip(cpu, cuda, tolerances)
    ):
        assert_close(
            f"{label} output {index}",
            cuda_value,
            cpu_value,
            atol=tolerance,
        )


def main():
    ordinary_model = os.environ["NEP_ADAPTERS_PYTHON_TEST_MODEL"]
    ordinary_xyz = os.environ["NEP_ADAPTERS_PYTHON_TEST_XYZ"]
    spin_model = os.environ["NEP_ADAPTERS_PYTHON_SPIN_MODEL"]

    nep_adapters.register_cuda()
    backend_names = {
        nep_adapters.backend_info(index).name
        for index in range(nep_adapters.backend_count())
    }
    if "cuda" not in backend_names:
        raise AssertionError("CUDA backend is not registered")

    structure = read_labeled_structure(ordinary_xyz)
    type_map = read_type_map(ordinary_model)
    types = np.asarray(
        [type_map[symbol] for symbol in structure.symbols],
        dtype=np.int32,
    )
    positions = np.ascontiguousarray(structure.positions, dtype=np.float64)
    box = np.ascontiguousarray(structure.cell.reshape(9), dtype=np.float64)
    atom_counts = np.asarray([len(types)], dtype=np.int32)
    ordinary = evaluate(
        ordinary_model,
        types,
        positions,
        box,
        atom_counts,
    )
    compare_outputs(
        "ordinary",
        ordinary,
        (5.0e-4, 1.0e-3, 5.0e-3, 5.0e-4),
    )

    spin_types = np.zeros(4, dtype=np.int32)
    spin_positions = np.asarray(
        [
            [0.2, 0.2, 0.2],
            [3.7, 0.3, 0.2],
            [0.4, 3.6, 0.5],
            [1.8, 1.7, 3.5],
        ],
        dtype=np.float64,
    )
    spins = np.asarray(
        [
            [1.0, 0.2, 0.0],
            [0.4, -0.3, 0.7],
            [-0.2, 0.8, 0.5],
            [0.6, 0.1, -0.4],
        ],
        dtype=np.float64,
    )
    # Match the existing CUDA spin fixture: fully periodic, with a box large
    # enough that the cutoff does not introduce repeated periodic images.
    spin_box = np.diag([16.0, 16.0, 16.0]).reshape(9)
    spin_counts = np.asarray([4], dtype=np.int32)
    spin = evaluate(
        spin_model,
        spin_types,
        spin_positions,
        spin_box,
        spin_counts,
        spins,
    )
    compare_outputs(
        "spin",
        spin,
        (2.0e-4, 2.0e-4, 1.0e-2, 2.0e-4, 2.0e-4, 2.0e-4),
    )

    with nep_adapters.NEPCalculator(ordinary_model, backend="cuda") as calculator:
        prediction = calculator.predict_arrays(
            types,
            positions,
            box,
            atom_counts,
        )
        descriptors = calculator.predict_descriptors_arrays(
            types,
            positions,
            box,
            atom_counts,
        )
    assert_close("high-level ordinary potential", prediction.potential, ordinary["cuda"][0])
    assert_close("high-level ordinary descriptor", descriptors, ordinary["cuda"][3])

    with nep_adapters.NEPCalculator(spin_model, backend="cuda") as calculator:
        prediction = calculator.predict_spin_arrays(
            spin_types,
            spin_positions,
            spins,
            spin_box,
            spin_counts,
        )
        descriptors = calculator.predict_spin_descriptors_arrays(
            spin_types,
            spin_positions,
            spins,
            spin_box,
            spin_counts,
        )
    assert_close("high-level spin mforce", prediction.mforces, spin["cuda"][3])
    assert_close("high-level spin descriptor", descriptors, spin["cuda"][5])

    print(
        "python CUDA smoke:",
        f"ordinary_atoms={len(types)}",
        f"spin_atoms={len(spin_types)}",
        f"ordinary_descriptor_dim={ordinary['cuda'][3].shape[1]}",
        f"spin_descriptor_dim={spin['cuda'][5].shape[1]}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
