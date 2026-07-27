import os
from concurrent.futures import ThreadPoolExecutor

import nep_adapters
import numpy as np

from baseline_utils import read_labeled_structure, read_type_map


def main():
    model_path = os.environ["NEP_ADAPTERS_PYTHON_TEST_MODEL"]
    xyz_path = os.environ["NEP_ADAPTERS_PYTHON_TEST_XYZ"]
    structure = read_labeled_structure(xyz_path)
    type_map = read_type_map(model_path)
    types = np.asarray(
        [type_map[symbol] for symbol in structure.symbols],
        dtype=np.int32,
    )
    positions = np.ascontiguousarray(structure.positions, dtype=np.float64)
    box = np.ascontiguousarray(structure.cell.reshape(9), dtype=np.float64)
    atom_counts = np.asarray([len(types)], dtype=np.int32)

    nep_adapters.register_cpu()
    with nep_adapters.load_model("cpu", model_path) as model:
        reference = model.calculate(types, box, positions, atom_counts)

        def calculate_once(_):
            prediction = model.calculate(types, box, positions, atom_counts)
            return max(
                float(np.max(np.abs(actual - expected)))
                for actual, expected in zip(prediction, reference)
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            max_drift = max(executor.map(calculate_once, range(800)))

    if max_drift > 1.0e-10:
        raise AssertionError(
            f"concurrent calculations on one model changed output: {max_drift}"
        )

    print(f"python model concurrency: calls=800 max_drift={max_drift:.3e}")


if __name__ == "__main__":
    main()
