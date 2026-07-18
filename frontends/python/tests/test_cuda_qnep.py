import os
from pathlib import Path
import sys

import numpy as np

import nep_adapters

from baseline_utils import read_type_map


def numeric_tokens(path):
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            values.extend(float(value) for value in line.split())
    return np.asarray(values, dtype=np.float64)


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


def error_stats(actual, expected, atol, rtol):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"shape mismatch: actual={actual.shape} expected={expected.shape}"
        )
    difference = np.abs(actual - expected)
    flat_index = int(np.argmax(difference))
    index = np.unravel_index(flat_index, difference.shape)
    reference_scale = float(np.max(np.abs(expected)))
    relative_floor = max(1.0e-12, reference_scale * 1.0e-8)
    relative = difference / np.maximum(np.abs(expected), relative_floor)
    limit = atol + rtol * np.abs(expected)
    return {
        "max_abs": float(difference[index]),
        "max_rel": float(np.max(relative)),
        "rms": float(np.sqrt(np.mean(np.square(difference)))),
        "index": tuple(int(value) for value in index),
        "actual": float(actual[index]),
        "expected": float(expected[index]),
        "violations": int(np.count_nonzero(difference > limit)),
    }


def require_within_budget(label, actual, expected, atol, rtol):
    stats = error_stats(actual, expected, atol, rtol)
    if stats["violations"]:
        raise AssertionError(
            f"{label} mismatch: max_abs={stats['max_abs']:.12g} "
            f"max_rel={stats['max_rel']:.12g} rms={stats['rms']:.12g} "
            f"index={stats['index']} actual={stats['actual']:.12g} "
            f"expected={stats['expected']:.12g} "
            f"violations={stats['violations']}"
        )
    return stats


def main():
    data_dir = Path(os.environ["NEP_ADAPTERS_QNEP_TEST_DATA_DIR"])
    production_fixture_dir = Path(
        os.environ["NEP_ADAPTERS_PRODUCTION_FIXTURE_DIR"]
    )
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
        with np.testing.assert_raises_regex(
            (ValueError, RuntimeError), "calculate_charge"
        ):
            model.calculate(types, box, positions, atom_counts)
        potentials, forces, virials, charges, becs = model.calculate_charge(
            types,
            box,
            positions,
            atom_counts,
        )
        if not all(
            np.all(np.isfinite(values))
            for values in (potentials, forces, virials, charges, becs)
        ):
            raise AssertionError("qNEP CUDA output contains non-finite values")
        descriptors = model.descriptors(types, box, positions, atom_counts)
        force_stats = require_within_budget(
            "qNEP force", forces, force_reference, 2.0e-4, 2.0e-6
        )
        virial_stats = require_within_budget(
            "qNEP virial", virials, virial_reference, 1.5e-3, 2.0e-6
        )
        descriptor_stats = require_within_budget(
            "qNEP descriptor",
            descriptors,
            descriptor_reference,
            1.0e-5,
            2.0e-6,
        )

        with nep_adapters.load_model("cpu", str(model_path)) as cpu_model:
            cpu_outputs = cpu_model.calculate_charge(
                types, box, positions, atom_counts
            )
        potential_stats = require_within_budget(
            "qNEP potential", potentials, cpu_outputs[0], 2.0e-4, 2.0e-6
        )
        energy_golden = numeric_tokens(
            production_fixture_dir / "qnep_charge_bec_golden.txt"
        )[85]
        if not np.isclose(
            potentials.sum(), energy_golden, atol=2.0e-4, rtol=2.0e-6
        ):
            raise AssertionError(
                f"qNEP energy mismatch: {potentials.sum()} vs {energy_golden}"
            )
        charge_stats = require_within_budget(
            "qNEP charge", charges, cpu_outputs[3], 2.0e-5, 2.0e-6
        )
        bec_stats = require_within_budget(
            "qNEP BEC", becs, cpu_outputs[4], 3.0e-3, 2.0e-5
        )

        with nep_adapters.NEPCalculator(model_path, backend="cuda") as calculator:
            prediction = calculator.predict_charge_arrays(
                np.tile(types, 2),
                np.tile(positions, (2, 1)),
                np.tile(box, (2, 1)),
                np.asarray([len(types), len(types)], dtype=np.int32),
            )
        if prediction.charges.shape != (2 * len(types),) or prediction.becs.shape != (
            2 * len(types),
            9,
        ):
            raise AssertionError("qNEP CUDA high-level batch shape mismatch")
        require_within_budget(
            "qNEP batch charge 0", prediction.charges[: len(types)], charges, 2.0e-5, 2.0e-6
        )
        require_within_budget(
            "qNEP batch BEC 1", prediction.becs[len(types) :], becs, 3.0e-3, 2.0e-5
        )

        os.environ["NEP_ADAPTERS_QNEP_KSPACE"] = "pppm"
        try:
            model.calculate_charge(types, box, positions, atom_counts)
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
        f"force_max_abs={force_stats['max_abs']:.12g}",
        f"force_rms={force_stats['rms']:.12g}",
        f"force_worst={force_stats['index']}",
        f"virial_max_abs={virial_stats['max_abs']:.12g}",
        f"virial_rms={virial_stats['rms']:.12g}",
        f"virial_worst={virial_stats['index']}",
        f"descriptor_max_abs={descriptor_stats['max_abs']:.12g}",
        f"descriptor_rms={descriptor_stats['rms']:.12g}",
        f"descriptor_worst={descriptor_stats['index']}",
        f"potential_max_abs={potential_stats['max_abs']:.12g}",
        f"charge_max_abs={charge_stats['max_abs']:.12g}",
        f"bec_max_abs={bec_stats['max_abs']:.12g}",
        "pppm=disabled-fail-closed",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
