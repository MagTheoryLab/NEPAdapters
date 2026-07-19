import os
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from baseline_utils import LabeledStructure, read_labeled_structure
from nep_adapters import ChargePrediction, NEPCalculator, load_model


FIXTURE_ROOT = Path(os.environ["NEP_ADAPTERS_PRODUCTION_FIXTURE_DIR"])
QNEP_ROOT = Path(os.environ["NEP_ADAPTERS_QNEP_TEST_DATA_DIR"])
ORDINARY_MODEL = Path(os.environ["NEP_ADAPTERS_PYTHON_TEST_MODEL"])


def numeric_tokens(path):
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            values.extend(float(value) for value in line.split())
    return np.asarray(values, dtype=np.float64)


def read_qnep_structure():
    lines = (QNEP_ROOT / "xyz.in").read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[0])
    box_nep = np.asarray(lines[1].split(), dtype=np.float64).reshape(3, 3)
    symbols = []
    positions = []
    for line in lines[2 : 2 + atom_count]:
        fields = line.split()
        symbols.append(fields[0])
        positions.append([float(value) for value in fields[1:4]])
    return LabeledStructure(
        symbols=symbols,
        positions=np.asarray(positions, dtype=np.float64),
        cell=box_nep.T,
        pbc=np.ones(3, dtype=np.int32),
        energy=0.0,
        forces=np.zeros((atom_count, 3), dtype=np.float64),
        virial=np.zeros(9, dtype=np.float64),
    )


class ProductionApiTest(unittest.TestCase):
    def test_qnep_explicit_low_and_high_level_api(self):
        structure = read_qnep_structure()
        model_path = QNEP_ROOT / "nep.txt"
        with NEPCalculator(model_path) as calculator:
            info = calculator.model.model_info()
            self.assertEqual(info["model_type"], "charge")
            self.assertEqual(info["capabilities"] & (1 << 4), 1 << 4)
            with self.assertRaisesRegex((ValueError, RuntimeError), "calculate_charge"):
                calculator.predict_structures([structure])

            prediction = calculator.predict_charge_structures([structure])
            self.assertIsInstance(prediction, ChargePrediction)
            self.assertEqual(prediction.energy.shape, (1,))
            self.assertEqual(prediction.potential.shape, (250,))
            self.assertEqual(prediction.forces.shape, (250, 3))
            self.assertEqual(prediction.virials.shape, (250, 9))
            self.assertEqual(prediction.charges.shape, (250,))
            self.assertEqual(prediction.becs.shape, (250, 9))
            for array in (
                prediction.energy,
                prediction.potential,
                prediction.forces,
                prediction.virials,
                prediction.charges,
                prediction.becs,
            ):
                self.assertEqual(array.dtype, np.float64)

            np.testing.assert_allclose(
                prediction.forces,
                np.loadtxt(QNEP_ROOT / "force_analytical_ref.out"),
                atol=1e-10,
                rtol=1e-10,
            )
            np.testing.assert_allclose(
                prediction.virials,
                np.loadtxt(QNEP_ROOT / "virial_ref.out"),
                atol=1e-10,
                rtol=1e-10,
            )
            descriptors = calculator.predict_descriptors([structure])
            descriptor_reference = numeric_tokens(QNEP_ROOT / "descriptor_ref.out")
            np.testing.assert_allclose(
                descriptors.ravel(), descriptor_reference, atol=1e-10, rtol=1e-10
            )

            golden = numeric_tokens(
                FIXTURE_ROOT / "qnep_charge_bec_golden.txt"
            )
            indices = golden[:7].astype(np.int64)
            charges = golden[7:14]
            charge_stats = golden[14:18]
            becs = golden[18:81].reshape(7, 9)
            bec_stats = golden[81:85]
            expected_energy = golden[85]
            np.testing.assert_allclose(
                prediction.energy, expected_energy, atol=1e-10, rtol=2e-11
            )
            np.testing.assert_allclose(
                prediction.charges[indices], charges, atol=1e-12, rtol=1e-12
            )
            np.testing.assert_allclose(
                [
                    np.linalg.norm(prediction.charges),
                    prediction.charges.sum(),
                    prediction.charges.min(),
                    prediction.charges.max(),
                ],
                charge_stats,
                atol=1e-12,
                rtol=1e-12,
            )
            np.testing.assert_allclose(
                prediction.becs[indices], becs, atol=1e-12, rtol=1e-12
            )
            np.testing.assert_allclose(
                [
                    np.linalg.norm(prediction.becs),
                    prediction.becs.sum(),
                    prediction.becs.min(),
                    prediction.becs.max(),
                ],
                bec_stats,
                atol=1e-12,
                rtol=1e-12,
            )
            energies, forces, virials, charges_out, becs_out = (
                calculator.calculate_charge([structure])
            )
            self.assertEqual(energies.shape, (1,))
            self.assertEqual(forces[0].shape, (250, 3))
            self.assertEqual(virials[0].shape, (9,))
            self.assertEqual(charges_out[0].shape, (250,))
            self.assertEqual(becs_out[0].shape, (250, 9))

        ordinary = load_model("cpu", str(ORDINARY_MODEL))
        try:
            types = np.zeros(1, dtype=np.int32)
            positions = np.zeros((1, 3), dtype=np.float64)
            box = np.eye(3, dtype=np.float64).reshape(9)
            counts = np.ones(1, dtype=np.int32)
            with self.assertRaisesRegex((ValueError, RuntimeError), "charge"):
                ordinary.calculate_charge(types, box, positions, counts)
        finally:
            ordinary.close()

    def test_response_models_golden_batch_and_rejection(self):
        cases = (
            ("dipole", "predict_dipoles", (2, 3), 1e-5),
            ("polarizability", "predict_polarizabilities", (2, 6), 2e-4),
        )
        for name, method_name, shape, atol in cases:
            with self.subTest(name=name):
                structure = read_labeled_structure(
                    FIXTURE_ROOT / name / "structure.xyz"
                )
                golden = np.loadtxt(FIXTURE_ROOT / name / "golden.txt")
                descriptor_golden = np.loadtxt(
                    FIXTURE_ROOT / name / "descriptor_golden.txt",
                    comments="#",
                )
                with NEPCalculator(FIXTURE_ROOT / name / "nep.txt") as calculator:
                    info = calculator.model.model_info()
                    self.assertEqual(info["capabilities"] & (1 << 6), 1 << 6)
                    output = getattr(calculator, method_name)([structure, structure])
                    self.assertEqual(output.shape, shape)
                    self.assertEqual(output.dtype, np.float64)
                    np.testing.assert_allclose(
                        output, np.tile(golden, (2, 1)), atol=atol, rtol=5e-4
                    )
                    types, positions, boxes, pbc = calculator.compose_structures(
                        [structure, structure]
                    )
                    counts = np.asarray([len(structure), len(structure)], dtype=np.int32)
                    low_method = (
                        calculator.model.dipoles
                        if name == "dipole"
                        else calculator.model.polarizabilities
                    )
                    low = low_method(types, boxes, positions, counts, pbc)
                    np.testing.assert_allclose(low, output, atol=0.0, rtol=0.0)
                    descriptors = calculator.predict_descriptors(
                        [structure, structure]
                    )
                    self.assertEqual(
                        descriptors.shape,
                        (2 * len(structure), info["descriptor_dim"]),
                    )
                    self.assertEqual(descriptors.dtype, np.float64)
                    np.testing.assert_allclose(
                        descriptors,
                        np.tile(descriptor_golden, (2, 1)),
                        atol=1e-10,
                        rtol=1e-10,
                    )
                    low_descriptors = calculator.model.descriptors(
                        types, boxes, positions, counts, pbc
                    )
                    np.testing.assert_allclose(
                        low_descriptors, descriptors, atol=0.0, rtol=0.0
                    )
                    empty_descriptors = calculator.predict_descriptors([])
                    self.assertEqual(
                        empty_descriptors.shape, (0, info["descriptor_dim"])
                    )
                    self.assertEqual(empty_descriptors.dtype, np.float64)
                    with self.assertRaisesRegex(
                        (ValueError, RuntimeError), "explicit|response"
                    ):
                        calculator.predict_structures([structure])

        ordinary_structure = read_labeled_structure(
            FIXTURE_ROOT / "dftd3" / "structure.xyz"
        )
        with NEPCalculator(FIXTURE_ROOT / "dftd3" / "nep.txt") as ordinary:
            with self.assertRaisesRegex((ValueError, RuntimeError), "dipole"):
                ordinary.predict_dipoles([ordinary_structure])
            with self.assertRaisesRegex(
                (ValueError, RuntimeError), "polarizability"
            ):
                ordinary.predict_polarizabilities([ordinary_structure])

    def test_dftd3_low_high_batch_golden_and_rejection(self):
        structure = read_labeled_structure(
            FIXTURE_ROOT / "dftd3" / "structure.xyz"
        )
        with NEPCalculator(FIXTURE_ROOT / "dftd3" / "nep.txt") as calculator:
            for combined, golden_name in (
                (False, "pure_golden.txt"),
                (True, "combined_golden.txt"),
            ):
                golden = numeric_tokens(FIXTURE_ROOT / "dftd3" / golden_name)
                expected_energy = golden[0]
                expected_force = golden[1:13].reshape(4, 3)
                expected_virial = golden[13:22]
                method = (
                    calculator.predict_with_dftd3_structures
                    if combined
                    else calculator.predict_dftd3_structures
                )
                prediction = method([structure, structure], "pbe", 12.0, 10.0)
                self.assertEqual(prediction.energy.shape, (2,))
                self.assertEqual(prediction.potential.shape, (8,))
                self.assertEqual(prediction.forces.shape, (8, 3))
                self.assertEqual(prediction.virials.shape, (8, 9))
                self.assertEqual(prediction.structure_virials.shape, (2, 9))
                np.testing.assert_allclose(
                    prediction.energy, expected_energy, atol=1e-10, rtol=2e-11
                )
                np.testing.assert_allclose(
                    prediction.forces,
                    np.tile(expected_force, (2, 1)),
                    atol=1e-10,
                    rtol=2e-11,
                )
                for virial_block in prediction.virial_blocks(mean=False):
                    np.testing.assert_allclose(
                        virial_block.sum(axis=0),
                        expected_virial,
                        atol=1e-10,
                        rtol=2e-11,
                    )
                types, positions, boxes, pbc = calculator.compose_structures(
                    [structure]
                )
                counts = np.asarray([4], dtype=np.int32)
                low_method = (
                    calculator.model.calculate_with_dftd3
                    if combined
                    else calculator.model.calculate_dftd3
                )
                potential, force, virial = low_method(
                    types, boxes, positions, counts, "pbe", 12.0, 10.0, pbc
                )
                self.assertEqual(potential.dtype, np.float64)
                np.testing.assert_allclose(force, expected_force, atol=1e-10, rtol=2e-11)
                np.testing.assert_allclose(
                    virial.sum(axis=0), expected_virial, atol=1e-10, rtol=2e-11
                )

            with self.assertRaises((ValueError, RuntimeError)):
                calculator.predict_dftd3_structures(
                    [structure], "not-a-functional", 12.0, 10.0
                )

        charge_structure = read_qnep_structure()
        with NEPCalculator(QNEP_ROOT / "nep.txt") as charge:
            with self.assertRaisesRegex((ValueError, RuntimeError), "ordinary"):
                charge.predict_dftd3_structures(
                    [charge_structure], "pbe", 12.0, 10.0
                )

    def test_empty_batches_are_stable_float64(self):
        with NEPCalculator(QNEP_ROOT / "nep.txt") as charge:
            prediction = charge.predict_charge_structures([])
            self.assertEqual(prediction.energy.shape, (0,))
            self.assertEqual(prediction.forces.shape, (0, 3))
            self.assertEqual(prediction.virials.shape, (0, 9))
            self.assertEqual(prediction.charges.shape, (0,))
            self.assertEqual(prediction.becs.shape, (0, 9))
            self.assertTrue(all(
                array.dtype == np.float64
                for array in (
                    prediction.energy,
                    prediction.forces,
                    prediction.virials,
                    prediction.charges,
                    prediction.becs,
                )
            ))
        for name, method, shape in (
            ("dipole", "predict_dipoles", (0, 3)),
            ("polarizability", "predict_polarizabilities", (0, 6)),
        ):
            with NEPCalculator(FIXTURE_ROOT / name / "nep.txt") as calculator:
                output = getattr(calculator, method)([])
                self.assertEqual(output.shape, shape)
                self.assertEqual(output.dtype, np.float64)
        with NEPCalculator(FIXTURE_ROOT / "dftd3" / "nep.txt") as calculator:
            for method in (
                calculator.predict_dftd3_structures,
                calculator.predict_with_dftd3_structures,
            ):
                prediction = method([], "pbe", 12.0, 10.0)
                self.assertEqual(prediction.energy.shape, (0,))
                self.assertEqual(prediction.forces.shape, (0, 3))
                self.assertEqual(prediction.virials.shape, (0, 9))
                self.assertEqual(prediction.structure_virials.shape, (0, 9))

    def test_threaded_cancellation_reset_and_close(self):
        structure = read_qnep_structure()
        calculator = NEPCalculator(QNEP_ROOT / "nep.txt")
        started = threading.Event()
        errors = []
        successes = []

        def worker():
            started.set()
            try:
                successes.append(
                    calculator.predict_charge_structures([structure] * 32)
                )
            except Exception as error:  # the exact Python type is an API detail
                errors.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(started.wait(1.0))
        time.sleep(0.01)
        calculator.cancel()
        thread.join(10.0)
        self.assertFalse(thread.is_alive())
        self.assertFalse(successes, "cancelled calculation returned partial success")
        self.assertEqual(len(errors), 1)
        self.assertIn("cancelled", str(errors[0]).lower())
        calculator.reset_cancel()
        recovered = calculator.predict_charge_structures([structure])
        self.assertEqual(recovered.charges.shape, (250,))
        calculator.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            calculator.predict_charge_structures([structure])


if __name__ == "__main__":
    unittest.main()
