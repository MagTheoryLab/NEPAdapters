"""New spin_type semantics and immutable historical spin-selection behavior."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from ase import Atoms
from nep_adapters import NEPCalculator, inspect_model, load_model
from nep_adapters.ase import NepAseCalculator
from test_spin2_batch import read_noncollinear

ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures"
BACKEND = os.environ.get("NEP_BLOCK_TEST_BACKEND", "cpu")


class SpinTypeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nep-spin-type-")
        self.addCleanup(self.temp.cleanup)
        self.path = ROOT / "spin_type/nep_spin_o3c2_uniform.nep"
        self.case = read_noncollinear(ROOT / "spin_type/spin3_o3c2_uniform_oracle.txt")
        c = self.case
        self.args = (c["types"], c["cell"].reshape(1, 9), c["positions"].reshape(-1, 3),
                     c["spins"].reshape(-1, 3), np.array([len(c["types"])], dtype=np.int32))

    def variant(self, name, text):
        path = Path(self.temp.name) / name
        path.write_text(text)
        return path

    def evaluate(self, path, args=None):
        with load_model(BACKEND, str(path)) as model:
            args = self.args if args is None else args
            return (*model.calculate_spin(*args), model.descriptors_spin(*args))

    def assert_same(self, a, b):
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=0, atol=2e-10 if BACKEND == "cpu" else 2e-6)

    def test_independent_oracle_and_ase(self):
        for channels in (2, 3):
            with self.subTest(channels=channels):
                model_path = ROOT / f"spin_type/nep_spin_o3c{channels}_uniform.nep"
                oracle_path = ROOT / f"spin_type/spin3_o3c{channels}_uniform_oracle.txt"
                self.assertEqual(hashlib.sha256(model_path.read_bytes()).hexdigest(),
                                 oracle_path.read_text().splitlines()[1].split()[1])
                c = read_noncollinear(oracle_path)
                info = inspect_model(model_path)  # Same native inspection used by NepTrain.
                self.assertTrue(info.supports("spin"))
                self.assertEqual(info.descriptor_dim, c["descriptor_dim"])
                atoms = Atoms(["Fe" if t == 0 else "Ge" for t in c["types"]],
                              positions=c["positions"].reshape(-1, 3),
                              cell=c["cell"].reshape(3, 3), pbc=True)
                atoms.set_array("spin", c["spins"].reshape(-1, 3))
                calc = NEPCalculator(model_path, backend=BACKEND)
                result = calc.predict_spin_structures([atoms])
                desc = calc.get_spin_descriptor(atoms)
                for key, actual in [("potential", result.potential), ("forces", result.forces),
                                    ("mforces", result.mforces), ("virial", result.virials),
                                    ("descriptor", desc)]:
                    np.testing.assert_allclose(np.asarray(actual).ravel(), c[key],
                                               rtol=1e-6, atol=2e-6 if BACKEND == "cpu" else 1.2e-3,
                                               err_msg=key)
                atoms.calc = NepAseCalculator(model_path, backend=BACKEND)
                np.testing.assert_allclose(atoms.get_forces(), result.forces, rtol=0, atol=2e-6)
                self.assertAlmostEqual(atoms.get_potential_energy(), result.energy[0], places=6)
                np.testing.assert_array_equal(atoms.calc.results["mforces"][c["types"] == 1], 0)
                inactive = c["types"] == 1
                np.testing.assert_array_equal(result.mforces[inactive], 0)
                np.testing.assert_array_equal(desc[inactive, 30:], 0)
                self.assertGreater(np.linalg.norm(result.forces[inactive]), 1e-4)
                atoms.arrays["spin"][inactive] = [0.4, -0.7, 1.2]
                atoms.set_array("mforce", np.full((len(atoms), 3), 123.0))
                atoms.set_array("force", np.full((len(atoms), 3), 456.0))
                atoms.info["energy"] = -789.0
                original = {key: value.copy() for key, value in atoms.arrays.items()}
                raw = calc.predict_spin_structures([atoms])
                for field in ("potential", "forces", "mforces", "virials"):
                    np.testing.assert_allclose(getattr(raw, field), getattr(result, field), rtol=0, atol=2e-6)
                atoms.calc = NepAseCalculator(model_path, backend=BACKEND)
                np.testing.assert_allclose(atoms.get_forces(), result.forces, rtol=0, atol=2e-6)
                for key, value in original.items():
                    np.testing.assert_array_equal(atoms.arrays[key], value)
                self.assertEqual(atoms.info["energy"], -789.0)


    def test_new_geometry_and_legacy_default_remain_distinct(self):
        text = self.path.read_text()
        new = self.evaluate(self.path)
        explicit = self.variant("legacy-all.nep", text.replace("spin_type Fe", "spin_dof_type Fe\nspin_env_type Fe Ge"))
        self.assert_same(new, self.evaluate(explicit))
        omitted = self.variant("legacy-default.nep", text.replace("spin_type Fe", "spin_dof_type Fe"))
        excluded = self.variant("legacy-fe.nep", text.replace("spin_type Fe", "spin_dof_type Fe\nspin_env_type Fe"))
        old = self.evaluate(omitted)
        self.assert_same(old, self.evaluate(excluded))
        self.assertGreater(np.max(np.abs(new[-1] - old[-1])), 1e-5)
        inactive = self.case["types"] == 1
        # Nonmagnetic atoms retain magnetic-geometry forces, not just structural forces.
        self.assertGreater(np.max(np.abs(new[1][inactive] - old[1][inactive])), 1e-5)
        all_types = self.variant("all.nep", text.replace("spin_type Fe", "spin_type Fe Ge"))
        default = self.variant("default.nep", text.replace("spin_type Fe\n", ""))
        self.assert_same(self.evaluate(all_types), self.evaluate(default))

    def test_effective_zero_spin_and_legacy_input_compatibility(self):
        args = list(self.args)
        args[3] = args[3].copy()
        args[3][self.case["types"] == 1] = [0.125, -0.75, 1.5]
        original = args[3].copy()
        self.assert_same(self.evaluate(self.path, args), self.evaluate(self.path))
        np.testing.assert_array_equal(args[3], original)
        legacy = self.variant("legacy.nep", self.path.read_text().replace("spin_type Fe", "spin_dof_type Fe\nspin_env_type Fe Ge"))
        result = self.evaluate(legacy, args)
        clean = self.evaluate(legacy)
        self.assertTrue(all(np.isfinite(x).all() for x in result))
        self.assertGreater(np.max(np.abs(result[-1] - clean[-1])), 1e-5)
        np.testing.assert_array_equal(result[3][self.case["types"] == 1], 0)

    def test_selector_errors(self):
        for selector in ("spin_type Fe\nspin_dof_type Fe", "spin_type Fe\nspin_env_type Fe Ge",
                         "spin_type Fe Fe", "spin_type Xe", "spin_type Fe\nspin_type Ge"):
            with self.subTest(selector=selector):
                path = self.variant("invalid.nep", self.path.read_text().replace("spin_type Fe", selector))
                with self.assertRaises(Exception):
                    load_model(BACKEND, str(path))


if __name__ == "__main__":
    unittest.main()
