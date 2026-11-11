"""Density model parsing, batching and ASE contracts against frozen independent labels."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from ase import Atoms
from nep_adapters import NEPCalculator, inspect_model, load_model
from nep_adapters.ase import NepAseCalculator

ROOT = Path(__file__).resolve().parents[3] / 'tests/fixtures/spin_density'
BACKEND = os.environ.get('NEP_BLOCK_TEST_BACKEND', 'cpu')


def arguments(case):
    return (np.array([['Fe', 'Bi', 'O'].index(t) for t in case['types']], np.int32),
            np.array(case['cell']).reshape(3, 3).T.reshape(1, 9),
            np.array(case['positions']), np.array(case['spins']), np.array([len(case['types'])], np.int32))


class DensitySpinTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = ROOT / 'soc/c3l2s2.nep'
        self.case = json.loads(self.path.with_suffix('.json').read_text())['cases']['large']

    def variant(self, text):
        path = Path(self.temp.name) / 'model.nep'
        path.write_text(text)
        return path

    def test_frozen_oracles_and_provenance(self):
        for record in sorted(ROOT.glob('*/*.json')):
            if record.parent.name == 'lammps': continue
            data = json.loads(record.read_text())
            model_path = record.with_suffix('.nep')
            self.assertEqual(hashlib.sha256(model_path.read_bytes()).hexdigest(), data['model_sha256'])
            cases = data['cases']
            cases = cases.values() if isinstance(cases, dict) else cases
            with load_model(BACKEND, str(model_path)) as model:
                for case in cases:
                    with self.subTest(model=record.stem, case=case.get('name', 'soc')):
                        potential, force, virial, mforce = model.calculate_spin(*arguments(case))
                        for actual, expected in [(potential, case['potential']), (force, case['force']),
                                                 (mforce, case['mforce']), (virial.sum(0), case['virial'])]:
                            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-5)

    def test_multibox_batch_and_descriptors(self):
        data = json.loads(self.path.with_suffix('.json').read_text())['cases']
        cases = [data['large'], data['triclinic']]
        serial_args = [arguments(case) for case in cases]
        args = tuple(np.concatenate([a[k] for a in serial_args]) for k in range(5))
        with load_model(BACKEND, str(self.path)) as model:
            for actual, index in zip(model.calculate_spin(*args), range(4)):
                expected = np.concatenate([model.calculate_spin(*a)[index] for a in serial_args])
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5)
            expected = np.concatenate([model.descriptors_spin(*a) for a in serial_args])
            np.testing.assert_allclose(model.descriptors_spin(*args), expected, rtol=1e-5, atol=2e-5)

    def test_ase_masking_and_nonmagnetic_energy_centers(self):
        c = self.case
        atoms = Atoms(c['types'], positions=c['positions'], cell=np.array(c['cell']).reshape(3, 3), pbc=True)
        atoms.set_array('spin', np.array(c['spins']))
        original = atoms.arrays['spin'].copy()
        calc = NEPCalculator(self.path, backend=BACKEND)
        result = calc.predict_spin_structures([atoms])
        descriptors = calc.get_spin_descriptor(atoms)
        self.assertGreater(np.linalg.norm(descriptors[2:, 30:]), 1e-4)
        np.testing.assert_array_equal(result.mforces[2:], 0.)
        atoms.calc = NepAseCalculator(self.path, backend=BACKEND)
        np.testing.assert_allclose(atoms.get_forces(), result.forces, atol=2e-5)
        self.assertAlmostEqual(atoms.get_potential_energy(), sum(c['potential']), places=5)
        np.testing.assert_array_equal(atoms.arrays['spin'], original)
        atoms.arrays['spin'][2:] *= 7.3
        changed = calc.predict_spin_structures([atoms])
        np.testing.assert_allclose(changed.potential, result.potential, atol=2e-6)
        np.testing.assert_allclose(changed.mforces, result.mforces, atol=2e-6)
        self.assertEqual(inspect_model(self.path).descriptor_dim, 64)

    def test_reference_energy_is_double_precision_only(self):
        shifts = np.array([-123.1234567890123, .234567890123456, 12.3456789012345])
        text = self.path.read_text().replace('energy_reference 0 0 0', 'energy_reference ' + ' '.join(map(str, shifts)))
        changed = self.variant(text)
        args = arguments(self.case)
        with load_model(BACKEND, str(self.path)) as a, load_model(BACKEND, str(changed)) as b:
            before, after = a.calculate_spin(*args), b.calculate_spin(*args)
            np.testing.assert_allclose(after[0] - before[0], shifts[args[0]], atol=1e-12, rtol=0)
            for x, y in zip(before[1:], after[1:]): np.testing.assert_allclose(x, y, atol=2e-6)
            np.testing.assert_allclose(a.descriptors_spin(*args), b.descriptors_spin(*args), atol=2e-6)

    def test_removed_models_and_invalid_density_shapes(self):
        text = self.path.read_text()
        variants = ['nep4_spin3 1 Fe\n', text.replace('model nep_spin', 'model nep_spin3'),
                    text.replace('spin_version 1', 'spin_version 2'),
                    text + 'spin_projection_size 36\n', text + 'spin_dof_type Fe\n',
                    text.replace('spin_basis_size 8', 'spin_basis_size 7'),
                    text.replace('spin_l_max 2', 'spin_l_max 5'),
                    text.replace('spin_order 2', 'spin_order 0'),
                    text.replace('spin_cutoff 6', 'spin_cutoff 7'),
                    text.replace('spin_type Fe', 'spin_type Fe Fe'),
                    text.replace('l_max 4 2 0', 'l_max 4 1 0')]
        for i, variant in enumerate(variants):
            with self.subTest(case=i):
                path = self.variant(variant)
                with self.assertRaises((ValueError, RuntimeError)):
                    load_model(BACKEND, str(path))

    def test_zbl_force_finite_difference(self):
        path = self.variant(self.path.read_text() + 'zbl 0.8 3.0\n')
        args = arguments(self.case)
        with load_model(BACKEND, str(path)) as model:
            base = model.calculate_spin(*args)
            step = 1e-5 if BACKEND == 'cpu' else 2e-3
            plus, minus = list(args), list(args)
            plus[2], minus[2] = args[2].copy(), args[2].copy()
            plus[2][0, 0] += step
            minus[2][0, 0] -= step
            fd = -(model.calculate_spin(*plus)[0].sum() - model.calculate_spin(*minus)[0].sum()) / (2 * step)
            self.assertLess(abs(fd - base[1][0, 0]), 1e-3)


if __name__ == '__main__':
    unittest.main()
