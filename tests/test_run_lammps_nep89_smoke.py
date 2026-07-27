"""Focused unit tests for the non-spin nep89 LAMMPS regression harness."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "run_lammps_nep89_smoke.py"
SPEC = importlib.util.spec_from_file_location("lammps_nep89_smoke", MODULE_PATH)
smoke = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(smoke)


class LammpsNep89SmokeTest(unittest.TestCase):

  def test_fcc_fixture_has_256_atoms_and_expected_box(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "data.nep89"
      atom_count = smoke.write_fcc_data(path, "Fe", 4, 3.6)
      contents = path.read_text(encoding="utf-8")

    self.assertEqual(atom_count, 256)
    self.assertIn("256 atoms", contents)
    self.assertIn("0.0 14.4 xlo xhi", contents)

  def test_gpu_input_uses_explicit_kokkos_pair(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "in.nep89"
      smoke.write_input(path, "/tmp/nep89.txt", "Fe", "nep/gpu/kk")
      contents = path.read_text(encoding="utf-8")

    self.assertIn("atom_style atomic/kk", contents)
    self.assertIn("run_style verlet/kk", contents)
    self.assertIn("pair_style nep/gpu/kk", contents)
    self.assertIn("pair_coeff * * /tmp/nep89.txt Fe", contents)

  def test_compare_reports_cpu_gpu_differences(self):
    columns = {
        "id": 1.0,
        "fx": 1.0,
        "fy": 2.0,
        "fz": 3.0,
        "c_eat": 4.0,
        **{f"c_satom[{index}]": float(index) for index in range(1, 7)},
    }
    gpu = dict(columns)
    gpu["fx"] += 0.25
    gpu["c_eat"] += 0.5
    gpu["c_satom[4]"] += 0.75

    result = smoke.compare(
        {1: columns},
        {"Step": 0.0, "Atoms": 1.0, "PotEng": 4.0},
        {1: gpu},
        {"Step": 0.0, "Atoms": 1.0, "PotEng": 4.125},
    )

    self.assertEqual(result["energy_diff"], 0.125)
    self.assertEqual(result["energy_relative_diff"], 0.125 / 4.125)
    self.assertEqual(result["max_force_component_diff"], 0.25)
    self.assertEqual(result["max_per_atom_energy_diff"], 0.5)
    self.assertEqual(result["max_stress_sum_component_diff"], 0.75)
    self.assertEqual(result["max_stress_sum_relative_diff"], 0.75 / 4.75)
    self.assertEqual(result["max_stress_sum_normalized_error"], 0.75 / 5.0)


if __name__ == "__main__":
  unittest.main()
