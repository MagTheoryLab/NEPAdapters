import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "run_lammps_spin_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_lammps_spin_smoke", MODULE_PATH)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class LammpsSpinSmokeTest(unittest.TestCase):
  def setUp(self):
    self.structure = {
        "box": [8.0, 8.0, 8.0],
        "elements": ["Fe"],
        "atoms": [
            {"type": 1, "position": [1.0, 2.0, 3.0], "spin": [1.0, 2.0, 2.0]}
        ],
    }

  def test_cpu_input_uses_real_spin_atom_style_and_mforce_dump(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "in.spin"
      smoke.write_input(path, "cpu.so", "nep.txt", self.structure, "nep/cpu", "command")
      contents = path.read_text(encoding="utf-8")

    self.assertIn("atom_style spin\n", contents)
    self.assertNotIn("atom_style spin/kk", contents)
    self.assertIn("plugin load cpu.so", contents)
    self.assertIn("compute spin all property/atom spx spy spz sp fmx fmy fmz", contents)
    self.assertIn("pair_style nep/cpu", contents)

  def test_gpu_input_uses_kokkos_spin_atom_style(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "in.spin"
      smoke.write_input(path, "gpu.so", "nep.txt", self.structure, "nep/gpu/kk", "environment")
      contents = path.read_text(encoding="utf-8")

    self.assertIn("atom_style spin/kk", contents)
    self.assertIn("run_style verlet/kk", contents)
    self.assertIn("pair_style nep/gpu/kk", contents)
    self.assertIn("newton off", contents)
    self.assertNotIn("plugin load", contents)

  def test_cpu_input_writes_requested_newton_mode(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "in.spin"
      smoke.write_input(
          path,
          "cpu.so",
          "nep.txt",
          self.structure,
          "nep/cpu",
          "environment",
          newton="on",
      )
      contents = path.read_text(encoding="utf-8")

    self.assertIn("newton on\n", contents)
    self.assertNotIn("newton off\n", contents)

  def test_builtin_input_needs_no_plugin(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "in.spin"
      smoke.write_input(path, "", "nep.txt", self.structure, "nep/gpu/kk", "builtin")
      contents = path.read_text(encoding="utf-8")

    self.assertIn("pair_style nep/gpu/kk", contents)
    self.assertNotIn("plugin load", contents)

  def test_set_spin_preserves_magnitude_and_direction(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "in.spin"
      smoke.write_input(path, "cpu.so", "nep.txt", self.structure, "nep/cpu", "environment")
      set_line = next(
          line for line in path.read_text(encoding="utf-8").splitlines()
          if line.startswith("set atom 1 spin")
      )
    values = [float(value) for value in set_line.split()[4:]]
    self.assertAlmostEqual(values[0], 3.0)
    self.assertAlmostEqual(math.sqrt(sum(value * value for value in values[1:])), 1.0)

  def test_multi_rank_command_requires_mpiexec(self):
    with self.assertRaisesRegex(ValueError, "--mpiexec"):
      smoke.lammps_command("lmp", "nep/cpu", "in.spin", 2, "")

  def test_multi_rank_command_prefixes_launcher(self):
    command = smoke.lammps_command("lmp", "nep/cpu", "in.spin", 2, "mpirun")
    self.assertEqual(command[:4], ["mpirun", "-np", "2", "lmp"])

  def test_multi_rank_command_uses_configured_numproc_flag(self):
    command = smoke.lammps_command(
        "lmp", "nep/cpu", "in.spin", 4, "mpiexec", "-n"
    )
    self.assertEqual(command[:4], ["mpiexec", "-n", "4", "lmp"])

  def test_mpi_processor_counts_reads_lammps_grid(self):
    screen = """
LAMMPS (22 Jul 2025)
  1 by 2 by 4 MPI processor grid
  1 by 2 by 4 MPI processor grid
"""
    self.assertEqual(smoke.mpi_processor_counts(screen), [8, 8])

  def test_committed_structure_is_valid(self):
    path = ROOT / "tests/fixtures/spin_chiral_protocol/lammps_structure.json"
    structure = smoke.read_structure(path)
    self.assertEqual(len(structure["atoms"]), 4)
    self.assertEqual(structure["elements"], ["Fe"])
    self.assertEqual(structure["box"], [4.0, 4.0, 4.0])


if __name__ == "__main__":
  unittest.main()
