"""Focused unit tests for the real LAMMPS baseline smoke harness."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "run_lammps_baseline_smoke.py"
SPEC = importlib.util.spec_from_file_location("lammps_baseline_smoke", MODULE_PATH)
smoke = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(smoke)


class LammpsBaselineSmokeTest(unittest.TestCase):

  def test_gpu_command_enables_one_kokkos_device(self):
    self.assertEqual(
        smoke.lammps_command("lmp", "nep/gpu"),
        [
            "lmp", "-k", "on", "g", "1",
            "-in", "in.baseline", "-log", "log.lammps",
        ],
    )

  def test_cpu_command_does_not_enable_kokkos(self):
    self.assertEqual(
        smoke.lammps_command("lmp", "nep/cpu"),
        ["lmp", "-in", "in.baseline", "-log", "log.lammps"],
    )

  def test_input_uses_requested_gpu_pair_style(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "in.baseline"
      smoke.write_input(path, "plugin.so", "nep.txt", ["Fe"], "nep/gpu")

      contents = path.read_text(encoding="utf-8")

    self.assertIn("pair_style nep/gpu", contents)
    self.assertIn("atom_style atomic/kk", contents)
    self.assertIn("run_style verlet/kk", contents)
    self.assertNotIn("pair_style nep/cpu", contents)
    self.assertNotIn("plugin load", contents)

  def test_command_load_mode_writes_explicit_plugin_command(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "in.baseline"
      smoke.write_input(
          path,
          "nepadapterscpuplugin.so",
          "nep.txt",
          ["Fe"],
          plugin_load_mode="command",
      )

      contents = path.read_text(encoding="utf-8")

    self.assertIn("plugin load nepadapterscpuplugin.so", contents)

  def test_plugin_environment_points_to_plugin_directory(self):
    env = smoke.plugin_environment("/opt/nepadapters/lib/nepadapterscpuplugin.so")

    self.assertEqual(env["LAMMPS_PLUGIN_PATH"], "/opt/nepadapters/lib")

  def test_plugin_environment_rejects_non_plugin_filename(self):
    with self.assertRaisesRegex(ValueError, "ending in 'plugin.so'"):
      smoke.plugin_environment("/opt/nepadapters/lib/nepadapters.so")


if __name__ == "__main__":
  unittest.main()
