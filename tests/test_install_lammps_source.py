"""Unit tests for the managed LAMMPS source-tree installer."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "install_lammps_source.py"
SPEC = importlib.util.spec_from_file_location("install_lammps_source", MODULE_PATH)
installer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(installer)


class InstallLammpsSourceTest(unittest.TestCase):

  def make_lammps_tree(self, parent: Path) -> Path:
    lammps = parent / "lammps"
    (lammps / "src").mkdir(parents=True)
    (lammps / "cmake").mkdir()
    (lammps / "src/pair.h").write_text("// pair\n", encoding="utf-8")
    (lammps / "cmake/CMakeLists.txt").write_text(
        "project(lammps)\n", encoding="utf-8"
    )
    return lammps

  def test_cpu_install_copies_only_cpu_pair(self):
    with tempfile.TemporaryDirectory() as directory:
      lammps = self.make_lammps_tree(Path(directory))
      installed = installer.install(lammps, "cpu")

      resolved = lammps.resolve()
      self.assertIn(resolved / "src/pair_nep_adapters_cpu.cpp", installed)
      self.assertTrue((resolved / installer.HOOK_RELATIVE).is_file())
      self.assertFalse((resolved / "src/pair_nep_adapters_cuda.cpp").exists())
      manifest = json.loads(
          (resolved / installer.MANIFEST_RELATIVE).read_text(encoding="utf-8")
      )
      self.assertEqual(manifest["backend"], "cpu")

  def test_reinstall_can_switch_from_both_to_cpu(self):
    with tempfile.TemporaryDirectory() as directory:
      lammps = self.make_lammps_tree(Path(directory))
      installer.install(lammps, "both")
      installer.install(lammps, "cpu")

      self.assertTrue((lammps / "src/pair_nep_adapters_cpu.cpp").is_file())
      self.assertFalse((lammps / "src/pair_nep_adapters_cuda.cpp").exists())

  def test_reinstall_refuses_modified_managed_file(self):
    with tempfile.TemporaryDirectory() as directory:
      lammps = self.make_lammps_tree(Path(directory))
      installer.install(lammps, "cpu")
      pair = lammps / "src/pair_nep_adapters_cpu.cpp"
      pair.write_text("// local edit\n", encoding="utf-8")

      with self.assertRaisesRegex(
          installer.InstallError, "locally modified managed file"
      ):
        installer.install(lammps, "cpu")

      self.assertEqual(pair.read_text(encoding="utf-8"), "// local edit\n")

  def test_install_refuses_unmanaged_existing_pair(self):
    with tempfile.TemporaryDirectory() as directory:
      lammps = self.make_lammps_tree(Path(directory))
      pair = lammps / "src/pair_nep_adapters_cpu.cpp"
      pair.write_bytes((installer.FRONTEND / pair.name).read_bytes())

      with self.assertRaisesRegex(installer.InstallError, "existing file"):
        installer.install(lammps, "cpu")

      self.assertTrue(pair.is_file())

  def test_uninstall_rejects_manifest_with_unmanaged_path(self):
    with tempfile.TemporaryDirectory() as directory:
      lammps = self.make_lammps_tree(Path(directory))
      keep = lammps / "keep.txt"
      keep.write_text("keep\n", encoding="utf-8")
      manifest = lammps / installer.MANIFEST_RELATIVE
      manifest.write_text(
          json.dumps(
              {
                  "format": 1,
                  "files": {"keep.txt": installer.sha256(keep)},
              }
          ),
          encoding="utf-8",
      )

      with self.assertRaisesRegex(installer.InstallError, "unmanaged paths"):
        installer.uninstall(lammps)

      self.assertTrue(keep.is_file())

  def test_uninstall_removes_only_managed_files(self):
    with tempfile.TemporaryDirectory() as directory:
      lammps = self.make_lammps_tree(Path(directory))
      keep = lammps / "src/user_pair.cpp"
      keep.write_text("// keep\n", encoding="utf-8")
      installer.install(lammps, "cpu")

      installer.uninstall(lammps)

      self.assertTrue(keep.is_file())
      self.assertFalse((lammps / "src/pair_nep_adapters_cpu.cpp").exists())
      self.assertFalse((lammps / installer.MANIFEST_RELATIVE).exists())


if __name__ == "__main__":
  unittest.main()
