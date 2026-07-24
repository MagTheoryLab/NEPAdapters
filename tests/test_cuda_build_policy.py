from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CUDA_POLICY = ROOT / "cmake" / "NEPAdaptersCUDAAutoDetect.cmake"
CMAKE = shutil.which("cmake")


@unittest.skipUnless(CMAKE, "cmake is required")
class CUDAAutoDetectTests(unittest.TestCase):
    def run_policy(
        self,
        *,
        path: str,
        nep_cuda: str | None = None,
        cudacxx: str | None = None,
        cuda_root: str | None = None,
        python_enabled: bool = True,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            result_path = temporary_path / "result.txt"
            script_path = temporary_path / "probe.cmake"
            script_path.write_text(
                "\n".join(
                    [
                        f"set(NEP_ADAPTERS_ENABLE_PYTHON {'ON' if python_enabled else 'OFF'})",
                        f'include("{CUDA_POLICY.as_posix()}")',
                        "nep_adapters_resolve_cuda_default(_default)",
                        f'file(WRITE "{result_path.as_posix()}"',
                        '  "default=${_default}\\n"',
                        '  "compiler=${CMAKE_CUDA_COMPILER}\\n")',
                    ]
                ),
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PATH"] = path
            for name in (
                "NEP_CUDA",
                "CUDACXX",
                "CUDAToolkit_ROOT",
                "CUDA_TOOLKIT_ROOT_DIR",
                "CUDA_PATH",
                "CUDA_HOME",
            ):
                environment.pop(name, None)
            if nep_cuda is not None:
                environment["NEP_CUDA"] = nep_cuda
            if cudacxx is not None:
                environment["CUDACXX"] = cudacxx
            if cuda_root is not None:
                environment["CUDAToolkit_ROOT"] = cuda_root

            completed = subprocess.run(
                [CMAKE, "-P", str(script_path)],
                text=True,
                capture_output=True,
                env=environment,
                check=False,
            )
            values: dict[str, str] = {}
            if result_path.exists():
                for line in result_path.read_text(encoding="utf-8").splitlines():
                    key, value = line.split("=", 1)
                    values[key] = value
            return completed, values

    @staticmethod
    def make_fake_nvcc(directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        nvcc = directory / ("nvcc.exe" if os.name == "nt" else "nvcc")
        nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        nvcc.chmod(0o755)
        return nvcc

    def test_auto_enables_cuda_when_nvcc_is_on_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            nvcc = self.make_fake_nvcc(Path(temporary))
            completed, values = self.run_policy(path=str(nvcc.parent))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(values["default"], "ON")
        self.assertEqual(Path(values["compiler"]), nvcc)

    def test_auto_stays_cpu_only_without_nvcc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed, values = self.run_policy(path=temporary)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(values["default"], "OFF")
        self.assertEqual(values["compiler"], "")

    def test_cuda_root_is_searched_without_path_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nvcc = self.make_fake_nvcc(root / "bin")
            completed, values = self.run_policy(
                path=str(root / "empty"),
                cuda_root=str(root),
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(values["default"], "ON")
        self.assertEqual(Path(values["compiler"]), nvcc)

    def test_cudacxx_is_used_without_path_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nvcc = self.make_fake_nvcc(root / "toolkit")
            completed, values = self.run_policy(
                path=str(root / "empty"),
                cudacxx=str(nvcc),
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(values["default"], "ON")
        self.assertEqual(Path(values["compiler"]), nvcc)

    def test_nep_cuda_zero_forces_cpu_with_nvcc_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            nvcc = self.make_fake_nvcc(Path(temporary))
            completed, values = self.run_policy(
                path=str(nvcc.parent),
                nep_cuda="0",
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(values["default"], "OFF")
        self.assertEqual(values["compiler"], "")

    def test_non_python_cmake_build_does_not_auto_enable_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            nvcc = self.make_fake_nvcc(Path(temporary))
            completed, values = self.run_policy(
                path=str(nvcc.parent),
                python_enabled=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(values["default"], "OFF")
        self.assertEqual(values["compiler"], "")

    def test_nep_cuda_one_forces_cuda_without_nvcc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed, values = self.run_policy(
                path=temporary,
                nep_cuda="1",
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(values["default"], "ON")

    def test_invalid_nep_cuda_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed, _ = self.run_policy(
                path=temporary,
                nep_cuda="maybe",
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("NEP_CUDA must be one of", completed.stderr)


if __name__ == "__main__":
    unittest.main()
