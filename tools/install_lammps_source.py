#!/usr/bin/env python3
"""Install or remove the NEPAdapters pair sources in a LAMMPS source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontends" / "lammps"
HOOK_SOURCE = ROOT / "cmake" / "NEPAdaptersLAMMPSSource.cmake"
HOOK_RELATIVE = Path("cmake/Modules/NEPAdaptersLAMMPSSource.cmake")
MANIFEST_RELATIVE = Path("src/.nep_adapters_source_manifest.json")

COMMON_FILES = (
    "pair_nep_adapters_common.cpp",
    "pair_nep_adapters_common.h",
)
CPU_FILES = (
    "pair_nep_adapters_cpu.cpp",
    "pair_nep_adapters_cpu.h",
)
CUDA_FILES = (
    "kokkos_view_strides.hpp",
    "pair_nep_adapters_cuda.cpp",
    "pair_nep_adapters_cuda.h",
)
MANAGED_RELATIVES = {
    str(Path("src") / name)
    for name in COMMON_FILES + CPU_FILES + CUDA_FILES
}
MANAGED_RELATIVES.add(str(HOOK_RELATIVE))


class InstallError(RuntimeError):
  """Raised when a source-tree install cannot be completed safely."""


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def validate_lammps_tree(lammps: Path) -> Path:
  resolved = lammps.expanduser().resolve()
  required = (resolved / "src/pair.h", resolved / "cmake/CMakeLists.txt")
  if not all(path.is_file() for path in required):
    raise InstallError(
        f"{resolved} is not a LAMMPS source tree containing "
        "src/pair.h and cmake/CMakeLists.txt"
    )
  return resolved


def selected_sources(backend: str) -> dict[Path, Path]:
  names = list(COMMON_FILES)
  if backend in ("cpu", "both"):
    names.extend(CPU_FILES)
  if backend in ("cuda", "both"):
    names.extend(CUDA_FILES)
  result = {Path("src") / name: FRONTEND / name for name in names}
  result[HOOK_RELATIVE] = HOOK_SOURCE
  missing = [str(path) for path in result.values() if not path.is_file()]
  if missing:
    raise InstallError("NEPAdapters source files are missing: " + ", ".join(missing))
  return result


def load_manifest(lammps: Path) -> dict | None:
  path = lammps / MANIFEST_RELATIVE
  if not path.exists():
    return None
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise InstallError(f"failed to read {path}: {error}") from error
  if data.get("format") != 1 or not isinstance(data.get("files"), dict):
    raise InstallError(f"unsupported NEPAdapters source manifest: {path}")
  unknown = sorted(set(data["files"]) - MANAGED_RELATIVES)
  if unknown:
    raise InstallError(
        "source manifest contains unmanaged paths: " + ", ".join(unknown)
    )
  return data


def verify_managed_file(lammps: Path, relative: str, expected_hash: str) -> None:
  path = lammps / relative
  if not path.is_file():
    raise InstallError(f"managed file is missing: {path}")
  if sha256(path) != expected_hash:
    raise InstallError(
        f"refusing to overwrite locally modified managed file: {path}"
    )


def write_manifest(lammps: Path, data: dict) -> None:
  path = lammps / MANIFEST_RELATIVE
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
      mode="w",
      encoding="utf-8",
      dir=path.parent,
      prefix=path.name + ".",
      delete=False,
  ) as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
  temporary.replace(path)


def install(lammps_source: Path, backend: str) -> list[Path]:
  lammps = validate_lammps_tree(lammps_source)
  sources = selected_sources(backend)
  previous = load_manifest(lammps)
  previous_files = previous["files"] if previous else {}

  desired = {str(relative): sha256(source) for relative, source in sources.items()}
  stale = set(previous_files) - set(desired)

  # Preflight every mutation so a conflict cannot leave a half-updated tree.
  for relative, expected_hash in previous_files.items():
    if relative in desired or relative in stale:
      verify_managed_file(lammps, relative, expected_hash)
  for relative, source in sources.items():
    destination = lammps / relative
    if destination.exists() and str(relative) not in previous_files:
      raise InstallError(f"refusing to overwrite existing file: {destination}")

  for relative in sorted(stale):
    (lammps / relative).unlink()

  installed = []
  for relative, source in sources.items():
    destination = lammps / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    installed.append(destination)

  write_manifest(
      lammps,
      {
          "format": 1,
          "backend": backend,
          "nep_adapters_source": str(ROOT),
          "files": desired,
      },
  )
  return installed


def uninstall(lammps_source: Path) -> list[Path]:
  lammps = validate_lammps_tree(lammps_source)
  manifest = load_manifest(lammps)
  if manifest is None:
    raise InstallError(f"no NEPAdapters source install found in {lammps}")

  for relative, expected_hash in manifest["files"].items():
    verify_managed_file(lammps, relative, expected_hash)

  removed = []
  for relative in manifest["files"]:
    path = lammps / relative
    path.unlink()
    removed.append(path)
  manifest_path = lammps / MANIFEST_RELATIVE
  manifest_path.unlink()
  removed.append(manifest_path)
  return removed


def configure_command(lammps: Path, backend: str) -> str:
  hook = lammps / HOOK_RELATIVE
  build = lammps / f".build/nep-adapters-{backend}"
  arguments = [
      "cmake",
      "-S", str(lammps / "cmake"),
      "-B", str(build),
      "-DCMAKE_BUILD_TYPE=Release",
      f"-DCMAKE_PROJECT_INCLUDE={hook}",
      f"-DNEP_ADAPTERS_SOURCE_DIR={ROOT}",
      f"-DNEP_ADAPTERS_LAMMPS_SOURCE_BACKEND={backend}",
  ]
  if backend in ("cuda", "both"):
    arguments.extend((
        "-DPKG_KOKKOS=ON",
        "-DKokkos_ENABLE_CUDA=ON",
    ))
  separator = " \\" + "\n  "
  return separator.join(shlex.quote(argument) for argument in arguments)


def parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Install NEPAdapters pair styles into a LAMMPS source tree"
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  install_parser = subparsers.add_parser("install")
  install_parser.add_argument("--lammps-source", type=Path, required=True)
  install_parser.add_argument(
      "--backend", choices=("cpu", "cuda", "both"), default="cpu"
  )

  uninstall_parser = subparsers.add_parser("uninstall")
  uninstall_parser.add_argument("--lammps-source", type=Path, required=True)
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(sys.argv[1:] if argv is None else argv)
  try:
    if args.command == "install":
      lammps = validate_lammps_tree(args.lammps_source)
      installed = install(lammps, args.backend)
      print(f"Installed {len(installed)} files into {lammps}")
      print("\nConfigure the embedded LAMMPS build with:\n")
      print(configure_command(lammps, args.backend))
    else:
      removed = uninstall(args.lammps_source)
      print(f"Removed {len(removed)} managed files")
  except InstallError as error:
    print(f"error: {error}", file=sys.stderr)
    return 2
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
