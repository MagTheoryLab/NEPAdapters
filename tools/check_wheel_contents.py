#!/usr/bin/env python3
"""Fail closed when a release wheel contains unexpected native libraries."""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

from normalize_macos_openmp import verify_tree

NATIVE_SUFFIXES = (".so", ".pyd", ".dylib", ".dll")
FORBIDDEN_NAMES = (
    "libcuda",
    "libcudart",
    "libcufft",
    "libcublas",
    "libcusolver",
    "libcurand",
)
ALLOWED_OPENMP_RUNTIME_NAMES = (
    "libgomp",
    "libiomp",
    "libomp",
    "vcomp",
)


def backend_name(member: str) -> str | None:
    name = PurePosixPath(member).name
    if name.startswith("nep_cpu.") and name.endswith(NATIVE_SUFFIXES):
        return "nep_cpu"
    if name.startswith("nep_gpu.") and name.endswith(NATIVE_SUFFIXES):
        return "nep_gpu"
    return None


def inspect_wheel(path: Path, variant: str) -> None:
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise AssertionError(f"{path.name}: corrupt member {corrupt}")
        members = archive.namelist()
        lower_members = [member.lower() for member in members]

        for forbidden in FORBIDDEN_NAMES:
            if any(forbidden in member for member in lower_members):
                raise AssertionError(f"{path.name}: bundled forbidden library {forbidden}")

        native_members = [
            member for member in members if member.lower().endswith(NATIVE_SUFFIXES)
        ]
        backend_members = [
            member for member in native_members if backend_name(member) is not None
        ]
        runtime_members = [
            member for member in native_members if backend_name(member) is None
        ]
        unexpected_runtime_members = [
            member
            for member in runtime_members
            if not PurePosixPath(member).name.lower().startswith(
                ALLOWED_OPENMP_RUNTIME_NAMES
            )
        ]
        if unexpected_runtime_members:
            raise AssertionError(
                f"{path.name}: bundled unexpected native libraries "
                f"{unexpected_runtime_members!r}"
            )
        backends = [backend_name(member) for member in backend_members]
        expected = ["nep_cpu"] if variant == "cpu" else ["nep_cpu", "nep_gpu"]
        if sorted(backends) != expected:
            raise AssertionError(
                f"{path.name}: native members {native_members!r}, expected {expected!r}"
            )

        metadata_members = [
            member for member in members if member.endswith(".dist-info/METADATA")
        ]
        if len(metadata_members) != 1:
            raise AssertionError(f"{path.name}: expected one METADATA file")
        metadata = Parser().parsestr(
            archive.read(metadata_members[0]).decode("utf-8")
        )
        if metadata["Name"] != "nep-adapters":
            raise AssertionError(f"{path.name}: unexpected distribution name")
        requirements = metadata.get_all("Requires-Dist", [])
        if not any(requirement.lower().startswith("numpy") for requirement in requirements):
            raise AssertionError(f"{path.name}: NumPy runtime dependency is missing")

    if sys.platform == "darwin":
        with tempfile.TemporaryDirectory(prefix="nep-adapters-wheel-check-") as tmp_dir:
            root = Path(tmp_dir)
            with zipfile.ZipFile(path) as archive:
                archive.extractall(root)
            verify_tree(root)

    print(
        f"{path.name}: variant={variant} native={','.join(expected)} "
        f"openmp_runtime={','.join(PurePosixPath(member).name for member in runtime_members) or 'system'} "
        f"size={path.stat().st_size}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("cpu", "combined"), required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    args = parser.parse_args()
    wheels = sorted(args.wheel_dir.rglob("*.whl"))
    if not wheels:
        raise SystemExit(f"no wheels found under {args.wheel_dir}")
    for wheel in wheels:
        inspect_wheel(wheel, args.variant)


if __name__ == "__main__":
    main()
