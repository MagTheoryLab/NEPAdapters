#!/usr/bin/env python3
"""Build and run CUDA backend correctness tests."""

import argparse
import os
import subprocess
from pathlib import Path


def run(args, cwd):
    print("+", " ".join(str(arg) for arg in args))
    subprocess.run(args, cwd=cwd, check=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build-cuda-tests")
    parser.add_argument(
        "--cuda-arch",
        default=os.environ.get("CMAKE_CUDA_ARCHITECTURES", "native"),
        help="CMAKE_CUDA_ARCHITECTURES value, e.g. native, 89, or 70",
    )
    parser.add_argument("--label", default="cuda")
    parser.add_argument(
        "--cpu-nep3-source-dir",
        default=os.environ.get("NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR", ""),
    )
    parser.add_argument("-j", "--jobs", default=str(os.cpu_count() or 2))
    parser.add_argument("--no-configure", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    build_dir = root / args.build_dir
    if not args.no_configure:
        configure_args = [
            "cmake",
            "-S",
            root,
            "-B",
            build_dir,
            "-DNEP_ADAPTERS_BUILD_TESTS=ON",
            "-DNEP_ADAPTERS_ENABLE_CUDA=ON",
            "-DNEP_ADAPTERS_CUDA_ENABLE_DEVICE_RUNTIME=ON",
            "-DNEP_ADAPTERS_ENABLE_CPU_NEP3=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}",
        ]
        if args.cpu_nep3_source_dir:
            configure_args.append(
                f"-DNEP_ADAPTERS_CPU_NEP3_SOURCE_DIR={args.cpu_nep3_source_dir}"
            )
        run(configure_args, root)

    if not args.no_build:
        run(["cmake", "--build", build_dir, "-j", args.jobs], root)

    run(["ctest", "--test-dir", build_dir, "-L", args.label, "--output-on-failure"], root)


if __name__ == "__main__":
    main()
