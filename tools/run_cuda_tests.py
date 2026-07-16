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
    parser.add_argument("--build-dir", default=".build/cuda-tests")
    parser.add_argument(
        "--cuda-arch",
        default=os.environ.get("CMAKE_CUDA_ARCHITECTURES", "native"),
        help="CMAKE_CUDA_ARCHITECTURES value, e.g. native, 89, or 70",
    )
    parser.add_argument("--label", default="cuda")
    parser.add_argument(
        "--qnep-pppm",
        action="store_true",
        help="compile the optional qNEP PPPM path and link cuFFT",
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
            "-DNEP_ADAPTERS_ENABLE_CPU=ON",
            "-DNEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM="
            + ("ON" if args.qnep_pppm else "OFF"),
            f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}",
        ]
        run(configure_args, root)

    if not args.no_build:
        run(["cmake", "--build", build_dir, "-j", args.jobs], root)

    run(["ctest", "--test-dir", build_dir, "-L", args.label, "--output-on-failure"], root)


if __name__ == "__main__":
    main()
