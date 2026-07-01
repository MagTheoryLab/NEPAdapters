#!/usr/bin/env python3
"""Build, test, benchmark, and write a compact NEPAdapters report."""

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def run_command(args, cwd, check=True, env=None):
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and completed.returncode != 0:
        print(completed.stdout)
        raise SystemExit(completed.returncode)
    return completed


def parse_junit(path):
    if not path.exists():
        return []

    root = ET.parse(path).getroot()
    tests = []
    for testcase in root.iter("testcase"):
        status = "passed"
        if testcase.find("failure") is not None or testcase.find("error") is not None:
            status = "failed"
        elif testcase.find("skipped") is not None:
            status = "skipped"
        tests.append(
            {
                "name": testcase.attrib.get("name", ""),
                "time": float(testcase.attrib.get("time", "0") or 0),
                "status": status,
            }
        )
    return tests


def ctest_labels(build_dir, cwd):
    completed = run_command(
        ["ctest", "--test-dir", str(build_dir), "--show-only=json-v1"],
        cwd=cwd,
    )
    labels = {}
    data = json.loads(completed.stdout)
    for test in data.get("tests", []):
        test_labels = []
        for prop in test.get("properties", []):
            if prop.get("name") == "LABELS":
                test_labels = list(prop.get("value", []))
        labels[test.get("name", "")] = test_labels
    return labels


def parse_json_line(output):
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("benchmark did not print a JSON line")


def benchmark_once(build_dir, cwd, scale, iterations, warmup, openmp_threads):
    executable = build_dir / "benchmarks" / "nep_adapters_bench_cpu_nep89"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(openmp_threads)
    completed = run_command(
        [
            str(executable),
            "--iterations",
            str(iterations),
            "--warmup",
            str(warmup),
            "--replicate",
            scale,
        ],
        cwd=cwd,
        env=env,
    )
    result = parse_json_line(completed.stdout)
    if not result.get("openmp_enabled"):
        raise RuntimeError("benchmark binary was not built with OpenMP")
    result["requested_omp_threads"] = openmp_threads
    result["parallel_mode"] = f"openmp_threads_{result.get('omp_threads', openmp_threads)}"
    return result


def benchmark_sweep(build_dir, cwd, scales, iterations, warmup, openmp_threads):
    results = []
    for scale in scales:
        results.append(
            benchmark_once(
                build_dir, cwd, scale, iterations, warmup, openmp_threads
            )
        )
    return results


def auto_scale_sweep(
    build_dir,
    cwd,
    candidates,
    iterations,
    warmup,
    openmp_threads,
    plateau_growth_ratio,
    min_saturation_atoms,
):
    results = []
    best_throughput = 0.0
    fixed_scale = None
    saturated = False
    for scale in candidates:
        result = benchmark_once(
            build_dir, cwd, scale, iterations, warmup, openmp_threads
        )
        throughput = result["atom_steps_per_second"]
        if throughput > best_throughput:
            best_throughput = throughput
        results.append(result)
        previous = results[-2] if len(results) >= 2 else None
        if (
            len(results) >= 3
            and result["atoms"] >= min_saturation_atoms
            and previous is not None
            and throughput <= previous["atom_steps_per_second"] * plateau_growth_ratio
        ):
            saturated = True
            fixed_scale = results[-2]["replicate"]
            break

    if fixed_scale is None and results:
        fixed_scale = results[-1]["replicate"]
    return results, fixed_scale, saturated


def label_text(labels):
    return ", ".join(labels) if labels else "-"


def write_report(
    path,
    build_dir,
    correctness_tests,
    bench_smoke_tests,
    labels,
    bench_results,
    fixed_scale,
    saturated,
    conditions_path,
    commands,
    lammps_baseline_payload,
    lammps_mpi_payload,
):
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    passed = sum(1 for test in correctness_tests if test["status"] == "passed")
    failed = sum(1 for test in correctness_tests if test["status"] == "failed")
    skipped = sum(1 for test in correctness_tests if test["status"] == "skipped")

    bench_passed = sum(1 for test in bench_smoke_tests if test["status"] == "passed")
    bench_failed = sum(1 for test in bench_smoke_tests if test["status"] == "failed")

    lines = [
        "# NEPAdapters Test Report",
        "",
        f"- generated_at: `{now}`",
        f"- host: `{platform.platform()}`",
        f"- python: `{platform.python_version()}`",
        f"- build_dir: `{build_dir}`",
        f"- performance_conditions: `{conditions_path}`",
        "",
        "## Summary",
        "",
        f"- correctness: `{passed} passed, {failed} failed, {skipped} skipped`",
        f"- benchmark smoke: `{bench_passed} passed, {bench_failed} failed`",
        f"- atom-scaling points: `{len(bench_results)}`",
        "",
        "## Correctness",
        "",
        "| Test | Labels | Status | Time (s) |",
        "| --- | --- | --- | ---: |",
    ]

    for test in correctness_tests:
        lines.append(
            f"| `{test['name']}` | `{label_text(labels.get(test['name'], []))}` "
            f"| `{test['status']}` | {test['time']:.3f} |"
        )

    lines.extend(
        [
            "",
            "Parity tests compare the adapter path against direct NEP CPU calls for "
            "energy, force, and summed virial. A passing parity row means the current "
            "adapter-owned type mapping, AoS/SoA conversion, batch offsets, and virial "
            "reduction match the oracle within the test tolerance.",
            "",
            "## Benchmark Smoke",
            "",
            "| Test | Labels | Status | Time (s) |",
            "| --- | --- | --- | ---: |",
        ]
    )

    for test in bench_smoke_tests:
        lines.append(
            f"| `{test['name']}` | `{label_text(labels.get(test['name'], []))}` "
            f"| `{test['status']}` | {test['time']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Atom Scaling",
            "",
            "This section measures one model instance using OpenMP inside the NEP CPU kernels.",
            "",
            "| Engine | Model | Replicate | Atoms | OpenMP Threads | Mode | Iterations | Seconds | Eval/s | Atom-steps/s |",
            "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for result in bench_results:
        lines.append(
            f"| `{result['engine']}` | `{result['model']}` | `{result['replicate']}` "
            f"| {result['atoms']} | {result.get('omp_threads', 0)} "
            f"| `{result.get('parallel_mode', '-')}` | {result['iterations']} "
            f"| {result['seconds']:.6f} "
            f"| {result['evals_per_second']:.3f} "
            f"| {result['atom_steps_per_second']:.3f} |"
        )

    if bench_results:
        best = max(bench_results, key=lambda item: item["atom_steps_per_second"])
        largest = max(bench_results, key=lambda item: item["atoms"])
        lines.extend(
            [
                "",
                "## Readout",
                "",
                f"- best atom throughput in this sweep: `{best['atom_steps_per_second']:.3f}` "
                f"atom-steps/s at `{best['atoms']}` atoms.",
                f"- largest tested system: `{largest['atoms']}` atoms with "
                f"`{largest['evals_per_second']:.3f}` eval/s.",
                f"- fixed future test scale: `{fixed_scale}` "
                f"({'saturation detected' if saturated else 'scan cap reached before saturation'}).",
                "- no performance pass/fail threshold is applied yet; this report records "
                "baseline evidence for later CPU/GPU/LAMMPS comparisons.",
            ]
        )

    lines.extend(["", "## LAMMPS Baseline Smoke", ""])
    if lammps_baseline_payload is None:
        lines.append(
            "LAMMPS baseline smoke was not run. Pass `--lmp-executable` to "
            "compare the plugin against the committed golden-label fixture."
        )
    else:
        result = lammps_baseline_payload["result"]
        lines.extend(
            [
                "| Quantity | Diff |",
                "| --- | ---: |",
                f"| energy | {result['energy_diff']:.3e} |",
                f"| per-atom energy sum | {result['eatom_sum_diff']:.3e} |",
                f"| max force component | {result['max_force_component_diff']:.3e} |",
                f"| max virial component | {result['max_virial_component_diff']:.3e} |",
            ]
        )

    lines.extend(["", "## LAMMPS MPI Smoke", ""])
    if lammps_mpi_payload is None:
        lines.append(
            "LAMMPS MPI smoke was not run. Pass `--lmp-executable` to include "
            "`mpirun -np 1/2/4` correctness evidence."
        )
    else:
        lines.extend(
            [
                "| Case | MPI ranks | Force diff | Eatom diff | Stress diff | PE diff | Press diff |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for case in lammps_mpi_payload["cases"]:
            for row in case["comparisons"]:
                lines.append(
                    f"| `{case['name']}` | {row['np']} "
                    f"| {row['max_force_component_diff']:.3e} "
                    f"| {row['max_eatom_diff']:.3e} "
                    f"| {row['max_stress_component_diff']:.3e} "
                    f"| {row['pe_diff']:.3e} "
                    f"| {row['max_press_component_diff']:.3e} |"
                )

    lines.extend(["", "## Commands", ""])
    for command in commands:
        lines.extend(["```sh", command, "```", ""])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_conditions(path, bench_results, fixed_scale, saturated, args):
    selected = None
    for result in bench_results:
        if result.get("replicate") == fixed_scale:
            selected = result
            break
    payload = {
        "benchmark": "cpu_nep3_nep89_find_force_batch",
        "fixed_scale": fixed_scale,
        "saturated": saturated,
        "openmp_threads": args.openmp_threads,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "plateau_growth_ratio": args.plateau_growth_ratio,
        "min_saturation_atoms": args.min_saturation_atoms,
        "selected_result": selected,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build-report")
    parser.add_argument("--output", default="reports/nep_adapters_report.md")
    parser.add_argument("--conditions-output", default="reports/performance_conditions.json")
    parser.add_argument(
        "--python-executable",
        default="/Users/superbing/miniconda3/envs/mysci/bin/python",
    )
    parser.add_argument("--lammps-source-dir", default="")
    parser.add_argument("--lmp-executable", default="")
    parser.add_argument("--lammps-plugin", default="")
    parser.add_argument("--mpirun", default="/opt/homebrew/bin/mpirun")
    parser.add_argument("--jobs", default="2")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--openmp-threads", type=int, default=4)
    parser.add_argument("--scales", default="auto")
    parser.add_argument(
        "--auto-scales",
        default="1x1x1,2x1x1,2x2x1,2x2x2,3x3x3,4x4x4",
    )
    parser.add_argument("--plateau-growth-ratio", type=float, default=1.05)
    parser.add_argument("--min-saturation-atoms", type=int, default=2000)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    build_dir = (repo / args.build_dir).resolve()
    output = (repo / args.output).resolve()
    conditions_output = (repo / args.conditions_output).resolve()
    report_dir = output.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        f"cmake -S . -B {build_dir} -DBUILD_SHARED_LIBS=ON -DNEP_ADAPTERS_BUILD_TESTS=ON -DNEP_ADAPTERS_BUILD_BENCHMARKS=ON -DNEP_ADAPTERS_ENABLE_PYTHON=ON -DNEP_ADAPTERS_ENABLE_LAMMPS=ON -DPython3_EXECUTABLE={args.python_executable} -DNEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR={repo / 'tests' / 'fixtures' / 'cpu_nep3_baseline'}",
        f"cmake --build {build_dir} -j{args.jobs}",
        f"ctest --test-dir {build_dir} -LE bench --output-on-failure",
        f"ctest --test-dir {build_dir} -L bench --output-on-failure",
    ]

    configure_args = [
        "cmake",
        "-S",
        str(repo),
        "-B",
        str(build_dir),
        "-DBUILD_SHARED_LIBS=ON",
        "-DNEP_ADAPTERS_BUILD_TESTS=ON",
        "-DNEP_ADAPTERS_BUILD_BENCHMARKS=ON",
        "-DNEP_ADAPTERS_ENABLE_PYTHON=ON",
        "-DNEP_ADAPTERS_ENABLE_LAMMPS=ON",
        f"-DPython3_EXECUTABLE={args.python_executable}",
        f"-DNEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR={repo / 'tests' / 'fixtures' / 'cpu_nep3_baseline'}",
    ]
    if args.lmp_executable:
        lmp_path = Path(args.lmp_executable).resolve()
        configure_args.append(f"-DNEP_ADAPTERS_LAMMPS_EXECUTABLE={lmp_path}")
        commands[0] += f" -DNEP_ADAPTERS_LAMMPS_EXECUTABLE={lmp_path}"
    configure_args.append("-DNEP_ADAPTERS_CPU_NEP3_ENABLE_OPENMP=ON")
    commands[0] += " -DNEP_ADAPTERS_CPU_NEP3_ENABLE_OPENMP=ON"

    libomp_env = os.environ.get("LIBOMP_PREFIX", "")
    libomp_prefix = Path(libomp_env) if libomp_env else Path()
    if not libomp_env or not libomp_prefix.exists():
        try:
            probe = subprocess.run(
                ["brew", "--prefix", "libomp"],
                cwd=repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if probe.returncode == 0:
                libomp_prefix = Path(probe.stdout.strip())
        except OSError:
            libomp_prefix = Path("")
    if libomp_prefix.exists():
        include_dir = libomp_prefix / "include"
        lib_path = libomp_prefix / "lib" / "libomp.dylib"
        if include_dir.exists() and lib_path.exists():
            openmp_flags = f"-Xpreprocessor -fopenmp -I{include_dir}"
            configure_args.extend(
                [
                    f"-DOpenMP_CXX_FLAGS={openmp_flags}",
                    "-DOpenMP_CXX_LIB_NAMES=omp",
                    f"-DOpenMP_omp_LIBRARY={lib_path}",
                ]
            )
            commands[0] += (
                f" -DOpenMP_CXX_FLAGS='{openmp_flags}'"
                " -DOpenMP_CXX_LIB_NAMES=omp"
                f" -DOpenMP_omp_LIBRARY={lib_path}"
            )
    if args.lammps_source_dir:
        configure_args.append(f"-DNEP_ADAPTERS_LAMMPS_SOURCE_DIR={args.lammps_source_dir}")
        commands[0] += f" -DNEP_ADAPTERS_LAMMPS_SOURCE_DIR={args.lammps_source_dir}"

    run_command(
        configure_args,
        cwd=repo,
    )
    run_command(["cmake", "--build", str(build_dir), f"-j{args.jobs}"], cwd=repo)

    labels = ctest_labels(build_dir, repo)

    correctness_xml = report_dir / "correctness.xml"
    bench_xml = report_dir / "benchmark_smoke.xml"
    correctness = run_command(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "-LE",
            "bench",
            "--output-on-failure",
            "--output-junit",
            str(correctness_xml),
        ],
        cwd=repo,
        check=False,
    )
    bench_smoke = run_command(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "-L",
            "bench",
            "--output-on-failure",
            "--output-junit",
            str(bench_xml),
        ],
        cwd=repo,
        check=False,
    )

    if args.scales == "auto":
        candidates = [item.strip() for item in args.auto_scales.split(",") if item.strip()]
        bench_results, fixed_scale, saturated = auto_scale_sweep(
            build_dir,
            repo,
            candidates,
            args.iterations,
            args.warmup,
            args.openmp_threads,
            args.plateau_growth_ratio,
            args.min_saturation_atoms,
        )
    else:
        scales = [item.strip() for item in args.scales.split(",") if item.strip()]
        bench_results = benchmark_sweep(
            build_dir,
            repo,
            scales,
            args.iterations,
            args.warmup,
            args.openmp_threads,
        )
        fixed_scale = bench_results[-1]["replicate"] if bench_results else "none"
        saturated = False

    lammps_baseline_payload = None
    lammps_mpi_payload = None
    if args.lmp_executable:
        plugin_path = (
            Path(args.lammps_plugin).resolve()
            if args.lammps_plugin
            else build_dir / "frontends" / "lammps" / "nepadaptersplugin.so"
        )
        model_env = os.environ.get("NEP_ADAPTERS_NEP89_MODEL_PATH", "")
        model_path = Path(model_env) if model_env else Path()
        if not model_env or not model_path.exists():
            model_path = (repo / "tests/fixtures/cpu_nep3_baseline/nep.txt").resolve()
        fixture_path = (repo / "tests/fixtures/cpu_nep3_baseline/train.xyz").resolve()
        baseline_json = report_dir / "lammps_baseline_smoke.json"
        baseline_md = report_dir / "lammps_baseline_smoke.md"
        baseline_command = [
            args.python_executable,
            str(repo / "tools" / "run_lammps_baseline_smoke.py"),
            "--lmp",
            str(Path(args.lmp_executable).resolve()),
            "--plugin",
            str(plugin_path),
            "--model",
            str(model_path.resolve()),
            "--fixture",
            str(fixture_path),
            "--work-dir",
            str(build_dir / "lammps_baseline_smoke"),
            "--json-output",
            str(baseline_json),
            "--markdown-output",
            str(baseline_md),
        ]
        commands.append(" ".join(baseline_command))
        run_command(baseline_command, cwd=repo)
        lammps_baseline_payload = json.loads(
            baseline_json.read_text(encoding="utf-8")
        )

        lammps_json = report_dir / "lammps_mpi_smoke.json"
        lammps_md = report_dir / "lammps_mpi_smoke.md"
        command = [
            args.python_executable,
            str(repo / "tools" / "run_lammps_mpi_smoke.py"),
            "--lmp",
            str(Path(args.lmp_executable).resolve()),
            "--plugin",
            str(plugin_path),
            "--model",
            str(model_path.resolve()),
            "--mpirun",
            args.mpirun,
            "--work-dir",
            str(build_dir / "lammps_mpi_smoke"),
            "--json-output",
            str(lammps_json),
            "--markdown-output",
            str(lammps_md),
        ]
        commands.append(" ".join(command))
        run_command(command, cwd=repo)
        lammps_mpi_payload = json.loads(lammps_json.read_text(encoding="utf-8"))

    write_report(
        output,
        build_dir,
        parse_junit(correctness_xml),
        parse_junit(bench_xml),
        labels,
        bench_results,
        fixed_scale,
        saturated,
        conditions_output,
        commands,
        lammps_baseline_payload,
        lammps_mpi_payload,
    )
    write_conditions(conditions_output, bench_results, fixed_scale, saturated, args)

    print(f"Wrote {output}")
    if correctness.returncode != 0 or bench_smoke.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
