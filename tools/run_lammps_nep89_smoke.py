#!/usr/bin/env python3
"""Compare a dense non-spin nep89 case through CPU and Kokkos CUDA LAMMPS."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_lammps_baseline_smoke import (
    parse_dump,
    parse_thermo,
    plugin_environment,
    read_type_map,
    run_command,
    validate_startup_banner,
)


def write_fcc_data(path, element, cells, lattice_constant):
    basis = (
        (0.0, 0.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    )
    box_length = cells * lattice_constant
    atoms = []
    for ix in range(cells):
        for iy in range(cells):
            for iz in range(cells):
                for bx, by, bz in basis:
                    atoms.append(
                        (
                            (ix + bx) * lattice_constant,
                            (iy + by) * lattice_constant,
                            (iz + bz) * lattice_constant,
                        )
                    )
    lines = [
        f"NEPAdapters nep89 dense FCC {element}",
        "",
        f"{len(atoms)} atoms",
        "1 atom types",
        "",
        f"0.0 {box_length:.17g} xlo xhi",
        f"0.0 {box_length:.17g} ylo yhi",
        f"0.0 {box_length:.17g} zlo zhi",
        "",
        "Masses",
        "",
        "1 1.0",
        "",
        "Atoms # atomic",
        "",
    ]
    for atom_id, (x, y, z) in enumerate(atoms, start=1):
        lines.append(f"{atom_id} 1 {x:.17g} {y:.17g} {z:.17g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(atoms)


def write_input(path, model, element, pair_style):
    is_gpu = pair_style.startswith("nep/gpu")
    atom_style = "atomic/kk" if is_gpu else "atomic"
    run_style = "verlet/kk" if is_gpu else "verlet"
    stress_cols = " ".join(f"c_satom[{index}]" for index in range(1, 7))
    path.write_text(
        "\n".join(
            (
                "clear",
                "units metal",
                f"atom_style {atom_style}",
                "boundary p p p",
                "read_data data.nep89",
                f"run_style {run_style}",
                f"pair_style {pair_style}",
                f"pair_coeff * * {model} {element}",
                "neighbor 1.0 bin",
                "neigh_modify every 1 delay 0 check yes",
                "compute eat all pe/atom",
                "compute satom all stress/atom NULL",
                "thermo 1",
                "thermo_style custom step atoms pe",
                "thermo_modify format float %.17g",
                f"dump d all custom 1 dump.out id fx fy fz c_eat {stress_cols}",
                "dump_modify d sort id format float %.17g",
                "run 0",
                "",
            )
        ),
        encoding="utf-8",
    )


def run_case(lmp, plugin, model, element, pair_style, work_dir, data_text):
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "data.nep89").write_text(data_text, encoding="utf-8")
    write_input(work_dir / "in.nep89", model, element, pair_style)
    command = [lmp]
    if pair_style.startswith("nep/gpu"):
        command.extend(("-k", "on", "g", "1"))
    command.extend(("-in", "in.nep89", "-log", "log.lammps"))
    screen = run_command(
        command,
        work_dir,
        env=plugin_environment(plugin),
    )
    banner_style = "nep/gpu" if pair_style.startswith("nep/gpu") else pair_style
    validate_startup_banner(screen, banner_style)
    (work_dir / "screen.out").write_text(screen, encoding="utf-8")
    return parse_dump(work_dir / "dump.out"), parse_thermo(
        work_dir / "log.lammps"
    )


def compare(
    cpu_rows,
    cpu_thermo,
    gpu_rows,
    gpu_thermo,
    stress_absolute_tolerance=5.0,
    stress_relative_tolerance=5.0e-6,
):
    if cpu_rows.keys() != gpu_rows.keys():
        raise RuntimeError("CPU and GPU LAMMPS atom ids differ")
    max_force_diff = 0.0
    max_eatom_diff = 0.0
    cpu_stress = [0.0] * 6
    gpu_stress = [0.0] * 6
    for atom_id in cpu_rows:
        cpu = cpu_rows[atom_id]
        gpu = gpu_rows[atom_id]
        for column in ("fx", "fy", "fz"):
            max_force_diff = max(
                max_force_diff, abs(cpu[column] - gpu[column])
            )
        max_eatom_diff = max(
            max_eatom_diff, abs(cpu["c_eat"] - gpu["c_eat"])
        )
        for component in range(6):
            column = f"c_satom[{component + 1}]"
            cpu_stress[component] += cpu[column]
            gpu_stress[component] += gpu[column]
    values = (
        list(cpu_thermo.values())
        + list(gpu_thermo.values())
        + [value for row in cpu_rows.values() for value in row.values()]
        + [value for row in gpu_rows.values() for value in row.values()]
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite value in nep89 CPU/GPU comparison")
    energy_diff = abs(cpu_thermo["PotEng"] - gpu_thermo["PotEng"])
    energy_scale = max(
        1.0, abs(cpu_thermo["PotEng"]), abs(gpu_thermo["PotEng"])
    )
    stress_diffs = [
        abs(cpu - gpu) for cpu, gpu in zip(cpu_stress, gpu_stress)
    ]
    stress_relative_diffs = [
        diff / max(1.0, abs(cpu), abs(gpu))
        for diff, cpu, gpu in zip(stress_diffs, cpu_stress, gpu_stress)
    ]
    stress_normalized_errors = [
        diff
        / max(
            stress_absolute_tolerance,
            stress_relative_tolerance * max(abs(cpu), abs(gpu)),
        )
        for diff, cpu, gpu in zip(stress_diffs, cpu_stress, gpu_stress)
    ]
    return {
        "energy_diff": energy_diff,
        "energy_relative_diff": energy_diff / energy_scale,
        "max_force_component_diff": max_force_diff,
        "max_per_atom_energy_diff": max_eatom_diff,
        "max_stress_sum_component_diff": max(stress_diffs),
        "max_stress_sum_relative_diff": max(stress_relative_diffs),
        "max_stress_sum_normalized_error": max(stress_normalized_errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmp", required=True)
    parser.add_argument("--cpu-plugin", required=True)
    parser.add_argument("--gpu-plugin", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--element", default="Fe")
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--lattice-constant", type=float, default=3.6)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--energy-relative-tolerance", type=float, default=5.0e-7
    )
    parser.add_argument("--force-tolerance", type=float, default=2.0e-5)
    parser.add_argument(
        "--per-atom-energy-tolerance", type=float, default=5.0e-6
    )
    parser.add_argument("--stress-absolute-tolerance", type=float, default=5.0)
    parser.add_argument(
        "--stress-relative-tolerance", type=float, default=5.0e-6
    )
    args = parser.parse_args()

    if args.cells <= 0 or args.lattice_constant <= 0.0:
        parser.error("--cells and --lattice-constant must be positive")
    elements = read_type_map(args.model)
    if args.element not in elements:
        parser.error(f"--element {args.element} is absent from the model")

    lmp = str(Path(args.lmp).resolve())
    cpu_plugin = str(Path(args.cpu_plugin).resolve())
    gpu_plugin = str(Path(args.gpu_plugin).resolve())
    model = str(Path(args.model).resolve())
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    atom_count = write_fcc_data(
        work_dir / "data.nep89",
        args.element,
        args.cells,
        args.lattice_constant,
    )
    data_text = (work_dir / "data.nep89").read_text(encoding="utf-8")

    cpu_rows, cpu_thermo = run_case(
        lmp,
        cpu_plugin,
        model,
        args.element,
        "nep/cpu",
        work_dir / "cpu",
        data_text,
    )
    gpu_rows, gpu_thermo = run_case(
        lmp,
        gpu_plugin,
        model,
        args.element,
        "nep/gpu/kk",
        work_dir / "gpu",
        data_text,
    )
    result = compare(
        cpu_rows,
        cpu_thermo,
        gpu_rows,
        gpu_thermo,
        args.stress_absolute_tolerance,
        args.stress_relative_tolerance,
    )
    payload = {
        "model": model,
        "element": args.element,
        "cells": args.cells,
        "atom_count": atom_count,
        "lattice_constant": args.lattice_constant,
        "result": result,
    }
    print(json.dumps(payload, sort_keys=True))
    checks = {
        "energy_relative_diff": args.energy_relative_tolerance,
        "max_force_component_diff": args.force_tolerance,
        "max_per_atom_energy_diff": args.per_atom_energy_tolerance,
        "max_stress_sum_normalized_error": 1.0,
    }
    failures = {
        key: result[key]
        for key, tolerance in checks.items()
        if result[key] > tolerance
    }
    if failures:
        raise SystemExit(
            "LAMMPS nep89 CPU/GPU comparison failed: "
            + ", ".join(f"{key}={value:.3e}" for key, value in failures.items())
        )


if __name__ == "__main__":
    main()
