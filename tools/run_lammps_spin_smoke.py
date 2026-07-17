#!/usr/bin/env python3
"""Run a real LAMMPS spin-plugin calculation against committed labels."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_lammps_baseline_smoke import (
    METAL_PRESSURE_CONVERSION,
    parse_dump,
    parse_thermo,
    plugin_environment,
    run_command,
    virial6_from_nep_raw9,
)


def read_reference(path):
    tokens = Path(path).read_text(encoding="utf-8").split()
    result = {}
    index = 0
    while index < len(tokens):
        key = tokens[index]
        count = int(tokens[index + 1])
        begin = index + 2
        end = begin + count
        result[key] = [float(value) for value in tokens[begin:end]]
        index = end
    return result


def read_structure(path):
    structure = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(structure["box"]) != 3 or not structure["atoms"]:
        raise ValueError("spin LAMMPS fixture requires a 3D box and atoms")
    return structure


def spin_magnitude_and_direction(spin):
    magnitude = math.sqrt(sum(component * component for component in spin))
    if magnitude <= 0.0:
        raise ValueError("LAMMPS spin fixture does not accept zero-length spins")
    return magnitude, [component / magnitude for component in spin]


def write_input(path, plugin, model, structure, pair_style, plugin_load_mode):
    gpu = pair_style in {"nep/gpu", "nep/gpu/kk"}
    atom_style = "spin/kk" if gpu else "spin"
    run_style = "verlet/kk" if gpu else "verlet"
    plugin_command = (
        [f"plugin load {plugin}"] if plugin_load_mode == "command" else []
    )
    xhi, yhi, zhi = structure["box"]
    lines = [
        "clear",
        "units metal",
        f"atom_style {atom_style}",
        "atom_modify map array",
        "boundary p p p",
        *plugin_command,
        f"region box block 0.0 {xhi:.17g} 0.0 {yhi:.17g} 0.0 {zhi:.17g} units box",
        f"create_box {len(structure['elements'])} box",
    ]
    for atom in structure["atoms"]:
        x, y, z = atom["position"]
        lines.append(
            f"create_atoms {atom['type']} single {x:.17g} {y:.17g} {z:.17g} units box"
        )
    for atom_type in range(1, len(structure["elements"]) + 1):
        lines.append(f"mass {atom_type} 1.0")
    for atom_id, atom in enumerate(structure["atoms"], start=1):
        magnitude, direction = spin_magnitude_and_direction(atom["spin"])
        lines.append(
            "set atom "
            f"{atom_id} spin {magnitude:.17g} "
            + " ".join(f"{component:.17g}" for component in direction)
        )
    stress_columns = " ".join(f"c_satom[{index}]" for index in range(1, 7))
    spin_columns = " ".join(f"c_spin[{index}]" for index in range(1, 8))
    lines.extend(
        [
            f"run_style {run_style}",
            f"pair_style {pair_style}",
            f"pair_coeff * * {model} {' '.join(structure['elements'])}",
            "neighbor 1.0 bin",
            "neigh_modify every 1 delay 0 check yes",
            "compute eat all pe/atom",
            "compute satom all stress/atom NULL",
            "compute spin all property/atom spx spy spz sp fmx fmy fmz",
            "thermo 1",
            "thermo_style custom step atoms pe press pxx pyy pzz pxy pxz pyz",
            "thermo_modify format float %.17g",
            (
                "dump d all custom 1 dump.out id type x y z fx fy fz c_eat "
                f"{spin_columns} {stress_columns}"
            ),
            "dump_modify d sort id format float %.17g",
            "run 0",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def lammps_command(lmp, pair_style, input_path, mpi_ranks, mpiexec):
    command = []
    if mpi_ranks > 1:
        if not mpiexec:
            raise ValueError("--mpiexec is required when --mpi-ranks is greater than 1")
        command.extend([mpiexec, "-np", str(mpi_ranks)])
    command.append(lmp)
    if pair_style in {"nep/gpu", "nep/gpu/kk"}:
        command.extend(["-k", "on", "g", "1"])
    command.extend(["-in", input_path, "-log", "log.lammps"])
    return command


def flatten(values):
    return [component for row in values for component in row]


def compare(structure, reference, dump_rows, thermo):
    expected_force = reference["force"]
    expected_mforce = reference["mforce"]
    actual_force = []
    actual_mforce = []
    actual_spin = []
    for atom_id in range(1, len(structure["atoms"]) + 1):
        row = dump_rows[atom_id]
        actual_force.extend(row[column] for column in ("fx", "fy", "fz"))
        actual_mforce.extend(row[f"c_spin[{index}]"] for index in range(5, 8))
        magnitude = row["c_spin[4]"]
        actual_spin.extend(row[f"c_spin[{index}]"] * magnitude for index in range(1, 4))

    stress_sum = [
        sum(row[f"c_satom[{component}]"] for row in dump_rows.values())
        for component in range(1, 7)
    ]
    actual_virial6 = [-value / METAL_PRESSURE_CONVERSION for value in stress_sum]
    expected_virial6 = virial6_from_nep_raw9(reference["virial9"])
    expected_spin = flatten(atom["spin"] for atom in structure["atoms"])

    return {
        "energy_diff": abs(thermo["PotEng"] - reference["energy_total"][0]),
        "eatom_sum_diff": abs(
            sum(row["c_eat"] for row in dump_rows.values())
            - reference["energy_total"][0]
        ),
        "max_force_component_diff": max(
            abs(actual - expected)
            for actual, expected in zip(actual_force, expected_force)
        ),
        "max_mforce_component_diff": max(
            abs(actual - expected)
            for actual, expected in zip(actual_mforce, expected_mforce)
        ),
        "max_virial_component_diff": max(
            abs(actual - expected)
            for actual, expected in zip(actual_virial6, expected_virial6)
        ),
        "max_spin_component_diff": max(
            abs(actual - expected)
            for actual, expected in zip(actual_spin, expected_spin)
        ),
        "lammps_virial6_from_stress_atom": actual_virial6,
        "reference_virial6": expected_virial6,
    }


def write_markdown(path, payload):
    result = payload["result"]
    rows = [
        ("energy", result["energy_diff"]),
        ("per-atom energy sum", result["eatom_sum_diff"]),
        ("force", result["max_force_component_diff"]),
        ("mforce", result["max_mforce_component_diff"]),
        ("virial", result["max_virial_component_diff"]),
        ("spin readback", result["max_spin_component_diff"]),
    ]
    lines = [
        "# LAMMPS Spin Plugin Smoke Report",
        "",
        f"- pair style: `{payload['pair_style']}`",
        f"- MPI ranks: `{payload['mpi_ranks']}`",
        f"- lmp: `{payload['lmp']}`",
        f"- plugin: `{payload['plugin']}`",
        "",
        "| Quantity | Diff |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {value:.3e} |" for name, value in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmp", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--pair-style", choices=("nep/cpu", "nep/gpu", "nep/gpu/kk"), default="nep/cpu"
    )
    parser.add_argument(
        "--plugin-load-mode", choices=("environment", "command"), default="environment"
    )
    parser.add_argument("--work-dir", default="build-lammps-spin-smoke")
    parser.add_argument("--mpiexec", default="")
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--force-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--mforce-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--virial-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--spin-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--json-output", default="")
    parser.add_argument("--markdown-output", default="")
    args = parser.parse_args()

    args.lmp = str(Path(args.lmp).resolve())
    args.plugin = str(Path(args.plugin).resolve())
    args.model = str(Path(args.model).resolve())
    structure = read_structure(args.structure)
    reference = read_reference(args.reference)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "in.spin"
    write_input(
        input_path,
        args.plugin,
        args.model,
        structure,
        args.pair_style,
        args.plugin_load_mode,
    )

    run_env = (
        plugin_environment(args.plugin)
        if args.plugin_load_mode == "environment"
        else os.environ.copy()
    )
    screen = run_command(
        lammps_command(
            args.lmp,
            args.pair_style,
            input_path.name,
            args.mpi_ranks,
            args.mpiexec,
        ),
        work_dir,
        env=run_env,
    )
    (work_dir / "screen.out").write_text(screen, encoding="utf-8")
    payload = {
        "lmp": args.lmp,
        "plugin": args.plugin,
        "model": args.model,
        "pair_style": args.pair_style,
        "mpi_ranks": args.mpi_ranks,
        "result": compare(
            structure,
            reference,
            parse_dump(work_dir / "dump.out"),
            parse_thermo(work_dir / "log.lammps"),
        ),
    }

    checks = {
        "energy_diff": args.energy_tolerance,
        "eatom_sum_diff": args.energy_tolerance,
        "max_force_component_diff": args.force_tolerance,
        "max_mforce_component_diff": args.mforce_tolerance,
        "max_virial_component_diff": args.virial_tolerance,
        "max_spin_component_diff": args.spin_tolerance,
    }
    failures = {
        key: payload["result"][key]
        for key, tolerance in checks.items()
        if payload["result"][key] > tolerance
    }
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_output:
        output = Path(args.markdown_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(output, payload)

    print(json.dumps(payload, sort_keys=True))
    if failures:
        raise SystemExit(
            "LAMMPS spin smoke failed: "
            + ", ".join(f"{key}={value:.3e}" for key, value in failures.items())
        )


if __name__ == "__main__":
    main()
