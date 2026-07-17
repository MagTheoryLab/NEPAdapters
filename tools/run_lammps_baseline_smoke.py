#!/usr/bin/env python3
"""Run real LAMMPS plugin output against the committed CPU baseline labels."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


METAL_PRESSURE_CONVERSION = 1602176.6208


def read_type_map(model_path):
    header = Path(model_path).read_text(encoding="utf-8").splitlines()[0].split()
    count = int(header[1])
    return header[2 : 2 + count]


def quoted_array(comment, key):
    match = re.search(rf'{key}="([^"]+)"', comment)
    if match is None:
        raise ValueError(f"missing {key} in fixture comment")
    return [float(value) for value in match.group(1).split()]


def scalar(comment, key):
    match = re.search(rf"{key}=([^ ]+)", comment)
    if match is None:
        raise ValueError(f"missing {key} in fixture comment")
    return float(match.group(1))


def read_fixture(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[0])
    comment = lines[1]
    lattice = quoted_array(comment, "Lattice")
    virial_raw9 = quoted_array(comment, "virial")
    atoms = []
    for atom_id, line in enumerate(lines[2 : 2 + atom_count], start=1):
        fields = line.split()
        atoms.append(
            {
                "id": atom_id,
                "symbol": fields[0],
                "position": [float(value) for value in fields[1:4]],
                "force": [float(value) for value in fields[4:7]],
            }
        )
    return {
        "energy": scalar(comment, "energy"),
        "lattice": lattice,
        "virial_raw9": virial_raw9,
        "atoms": atoms,
    }


def ensure_orthogonal_cell(lattice):
    off_diagonal = [lattice[index] for index in (1, 2, 3, 5, 6, 7)]
    if any(abs(value) > 1.0e-12 for value in off_diagonal):
        raise ValueError("baseline LAMMPS smoke currently expects an orthogonal cell")
    return lattice[0], lattice[4], lattice[8]


def virial6_from_nep_raw9(raw9):
    return [
        raw9[0],
        raw9[4],
        raw9[8],
        0.5 * (raw9[1] + raw9[3]),
        0.5 * (raw9[2] + raw9[6]),
        0.5 * (raw9[5] + raw9[7]),
    ]


def write_data(path, fixture, elements):
    type_index = {symbol: index + 1 for index, symbol in enumerate(elements)}
    xhi, yhi, zhi = ensure_orthogonal_cell(fixture["lattice"])
    lines = [
        "NEPAdapters fixed baseline",
        "",
        f"{len(fixture['atoms'])} atoms",
        f"{len(elements)} atom types",
        "",
        f"0.0 {xhi:.17g} xlo xhi",
        f"0.0 {yhi:.17g} ylo yhi",
        f"0.0 {zhi:.17g} zlo zhi",
        "",
        "Masses",
        "",
    ]
    lines.extend(f"{index + 1} 1.0" for index in range(len(elements)))
    lines.extend(["", "Atoms # atomic", ""])
    for atom in fixture["atoms"]:
        atom_type = type_index[atom["symbol"]]
        x, y, z = atom["position"]
        lines.append(f"{atom['id']} {atom_type} {x:.17g} {y:.17g} {z:.17g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_input(
    path,
    plugin,
    model,
    elements,
    pair_style="nep/cpu",
    plugin_load_mode="environment",
):
    stress_cols = " ".join(f"c_satom[{index}]" for index in range(1, 7))
    atom_style = "atomic/kk" if pair_style == "nep/gpu" else "atomic"
    run_style = "verlet/kk" if pair_style == "nep/gpu" else "verlet"
    plugin_command = (
        [f"plugin load {plugin}"] if plugin_load_mode == "command" else []
    )
    path.write_text(
        "\n".join(
            [
                "clear",
                "units metal",
                f"atom_style {atom_style}",
                "boundary p p p",
                *plugin_command,
                "read_data data.baseline",
                f"run_style {run_style}",
                f"pair_style {pair_style}",
                f"pair_coeff * * {model} {' '.join(elements)}",
                "neighbor 1.0 bin",
                "neigh_modify every 1 delay 0 check yes",
                "compute eat all pe/atom",
                "compute satom all stress/atom NULL",
                "thermo 1",
                "thermo_style custom step atoms pe press pxx pyy pzz pxy pxz pyz",
                "thermo_modify format float %.17g",
                f"dump d all custom 1 dump.out id type x y z fx fy fz c_eat {stress_cols}",
                "dump_modify d sort id format float %.17g",
                "run 0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_command(args, cwd, env=None):
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout)
    return completed.stdout


def plugin_environment(plugin):
    plugin_path = Path(plugin)
    if not plugin_path.name.endswith("plugin.so"):
        raise ValueError(
            "LAMMPS_PLUGIN_PATH auto-loading requires a filename ending in "
            f"'plugin.so': {plugin_path.name}"
        )
    env = os.environ.copy()
    env["LAMMPS_PLUGIN_PATH"] = str(plugin_path.parent)
    return env


def lammps_command(lmp, pair_style):
    command = [lmp]
    if pair_style == "nep/gpu":
        command.extend(["-k", "on", "g", "1"])
    command.extend(["-in", "in.baseline", "-log", "log.lammps"])
    return command


def parse_dump(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    frame = None
    index = 0
    while index < len(lines):
        if lines[index] != "ITEM: TIMESTEP":
            index += 1
            continue
        atom_count = int(lines[index + 3])
        columns = lines[index + 8].split()[2:]
        rows = {}
        for line in lines[index + 9 : index + 9 + atom_count]:
            values = line.split()
            record = {column: float(value) for column, value in zip(columns, values)}
            rows[int(record["id"])] = record
        frame = rows
        index += 9 + atom_count
    if frame is None:
        raise ValueError(f"no dump frame found in {path}")
    return frame


def parse_thermo(path):
    headers = None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[:3] == ["Step", "Atoms", "PotEng"]:
            headers = parts
            continue
        if headers and len(parts) == len(headers):
            try:
                rows.append(dict(zip(headers, [float(value) for value in parts])))
            except ValueError:
                pass
    if not rows:
        raise ValueError(f"no thermo row found in {path}")
    return rows[-1]


def compare(fixture, dump_rows, thermo):
    force_columns = ["fx", "fy", "fz"]
    max_force_diff = 0.0
    for atom in fixture["atoms"]:
        row = dump_rows[atom["id"]]
        for column, expected in zip(force_columns, atom["force"]):
            max_force_diff = max(max_force_diff, abs(row[column] - expected))

    eatom_sum = sum(row["c_eat"] for row in dump_rows.values())
    stress_sum = [
        sum(row[f"c_satom[{component}]"] for row in dump_rows.values())
        for component in range(1, 7)
    ]
    lammps_virial6 = [
        -value / METAL_PRESSURE_CONVERSION for value in stress_sum
    ]
    reference_virial6 = virial6_from_nep_raw9(fixture["virial_raw9"])
    max_virial_diff = max(
        abs(actual - expected)
        for actual, expected in zip(lammps_virial6, reference_virial6)
    )
    return {
        "energy_diff": abs(thermo["PotEng"] - fixture["energy"]),
        "eatom_sum_diff": abs(eatom_sum - fixture["energy"]),
        "max_force_component_diff": max_force_diff,
        "max_virial_component_diff": max_virial_diff,
        "lammps_virial6_from_stress_atom": lammps_virial6,
        "reference_virial6": reference_virial6,
    }


def write_markdown(path, payload):
    result = payload["result"]
    lines = [
        "# LAMMPS Baseline Smoke Report",
        "",
        f"- lmp: `{payload['lmp']}`",
        f"- plugin: `{payload['plugin']}`",
        f"- model: `{payload['model']}`",
        f"- fixture: `{payload['fixture']}`",
        "",
        "| Quantity | Diff |",
        "| --- | ---: |",
        f"| energy | {result['energy_diff']:.3e} |",
        f"| per-atom energy sum | {result['eatom_sum_diff']:.3e} |",
        f"| max force component | {result['max_force_component_diff']:.3e} |",
        f"| max virial component | {result['max_virial_component_diff']:.3e} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmp", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument(
        "--pair-style", choices=("nep/cpu", "nep/gpu"), default="nep/cpu"
    )
    parser.add_argument(
        "--plugin-load-mode",
        choices=("environment", "command"),
        default="environment",
        help=(
            "Load through LAMMPS_PLUGIN_PATH by default; use 'command' only "
            "to exercise an explicit plugin load line."
        ),
    )
    parser.add_argument("--work-dir", default="build-lammps-baseline-smoke")
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--force-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--virial-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--json-output", default="")
    parser.add_argument("--markdown-output", default="")
    args = parser.parse_args()

    args.lmp = str(Path(args.lmp).resolve())
    args.plugin = str(Path(args.plugin).resolve())
    args.model = str(Path(args.model).resolve())
    args.fixture = str(Path(args.fixture).resolve())

    fixture = read_fixture(args.fixture)
    elements = read_type_map(args.model)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    write_data(work_dir / "data.baseline", fixture, elements)
    write_input(
        work_dir / "in.baseline",
        args.plugin,
        args.model,
        elements,
        args.pair_style,
        args.plugin_load_mode,
    )

    run_env = (
        plugin_environment(args.plugin)
        if args.plugin_load_mode == "environment"
        else None
    )
    screen = run_command(
        lammps_command(args.lmp, args.pair_style),
        work_dir,
        env=run_env,
    )
    (work_dir / "screen.out").write_text(screen, encoding="utf-8")
    payload = {
        "lmp": args.lmp,
        "plugin": args.plugin,
        "model": args.model,
        "fixture": args.fixture,
        "pair_style": args.pair_style,
        "plugin_load_mode": args.plugin_load_mode,
        "result": compare(
            fixture,
            parse_dump(work_dir / "dump.out"),
            parse_thermo(work_dir / "log.lammps"),
        ),
    }

    checks = {
        "energy_diff": args.energy_tolerance,
        "eatom_sum_diff": args.energy_tolerance,
        "max_force_component_diff": args.force_tolerance,
        "max_virial_component_diff": args.virial_tolerance,
    }
    failures = {
        key: payload["result"][key]
        for key, tolerance in checks.items()
        if payload["result"][key] > tolerance
    }
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_output:
        md_path = Path(args.markdown_output)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(md_path, payload)

    print(json.dumps(payload, sort_keys=True))
    if failures:
        raise SystemExit(
            "LAMMPS baseline smoke failed: " +
            ", ".join(f"{key}={value:.3e}" for key, value in failures.items())
        )


if __name__ == "__main__":
    main()
