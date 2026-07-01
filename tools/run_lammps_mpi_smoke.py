#!/usr/bin/env python3
"""Run a small MPI LAMMPS correctness smoke for the NEPAdapters plugin."""

import argparse
import json
import math
import subprocess
from pathlib import Path


def write_data(path):
    nx, ny, nz = 4, 4, 4
    spacing = 5.0
    rows = []
    atom_id = 1
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                atom_type = 1 if (ix + iy + iz) % 2 == 0 else 2
                x = (ix + 0.5) * spacing + 0.013 * math.sin(atom_id)
                y = (iy + 0.5) * spacing + 0.017 * math.cos(atom_id * 1.7)
                z = (iz + 0.5) * spacing + 0.011 * math.sin(atom_id * 2.3)
                rows.append((atom_id, atom_type, x, y, z))
                atom_id += 1

    lines = [
        "NEPAdapters MPI smoke",
        "",
        f"{len(rows)} atoms",
        "2 atom types",
        "",
        f"0.0 {nx * spacing:.16g} xlo xhi",
        f"0.0 {ny * spacing:.16g} ylo yhi",
        f"0.0 {nz * spacing:.16g} zlo zhi",
        "",
        "Masses",
        "",
        "1 127.6",
        "2 207.2",
        "",
        "Atoms # atomic",
        "",
    ]
    lines.extend(f"{i} {t} {x:.17g} {y:.17g} {z:.17g}" for i, t, x, y, z in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_input(path, plugin, model, compute_style):
    stress_cols = " ".join(f"c_satom[{i}]" for i in range(1, 10 if compute_style.startswith("centroid") else 7))
    path.write_text(
        "\n".join(
            [
                "clear",
                "units metal",
                "atom_style atomic",
                "boundary p p p",
                "processors ${px} ${py} ${pz}",
                f"plugin load {plugin}",
                "read_data data.smoke",
                "pair_style nep/cpu",
                f"pair_coeff * * {model} Te Pb",
                "neighbor 1.0 bin",
                "neigh_modify every 1 delay 0 check yes",
                "compute eat all pe/atom",
                f"compute satom all {compute_style} NULL",
                "thermo 1",
                "thermo_style custom step atoms pe press pxx pyy pzz pxy pxz pyz",
                f"dump d all custom 1 dump.out id type x y z fx fy fz c_eat {stress_cols}",
                "dump_modify d sort id format float %.17g",
                "run 0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_command(args, cwd):
    completed = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout)
    return completed.stdout


def parse_dump(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = 0
    frame = None
    while idx < len(lines):
        if lines[idx] != "ITEM: TIMESTEP":
            idx += 1
            continue
        atom_count = int(lines[idx + 3])
        cols = lines[idx + 8].split()[2:]
        rows = {}
        for line in lines[idx + 9 : idx + 9 + atom_count]:
            values = line.split()
            record = {col: float(value) for col, value in zip(cols, values)}
            rows[int(record["id"])] = record
        frame = rows
        idx += 9 + atom_count
    if frame is None:
        raise ValueError(f"no dump frame found in {path}")
    return frame


def parse_thermo(path):
    headers = None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 10 and parts[:3] == ["Step", "Atoms", "PotEng"]:
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


def compare_case(work_dir, prefix, stress_components):
    ref = parse_dump(work_dir / f"{prefix}.np1.dump")
    ref_thermo = parse_thermo(work_dir / f"{prefix}.np1.log")
    comparisons = []
    for np in (2, 4):
        cur = parse_dump(work_dir / f"{prefix}.np{np}.dump")
        thermo = parse_thermo(work_dir / f"{prefix}.np{np}.log")
        ids = sorted(ref)
        force_cols = ["fx", "fy", "fz"]
        stress_cols = [f"c_satom[{i}]" for i in range(1, stress_components + 1)]
        press_cols = ["Press", "Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]
        comparisons.append(
            {
                "np": np,
                "max_force_component_diff": max(
                    abs(cur[i][col] - ref[i][col]) for i in ids for col in force_cols
                ),
                "max_eatom_diff": max(abs(cur[i]["c_eat"] - ref[i]["c_eat"]) for i in ids),
                "max_stress_component_diff": max(
                    abs(cur[i][col] - ref[i][col]) for i in ids for col in stress_cols
                ),
                "pe_diff": abs(thermo["PotEng"] - ref_thermo["PotEng"]),
                "max_press_component_diff": max(
                    abs(thermo[col] - ref_thermo[col]) for col in press_cols
                ),
            }
        )
    return comparisons


def run_lammps_case(args, work_dir, case_name, compute_style):
    input_path = work_dir / f"in.{case_name}"
    write_input(input_path, args.plugin, args.model, compute_style)
    prefix = f"{case_name}_forces"
    proc_grids = {
        1: (1, 1, 1),
        2: (2, 1, 1),
        4: (2, 2, 1),
    }
    for np, grid in proc_grids.items():
        dump = work_dir / "dump.out"
        if dump.exists():
            dump.unlink()
        command = [
            args.mpirun,
            "--oversubscribe",
            "-np",
            str(np),
            args.lmp,
            "-var",
            "px",
            str(grid[0]),
            "-var",
            "py",
            str(grid[1]),
            "-var",
            "pz",
            str(grid[2]),
            "-in",
            str(input_path.name),
            "-log",
            f"{prefix}.np{np}.log",
        ]
        screen = run_command(command, work_dir)
        (work_dir / f"{prefix}.np{np}.screen").write_text(screen, encoding="utf-8")
        dump.rename(work_dir / f"{prefix}.np{np}.dump")
    return compare_case(work_dir, prefix, 9 if compute_style.startswith("centroid") else 6)


def write_markdown(path, payload):
    lines = [
        "# LAMMPS MPI Smoke Report",
        "",
        f"- lmp: `{payload['lmp']}`",
        f"- plugin: `{payload['plugin']}`",
        f"- model: `{payload['model']}`",
        "",
        "| Case | MPI ranks | Force diff | Eatom diff | Stress diff | PE diff | Press diff |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        for row in case["comparisons"]:
            lines.append(
                f"| `{case['name']}` | {row['np']} "
                f"| {row['max_force_component_diff']:.3e} "
                f"| {row['max_eatom_diff']:.3e} "
                f"| {row['max_stress_component_diff']:.3e} "
                f"| {row['pe_diff']:.3e} "
                f"| {row['max_press_component_diff']:.3e} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmp", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mpirun", default="/opt/homebrew/bin/mpirun")
    parser.add_argument("--work-dir", default="build-lammps-mpi-smoke")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--markdown-output", default="")
    args = parser.parse_args()

    args.lmp = str(Path(args.lmp).resolve())
    args.plugin = str(Path(args.plugin).resolve())
    args.model = str(Path(args.model).resolve())

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    write_data(work_dir / "data.smoke")

    payload = {
        "lmp": args.lmp,
        "plugin": args.plugin,
        "model": args.model,
        "cases": [
            {
                "name": "stress_atom",
                "compute": "stress/atom",
                "comparisons": run_lammps_case(args, work_dir, "stress_atom", "stress/atom"),
            },
            {
                "name": "centroid_stress_atom",
                "compute": "centroid/stress/atom",
                "comparisons": run_lammps_case(
                    args,
                    work_dir,
                    "centroid_stress_atom",
                    "centroid/stress/atom",
                ),
            },
        ],
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


if __name__ == "__main__":
    main()
