#!/usr/bin/env python3
"""Generate deterministic versioned O/C spin models and FP64 oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from torchnep.magnetic_polynomial import MagneticPolynomialDescriptor
from torchnep.model import NEPModel
from torchnep.nep import NEPCalculator


DTYPE = torch.float64


def spin_descriptor_dim(compress: int, lmax: int, order: int, soc: int,
                        spin_mode: int = 2) -> int:
    pairs = compress * (compress + 1) // 2
    dim = 1 + 2 * compress
    if soc and lmax >= 2:
        dim += 2 * compress
    if order >= 2:
        dim += 2 * compress
        if lmax >= 1:
            dim += (3 if soc else 1) * compress
        if lmax >= 2:
            dim += compress
        dim += compress + 2 * pairs
        if soc and lmax >= 1:
            dim += (2 if compress >= 2 else 1) * compress
        if soc and lmax >= 2:
            dim += (2 if compress >= 2 else 1) * compress
    if order >= 3:
        dim += compress
        if soc and lmax >= 1:
            dim += (2 if compress >= 2 else 1) * compress
        if soc and lmax >= 2:
            dim += (3 if compress >= 2 else 1) * compress
        if soc and lmax >= 1 and compress >= 3:
            dim += compress
    if spin_mode == 3 and order >= 2:
        dim += lmax * pairs
    return dim


def model_config(order: int = 3, compress: int = 2,
                 lmax: int = 2, soc: int = 1,
                 zbl: float | None = None, spin_mode: int = 2,
                 spin_cutoff=(6.0, 6.0)):
    config = {
        "num_types": 2,
        "type_names": ["Fe", "Ge"],
        "cutoff_radial": 6.0,
        "cutoff_angular": 5.0,
        "n_max_radial": 4,
        "n_max_angular": 4,
        "basis_size_radial": 8,
        "basis_size_angular": 8,
        "l_max": [4, 2, 0],
        "neuron": 30,
        "spin_mode": spin_mode,
        "spin_compress": compress,
        "spin_basis_size": [8, 0],
        "spin_l_max": [lmax, 0, 0],
        "spin_cutoff": list(spin_cutoff),
        "spin_cutoff_by_type": list(spin_cutoff),
        "spin_order": order,
        "spin_soc": soc,
        "spin_dof_type": ["Fe"],
        "spin_env_type": ["Fe", "Ge"],
    }
    if zbl is not None:
        config["zbl"] = zbl
    return config


def write_model(path: Path, order: int = 3, compress: int = 2,
                lmax: int = 2, soc: int = 1,
                zbl: float | None = None, spin_mode: int = 2,
                spin_cutoff=(6.0, 6.0)):
    torch.manual_seed(20260812)
    model = NEPModel(model_config(
        order=order, compress=compress, lmax=lmax, soc=soc, zbl=zbl,
        spin_mode=spin_mode, spin_cutoff=spin_cutoff)).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(0.08 * torch.randn_like(parameter))
        # A dense, non-symmetric leg mix catches flatten-order mistakes.
        model.spin_descriptor.radial_leg_mix.copy_(
            torch.linspace(-0.31, 0.43, 4 * compress * compress,
                           dtype=DTYPE).reshape(4, compress, compress)
        )
        descriptor_dim = 30 + spin_descriptor_dim(
            compress, lmax, order, soc, spin_mode)
        model.q_scaler.copy_(
            torch.linspace(0.55, 1.35, descriptor_dim, dtype=DTYPE))
        model.energy_baseline.copy_(torch.tensor([-0.17, 0.23], dtype=DTYPE))
    model.save_nep_txt(path, max_NN_radial=128, max_NN_angular=128)


def cases():
    positions = np.array(
        [[1.0, 1.0, 1.0], [2.4, 1.3, 0.8], [0.7, 2.6, 1.4],
         [1.3, 0.5, 2.7], [3.1, 2.8, 2.2]], dtype=np.float64)
    spins = np.array(
        [[0.8, 0.1, 0.2], [0.2, 0.9, -0.1], [-0.3, 0.4, 0.7],
         [0.5, -0.2, 0.6], [-0.4, -0.3, 0.8]], dtype=np.float64)
    cell = np.diag([14.0, 13.0, 15.0])
    species = ["Fe", "Ge", "Fe", "Ge", "Fe"]
    yield "zero", species, positions, cell, np.zeros_like(spins)
    yield "collinear", species, positions, cell, np.array(
        [[0.0, 0.0, x] for x in (0.8, -0.6, 1.1, -0.3, 0.5)])
    yield "noncollinear", species, positions, cell, spins
    yield "chiral_soc", species, positions + np.array(
        [[0, 0, 0], [.1, -.2, .3], [-.2, .1, .2],
         [.3, .2, -.1], [-.1, .3, .1]]), cell, np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [.6, -.2, .7], [-.4, .8, .3]])
    strain = np.array([[1.015, .012, -.006], [0, .987, .009], [.004, 0, 1.008]])
    yield "strained", species, positions @ strain.T, cell @ strain.T, spins
    rng = np.random.default_rng(20260812)
    yield "random_multitype", ["Ge", "Fe", "Ge", "Ge", "Fe", "Fe", "Ge", "Fe"], \
        rng.uniform(.7, 8.0, (8, 3)), \
        np.array([[13., .4, -.2], [0, 14., .3], [0, 0, 13.5]]), \
        rng.normal(0, .65, (8, 3))
    yield "pbc_wrap", ["Fe", "Ge", "Fe", "Ge"], np.array(
        [[.25, 1.1, 1.2], [13.45, 1.4, 1.0], [6.8, .2, 14.4], [6.5, 12.5, .45]]), \
        cell, np.array([[.7, -.1, .3], [-.2, .8, .4], [.5, .2, -.6], [-.4, .3, .9]])
    yield "cutoff_boundary", ["Fe", "Ge", "Fe"], np.array(
        [[10., 10., 10.], [15.999, 10., 10.], [16.001, 10., 10.]]), \
        np.diag([30., 30., 30.]), np.array([[.8, .2, -.1], [.1, -.7, .4], [-.2, .3, .9]])


def write_array(handle, name, values):
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    handle.write(f"{name} {flat.size}")
    for value in flat:
        handle.write(f" {value:.17e}")
    handle.write("\n")


def write_lammps_fixture(structure_path, reference_path, species, positions,
                         cell, spins, result):
    elements = list(dict.fromkeys(species))
    structure = {
        "box": [float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])],
        "elements": elements,
        "atoms": [
            {
                "type": elements.index(symbol) + 1,
                "position": [float(value) for value in position],
                "spin": [float(value) for value in spin],
            }
            for symbol, position, spin in zip(species, positions, spins)
        ],
    }
    Path(structure_path).parent.mkdir(parents=True, exist_ok=True)
    Path(structure_path).write_text(
        json.dumps(structure, indent=2) + "\n", encoding="utf-8")
    potential = result["energy"].detach().cpu().numpy().reshape(-1)
    virial = result["virial"].detach().cpu().numpy().reshape(-1, 9).sum(axis=0)
    with Path(reference_path).open("w", encoding="utf-8") as handle:
        for name, values in (
            ("energy_total", [potential.sum()]),
            ("energy_atom", potential),
            ("force", result["forces"].detach().cpu().numpy()),
            ("mforce", result["mforces"].detach().cpu().numpy()),
            ("virial9", virial),
        ):
            write_array(handle, name, values)


def assert_literal_o2_prefix(model):
    o3 = model.spin_descriptor
    o2 = MagneticPolynomialDescriptor(
        spin_compress=2, spin_l_max=2, spin_order=2, spin_soc=True,
        basis_size_radial=8, cutoff_radial=6.0).double()
    with torch.no_grad():
        o2.radial_leg_mix.copy_(o3.radial_leg_mix)
    positions = torch.tensor(
        [[0., 0., 0.], [1.2, .1, .2], [-.2, 1.3, .3]], dtype=DTYPE)
    spins = torch.tensor(
        [[.8, .1, .2], [.2, .9, -.1], [-.3, .4, .7]], dtype=DTYPE)
    pair_i = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    pair_j = torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.long)
    rij = positions.index_select(0, pair_j) - positions.index_select(0, pair_i)
    types = torch.tensor([0, 1, 0], dtype=torch.long)
    args = (rij, pair_i, pair_j, spins, 3, types, model.c_spin, None,
            model.spin_dof_type_active, model.spin_env_type_active)
    q3 = o3(*args)
    q2 = o2(*args)
    torch.testing.assert_close(q3[:, :37], q2, rtol=0, atol=1.0e-13)


def assert_symmetries(calculator: NEPCalculator):
    name, species, positions, cell, spins = next(
        case for case in cases() if case[0] == "noncollinear")
    del name
    base = calculator.compute(species, positions, cell, spins, compute_descriptor=True)
    axis = np.array([.3, -.4, .5]); axis /= np.linalg.norm(axis)
    angle = .73
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    rotation = np.eye(3) * np.cos(angle) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * cross
    rotated = calculator.compute(species, positions @ rotation.T,
                                 cell @ rotation.T, spins @ rotation.T,
                                 compute_descriptor=True)
    torch.testing.assert_close(rotated["descriptor"], base["descriptor"], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(rotated["forces"], base["forces"] @ torch.as_tensor(rotation.T), rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(rotated["mforces"], base["mforces"] @ torch.as_tensor(rotation.T), rtol=2e-11, atol=2e-11)
    reversed_result = calculator.compute(species, positions, cell, -spins, compute_descriptor=True)
    torch.testing.assert_close(reversed_result["descriptor"], base["descriptor"], rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(reversed_result["forces"], base["forces"], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(reversed_result["mforces"], -base["mforces"], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(reversed_result["tau"], base["tau"], rtol=2e-12, atol=2e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--create-model", action="store_true")
    parser.add_argument("--lammps-structure")
    parser.add_argument("--lammps-reference")
    parser.add_argument("--order", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--compress", type=int, choices=range(1, 10), default=2)
    parser.add_argument("--lmax", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--soc", type=int, choices=(0, 1), default=1)
    parser.add_argument("--zbl", type=float)
    parser.add_argument("--spin-mode", type=int, choices=(2, 3), default=2)
    parser.add_argument("--spin-cutoff", type=float, nargs="+", default=[6.0])
    args = parser.parse_args()
    model_path = Path(args.model)
    if args.create_model:
        write_model(model_path, order=args.order, compress=args.compress,
                    lmax=args.lmax, soc=args.soc, zbl=args.zbl,
                    spin_mode=args.spin_mode, spin_cutoff=args.spin_cutoff)
    calculator = NEPCalculator(str(model_path), dtype=DTYPE)
    if (args.spin_mode, args.order, args.compress, args.lmax, args.soc) == (
            2, 3, 2, 2, 1):
        assert_literal_o2_prefix(calculator)
    assert_symmetries(calculator)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    with Path(args.output).open("w", encoding="utf-8") as handle:
        handle.write(f"spin{args.spin_mode}_oc_oracle_v1\n")
        handle.write(f"model_sha256 {digest}\n")
        spin_dim = spin_descriptor_dim(
            args.compress, args.lmax, args.order, args.soc, args.spin_mode)
        handle.write(f"descriptor_dim {30 + spin_dim}\n")
        handle.write(f"spin_descriptor_dim {spin_dim}\n")
        for name, species, positions, cell, spins in cases():
            result = calculator.compute(species, positions, cell, spins,
                                        compute_descriptor=True)
            if name == "noncollinear" and args.lammps_structure and args.lammps_reference:
                write_lammps_fixture(
                    args.lammps_structure, args.lammps_reference, species,
                    positions, cell, spins, result)
            handle.write(f"case {name} {len(species)}\n")
            handle.write("types " + " ".join(species) + "\n")
            for key, value in (("cell", cell), ("positions", positions),
                               ("spins", spins), ("potential", result["energy"].cpu()),
                               ("forces", result["forces"].cpu()),
                               ("virial", result["virial"].cpu()),
                               ("mforces", result["mforces"].cpu()),
                               ("tau", result["tau"].cpu()),
                               ("descriptor", result["descriptor"].cpu())):
                write_array(handle, key, value)
            handle.write("end\n")


if __name__ == "__main__":
    main()
