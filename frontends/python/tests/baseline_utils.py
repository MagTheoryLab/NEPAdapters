from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LabeledStructure:
    symbols: list[str]
    positions: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    energy: float
    forces: np.ndarray
    virial: np.ndarray

    def get_chemical_symbols(self):
        return list(self.symbols)

    def __len__(self):
        return len(self.symbols)


def read_type_map(model_path):
    header = Path(model_path).read_text(encoding="utf-8").splitlines()[0].split()
    count = int(header[1])
    return {symbol: index for index, symbol in enumerate(header[2 : 2 + count])}


def _parse_scalar(comment, key):
    prefix = f"{key}="
    begin = comment.index(prefix) + len(prefix)
    end = comment.find(" ", begin)
    if end < 0:
        end = len(comment)
    return float(comment[begin:end])


def _parse_quoted_array(comment, key):
    prefix = f'{key}="'
    begin = comment.index(prefix) + len(prefix)
    end = comment.index('"', begin)
    return np.asarray([float(value) for value in comment[begin:end].split()], dtype=np.float64)


def read_labeled_structure(xyz_path):
    with open(xyz_path, encoding="utf-8") as handle:
        atom_count = int(handle.readline())
        comment = handle.readline()
        cell = _parse_quoted_array(comment, "Lattice").reshape(3, 3)
        energy = _parse_scalar(comment, "energy")
        virial = _parse_quoted_array(comment, "virial")
        symbols = []
        positions = []
        forces = []
        for _ in range(atom_count):
            fields = handle.readline().split()
            symbols.append(fields[0])
            positions.append([float(value) for value in fields[1:4]])
            forces.append([float(value) for value in fields[4:7]])
    return LabeledStructure(
        symbols=symbols,
        positions=np.asarray(positions, dtype=np.float64),
        cell=cell,
        pbc=np.asarray([1, 1, 1], dtype=np.int32),
        energy=energy,
        forces=np.asarray(forces, dtype=np.float64),
        virial=virial,
    )


def read_descriptor_fixture(path):
    with open(path, encoding="utf-8") as handle:
        fields = handle.readline().split()
        if fields[:2] != ["#", "shape"] or len(fields) != 4:
            raise ValueError("descriptor fixture header must be '# shape <rows> <cols>'")
        rows = int(fields[2])
        cols = int(fields[3])
        values = np.loadtxt(handle, dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape(rows, cols)
