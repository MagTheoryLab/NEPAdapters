"""Small NumPy-first calculator facade for NEPAdapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._native import Model, load_model


@dataclass(frozen=True)
class Prediction:
    energy: np.ndarray
    potential: np.ndarray
    forces: np.ndarray
    virials: np.ndarray
    structure_virials: np.ndarray
    atom_counts: np.ndarray

    def force_blocks(self) -> list[np.ndarray]:
        if len(self.atom_counts) == 0:
            return []
        return list(np.split(self.forces, np.cumsum(self.atom_counts)[:-1]))

    def virial_blocks(self, mean: bool = True) -> list[np.ndarray]:
        if len(self.atom_counts) == 0:
            return []
        blocks = np.split(self.virials, np.cumsum(self.atom_counts)[:-1])
        if not mean:
            return list(blocks)
        return [block.mean(axis=0) for block in blocks]


def _read_type_map(model_path: str | Path) -> dict[str, int]:
    fields = Path(model_path).read_text(encoding="utf-8").splitlines()[0].split()
    if len(fields) < 3 or not fields[0].startswith("nep"):
        raise ValueError("failed to read NEP element list from model header")
    num_types = int(fields[1])
    if len(fields) < 2 + num_types:
        raise ValueError("failed to read NEP element list from model header")
    return {symbol: index for index, symbol in enumerate(fields[2 : 2 + num_types])}


def _as_structure_list(structures) -> list:
    if isinstance(structures, (str, bytes)):
        raise TypeError("structures must be objects, not strings")
    if hasattr(structures, "get_chemical_symbols") or hasattr(structures, "positions"):
        return [structures]
    return list(structures)


def _cell_to_nep_box(cell) -> np.ndarray:
    array = np.asarray(cell, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError("structure cell must have shape (3, 3)")
    return np.ascontiguousarray(array.T.reshape(9), dtype=np.float64)


def _structure_pbc(structure) -> np.ndarray:
    pbc = getattr(structure, "pbc", None)
    if pbc is None:
        return np.asarray([1, 1, 1], dtype=np.int32)
    return np.asarray(pbc, dtype=np.int32).reshape(3)


def _structure_symbols(structure) -> list[str]:
    if hasattr(structure, "get_chemical_symbols"):
        return list(structure.get_chemical_symbols())
    symbols = getattr(structure, "symbols", None)
    if symbols is None:
        raise TypeError("structure must provide get_chemical_symbols() or symbols")
    return [str(symbol) for symbol in symbols]


def _empty_prediction() -> Prediction:
    return Prediction(
        energy=np.asarray([], dtype=np.float64),
        potential=np.asarray([], dtype=np.float64),
        forces=np.empty((0, 3), dtype=np.float64),
        virials=np.empty((0, 9), dtype=np.float64),
        structure_virials=np.empty((0, 9), dtype=np.float64),
        atom_counts=np.asarray([], dtype=np.int32),
    )


class NEPCalculator:
    def __init__(self, model_file: str | Path = "nep.txt", backend: str = "cpu_nep3"):
        self.model_path = Path(model_file)
        self.backend = backend
        self.type_dict = _read_type_map(self.model_path)
        self.element_list = list(self.type_dict)
        self.model: Model = load_model(backend, str(self.model_path))
        self.descriptor_dim = int(self.model.model_info().get("descriptor_dim", 0))
        self.initialized = True

    def close(self) -> None:
        if self.model is not None:
            self.model.close()

    def __enter__(self) -> "NEPCalculator":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def compose_structures(self, structures) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        structure_list = _as_structure_list(structures)
        atom_counts = np.asarray([len(_structure_symbols(item)) for item in structure_list], dtype=np.int32)
        types = np.empty(int(atom_counts.sum()), dtype=np.int32)
        positions = np.empty((int(atom_counts.sum()), 3), dtype=np.float64)
        boxes = np.empty((len(structure_list), 9), dtype=np.float64)
        pbc = np.empty((len(structure_list), 3), dtype=np.int32)

        cursor = 0
        for structure_index, structure in enumerate(structure_list):
            symbols = _structure_symbols(structure)
            count = len(symbols)
            try:
                types[cursor : cursor + count] = [self.type_dict[symbol] for symbol in symbols]
            except KeyError as exc:
                raise ValueError(f"element {exc.args[0]!r} is absent from the NEP model") from exc
            positions[cursor : cursor + count] = np.asarray(structure.positions, dtype=np.float64)
            boxes[structure_index] = _cell_to_nep_box(structure.cell)
            pbc[structure_index] = _structure_pbc(structure)
            cursor += count
        return types, positions, boxes, pbc

    def predict_arrays(
        self,
        types,
        positions,
        boxes,
        atom_counts=None,
        pbc=None,
    ) -> Prediction:
        types_array = np.ascontiguousarray(types, dtype=np.int32)
        positions_array = np.ascontiguousarray(positions, dtype=np.float64)
        if atom_counts is None:
            atom_counts_array = np.asarray([len(types_array)], dtype=np.int32)
        else:
            atom_counts_array = np.ascontiguousarray(atom_counts, dtype=np.int32)
        if len(atom_counts_array) == 0:
            return _empty_prediction()
        boxes_array = np.ascontiguousarray(boxes, dtype=np.float64)
        if boxes_array.ndim == 1 and len(atom_counts_array) > 1:
            boxes_array = np.tile(boxes_array.reshape(1, 9), (len(atom_counts_array), 1))

        potentials, forces, virials = self.model.calculate(
            types_array,
            boxes_array,
            positions_array,
            atom_counts_array,
            pbc,
        )
        offsets = np.r_[0, np.cumsum(atom_counts_array)]
        energy = np.asarray(
            [potentials[offsets[i] : offsets[i + 1]].sum() for i in range(len(atom_counts_array))],
            dtype=np.float64,
        )
        structure_virials = np.asarray(
            [virials[offsets[i] : offsets[i + 1]].mean(axis=0) for i in range(len(atom_counts_array))],
            dtype=np.float64,
        )
        return Prediction(
            energy=energy,
            potential=np.asarray(potentials, dtype=np.float64),
            forces=np.asarray(forces, dtype=np.float64),
            virials=np.asarray(virials, dtype=np.float64),
            structure_virials=structure_virials,
            atom_counts=atom_counts_array,
        )

    def predict_structures(self, structures) -> Prediction:
        structure_list = _as_structure_list(structures)
        if not structure_list:
            return _empty_prediction()
        types, positions, boxes, pbc = self.compose_structures(structure_list)
        atom_counts = np.asarray([len(_structure_symbols(item)) for item in structure_list], dtype=np.int32)
        return self.predict_arrays(types, positions, boxes, atom_counts, pbc)

    def predict_descriptors_arrays(
        self,
        types,
        positions,
        boxes,
        atom_counts=None,
        pbc=None,
    ) -> np.ndarray:
        types_array = np.ascontiguousarray(types, dtype=np.int32)
        positions_array = np.ascontiguousarray(positions, dtype=np.float64)
        if atom_counts is None:
            atom_counts_array = np.asarray([len(types_array)], dtype=np.int32)
        else:
            atom_counts_array = np.ascontiguousarray(atom_counts, dtype=np.int32)
        if len(atom_counts_array) == 0:
            return np.empty((0, self.descriptor_dim), dtype=np.float64)
        boxes_array = np.ascontiguousarray(boxes, dtype=np.float64)
        if boxes_array.ndim == 1 and len(atom_counts_array) > 1:
            boxes_array = np.tile(boxes_array.reshape(1, 9), (len(atom_counts_array), 1))
        return np.asarray(
            self.model.descriptors(
                types_array,
                boxes_array,
                positions_array,
                atom_counts_array,
                pbc,
            ),
            dtype=np.float64,
        )

    def predict_descriptors(self, structures) -> np.ndarray:
        structure_list = _as_structure_list(structures)
        if not structure_list:
            return np.empty((0, self.descriptor_dim), dtype=np.float64)
        types, positions, boxes, pbc = self.compose_structures(structure_list)
        atom_counts = np.asarray([len(_structure_symbols(item)) for item in structure_list], dtype=np.int32)
        return self.predict_descriptors_arrays(types, positions, boxes, atom_counts, pbc)

    def calculate(self, structures, mean_virial: bool = True):
        prediction = self.predict_structures(structures)
        return (
            prediction.energy,
            prediction.force_blocks(),
            prediction.virial_blocks(mean=mean_virial),
        )

    def get_descriptor(self, structure) -> np.ndarray:
        return self.get_structures_descriptor([structure], mean_descriptor=False)

    def get_structures_descriptor(self, structures, mean_descriptor: bool = True) -> np.ndarray:
        structure_list = _as_structure_list(structures)
        if not structure_list:
            return np.empty((0, self.descriptor_dim), dtype=np.float32)
        descriptors = self.predict_descriptors(structure_list).astype(np.float32, copy=False)
        if not mean_descriptor:
            return descriptors

        atom_counts = np.asarray([len(_structure_symbols(item)) for item in structure_list], dtype=np.int32)
        offsets = np.r_[0, np.cumsum(atom_counts)]
        return np.asarray(
            [descriptors[offsets[i] : offsets[i + 1]].mean(axis=0) for i in range(len(atom_counts))],
            dtype=np.float32,
        )


NepCalculator = NEPCalculator
Nep3Calculator = NEPCalculator
