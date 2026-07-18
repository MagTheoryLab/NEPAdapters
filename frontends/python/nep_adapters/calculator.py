"""Small NumPy-first calculator facade for NEPAdapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .runtime import Model, load_model


_FULLY_PERIODIC_PBC = (1, 1, 1)


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


@dataclass(frozen=True)
class SpinPrediction(Prediction):
    mforces: np.ndarray

    def mforce_blocks(self) -> list[np.ndarray]:
        if len(self.atom_counts) == 0:
            return []
        return list(np.split(self.mforces, np.cumsum(self.atom_counts)[:-1]))


@dataclass(frozen=True)
class ChargePrediction(Prediction):
    charges: np.ndarray
    becs: np.ndarray

    def charge_blocks(self) -> list[np.ndarray]:
        if len(self.atom_counts) == 0:
            return []
        return list(np.split(self.charges, np.cumsum(self.atom_counts)[:-1]))

    def bec_blocks(self) -> list[np.ndarray]:
        if len(self.atom_counts) == 0:
            return []
        return list(np.split(self.becs, np.cumsum(self.atom_counts)[:-1]))


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
        return np.asarray(_FULLY_PERIODIC_PBC, dtype=np.int32)
    return np.asarray(pbc, dtype=np.int32).reshape(3)


def _structure_symbols(structure) -> list[str]:
    if hasattr(structure, "get_chemical_symbols"):
        return list(structure.get_chemical_symbols())
    symbols = getattr(structure, "symbols", None)
    if symbols is None:
        raise TypeError("structure must provide get_chemical_symbols() or symbols")
    return [str(symbol) for symbol in symbols]


def _structure_spins(structure) -> np.ndarray:
    arrays = getattr(structure, "arrays", None)
    if arrays is not None and "spins" in arrays:
        spins = arrays["spins"]
    else:
        spins = getattr(structure, "spins", None)
    if spins is None:
        raise TypeError(
            "spin structures must provide a (natoms, 3) 'spins' attribute or arrays['spins']"
        )
    result = np.asarray(spins, dtype=np.float64)
    if result.shape != (len(_structure_symbols(structure)), 3):
        raise ValueError("structure spins must have shape (natoms, 3)")
    return result


def _empty_prediction() -> Prediction:
    return Prediction(
        energy=np.asarray([], dtype=np.float64),
        potential=np.asarray([], dtype=np.float64),
        forces=np.empty((0, 3), dtype=np.float64),
        virials=np.empty((0, 9), dtype=np.float64),
        structure_virials=np.empty((0, 9), dtype=np.float64),
        atom_counts=np.asarray([], dtype=np.int32),
    )


def _empty_spin_prediction() -> SpinPrediction:
    empty = _empty_prediction()
    return SpinPrediction(
        energy=empty.energy,
        potential=empty.potential,
        forces=empty.forces,
        virials=empty.virials,
        structure_virials=empty.structure_virials,
        atom_counts=empty.atom_counts,
        mforces=np.empty((0, 3), dtype=np.float64),
    )


def _empty_charge_prediction() -> ChargePrediction:
    empty = _empty_prediction()
    return ChargePrediction(
        energy=empty.energy,
        potential=empty.potential,
        forces=empty.forces,
        virials=empty.virials,
        structure_virials=empty.structure_virials,
        atom_counts=empty.atom_counts,
        charges=np.asarray([], dtype=np.float64),
        becs=np.empty((0, 9), dtype=np.float64),
    )


def _prediction_from_outputs(
    potentials,
    forces,
    virials,
    atom_counts: np.ndarray,
) -> Prediction:
    potential_array = np.asarray(potentials, dtype=np.float64)
    force_array = np.asarray(forces, dtype=np.float64)
    virial_array = np.asarray(virials, dtype=np.float64)
    offsets = np.r_[0, np.cumsum(atom_counts)]
    energy = np.asarray(
        [potential_array[offsets[i] : offsets[i + 1]].sum() for i in range(len(atom_counts))],
        dtype=np.float64,
    )
    structure_virials = np.asarray(
        [virial_array[offsets[i] : offsets[i + 1]].mean(axis=0) for i in range(len(atom_counts))],
        dtype=np.float64,
    )
    return Prediction(
        energy=energy,
        potential=potential_array,
        forces=force_array,
        virials=virial_array,
        structure_virials=structure_virials,
        atom_counts=atom_counts,
    )


class NEPCalculator:
    def __init__(self, model_file: str | Path = "nep.txt", backend: str = "cpu"):
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

    def cancel(self) -> None:
        self.model.cancel()

    def reset_cancel(self) -> None:
        self.model.reset_cancel()

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

    def compose_spin_structures(
        self,
        structures,
        spins=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        structure_list = _as_structure_list(structures)
        types, positions, boxes, pbc = self.compose_structures(structure_list)
        atom_counts = [len(_structure_symbols(item)) for item in structure_list]
        if spins is None:
            spin_blocks = [_structure_spins(item) for item in structure_list]
            spin_array = np.concatenate(spin_blocks, axis=0)
        else:
            try:
                spin_array = np.asarray(spins, dtype=np.float64)
            except (TypeError, ValueError):
                spin_array = np.empty((0, 3), dtype=np.float64)
            if spin_array.shape != (len(types), 3):
                try:
                    spin_blocks = [
                        np.asarray(block, dtype=np.float64) for block in spins
                    ]
                except TypeError as exc:
                    raise ValueError("spins must have shape (total_atoms, 3)") from exc
                if len(spin_blocks) != len(atom_counts) or any(
                    block.shape != (count, 3)
                    for block, count in zip(spin_blocks, atom_counts)
                ):
                    raise ValueError(
                        "spins must have shape (total_atoms, 3) or one (natoms, 3) block per structure"
                    )
                spin_array = np.concatenate(spin_blocks, axis=0)
        return types, positions, boxes, pbc, np.ascontiguousarray(spin_array)

    def predict_arrays(
        self,
        types,
        positions,
        boxes,
        atom_counts=None,
        pbc=_FULLY_PERIODIC_PBC,
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
        return _prediction_from_outputs(
            potentials, forces, virials, atom_counts_array
        )

    def predict_structures(self, structures) -> Prediction:
        structure_list = _as_structure_list(structures)
        if not structure_list:
            return _empty_prediction()
        types, positions, boxes, pbc = self.compose_structures(structure_list)
        atom_counts = np.asarray([len(_structure_symbols(item)) for item in structure_list], dtype=np.int32)
        return self.predict_arrays(types, positions, boxes, atom_counts, pbc)

    def predict_charge_arrays(
        self,
        types,
        positions,
        boxes,
        atom_counts=None,
        pbc=_FULLY_PERIODIC_PBC,
    ) -> ChargePrediction:
        types_array = np.ascontiguousarray(types, dtype=np.int32)
        positions_array = np.ascontiguousarray(positions, dtype=np.float64)
        atom_counts_array = (
            np.asarray([len(types_array)], dtype=np.int32)
            if atom_counts is None
            else np.ascontiguousarray(atom_counts, dtype=np.int32)
        )
        if len(atom_counts_array) == 0:
            return _empty_charge_prediction()
        boxes_array = np.ascontiguousarray(boxes, dtype=np.float64)
        if boxes_array.ndim == 1 and len(atom_counts_array) > 1:
            boxes_array = np.tile(boxes_array.reshape(1, 9), (len(atom_counts_array), 1))
        potentials, forces, virials, charges, becs = self.model.calculate_charge(
            types_array,
            boxes_array,
            positions_array,
            atom_counts_array,
            pbc,
        )
        base = _prediction_from_outputs(
            potentials, forces, virials, atom_counts_array
        )
        return ChargePrediction(
            energy=base.energy,
            potential=base.potential,
            forces=base.forces,
            virials=base.virials,
            structure_virials=base.structure_virials,
            atom_counts=base.atom_counts,
            charges=np.asarray(charges, dtype=np.float64),
            becs=np.asarray(becs, dtype=np.float64),
        )

    def predict_charge_structures(self, structures) -> ChargePrediction:
        structure_list = _as_structure_list(structures)
        if not structure_list:
            return _empty_charge_prediction()
        types, positions, boxes, pbc = self.compose_structures(structure_list)
        atom_counts = np.asarray(
            [len(_structure_symbols(item)) for item in structure_list],
            dtype=np.int32,
        )
        return self.predict_charge_arrays(
            types, positions, boxes, atom_counts, pbc
        )

    def predict_dipoles(self, structures) -> np.ndarray:
        structure_list = _as_structure_list(structures)
        if not structure_list:
            return np.empty((0, 3), dtype=np.float64)
        types, positions, boxes, pbc = self.compose_structures(structure_list)
        atom_counts = np.asarray(
            [len(_structure_symbols(item)) for item in structure_list],
            dtype=np.int32,
        )
        return np.asarray(
            self.model.dipoles(types, boxes, positions, atom_counts, pbc),
            dtype=np.float64,
        )

    def predict_polarizabilities(self, structures) -> np.ndarray:
        structure_list = _as_structure_list(structures)
        if not structure_list:
            return np.empty((0, 6), dtype=np.float64)
        types, positions, boxes, pbc = self.compose_structures(structure_list)
        atom_counts = np.asarray(
            [len(_structure_symbols(item)) for item in structure_list],
            dtype=np.int32,
        )
        return np.asarray(
            self.model.polarizabilities(
                types, boxes, positions, atom_counts, pbc
            ),
            dtype=np.float64,
        )

    def get_structures_dipole(self, structures) -> np.ndarray:
        return self.predict_dipoles(structures)

    def get_structures_polarizability(self, structures) -> np.ndarray:
        return self.predict_polarizabilities(structures)

    def _predict_dftd3_structures(
        self,
        structures,
        functional: str,
        cutoff: float,
        cutoff_cn: float,
        include_nep: bool,
    ) -> Prediction:
        structure_list = _as_structure_list(structures)
        if not structure_list:
            return _empty_prediction()
        types, positions, boxes, pbc = self.compose_structures(structure_list)
        atom_counts = np.asarray(
            [len(_structure_symbols(item)) for item in structure_list],
            dtype=np.int32,
        )
        method = (
            self.model.calculate_with_dftd3
            if include_nep
            else self.model.calculate_dftd3
        )
        potentials, forces, virials = method(
            types,
            boxes,
            positions,
            atom_counts,
            functional,
            float(cutoff),
            float(cutoff_cn),
            pbc,
        )
        return _prediction_from_outputs(
            potentials, forces, virials, atom_counts
        )

    def predict_dftd3_structures(
        self,
        structures,
        functional: str,
        cutoff: float,
        cutoff_cn: float,
    ) -> Prediction:
        return self._predict_dftd3_structures(
            structures, functional, cutoff, cutoff_cn, False
        )

    def predict_with_dftd3_structures(
        self,
        structures,
        functional: str,
        cutoff: float,
        cutoff_cn: float,
    ) -> Prediction:
        return self._predict_dftd3_structures(
            structures, functional, cutoff, cutoff_cn, True
        )

    def predict_spin_arrays(
        self,
        types,
        positions,
        spins,
        boxes,
        atom_counts=None,
        pbc=_FULLY_PERIODIC_PBC,
    ) -> SpinPrediction:
        types_array = np.ascontiguousarray(types, dtype=np.int32)
        positions_array = np.ascontiguousarray(positions, dtype=np.float64)
        spins_array = np.ascontiguousarray(spins, dtype=np.float64)
        if atom_counts is None:
            atom_counts_array = np.asarray([len(types_array)], dtype=np.int32)
        else:
            atom_counts_array = np.ascontiguousarray(atom_counts, dtype=np.int32)
        if len(atom_counts_array) == 0:
            return _empty_spin_prediction()
        boxes_array = np.ascontiguousarray(boxes, dtype=np.float64)
        if boxes_array.ndim == 1 and len(atom_counts_array) > 1:
            boxes_array = np.tile(boxes_array.reshape(1, 9), (len(atom_counts_array), 1))

        potentials, forces, virials, mforces = self.model.calculate_spin(
            types_array,
            boxes_array,
            positions_array,
            spins_array,
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
        return SpinPrediction(
            energy=energy,
            potential=np.asarray(potentials, dtype=np.float64),
            forces=np.asarray(forces, dtype=np.float64),
            virials=np.asarray(virials, dtype=np.float64),
            structure_virials=structure_virials,
            atom_counts=atom_counts_array,
            mforces=np.asarray(mforces, dtype=np.float64),
        )

    def predict_spin_structures(self, structures, spins=None) -> SpinPrediction:
        structure_list = _as_structure_list(structures)
        if not structure_list:
            return _empty_spin_prediction()
        types, positions, boxes, pbc, spin_array = self.compose_spin_structures(
            structure_list,
            spins,
        )
        atom_counts = np.asarray(
            [len(_structure_symbols(item)) for item in structure_list],
            dtype=np.int32,
        )
        return self.predict_spin_arrays(
            types,
            positions,
            spin_array,
            boxes,
            atom_counts,
            pbc,
        )

    def predict_descriptors_arrays(
        self,
        types,
        positions,
        boxes,
        atom_counts=None,
        pbc=_FULLY_PERIODIC_PBC,
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

    def predict_spin_descriptors_arrays(
        self,
        types,
        positions,
        spins,
        boxes,
        atom_counts=None,
        pbc=_FULLY_PERIODIC_PBC,
    ) -> np.ndarray:
        types_array = np.ascontiguousarray(types, dtype=np.int32)
        positions_array = np.ascontiguousarray(positions, dtype=np.float64)
        spins_array = np.ascontiguousarray(spins, dtype=np.float64)
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
            self.model.descriptors_spin(
                types_array,
                boxes_array,
                positions_array,
                spins_array,
                atom_counts_array,
                pbc,
            ),
            dtype=np.float64,
        )

    def predict_spin_descriptors(self, structures, spins=None) -> np.ndarray:
        structure_list = _as_structure_list(structures)
        if not structure_list:
            return np.empty((0, self.descriptor_dim), dtype=np.float64)
        types, positions, boxes, pbc, spin_array = self.compose_spin_structures(
            structure_list,
            spins,
        )
        atom_counts = np.asarray(
            [len(_structure_symbols(item)) for item in structure_list],
            dtype=np.int32,
        )
        return self.predict_spin_descriptors_arrays(
            types,
            positions,
            spin_array,
            boxes,
            atom_counts,
            pbc,
        )

    def calculate(self, structures, mean_virial: bool = True):
        prediction = self.predict_structures(structures)
        return (
            prediction.energy,
            prediction.force_blocks(),
            prediction.virial_blocks(mean=mean_virial),
        )

    def calculate_charge(self, structures, mean_virial: bool = True):
        prediction = self.predict_charge_structures(structures)
        return (
            prediction.energy,
            prediction.force_blocks(),
            prediction.virial_blocks(mean=mean_virial),
            prediction.charge_blocks(),
            prediction.bec_blocks(),
        )

    def calculate_dftd3(
        self,
        structures,
        functional: str,
        cutoff: float,
        cutoff_cn: float,
        mean_virial: bool = True,
    ):
        prediction = self.predict_dftd3_structures(
            structures, functional, cutoff, cutoff_cn
        )
        return (
            prediction.energy,
            prediction.force_blocks(),
            prediction.virial_blocks(mean=mean_virial),
        )

    def calculate_with_dftd3(
        self,
        structures,
        functional: str,
        cutoff: float,
        cutoff_cn: float,
        mean_virial: bool = True,
    ):
        prediction = self.predict_with_dftd3_structures(
            structures, functional, cutoff, cutoff_cn
        )
        return (
            prediction.energy,
            prediction.force_blocks(),
            prediction.virial_blocks(mean=mean_virial),
        )

    def calculate_spin(self, structures, spins=None, mean_virial: bool = True):
        prediction = self.predict_spin_structures(structures, spins)
        return (
            prediction.energy,
            prediction.force_blocks(),
            prediction.virial_blocks(mean=mean_virial),
            prediction.mforce_blocks(),
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

    def get_spin_descriptor(self, structure, spins=None) -> np.ndarray:
        return self.get_spin_structures_descriptor(
            [structure],
            spins=spins,
            mean_descriptor=False,
        )

    def get_spin_structures_descriptor(
        self,
        structures,
        spins=None,
        mean_descriptor: bool = True,
    ) -> np.ndarray:
        structure_list = _as_structure_list(structures)
        if not structure_list:
            return np.empty((0, self.descriptor_dim), dtype=np.float32)
        descriptors = self.predict_spin_descriptors(
            structure_list,
            spins,
        ).astype(np.float32, copy=False)
        if not mean_descriptor:
            return descriptors

        atom_counts = np.asarray(
            [len(_structure_symbols(item)) for item in structure_list],
            dtype=np.int32,
        )
        offsets = np.r_[0, np.cumsum(atom_counts)]
        return np.asarray(
            [descriptors[offsets[i] : offsets[i + 1]].mean(axis=0) for i in range(len(atom_counts))],
            dtype=np.float32,
        )
