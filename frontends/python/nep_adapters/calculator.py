"""Small NumPy-first calculator facade for NEPAdapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .errors import InvalidInputError, OutOfMemoryError
from .runtime import Model, backend_status, load_model


_FULLY_PERIODIC_PBC = (1, 1, 1)


_CAPABILITY_NAMES = {
    1 << 0: "batch_find_force",
    1 << 1: "external_neighbors",
    1 << 2: "device_input",
    1 << 3: "spin",
    1 << 4: "charge",
    1 << 5: "virial",
    1 << 6: "descriptors",
    1 << 7: "spin_energy_transfer",
    1 << 8: "dipole",
    1 << 9: "polarizability",
    1 << 10: "dftd3",
    1 << 11: "evaluate_with_descriptors",
}


@dataclass(frozen=True)
class ModelInfo:
    model_type: str
    elements: tuple[str, ...]
    num_types: int
    descriptor_dim: int
    cutoff_radial: float
    cutoff_angular: float
    cutoff_max: float
    capabilities: int
    capability_names: frozenset[str]
    sha256: str
    backend: str

    def supports(self, capability: str) -> bool:
        return capability in self.capability_names


@dataclass(frozen=True)
class WorkspaceEstimate:
    model_bytes: int
    workspace_bytes: int
    total_bytes: int
    atom_capacity: int
    structure_capacity: int


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


def _read_model_header(model_path: str | Path) -> tuple[tuple[str, ...], str]:
    path = Path(model_path)
    try:
        content = path.read_bytes()
        first_line = content.decode("utf-8").splitlines()[0]
    except (OSError, UnicodeError, IndexError) as error:
        raise InvalidInputError(
            f"failed to read NEP model header: {error}",
            operation="inspect_model",
        ) from error
    fields = first_line.split()
    if len(fields) < 3 or not fields[0].startswith("nep"):
        raise InvalidInputError(
            "failed to read NEP element list from model header",
            operation="inspect_model",
        )
    try:
        num_types = int(fields[1])
    except ValueError as error:
        raise InvalidInputError(
            "failed to read NEP type count from model header",
            operation="inspect_model",
        ) from error
    if len(fields) < 2 + num_types:
        raise InvalidInputError(
            "failed to read NEP element list from model header",
            operation="inspect_model",
        )
    return tuple(fields[2 : 2 + num_types]), hashlib.sha256(content).hexdigest()


def _model_info_from_native(
    model_path: str | Path,
    backend: str,
    native_info: dict,
) -> ModelInfo:
    elements, digest = _read_model_header(model_path)
    capabilities = int(native_info.get("capabilities", 0))
    names = frozenset(
        name for flag, name in _CAPABILITY_NAMES.items() if capabilities & flag
    )
    return ModelInfo(
        model_type=str(native_info["model_type"]),
        elements=elements,
        num_types=int(native_info["num_types"]),
        descriptor_dim=int(native_info["descriptor_dim"]),
        cutoff_radial=float(native_info["cutoff_radial"]),
        cutoff_angular=float(native_info["cutoff_angular"]),
        cutoff_max=float(native_info["cutoff_max"]),
        capabilities=capabilities,
        capability_names=names,
        sha256=digest,
        backend=backend,
    )


def inspect_model(model_path: str | Path) -> ModelInfo:
    """Inspect model semantics through the canonical CPU parser."""
    path = Path(model_path)
    with load_model("cpu", str(path)) as model:
        return _model_info_from_native(path, "cpu", model.model_info())


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
    if isinstance(pbc, str):
        tokens = pbc.replace(",", " ").split()
        truth_values = {"1": 1, "t": 1, "true": 1, "0": 0, "f": 0, "false": 0}
        try:
            pbc = [truth_values[token.lower()] for token in tokens]
        except KeyError as error:
            raise ValueError(f"invalid structure pbc token: {error.args[0]!r}") from error
    return np.asarray(pbc, dtype=np.int32).reshape(3)


def _structure_symbols(structure) -> list[str]:
    if hasattr(structure, "get_chemical_symbols"):
        return list(structure.get_chemical_symbols())
    symbols = getattr(structure, "symbols", None)
    if symbols is None:
        raise TypeError("structure must provide get_chemical_symbols() or symbols")
    return [str(symbol) for symbol in symbols]


def _structure_spins(
    structure,
    *,
    required: bool = True,
) -> np.ndarray | None:
    atom_count = len(_structure_symbols(structure))
    explicit: list[tuple[str, np.ndarray]] = []
    atomic_properties = getattr(structure, "atomic_properties", None)
    arrays = getattr(structure, "arrays", None)
    for container_name, container in (
        ("atomic_properties", atomic_properties),
        ("arrays", arrays),
    ):
        if container is None:
            continue
        for key in ("spin", "spins"):
            if key in container:
                explicit.append(
                    (f"{container_name}[{key!r}]", np.asarray(container[key], dtype=np.float64))
                )
    for key in ("spin", "spins"):
        value = getattr(structure, key, None)
        if value is not None:
            explicit.append((key, np.asarray(value, dtype=np.float64)))

    for name, value in explicit:
        if value.shape != (atom_count, 3):
            raise InvalidInputError(
                f"{name} must have shape (natoms, 3)",
                operation="compose_spin_structures",
            )
        if not np.isfinite(value).all():
            raise InvalidInputError(
                f"{name} contains non-finite spin values",
                operation="compose_spin_structures",
            )
    if explicit:
        source_name, result = explicit[0]
        for other_name, other in explicit[1:]:
            if not np.array_equal(result, other):
                raise InvalidInputError(
                    f"ambiguous spin inputs: {source_name} and {other_name} differ",
                    operation="compose_spin_structures",
                )
    else:
        result = None

    initial = None
    if arrays is not None and "initial_magmoms" in arrays:
        initial = np.asarray(arrays["initial_magmoms"], dtype=np.float64)
        if initial.ndim == 1:
            raise InvalidInputError(
                "scalar ASE initial_magmoms cannot be used as vector spins; "
                "lift them explicitly before prediction",
                operation="compose_spin_structures",
            )
        if initial.shape != (atom_count, 3):
            raise InvalidInputError(
                "ASE initial_magmoms must have shape (natoms, 3)",
                operation="compose_spin_structures",
            )
        if not np.isfinite(initial).all():
            raise InvalidInputError(
                "ASE initial_magmoms contains non-finite values",
                operation="compose_spin_structures",
            )
        if result is not None and not np.array_equal(result, initial):
            raise InvalidInputError(
                "explicit spin values and ASE initial_magmoms differ",
                operation="compose_spin_structures",
            )
        if result is None:
            result = initial

    if result is None and required:
        raise InvalidInputError(
            "spin structures must provide vector spin/spins values or vector ASE initial_magmoms",
            operation="compose_spin_structures",
        )
    if result is None:
        return None
    return np.ascontiguousarray(result, dtype=np.float64)


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
        self.model: Model = load_model(backend, str(self.model_path))
        self.model_info = _model_info_from_native(
            self.model_path,
            backend,
            self.model.model_info(),
        )
        self.element_list = list(self.model_info.elements)
        self.type_dict = {
            symbol: index for index, symbol in enumerate(self.element_list)
        }
        self.descriptor_dim = self.model_info.descriptor_dim
        self.initialized = True

    def estimate_workspace(
        self,
        atom_capacity: int,
        structure_capacity: int = 1,
    ) -> WorkspaceEstimate:
        raw = self.model.workspace_estimate(atom_capacity, structure_capacity)
        return WorkspaceEstimate(
            model_bytes=int(raw["model_bytes"]),
            workspace_bytes=int(raw["workspace_bytes"]),
            total_bytes=int(raw["total_bytes"]),
            atom_capacity=int(raw["atom_capacity"]),
            structure_capacity=int(raw["structure_capacity"]),
        )

    def recommend_max_atoms(
        self,
        *,
        memory_fraction: float = 0.70,
        reserve_bytes: int = 256 * 1024 * 1024,
        upper_bound: int = 100_000_000,
    ) -> int | None:
        """Return a conservative CUDA workspace atom capacity."""
        if self.backend != "cuda":
            return None
        if not 0.0 < memory_fraction <= 1.0:
            raise InvalidInputError("memory_fraction must be in (0, 1]")
        status = backend_status("cuda")
        if not status.available or status.free_memory_bytes is None:
            return None
        budget = int(status.free_memory_bytes * memory_fraction) - int(reserve_bytes)
        if budget <= 0:
            raise OutOfMemoryError(
                "CUDA free memory is below the reserved safety margin",
                backend="cuda",
                operation="recommend_max_atoms",
            )

        def fits(atom_capacity: int) -> bool:
            return self.estimate_workspace(atom_capacity).workspace_bytes <= budget

        if not fits(1):
            raise OutOfMemoryError(
                "CUDA workspace cannot fit even one atom within the configured budget",
                backend="cuda",
                operation="recommend_max_atoms",
            )
        low = 1
        high = 2
        while high < upper_bound and fits(high):
            low = high
            high = min(upper_bound, high * 2)
        if high == upper_bound and fits(high):
            return high
        while low + 1 < high:
            middle = low + (high - low) // 2
            if fits(middle):
                low = middle
            else:
                high = middle
        return low

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
                    raise InvalidInputError(
                        "spins must have shape (total_atoms, 3)",
                        operation="compose_spin_structures",
                    ) from exc
                if len(spin_blocks) != len(atom_counts) or any(
                    block.shape != (count, 3)
                    for block, count in zip(spin_blocks, atom_counts)
                ):
                    raise InvalidInputError(
                        "spins must have shape (total_atoms, 3) or one (natoms, 3) block per structure",
                        operation="compose_spin_structures",
                    )
                spin_array = np.concatenate(spin_blocks, axis=0)
            if not np.isfinite(spin_array).all():
                raise InvalidInputError(
                    "spins contains non-finite values",
                    operation="compose_spin_structures",
                )
            cursor = 0
            for structure, count in zip(structure_list, atom_counts):
                embedded = _structure_spins(structure, required=False)
                if embedded is not None and not np.array_equal(
                    spin_array[cursor : cursor + count], embedded
                ):
                    raise InvalidInputError(
                        "explicit spins and structure spin metadata differ",
                        operation="compose_spin_structures",
                    )
                cursor += count
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

    def predict_with_descriptors_arrays(
        self,
        types,
        positions,
        boxes,
        atom_counts=None,
        pbc=_FULLY_PERIODIC_PBC,
    ) -> tuple[Prediction, np.ndarray]:
        types_array = np.ascontiguousarray(types, dtype=np.int32)
        positions_array = np.ascontiguousarray(positions, dtype=np.float64)
        if atom_counts is None:
            atom_counts_array = np.asarray([len(types_array)], dtype=np.int32)
        else:
            atom_counts_array = np.ascontiguousarray(atom_counts, dtype=np.int32)
        if len(atom_counts_array) == 0:
            return (
                _empty_prediction(),
                np.empty((0, self.descriptor_dim), dtype=np.float64),
            )
        boxes_array = np.ascontiguousarray(boxes, dtype=np.float64)
        if boxes_array.ndim == 1 and len(atom_counts_array) > 1:
            boxes_array = np.tile(
                boxes_array.reshape(1, 9), (len(atom_counts_array), 1)
            )
        potentials, forces, virials, descriptors = (
            self.model.calculate_with_descriptors(
                types_array,
                boxes_array,
                positions_array,
                atom_counts_array,
                pbc,
            )
        )
        return (
            _prediction_from_outputs(
                potentials, forces, virials, atom_counts_array
            ),
            np.asarray(descriptors, dtype=np.float64),
        )

    def predict_with_descriptors_structures(
        self, structures
    ) -> tuple[Prediction, np.ndarray]:
        structure_list = _as_structure_list(structures)
        if not structure_list:
            return (
                _empty_prediction(),
                np.empty((0, self.descriptor_dim), dtype=np.float64),
            )
        types, positions, boxes, pbc = self.compose_structures(structure_list)
        atom_counts = np.asarray(
            [len(_structure_symbols(item)) for item in structure_list],
            dtype=np.int32,
        )
        return self.predict_with_descriptors_arrays(
            types, positions, boxes, atom_counts, pbc
        )

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
