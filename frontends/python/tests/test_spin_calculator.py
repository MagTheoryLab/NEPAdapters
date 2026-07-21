import os
import sys

import numpy as np

import nep_adapters


class SpinStructure:
    def __init__(self, spins):
        self.symbols = ["Fe"] * 4
        self.positions = np.asarray(
            [
                [0.2, 0.2, 0.2],
                [3.7, 0.3, 0.2],
                [0.4, 3.6, 0.5],
                [1.8, 1.7, 3.5],
            ],
            dtype=np.float64,
        )
        self.cell = np.diag([4.0, 4.0, 4.0])
        self.pbc = np.asarray([1, 1, 1], dtype=np.int32)
        self.spins = np.asarray(spins, dtype=np.float64)


class SpinAliasStructure(SpinStructure):
    def __init__(self, spins, *, source, extra=None):
        super().__init__(spins)
        del self.spins
        values = np.asarray(spins, dtype=np.float64)
        if source == "atomic_spin":
            self.atomic_properties = {"spin": values}
        elif source == "atomic_spins":
            self.atomic_properties = {"spins": values}
        elif source == "arrays_spin":
            self.arrays = {"spin": values}
        elif source == "initial_magmoms":
            self.arrays = {"initial_magmoms": values}
        elif source == "scalar_initial_magmoms":
            self.arrays = {"initial_magmoms": values[:, 2]}
        elif source == "missing":
            pass
        else:
            raise AssertionError(f"unknown test source {source}")
        if extra is not None:
            if not hasattr(self, "arrays"):
                self.arrays = {}
            self.arrays.update(extra)


def read_reference(path):
    blocks = {}
    tokens = open(path, encoding="utf-8").read().split()
    cursor = 0
    while cursor < len(tokens):
        name = tokens[cursor]
        count = int(tokens[cursor + 1])
        cursor += 2
        blocks[name] = np.asarray(tokens[cursor : cursor + count], dtype=np.float64)
        cursor += count
    return blocks


def main():
    model_path = os.environ["NEP_ADAPTERS_PYTHON_SPIN_MODEL"]
    reference = read_reference(os.environ["NEP_ADAPTERS_PYTHON_SPIN_REFERENCE"])
    spins = np.asarray(
        [
            [1.0, 0.2, 0.0],
            [0.4, -0.3, 0.7],
            [-0.2, 0.8, 0.5],
            [0.6, 0.1, -0.4],
        ],
        dtype=np.float64,
    )
    structure = SpinStructure(spins)
    types = np.zeros(4, dtype=np.int32)
    atom_counts = np.asarray([4], dtype=np.int32)
    box = structure.cell.T.reshape(9)

    with nep_adapters.load_model("cpu", model_path) as model:
        info = model.model_info()
        if not info["capabilities"] & (1 << 3):
            raise AssertionError("spin capability is missing")
        outputs = model.calculate_spin(
            types,
            box,
            structure.positions,
            spins,
            atom_counts,
        )
        descriptors = model.descriptors_spin(
            types,
            box,
            structure.positions,
            spins,
            atom_counts,
        )
        batch_size = 32
        batch_descriptors = model.descriptors_spin(
            np.tile(types, batch_size),
            np.tile(box, (batch_size, 1)),
            np.tile(structure.positions, (batch_size, 1)),
            np.tile(spins, (batch_size, 1)),
            np.full(batch_size, len(types), dtype=np.int32),
        )
        try:
            model.calculate(types, box, structure.positions, atom_counts)
        except ValueError:
            pass
        else:
            raise AssertionError("spin model calculation without spins must fail")

    potentials, forces, virials, mforces = outputs
    if potentials.shape != (4,) or forces.shape != (4, 3):
        raise AssertionError("spin calculation output shape mismatch")
    if virials.shape != (4, 9) or mforces.shape != (4, 3):
        raise AssertionError("spin calculation output shape mismatch")
    if descriptors.shape != (4, info["descriptor_dim"]):
        raise AssertionError("spin descriptor output shape mismatch")
    if not np.allclose(potentials, reference["energy_atom"], rtol=0.0, atol=1.0e-10):
        raise AssertionError("spin per-atom energies differ from fixture")
    if not np.allclose(forces.reshape(-1), reference["force"], rtol=0.0, atol=1.0e-10):
        raise AssertionError("spin forces differ from fixture")
    if not np.allclose(mforces.reshape(-1), reference["mforce"], rtol=0.0, atol=1.0e-10):
        raise AssertionError("spin magnetic forces differ from fixture")
    if not np.allclose(virials.sum(axis=0), reference["virial9"], rtol=0.0, atol=1.0e-10):
        raise AssertionError("spin virial differs from fixture")
    if not np.allclose(descriptors.reshape(-1), reference["descriptor"], rtol=0.0, atol=1.0e-10):
        raise AssertionError("spin descriptors differ from fixture")
    if not np.allclose(
        batch_descriptors,
        np.tile(descriptors, (batch_size, 1)),
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise AssertionError("parallel spin descriptor batch differs from serial result")
    with nep_adapters.NEPCalculator(model_path) as calculator:
        prediction = calculator.predict_spin_structures(structure)
        explicit_prediction = calculator.predict_spin_structures(structure, spins)
        arrays_structure = SpinStructure(spins)
        arrays_structure.arrays = {"spins": arrays_structure.spins}
        del arrays_structure.spins
        batch_prediction = calculator.predict_spin_structures(
            [structure, arrays_structure]
        )
        block_prediction = calculator.predict_spin_structures(
            [structure, arrays_structure],
            [spins, spins],
        )
        empty_prediction = calculator.predict_spin_structures([])
        structure_descriptors = calculator.predict_spin_descriptors(structure)
        atom_descriptors = calculator.get_spin_descriptor(structure)
        mean_descriptors = calculator.get_spin_structures_descriptor([structure])
        calculated = calculator.calculate_spin(structure)
        for source in (
            "atomic_spin",
            "atomic_spins",
            "arrays_spin",
            "initial_magmoms",
        ):
            alias_prediction = calculator.predict_spin_structures(
                SpinAliasStructure(spins, source=source)
            )
            if not np.allclose(alias_prediction.mforces, mforces):
                raise AssertionError(f"{source} spin alias differs from explicit spins")
        matching_initial = SpinAliasStructure(
            spins,
            source="arrays_spin",
            extra={"initial_magmoms": spins.copy()},
        )
        calculator.predict_spin_structures(matching_initial)
        calculator.predict_spin_structures(matching_initial, spins=spins.copy())
        invalid_sources = [
            SpinAliasStructure(spins, source="scalar_initial_magmoms"),
            SpinAliasStructure(spins, source="missing"),
            SpinAliasStructure(
                spins,
                source="arrays_spin",
                extra={"initial_magmoms": spins + 1.0},
            ),
            SpinAliasStructure(
                spins,
                source="atomic_spin",
                extra={"spins": spins + 1.0},
            ),
        ]
        for invalid in invalid_sources:
            try:
                calculator.predict_spin_structures(invalid)
            except nep_adapters.InvalidInputError:
                pass
            else:
                raise AssertionError("invalid or ambiguous spin input must fail closed")
        try:
            calculator.predict_spin_structures(
                matching_initial,
                spins=spins + 1.0,
            )
        except nep_adapters.InvalidInputError:
            pass
        else:
            raise AssertionError(
                "explicit spins differing from ASE metadata must fail closed"
            )
        try:
            calculator.predict_spin_arrays(
                types,
                structure.positions,
                spins,
                box,
                pbc=np.asarray([1, 0, 1], dtype=np.int32),
            )
        except ValueError as error:
            if "fully periodic" not in str(error):
                raise
        else:
            raise AssertionError("non-periodic spin input must be rejected")

    if not isinstance(prediction, nep_adapters.SpinPrediction):
        raise AssertionError("high-level spin prediction type mismatch")
    if not np.allclose(prediction.mforces, mforces, rtol=0.0, atol=1.0e-10):
        raise AssertionError("high-level magnetic forces differ from native API")
    if not np.allclose(explicit_prediction.mforces, mforces, rtol=0.0, atol=1.0e-10):
        raise AssertionError("explicit spin input differs from structure spin input")
    if batch_prediction.atom_counts.tolist() != [4, 4]:
        raise AssertionError("spin structure batch atom counts mismatch")
    if len(batch_prediction.mforce_blocks()) != 2:
        raise AssertionError("spin structure batch block split mismatch")
    if not np.allclose(
        batch_prediction.energy,
        [prediction.energy[0], prediction.energy[0]],
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise AssertionError("spin structure batch energy mismatch")
    if not np.allclose(block_prediction.mforces, batch_prediction.mforces):
        raise AssertionError("spin block input differs from structure spin input")
    if empty_prediction.mforces.shape != (0, 3) or empty_prediction.energy.shape != (0,):
        raise AssertionError("empty spin prediction shape mismatch")
    if not np.allclose(structure_descriptors, descriptors, rtol=0.0, atol=1.0e-10):
        raise AssertionError("high-level spin descriptors differ from native API")
    if not np.allclose(atom_descriptors, descriptors.astype(np.float32), rtol=0.0, atol=0.0):
        raise AssertionError("get_spin_descriptor output mismatch")
    if mean_descriptors.shape != (1, info["descriptor_dim"]):
        raise AssertionError("mean spin descriptor shape mismatch")
    if len(calculated) != 4 or calculated[3][0].shape != (4, 3):
        raise AssertionError("calculate_spin facade result mismatch")

    print(
        "python spin calculator:",
        f"atoms={len(types)}",
        f"descriptor_dim={info['descriptor_dim']}",
        f"energy={float(prediction.energy[0]):.12g}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
