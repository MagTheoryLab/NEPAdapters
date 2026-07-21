"""Optional ASE calculator adapter for NEPAdapters."""

from __future__ import annotations

import numpy as np

from .calculator import NEPCalculator, SpinPrediction

try:
    from ase.calculators.calculator import Calculator, all_changes
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.stress import full_3x3_to_voigt_6_stress
except ImportError as exc:  # pragma: no cover - exercised only without ASE installed.
    raise ImportError(
        "nep_adapters.ase requires the optional 'ase' dependency. "
        "Install with: pip install nep-adapters[ase]"
    ) from exc


def _stress_from_nep_compute_virial(virial9, atoms):
    virial_matrix = np.asarray(virial9, dtype=np.float64).reshape(3, 3)
    stress_matrix = virial_matrix * len(atoms) / atoms.get_volume()
    return full_3x3_to_voigt_6_stress(stress_matrix)


class NepAseCalculator(Calculator):
    implemented_properties = [
        "energy",
        "energies",
        "forces",
        "stress",
        "descriptor",
        "mforces",
    ]

    def __init__(self, model_file="nep.txt", backend="cpu", **kwargs):
        super().__init__(**kwargs)
        self._calc = NEPCalculator(model_file, backend=backend)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)

        if self._calc.model_info.model_type == "spin":
            prediction = self._calc.predict_spin_structures([atoms])
        else:
            prediction = self._calc.predict_structures([atoms])
        self.results["energy"] = float(prediction.energy[0])
        self.results["energies"] = prediction.potential
        self.results["forces"] = prediction.forces
        self.results["stress"] = _stress_from_nep_compute_virial(
            prediction.structure_virials[0],
            atoms,
        )
        if "descriptor" in properties:
            if isinstance(prediction, SpinPrediction):
                self.results["descriptor"] = self._calc.get_spin_descriptor(atoms)
            else:
                self.results["descriptor"] = self._calc.get_descriptor(atoms)
        if isinstance(prediction, SpinPrediction):
            self.results["mforces"] = prediction.mforces


def attach_single_point(atoms_or_list, calculator: NEPCalculator, calc_descriptor: bool = False):
    atoms_list = [atoms_or_list] if hasattr(atoms_or_list, "get_chemical_symbols") else list(atoms_or_list)
    if calculator.model_info.model_type == "spin":
        prediction = calculator.predict_spin_structures(atoms_list)
    else:
        prediction = calculator.predict_structures(atoms_list)
    force_blocks = prediction.force_blocks()
    for index, atoms in enumerate(atoms_list):
        spc = SinglePointCalculator(
            atoms,
            energy=float(prediction.energy[index]),
            forces=force_blocks[index],
            stress=_stress_from_nep_compute_virial(prediction.structure_virials[index], atoms),
        )
        if calc_descriptor:
            if isinstance(prediction, SpinPrediction):
                spc.results["descriptor"] = calculator.get_spin_descriptor(atoms)
            else:
                spc.results["descriptor"] = calculator.get_descriptor(atoms)
        if isinstance(prediction, SpinPrediction):
            spc.results["mforces"] = prediction.mforce_blocks()[index]
        atoms.calc = spc
    return atoms_or_list
