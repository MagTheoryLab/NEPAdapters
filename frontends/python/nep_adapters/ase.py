"""Optional ASE calculator adapter for NEPAdapters."""

from __future__ import annotations

import numpy as np

from .calculator import NEPCalculator, SpinPrediction, _structure_spins

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
    """Convert a mean per-atom GPUMD virial to ASE Voigt stress.

    GPUMD uses a pressure-positive virial W.  ASE defines stress as the
    energy derivative with respect to strain, so for a static structure
    sigma_ASE = -W_total / V = -N * W_mean / V.
    """
    virial_matrix = np.asarray(virial9, dtype=np.float64).reshape(3, 3)
    stress_matrix = -virial_matrix * len(atoms) / atoms.get_volume()
    return full_3x3_to_voigt_6_stress(stress_matrix)


def _spin_state_changed(previous, atoms):
    return previous is not None and not np.array_equal(
        _structure_spins(previous, required=False),
        _structure_spins(atoms, required=False),
    )


class _SpinSinglePointCalculator(SinglePointCalculator):
    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol)
        if _spin_state_changed(self.atoms, atoms):
            changes.append("spins")
        return changes


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

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol)
        if self._calc.model_info.model_type == "spin" and _spin_state_changed(self.atoms, atoms):
            changes.append("spins")
        return changes

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
    single_point = _SpinSinglePointCalculator if isinstance(prediction, SpinPrediction) else SinglePointCalculator
    for index, atoms in enumerate(atoms_list):
        spc = single_point(
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
