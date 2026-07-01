import importlib.util
import os

import numpy as np


def main():
    if importlib.util.find_spec("ase") is None:
        print("optional ASE calculator smoke skipped: ase is not installed")
        return

    from ase.io import read
    from ase.stress import full_3x3_to_voigt_6_stress

    import nep_adapters
    from nep_adapters.ase import NepAseCalculator, attach_single_point
    from baseline_utils import read_descriptor_fixture, read_labeled_structure

    model_path = os.environ["NEP_ADAPTERS_PYTHON_TEST_MODEL"]
    xyz_path = os.environ["NEP_ADAPTERS_PYTHON_TEST_XYZ"]
    atoms = read(xyz_path, index=0)
    reference = read_labeled_structure(xyz_path)
    expected_descriptors = read_descriptor_fixture(
        os.path.join(os.path.dirname(xyz_path), "descriptor.txt")
    ).astype(np.float32)

    atoms.calc = NepAseCalculator(model_path)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    descriptors = atoms.calc.get_property("descriptor", atoms)

    core = nep_adapters.NEPCalculator(model_path)
    prediction = core.predict_structures([atoms])
    if abs(float(energy) - float(prediction.energy[0])) > 1.0e-10:
        raise AssertionError("ASE calculator energy differs from core calculator")
    if not np.allclose(forces, prediction.forces, rtol=0.0, atol=1.0e-10):
        raise AssertionError("ASE calculator forces differ from core calculator")
    if stress.shape != (6,):
        raise AssertionError("ASE calculator stress shape mismatch")
    if not np.allclose(descriptors, expected_descriptors, rtol=0.0, atol=1.0e-6):
        raise AssertionError("ASE calculator descriptors differ from fixed baseline labels")
    expected_stress = full_3x3_to_voigt_6_stress(
        reference.virial.reshape(3, 3) / atoms.get_volume()
    )
    if abs(float(energy) - reference.energy) > 1.0e-10:
        raise AssertionError("ASE calculator energy differs from fixed baseline label")
    if not np.allclose(forces, reference.forces, rtol=0.0, atol=1.0e-10):
        raise AssertionError("ASE calculator forces differ from fixed baseline labels")
    if not np.allclose(stress, expected_stress, rtol=0.0, atol=1.0e-10):
        raise AssertionError("ASE calculator stress differs from fixed baseline label")

    atoms2 = atoms.copy()
    attach_single_point(atoms2, core)
    if abs(float(atoms2.get_potential_energy()) - float(prediction.energy[0])) > 1.0e-10:
        raise AssertionError("single-point energy differs from core calculator")
    if not np.allclose(atoms2.get_forces(), prediction.forces, rtol=0.0, atol=1.0e-10):
        raise AssertionError("single-point forces differ from core calculator")
    if not np.allclose(atoms2.get_stress(), expected_stress, rtol=0.0, atol=1.0e-10):
        raise AssertionError("single-point stress differs from fixed baseline label")

    atoms3 = atoms.copy()
    attach_single_point(atoms3, core, calc_descriptor=True)
    if not np.allclose(atoms3.calc.results["descriptor"], expected_descriptors, rtol=0.0, atol=1.0e-6):
        raise AssertionError("single-point descriptor differs from fixed baseline label")

    print("optional ASE calculator smoke:", f"atoms={len(atoms)}", f"energy={energy:.12g}")


if __name__ == "__main__":
    main()
