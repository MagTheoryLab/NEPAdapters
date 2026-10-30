"""New GPUMD block syntax versus unchanged legacy fixtures and calculations."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nep_adapters import inspect_model, load_model

ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures"
BACKEND = os.environ.get("NEP_BLOCK_TEST_BACKEND", "cpu")


def block_model(path, reference=None, alias=False):
    """Independent test writer: transpose legacy matrices with NumPy."""
    lines = Path(path).read_text().splitlines()
    header = lines.pop(0).split()
    tag, types = header[0], int(header[1])
    fields = {}
    while lines and not _numeric(lines[0]):
        key, *values = lines.pop(0).split()
        fields[key.lower()] = values
    with load_model("cpu", str(path)) as native:
        dim = native.model_info()["descriptor_dim"]
    h1, h2 = map(int, fields["ann"])
    charge = "charge" in tag
    ann = ((dim + 1) * h1 + (h1 + 2) * h2 if h2 else (dim + 2 + charge) * h1)
    ann = ann * types + (2 if charge else 1)
    if "polarizability" in tag:
        ann *= 2
    nmax = list(map(int, fields["n_max"]))
    basis = list(map(int, fields["basis_size"]))
    sizes = [(n + 1) * (b + 1) * types**2 for n, b in zip(nmax, basis)]
    count = ann + sum(sizes)
    spin = "spin3" in tag
    if spin:
        count += types**2 * int(fields["spin_compress"][0]) * (int(fields["spin_basis_size"][0]) + 1)
        count += int(fields["spin_projection_size"][0])
        fields.pop("spin_mode")
        fields["spin_mode" if alias else "spin_version"] = ["3" if alias else "1"]
        fields["energy_reference"] = fields.pop("spin_baseline")
        model = "nep_spin3" if alias else "nep_spin"
    elif charge:
        model = "qnep" + tag[-1]
    elif "dipole" in tag:
        model = "nep_dipole"
    elif "polarizability" in tag:
        model = "nep_polarizability"
    else:
        model = "nep"
    if model in ("nep", "qnep1", "qnep2"):
        fields["energy_reference"] = list(map(str, reference if reference is not None else [0] * types))
    parameters = np.array([line.strip() for line in lines[:count]], dtype=object)
    offset = ann
    for size in sizes:
        parameters[offset:offset + size] = parameters[offset:offset + size].reshape(-1, types**2).T.reshape(-1)
        offset += size
    assert len(lines) >= count + dim
    blocks = {"parameters": parameters, "q_scaler": lines[count:count + dim]}
    if len(lines) > count + dim:
        blocks["zbl_parameters"] = lines[count + dim:]
    # Exercise case folding, reordered fields/blocks, comments and packed values.
    result = ["NEP " + " ".join(header[1:]), "# GPUMD format_version 1", "FORMAT_VERSION 1", "MODEL " + model]
    result += [key.upper() + " " + " ".join(values) for key, values in sorted(fields.items(), reverse=True)]
    for name, values in sorted(blocks.items(), reverse=True):
        result.append(f"@ {name} {len(values)}")
        result += [" ".join(values[i:i + 5]) + " # values" for i in range(0, len(values), 5)]
    return "\n".join(result) + "\n"


def _numeric(line):
    try:
        float(line)
        return True
    except ValueError:
        return False


class BlockFormatTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nep-block-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "模型.nep"
        self.types = np.array([0, 1, 0, 1], dtype=np.int32)
        self.boxes = np.diag([20., 21., 22.]).reshape(1, 9)
        self.positions = np.array([[1., 2., 3.], [3.1, 2.2, 3.3], [2.3, 4.2, 3.7], [3.9, 3.5, 4.7]])
        self.counts = np.array([4], dtype=np.int32)
        self.args = (self.types, self.boxes, self.positions, self.counts)

    def compare(self, fixture, method="calculate", reference=None, alias=False):
        self.path.write_text(block_model(fixture, reference, alias))
        info = inspect_model(self.path)
        self.assertEqual(info.elements, inspect_model(fixture).elements)
        args = self.args
        if "spin" in method:
            spins = np.array([[0.1, -0.2, 1.1], [0.3, 0.1, 0.4], [-0.4, 0.8, 1.2], [0.1, 0.2, 0.3]])
            args = (self.types, self.boxes, self.positions, spins, self.counts)
        with load_model(BACKEND, str(fixture)) as old, load_model(BACKEND, str(self.path)) as new:
            self.assertEqual(old.model_info(), new.model_info())
            before, after = getattr(old, method)(*args), getattr(new, method)(*args)
            if isinstance(before, np.ndarray):
                before, after = (before,), (after,)
            if reference is not None:
                before = list(before)
                before[0] = before[0] + np.asarray(reference)[self.types]
            for expected, actual in zip(before, after):
                np.testing.assert_allclose(actual, expected, atol=2e-10 if BACKEND == "cpu" else 2e-6, rtol=0)
            if method == "calculate" and BACKEND == "cpu":
                fused = new.calculate_with_descriptors(*args)
                for actual, expected in zip(fused[:3], after):
                    np.testing.assert_allclose(actual, expected, atol=2e-10 if BACKEND == "cpu" else 2e-6, rtol=0)
            if method in ("calculate", "calculate_spin"):
                descriptors = "descriptors_spin" if "spin" in method else "descriptors"
                np.testing.assert_allclose(getattr(new, descriptors)(*args), getattr(old, descriptors)(*args), atol=1e-12, rtol=0)

    def test_spin3_and_previous_block_alias(self):
        for name in ("o3c2_uniform", "o3c2_typewise", "o3c3_uniform", "o3c4_uniform"):
            for alias in (False, True):
                with self.subTest(name=name, alias=alias):
                    self.compare(ROOT / f"nep4_spin3_{name}.nep", "calculate_spin", alias=alias)

    def test_ordinary_reference_and_descriptors(self):
        self.compare(ROOT / "cpu_baseline/nep.txt", reference=[-123.1234567890123, 0.234567890123456])

    def test_spin3_zbl(self):
        text = (ROOT / "nep4_spin3_o3c2_typewise.nep").read_text()
        fixture = Path(self.temp.name) / "spin-zbl.nep"
        for flexible in (False, True):
            with self.subTest(flexible=flexible):
                zbl = "0 0" if flexible else "0.8 2.5"
                model = text.replace("nep4_spin3 ", "nep4_spin3_zbl ", 1).replace("\ncutoff ", f"\nzbl {zbl}\ncutoff ", 1)
                if flexible:
                    model += "\n".join(map(str, [0.8, 2.5, 0.18175, 3.1998, 0.50986, 0.94229, 0.28022, 0.4029, 0.02817, 0.20162] * 3)) + "\n"
                fixture.write_text(model)
                self.compare(fixture, "calculate_spin")

    def test_single_element_and_reference_with_dftd3(self):
        self.types[:] = 0
        fixture = ROOT / "production_api/dftd3/nep.txt"
        reference = [-123.1234567890123]
        self.compare(fixture, reference=reference)
        if BACKEND == "cpu":
            with load_model("cpu", str(fixture)) as old, load_model("cpu", str(self.path)) as new:
                args = (*self.args, "pbe", 8., 6.)
                before, after = old.calculate_with_dftd3(*args), new.calculate_with_dftd3(*args)
                np.testing.assert_allclose(after[0], before[0] + reference[0], atol=1e-10, rtol=0)
                for actual, expected in zip(after[1:], before[1:]):
                    np.testing.assert_array_equal(actual, expected)

    def test_charge_reference_bec_and_forces(self):
        fixture = ROOT / "nep_cpu_reference/qnep/nep.txt"
        self.compare(fixture, "calculate_charge", [-123.1234567890123, 0.234567890123456])
        # charge2 has the same weight layout and a different electrostatic path.
        charge2 = Path(self.temp.name) / "charge2.nep"
        charge2.write_text(fixture.read_text().replace("nep4_charge1", "nep4_charge2", 1))
        self.compare(charge2, "calculate_charge", [-0.1234567890123, 0.234567890123456])

    @unittest.skipIf(BACKEND == "cuda", "CUDA response kernels are not supported")
    def test_response_models(self):
        self.compare(ROOT / "production_api/dipole/nep.txt", "dipoles")
        self.compare(ROOT / "production_api/polarizability/nep.txt", "polarizabilities")

    def test_invalid_blocks_rejected(self):
        text = block_model(ROOT / "nep4_spin3_o3c2_uniform.nep")
        cases = {
            "version": text.replace("FORMAT_VERSION 1", "FORMAT_VERSION 2"),
            "spin version": text.replace("SPIN_VERSION 1", "SPIN_VERSION 2"),
            "missing spin version": text.replace("SPIN_VERSION 1\n", ""),
            "duplicate field": text + "format_version 1\n",
            "duplicate block": text + "@ parameters 0\n",
            "unknown field": text + "future_feature 1\n",
            "unknown block": text + "@ future_feature 0\n",
            "truncation": text.rsplit("\n", 3)[0] + "\n",
            "extra data": text + "1.0\n",
            "nonfinite": text.replace("# values", "nan", 1),
            "count overflow": text.replace("@ parameters ", "@ parameters 999999999999999999999999999999", 1),
            "missing parameters": text.replace("@ parameters", "@ another_parameters"),
            "model": text.replace("MODEL nep_spin", "MODEL nep_future"),
            "wrong scaler": text.replace("@ q_scaler ", "@ q_scaler 1", 1),
        }
        for name, invalid in cases.items():
            with self.subTest(name=name):
                self.path.write_text(invalid)
                with self.assertRaises((ValueError, RuntimeError)):
                    load_model(BACKEND, str(self.path))


if __name__ == "__main__":
    unittest.main()
