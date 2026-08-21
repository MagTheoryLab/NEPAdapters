import os

import numpy as np

import nep_adapters


def read_noncollinear(path):
    tokens = open(path, encoding="utf-8").read().split()
    descriptor_dim = int(tokens[tokens.index("descriptor_dim") + 1])
    cursor = tokens.index("noncollinear") - 1
    if tokens[cursor] != "case":
        raise AssertionError("missing noncollinear oracle case")
    atom_count = int(tokens[cursor + 2])
    cursor += 3
    if tokens[cursor] != "types":
        raise AssertionError("missing oracle types")
    symbols = tokens[cursor + 1 : cursor + 1 + atom_count]
    cursor += 1 + atom_count
    blocks = {}
    while tokens[cursor] != "end":
        name = tokens[cursor]
        count = int(tokens[cursor + 1])
        cursor += 2
        blocks[name] = np.asarray(tokens[cursor : cursor + count], dtype=np.float64)
        cursor += count
    blocks["types"] = np.asarray(
        [0 if symbol == "Fe" else 1 for symbol in symbols], dtype=np.int32
    )
    blocks["descriptor_dim"] = descriptor_dim
    return blocks


def main():
    model_path = os.environ["NEP_ADAPTERS_PYTHON_SPIN2_MODEL"]
    oracle = read_noncollinear(os.environ["NEP_ADAPTERS_PYTHON_SPIN2_ORACLE"])
    atom_count = len(oracle["types"])
    counts = np.asarray([atom_count], dtype=np.int32)
    positions = oracle["positions"].reshape(atom_count, 3)
    spins = oracle["spins"].reshape(atom_count, 3)
    box = oracle["cell"].reshape(1, 9)
    with nep_adapters.load_model("cpu", model_path) as model:
        serial = model.calculate_spin(oracle["types"], box, positions, spins, counts)
        descriptor = model.descriptors_spin(oracle["types"], box, positions, spins, counts)
        batch_size = 8
        batch = model.calculate_spin(
            np.tile(oracle["types"], batch_size),
            np.tile(box, (batch_size, 1)),
            np.tile(positions, (batch_size, 1)),
            np.tile(spins, (batch_size, 1)),
            np.full(batch_size, atom_count, dtype=np.int32),
        )
    potential, force, virial, mforce = serial
    np.testing.assert_allclose(potential, oracle["potential"], rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(force.reshape(-1), oracle["forces"], rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(mforce.reshape(-1), oracle["mforces"], rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(virial.reshape(-1), oracle["virial"], rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(descriptor.reshape(-1), oracle["descriptor"], rtol=0.0, atol=1.0e-10)
    for batched, reference in zip(batch, serial):
        expected = np.concatenate([reference] * batch_size, axis=0)
        np.testing.assert_allclose(batched, expected, rtol=0.0, atol=1.0e-10)
    if descriptor.shape != (atom_count, oracle["descriptor_dim"]):
        raise AssertionError("unexpected unified spin2 descriptor shape")


if __name__ == "__main__":
    main()
