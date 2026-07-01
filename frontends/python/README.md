# Python Frontend

This directory contains the minimal Python frontend.

The default package should be CPU-only and should not import or link CUDA unless
the user explicitly builds or installs a CUDA-enabled package.

M2 uses a small pybind11 binding over the public C ABI. Python owns user-facing
NumPy shapes, while CMake-built native libraries own runtime and engine
implementation. The `Model.calculate()` method follows NepTrainKit's native
calculator shape and returns three NumPy arrays directly:

- per-atom potential, shape `(natoms,)`;
- forces, shape `(natoms, 3)`;
- per-atom virial, shape `(natoms, 9)`.

`Model.model_info()` returns cutoff and capability metadata without running a
calculation.

The higher-level `NEPCalculator` facade is also NumPy-first and does not require
ASE. It accepts duck-typed structures with `get_chemical_symbols()` or `symbols`,
plus `positions`, `cell`, and optional `pbc`, then returns a `Prediction`
dataclass:

- `energy`, shape `(nstructures,)`;
- `potential`, shape `(natoms,)`;
- `forces`, shape `(natoms, 3)`;
- `virials`, shape `(natoms, 9)`;
- `structure_virials`, shape `(nstructures, 9)`, using NepTrainKit-style mean
  per-atom virials.

`NEPCalculator.calculate(structures)` returns `(energy, force_blocks,
virial_blocks)` for NepTrainKit-style callers. Empty batches return empty arrays
and empty block lists.

ASE support is optional and lives in `nep_adapters.ase`. Importing
`nep_adapters` does not import ASE. Users who want an ASE calculator can install
the optional extra and import the adapter explicitly:

```python
from nep_adapters.ase import NepAseCalculator

atoms.calc = NepAseCalculator("nep.txt")
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

Local smoke test:

```sh
cmake -S . -B build-python \
  -DBUILD_SHARED_LIBS=ON \
  -DNEP_ADAPTERS_BUILD_TESTS=ON \
  -DNEP_ADAPTERS_ENABLE_PYTHON=ON \
  -DPython3_EXECUTABLE=/Users/superbing/miniconda3/envs/mysci/bin/python
cmake --build build-python -j2
ctest --test-dir build-python -L python --output-on-failure
```

Wheel build:

```sh
python -m build --wheel
```
