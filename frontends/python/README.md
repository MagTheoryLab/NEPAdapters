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

`Model.descriptors()` returns per-atom descriptors directly as a NumPy array
with shape `(natoms, descriptor_dim)`. `Model.model_info()` returns cutoff,
capability, and `descriptor_dim` metadata without running a calculation.

Spin models use explicit methods so ordinary callers cannot accidentally omit
spin input. `Model.calculate_spin()` takes a `(natoms, 3)` spin array and
returns potential, force, per-atom virial, magnetic force, and torque arrays.
`Model.descriptors_spin()` returns the spin-model descriptor matrix. Torque is
defined as `spin x magnetic_force`.

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

`NEPCalculator.get_descriptor(structure)` returns per-atom descriptors for one
structure. `get_structures_descriptor(structures, mean_descriptor=True)` mirrors
NepTrainKit's descriptor helper: with `mean_descriptor=True` it returns one mean
descriptor per structure, otherwise it returns the concatenated per-atom matrix.

The matching high-level spin surface is `predict_spin_arrays()`,
`predict_spin_structures()`, `predict_spin_descriptors_arrays()`,
`predict_spin_descriptors()`, `calculate_spin()`, `get_spin_descriptor()`, and
`get_spin_structures_descriptor()`. A structure can provide spins through a
`(natoms, 3)` `spins` attribute or `arrays["spins"]`; callers may instead pass
the concatenated spin array explicitly.

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
cmake -S . -B .build/python \
  -DBUILD_SHARED_LIBS=ON \
  -DNEP_ADAPTERS_BUILD_TESTS=ON \
  -DNEP_ADAPTERS_ENABLE_PYTHON=ON \
  -DPython3_EXECUTABLE=/Users/superbing/miniconda3/envs/mysci/bin/python
cmake --build .build/python -j2
ctest --test-dir .build/python -L python --output-on-failure
```

Wheel build:

```sh
python -m build --wheel
```

The wheel configuration explicitly disables CUDA, qNEP PPPM/cuFFT, OpenMP,
and C/C++ development-file installation. The base wheel contains the Python
package and the ABI-qualified `nep_cpu` extension; standalone CMake installs
keep development libraries, headers, OpenMP, and package metadata enabled by
default.

A CUDA-enabled wheel can be built on a CUDA host without changing the base
wheel defaults:

```sh
python -m build --wheel \
  -Ccmake.define.NEP_ADAPTERS_ENABLE_CUDA=ON
```

That wheel contains separate ABI-qualified `nep_cpu` and `nep_gpu` extensions.
Importing `nep_adapters` or selecting `cpu` loads only `nep_cpu`; `nep_gpu`
is imported lazily after an explicit `backend="cuda"` request. CUDA loading
failures never switch to CPU. qNEP PPPM stays disabled, so the default CUDA
wheel does not link cuFFT.

CUDA accepts the implemented NEP4/NEP5 protocol surface. A NEP3 model passed
to `backend="cuda"` fails explicitly as unsupported and is not rerouted to
`cpu`. qNEP direct-mode models support both `calculate()` and
`descriptors()` through `nep_gpu`.

The repository default CUDA architecture is `sm_89`, intended for a local
RTX 4090 build. A broader precompiled wheel can embed SASS for V100, T4, A100,
A10/RTX 30, and RTX 4090 plus `compute_89` PTX for forward JIT:

```sh
CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=70-real;75-real;80-real;86-real;89-real;89-virtual' \
python -m build --wheel \
  -Ccmake.define.NEP_ADAPTERS_ENABLE_CUDA=ON
```

Release wheels must be compiled inside the chosen manylinux build image. An
`auditwheel repair` pass can add tags or vendor allowed libraries, but it cannot
remove GLIBC/GLIBCXX symbols introduced by a newer host toolchain.

## Wheel dependency boundary

The wheel does not copy the active Python environment into the package.
Specifically:

- NumPy is declared as a runtime dependency and installed separately; it is not
  embedded in the wheel.
- ASE is an optional dependency and is not embedded.
- pybind11, CMake, scikit-build-core, and the compiler toolchain are build-time
  dependencies only.
- The default Linux CUDA build does not vendor `libcuda`, `libcudart`, or
  `libcufft`; the CUDA runtime is linked statically into `nep_gpu`, and the
  NVIDIA driver remains a host requirement.
- glibc, libstdc++, libm, and libgcc remain external system libraries. Their
  required symbol versions are fixed by the manylinux build environment.

`auditwheel show` and an archive-content inspection are release gates. If a
future build links another non-system shared library, it must be reviewed
explicitly instead of being accepted as an accidental vendored dependency.
