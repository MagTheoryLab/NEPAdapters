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
