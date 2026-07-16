# NEPAdapters

NEPAdapters is the current working name for a NEP compute framework. The name may
change later; the important boundary is:

- `core`: stable NEP runtime semantics, public data views, capability reporting,
  errors, and engine dispatch.
- `engines`: concrete CPU/CUDA implementations.
- `frontends`: Python, LAMMPS, and future software integrations.

This repository is not just a thin adapter around external code. It should be
able to ship as a CPU-only Python package, optionally build CUDA engines, and let
users compile LAMMPS pair/plugin integrations against the same runtime.

## Ownership

The project owns:

- A public API for model loading and prediction.
- A narrow engine SPI used by CPU/CUDA implementations.
- Capability-driven dispatch across engines.
- Compatibility and parity tests across engines.
- Packaging policy for Python wheels, native libraries, and LAMMPS integrations.

The project must not let one frontend or one engine define the whole design:

- LAMMPS pair styles are frontends, not engines.
- Python bindings are frontends, not the runtime.
- CUDA dependencies stay in the CUDA engine and CUDA-enabled packages.
- The core runtime must remain buildable without CUDA, Python, or LAMMPS.

## Layout

- `include/nep_adapters/`: current public C/C++ headers.
  - `api.h`: experimental C ABI.
  - `engine.hpp`: C++ engine SPI and registration entry.
  - `views.hpp`: frontend/engine data-view vocabulary.
  - `capability.hpp`: capability flags and helpers.
- `src/`: core registry and API implementation.
- `engines/cpu/`: the supported CPU engine, including spin and charge paths.
- `engines/cuda/`: the CUDA engine used by batch and Kokkos LAMMPS frontends.
- `frontends/python/`: Python package boundary. M2 uses a minimal pybind11
  frontend over the C ABI, a NumPy-first calculator facade, and an optional
  `nep_adapters.ase` adapter.
- `frontends/lammps/`: LAMMPS pair/plugin boundary. The CPU pair style is
  `nep/cpu`; `nep/gpu` is built when the CUDA frontend is enabled.
- `tests/`: contract, parity, and fixture tests.
  - `tests/fixtures/cpu_baseline/`: committed CPU baseline model,
    structure, and golden labels for correctness tests. This is repository test
    data and is not included in the Python package.
- `benchmarks/`: throughput and scaling probes for engines/frontends.
- `docs/design.md`: architecture notes and staged plan.

Legacy `backends/` and `adapters/` directories are kept only as transitional
notes until the new layout is filled in.

## Current CPU Milestone

The useful CPU baseline is:

1. Keep the core buildable without CUDA, Python, or LAMMPS.
2. Use `cpu` as the only supported CPU engine.
3. Expose a CPU-only pybind11 Python frontend returning NumPy arrays directly.
4. Build a LAMMPS CPU pair/plugin frontend over the same runtime.
5. Generate a correctness and OpenMP throughput report for the CPU baseline.

The LAMMPS CPU pair uses `compute_for_lammps` through the external-neighbor
engine path. Local `mpirun -np 1/2/4` smoke tests compare 1-rank and multi-rank
energy, force, per-atom energy, `stress/atom`, and `centroid/stress/atom`.
When a local `lmp` executable is provided, the plugin is also tested directly
against the committed baseline fixture through `pair_style nep/cpu`.

## Tests And Benchmarks

Correctness tests use CTest labels:

- `contract`: core API and SPI behavior.
- `smoke`: minimal load/compute checks.
- `parity`: engine-to-oracle comparisons.
- `frontend`: Python, LAMMPS, or future software integrations.
- `cuda`: CUDA backend correctness tests.
- `kokkos`: CUDA device-neighbor simulation matching the LAMMPS/Kokkos caller
  shape.

Performance probes use the `bench` and `performance` labels and are built only
when `NEP_ADAPTERS_BUILD_BENCHMARKS=ON`.

```sh
cmake -S . -B build -DNEP_ADAPTERS_BUILD_TESTS=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure

cmake -S . -B build-bench \
  -DNEP_ADAPTERS_BUILD_TESTS=ON \
  -DNEP_ADAPTERS_BUILD_BENCHMARKS=ON
cmake --build build-bench -j2
ctest --test-dir build-bench -L bench --output-on-failure
```

CUDA backend correctness is separate from MD throughput benchmarking:

```sh
python3 tools/run_cuda_tests.py
```

On fixed-architecture clusters, pass the target explicitly:

```sh
python3 tools/run_cuda_tests.py --cuda-arch 70   # V100
python3 tools/run_cuda_tests.py --cuda-arch 89   # RTX 4090 / Ada
```

To compile and test the optional qNEP PPPM/cuFFT path, add `--qnep-pppm`.

This runs the `cuda` CTest label only: contract checks, device-model/layout
checks, batch force finite-difference gates, triclinic CPU parity, and the
LAMMPS/Kokkos-style device-neighbor simulation. It does not run benchmark
targets.

To produce a local Markdown report with correctness, benchmark smoke, and a
4-thread OpenMP atom-scaling sweep:

```sh
source /Users/superbing/miniconda3/etc/profile.d/conda.sh
conda activate mysci
python3 tools/generate_test_report.py
```

The report also writes `reports/performance_conditions.json`, which records the
fixed future benchmark scale selected by the automatic saturation scan.

The report builds `cpu` with OpenMP and measures a single model instance
with `OMP_NUM_THREADS=4`.

## Python Package

The CPU-only Python package can be built through `pyproject.toml`:

```sh
source /Users/superbing/miniconda3/etc/profile.d/conda.sh
conda activate mysci
python -m build --wheel
```

The CPU implementation is vendored under `engines/cpu/native`; no external
CPU engine source tree is required.

On a CUDA build host, build a GPU-capable wheel with:

```sh
python -m build --wheel \
  -Ccmake.define.NEP_ADAPTERS_ENABLE_CUDA=ON
```

The GPU-capable wheel keeps the explicit backends in separate ABI-qualified
extensions: `nep_cpu` for `cpu` and lazily imported `nep_gpu` for CUDA. Use
`backend="cuda"` to select CUDA; an unavailable GPU extension fails directly
and never falls back to CPU. PPPM remains disabled unless separately compiled
in, so this wheel does not require cuFFT.

The default CUDA wheel command targets `sm_89` for local RTX 4090 use. Release
artifacts should choose and record an explicit architecture set and must be
compiled in the target manylinux environment; repairing a wheel compiled on a
newer host cannot lower its required GLIBC or GLIBCXX symbol versions.

The default package depends on NumPy, not ASE. The optional ASE interface is
available as `nep_adapters.ase` and can be requested with the `ase` extra.
Repository fixtures under `tests/fixtures/` are not packaged into the wheel.

The wheel does not bundle NumPy, ASE, pybind11, CMake, or scikit-build-core.
NumPy is a runtime dependency installed separately by the package manager; ASE
is optional; pybind11, CMake, and scikit-build-core are build-only tools. The
Linux CUDA wheel also does not vendor `libcuda`, `libcudart`, or `libcufft`.
The CUDA runtime used by `nep_gpu` is linked statically, while the NVIDIA driver
is supplied by the target machine. System C/C++ libraries such as glibc,
libstdc++, libm, and libgcc remain external platform dependencies.

Ordinary and spin Python calls are separate: use `calculate`/`descriptors` for
ordinary models and `calculate_spin`/`descriptors_spin` for spin models. The
high-level `NEPCalculator` exposes the corresponding `predict_*`,
`calculate_spin`, and `get_spin_*descriptor` helpers.

CUDA qNEP builds do not link cuFFT by default. The direct reciprocal-space path
remains available; the experimental PPPM implementation is compiled only with
`-DNEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM=ON`. A default build that is explicitly
asked for `NEP_ADAPTERS_QNEP_KSPACE=pppm` returns an error instead of silently
switching algorithms.
