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
  - `backend.hpp`: transitional compatibility include for the old name.
- `src/`: core registry and API implementation.
- `engines/cpu_nep3/`: adapter for the official/existing NEP CPU class, used
  first as the CPU reference engine and parity oracle. The `nep3` name is
  historical; this is the current NEP CPU code path exposed by that class.
- `engines/cpu_opt/`: planned optimized CPU engine for OpenMP/SIMD/layout work.
- `engines/cuda/`: planned CUDA engine, currently corresponding to the maintained
  NEP_GPU direction.
- `frontends/python/`: Python package boundary. M2 uses a minimal pybind11
  frontend over the C ABI and returns NumPy arrays directly.
- `frontends/lammps/`: LAMMPS pair/plugin boundary. The current CPU pair style is
  `nep/adapters/cpu`; CUDA/Kokkos styles are intentionally deferred.
- `tests/`: contract, parity, and fixture tests.
- `benchmarks/`: throughput and scaling probes for engines/frontends.
- `docs/design.md`: architecture notes and staged plan.

Legacy `backends/` and `adapters/` directories are kept only as transitional
notes until the new layout is filled in.

## Current CPU Milestone

The useful CPU baseline is:

1. Keep the core buildable without CUDA, Python, or LAMMPS.
2. Use `cpu_nep3` as the reference CPU engine and parity oracle.
3. Expose a CPU-only pybind11 Python frontend returning NumPy arrays directly.
4. Build a LAMMPS CPU pair/plugin frontend over the same runtime.
5. Generate a correctness and OpenMP throughput report for the CPU baseline.

The LAMMPS CPU pair is currently a one-MPI-rank frontend. Multi-rank LAMMPS
support should be added through an external-neighbor engine path, not by hiding
LAMMPS-specific arrays inside the generic Python batch API.

## Tests And Benchmarks

Correctness tests use CTest labels:

- `contract`: core API and SPI behavior.
- `smoke`: minimal load/compute checks.
- `parity`: engine-to-oracle comparisons.
- `frontend`: Python, LAMMPS, or future software integrations.

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

To produce a local Markdown report with correctness, benchmark smoke, and a
4-thread OpenMP atom-scaling sweep:

```sh
source /Users/superbing/miniconda3/etc/profile.d/conda.sh
conda activate mysci
python3 tools/generate_test_report.py
```

The report also writes `reports/performance_conditions.json`, which records the
fixed future benchmark scale selected by the automatic saturation scan.

The report builds `cpu_nep3` with OpenMP and measures a single model instance
with `OMP_NUM_THREADS=4`.

## Python Package

The CPU-only Python package can be built through `pyproject.toml`:

```sh
source /Users/superbing/miniconda3/etc/profile.d/conda.sh
conda activate mysci
python -m build --wheel
```

The build currently needs a NEP CPU source tree. On this workstation CMake
auto-detects `../NepTrainKit/src/nep_cpu`; elsewhere pass
`-Ccmake.define.NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR=/path/to/nep_cpu`.
