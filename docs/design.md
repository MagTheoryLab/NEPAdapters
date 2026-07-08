# NEPAdapters Design

## Positioning

NEPAdapters is the current working name for a NEP compute framework. Its value is
not "wrapping" one implementation. Its value is keeping runtime semantics,
multiple engines, frontends, packaging, and parity tests aligned.

The design has three layers:

- `core`: model/runtime semantics, data views, capability reporting, error
  handling, and dispatch.
- `engines`: compute implementations such as `cpu_nep3` and `cpu_opt`.
- `frontends`: Python bindings, LAMMPS pair/plugin code, and future software
  integrations.

## Non-Negotiable Boundaries

- LAMMPS is a frontend, not an engine.
- Python is a frontend, not the runtime.
- The core target must not include or link Python or LAMMPS.
- Engines implement the core engine SPI and must not depend on frontends.
- Frontends consume the public runtime API and must not include engine internals.
- ASE must remain an optional Python adapter, not a dependency of the core
  calculator package.

## Engine Strategy

The first CPU engine should not be a new reference implementation written from
scratch. The official or existing NEP CPU class should be adapted as
`cpu_nep3` and used as the reference engine and parity oracle.

Planned engines:

- `cpu_nep3`: thin SPI shim around the official/existing NEP CPU implementation.
  The name is historical; it represents the current NEP CPU code path exposed by
  that class. It is the correctness baseline, not the performance target.
- `cpu_opt`: optimized CPU implementation for OpenMP/SIMD/layout/algorithm work.
  It must pass parity against `cpu_nep3`.
This avoids repeating the official CPU implementation while still allowing
independent CPU engineering.

The CUDA engine should be built normal-NEP first. Its protocol source of truth is
the current `torchnep/src/force` NEP implementation, while performance choices
should reuse the measured high-performance CUDA work only after the protocol
boundary is clear. Spin support should layer on top only after the non-spin
dataflow is stable: model parsing, owned-neighbor construction, host/device
staging, descriptor/ANN execution, force scatter, and virial reduction should be
ordinary NEP concepts first, with spin-specific buffers and kernels added later
instead of defining the base shape.

CUDA implementation files should keep human-facing boundaries explicit:
`model_protocol` parses the model and computes parameter counts, `workspace_plan`
names and sizes device buffers, and later `.cu` files should implement kernels
against those contracts. Avoid recreating a single large bridge file that mixes
text parsing, LAMMPS conventions, staging, kernels, and diagnostics.

CUDA-facing tests should keep caller shapes separate from the start: ordinary
batch API, LAMMPS host-neighbor simulation, and LAMMPS Kokkos/device-staging
simulation are different contracts. The Kokkos path should not be forced through
host pointer APIs; it needs an explicit device-input contract before kernels are
advertised as supported.

The CUDA force path has two first-class input modes:

- direct coordinates: the caller provides atom types, positions, boxes, PBC
  flags, and per-structure ranges. The CUDA engine owns neighbor construction
  and therefore keeps structure, box, and PBC metadata in its internal-neighbor
  workspace.
- external neighbors: the caller provides an already-built neighbor topology,
  such as LAMMPS host neighbors or later Kokkos/device neighbors. The CUDA engine
  stages active atoms and converts or aliases the topology into its slot-major
  execution layout without making this path depend on batch boxes.

Both modes should converge before descriptor and ANN kernels on the same
slot-major neighbor arrays and SoA atom/output buffers. That shared execution
layout is the performance-critical boundary; the staging code on either side may
remain caller-specific.

Internal CUDA neighbor generation should use a large-system cell-list algorithm,
not an all-pairs builder. The current CUDA scaffold starts with CSR-style cells:
count atoms per cell, prefix-scan offsets, scatter atom ids into cell-contiguous
storage, then traverse neighboring cells to fill radial and angular slot-major
neighbor arrays. That layout leaves room for later sorted-cell ordering and
Kokkos device-view input without changing descriptor kernels.

LAMMPS Kokkos should be treated as a third frontend shape, not as the same thing
as the host-neighbor API. In the local LAMMPS checkout, Kokkos coordinates are
`X_FLOAT*[3]` views accessed as `x(i,0..2)`, while neighbor data is held in
Kokkos `int**` views and sometimes a transpose view. That can support a later
device-view path with less copying, but the CUDA engine should still own the
final execution layout for NEP kernels instead of inheriting LAMMPS view layout
as the core ABI.

## Frontend Strategy

Python and LAMMPS have different shapes and should not force each other into the
same call pattern.

- Python primarily wants model loading, batch prediction, optional descriptors,
  optional descriptors, automatic or explicit engine selection, and direct
  NumPy arrays.
- LAMMPS primarily wants a pair/plugin frontend that translates LAMMPS atom,
  box, type-map, neighbor-list, energy, force, and virial conventions into core
  views.

LAMMPS pair styles should live under `frontends/lammps/`. They may support a
runtime plugin and/or a source package, but both are frontends over the same core
runtime and engines. The CPU LAMMPS pair enters through the LAMMPS-shaped
external-neighbor contract and the `cpu_nep3` engine forwards that path to the
underlying NEP CPU `compute_for_lammps` implementation. The Python frontend must
continue to use the regular batch `compute` path and should not inherit LAMMPS
neighbor-list or ghost-atom conventions.

LAMMPS pair-style names are user-facing and should not expose the internal
adapter framework name. The convention is:

- `nep/cpu`: ordinary NEP through the CPU backend.
- `nep/gpu`: ordinary NEP through the GPU backend.
- `nep/spin/cpu`: spin NEP through the CPU backend.
- `nep/spin/gpu`: spin NEP through the GPU backend.

## Public API vs Engine SPI

Public API is what Python, NepTrainKit, LAMMPS integrations, and C/C++ users can
see. It should change slowly and follow semver once v1 exists. In the current
M0 skeleton, the public vocabulary is split across `api.h`, `views.hpp`, and
`capability.hpp`.

Engine SPI is what engine authors implement. It lives in `engine.hpp` and may
change during v0 while CPU implementations are still being shaped.

For v0, the existing C API in `include/nep_adapters/api.h` is experimental. Do
not freeze ABI until spin, charge, descriptors, external-neighbor input,
owned-neighbor construction, and device input have clear data-view contracts.

## Packaging Strategy

The default Python package should be CPU-only:

- build core + `cpu_nep3` + possibly `cpu_opt`;
- not include LAMMPS.

LAMMPS integration should be buildable by users who download this repository:

- runtime plugin is preferred where practical;
- source package can remain as a fallback for older or constrained LAMMPS builds;
- it should link the runtime/engine libraries but never depend on Python.

## CPU Baseline Plan

1. Keep core buildable without CUDA, Python, and LAMMPS.
2. Keep `cpu_nep3` as the oracle-engine boundary.
3. Expose Python through pybind11 with NumPy arrays and no Python-side shape
   conversions after native return.
4. Expose a CPU LAMMPS pair/plugin through `compute_for_lammps` and external
   neighbor lists.
5. Record correctness, parity, LAMMPS MPI smoke, and OpenMP atom scaling in a
   generated report.
6. Keep small golden-label fixtures in `tests/fixtures/`; they are repository
   test data and must not be packaged into Python wheels.

## Test And Benchmark Strategy

Testing has two separate jobs:

- correctness: API/SPI contracts, smoke tests, parity against `cpu_nep3`, and
  frontend integration tests. The default CPU/Python/LAMMPS correctness tests
  must also compare against committed golden labels in
  `tests/fixtures/cpu_nep3_baseline/`;
- performance: throughput, scaling, and profiler-backed bottleneck evidence.

CTest is the top-level dispatcher for native tests and benchmarks. Tests must
use labels such as `contract`, `smoke`, `parity`, `frontend`, `engine`,
`python`, `lammps`, `bench`, and `performance` so CI and local runs can
select the right slice without inventing new runners.

Python-specific tests should use the `mysci` conda environment and the pybind11
frontend should exchange NumPy arrays directly with native code. The default
calculator facade should not import ASE; the optional `nep_adapters.ase` module
owns ASE `Calculator` and `SinglePointCalculator` integration. Python
performance work should use `pytest-benchmark` once broader Python APIs exist.
LAMMPS performance work should run real LAMMPS input decks under
`benchmarks/lammps/`; the repository should not grow a duplicate MD driver. The
current CPU report includes LAMMPS compile/plugin smoke plus local
`mpirun -np 1/2/4` correctness smoke.

Multi-rank LAMMPS correctness should start from a backend-neutral C++ domain
decomposition contract: full-system reference, rank-local owned atoms, ghost
atoms, compact external-neighbor lists, ghost-force foldback, and virial
reduction. Engines can each provide runners for that same contract.

## Red Lines

- Do not make LAMMPS neighbor-list shape define the Python batch API.
- Do not make a base Python wheel depend on CUDA libraries.
- Do not mix Python or LAMMPS headers into core.
- Do not rewrite the CPU reference engine if the official/existing NEP CPU class
  can serve as the oracle.
- Do not make `cpu_opt` depend on `cpu_nep3` outside tests.
- Do not expose engine SPI types through the public API.
- Do not mix correctness pass/fail thresholds with machine-specific benchmark
  baselines. Record throughput first; add regression gates only with explicit
  per-machine baselines.
