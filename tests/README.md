# Tests

CTest is the top-level test runner.

Current label meanings:

- `contract`: core API or SPI behavior.
- `smoke`: model loads and computes finite outputs.
- `parity`: adapter/runtime output matches a trusted oracle.
- `engine`: engine-level tests.
- `frontend`: Python, LAMMPS, or future software integration tests.
- `cuda`: CUDA backend tests.
- `device`: tests that require CUDA device runtime kernels.
- `kokkos`: LAMMPS/Kokkos-shaped device-neighbor simulation tests.
- `force`: finite-difference or parity gates for CUDA force components.
- `large_model`: tests that use a large model such as `nep89`.
- `domain_decomp`: backend-neutral local/ghost/foldback semantics for future
  multi-rank LAMMPS paths.
- `calculator`: Python calculator facade tests.
- `ase`: optional ASE adapter smoke tests. These skip cleanly when ASE is not
  installed.

The default `cpu` tests use `tests/fixtures/cpu_baseline/`. That
fixture contains the model, fixed structure, and golden `energy`, `force`, and
`virial` labels plus a per-atom `descriptor.txt` matrix. The labels are
repository test data only; they are not part of the Python wheel.

CPU correctness combines committed golden labels with an independently compiled
strict-FP64 oracle. This covers type mapping, AoS/SoA conversion, batch offsets,
energy and virial reduction, descriptor layout, spin derivatives, and LAMMPS
neighbor-view conversion without retaining a second CPU backend.

`tests/fixtures/nep_cpu_reference/` vendors the minimal 250-atom ordinary NEP3
and qNEP cases from the NEP_CPU test suite. The CPU engine is checked against
their force, per-atom raw9 virial, and descriptor references without requiring
another repository at configure or test time. CUDA checks qNEP calculation and
descriptor parity; the NEP3 case instead locks the explicit unsupported status,
with no CPU fallback.

Python has separate ordinary and spin gates. The spin calculator test checks
native and high-level calculation/descriptor parity, magnetic force, torque,
single- and multi-structure batching, structure-owned and explicit spin arrays,
empty results, wrong-model rejection, and the fully periodic input contract.

`nep_adapters_cpu_lammps_neighbors_test` covers the LAMMPS-shaped path:
`ilist`, `numneigh`, `firstneigh`, `type_map`, `double** x`, `double** f`, and
raw 9-component per-atom virials are passed through the adapter and checked
against committed golden labels. Python/batch tests continue to use the regular
`compute`-backed API.

`nep_adapters_lammps_plugin_baseline_test` is a real LAMMPS runtime test. It is
registered when `NEP_ADAPTERS_LAMMPS_EXECUTABLE` points to an `lmp` binary. It
loads `nepadaptersplugin.so`, runs `pair_style nep/cpu` on the committed baseline
fixture, and compares total energy, per-atom energy sum, forces, and virial from
`stress/atom` against the golden labels.

`nep_adapters_domain_decomp_contract_test` is pure C++. It does not call LAMMPS.
It compares a full-system reference against two synthetic rank-local
systems with ghost atoms, then folds ghost force contributions and reduces
virials. External-neighbor runners should reuse this contract shape when engines
are added.

`nep_adapters_virial_order_test` fixes the component-order contract between the
regular NEP `compute` path and the LAMMPS `compute_for_lammps` path.

CUDA backend closure is run with:

```sh
python3 tools/run_cuda_tests.py
```

Use `--cuda-arch 70` on V100 nodes and `--cuda-arch 89` on RTX 4090/Ada nodes
when `native` architecture detection is not wanted. The script configures
`NEP_ADAPTERS_ENABLE_CUDA=ON`, builds the tests, then runs
`ctest -L cuda`. This is a correctness gate only; MD throughput, `ncu`, and
`nsys` runs stay under `benchmarks/` or external job scripts.

The LAMMPS `nep/gpu` frontend is supported only with a CUDA-enabled Kokkos
build. Missing Kokkos device state is a hard error; tests must not rely on a
host-neighbor fallback from the GPU pair style.

When a CUDA Kokkos LAMMPS executable is supplied, CTest also loads the built
plugin into real LAMMPS and checks `nep/gpu` energy, per-atom energy, force, and
virial against the committed baseline fixture.

The CUDA gate covers these surfaces:

- engine registration and public capability reporting;
- host batch API and LAMMPS host-neighbor simulation;
- device model/workspace upload and internal neighbor construction;
- radial, angular, high-body, and ZBL force finite-difference checks;
- CPU/CUDA triclinic parity;
- LAMMPS/Kokkos-style strided device-neighbor input, output layout, type map,
  virial ordering, and neighbor-capacity failure behavior.
- qNEP direct reciprocal-space force, virial, and descriptor reference parity,
  plus either fail-closed PPPM
  behavior in the default build or PPPM reference parity in a PPPM-enabled
  build.

`tools/run_lammps_mpi_smoke.py` runs the local LAMMPS plugin under
`mpirun -np 1/2/4` and compares multi-rank output against the 1-rank reference
for forces, per-atom energy, per-atom stress, total potential energy, and
pressure components.
