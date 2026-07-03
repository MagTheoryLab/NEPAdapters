# Tests

CTest is the top-level test runner.

Current label meanings:

- `contract`: core API or SPI behavior.
- `smoke`: model loads and computes finite outputs.
- `parity`: adapter/runtime output matches a trusted oracle.
- `engine`: engine-level tests.
- `frontend`: Python, LAMMPS, or future software integration tests.
- `large_model`: tests that use a large model such as `nep89`.
- `domain_decomp`: backend-neutral local/ghost/foldback semantics for future
  multi-rank LAMMPS paths.
- `calculator`: Python calculator facade tests.
- `ase`: optional ASE adapter smoke tests. These skip cleanly when ASE is not
  installed.

The default `cpu_nep3` tests use `tests/fixtures/cpu_nep3_baseline/`. That
fixture contains the model, fixed structure, and golden `energy`, `force`, and
`virial` labels plus a per-atom `descriptor.txt` matrix. The labels are
repository test data only; they are not part of the Python wheel.

For `cpu_nep3`, parity compares the public adapter path against direct calls to
the underlying NEP CPU class. That keeps the test focused on adapter-owned
boundaries: type mapping, AoS/SoA conversion, batch offsets, energy reduction,
virial reduction, and descriptor layout conversion.

`nep_adapters_cpu_nep3_lammps_neighbors_test` covers the LAMMPS-shaped path:
`ilist`, `numneigh`, `firstneigh`, `type_map`, `double** x`, `double** f`, and
raw 9-component per-atom virials are passed through the adapter and compared
against direct `NEP::compute_for_lammps`. Python/batch tests continue to use
the regular `compute`-backed API.

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

`tools/run_lammps_mpi_smoke.py` runs the local LAMMPS plugin under
`mpirun -np 1/2/4` and compares multi-rank output against the 1-rank reference
for forces, per-atom energy, per-atom stress, total potential energy, and
pressure components.
