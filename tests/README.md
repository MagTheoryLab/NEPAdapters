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

For `cpu_nep3`, parity compares the public adapter path against direct calls to
the underlying NEP CPU class. That keeps the test focused on adapter-owned
boundaries: type mapping, AoS/SoA conversion, batch offsets, energy reduction,
and virial reduction.

`nep_adapters_cpu_nep3_lammps_neighbors_test` covers the LAMMPS-shaped path:
`ilist`, `numneigh`, `firstneigh`, `type_map`, `double** x`, `double** f`, and
raw 9-component per-atom virials are passed through the adapter and compared
against direct `NEP::compute_for_lammps`. Python/batch tests continue to use
the regular `compute`-backed API.

`nep_adapters_domain_decomp_contract_test` is pure C++. It does not call CUDA or
LAMMPS. It compares a full-system reference against two synthetic rank-local
systems with ghost atoms, then folds ghost force contributions and reduces
virials. CPU and CUDA external-neighbor runners should reuse this contract shape
when those backends are added.
