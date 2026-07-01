# Tests

CTest is the top-level test runner.

Current label meanings:

- `contract`: core API or SPI behavior.
- `smoke`: model loads and computes finite outputs.
- `parity`: adapter/runtime output matches a trusted oracle.
- `engine`: engine-level tests.
- `frontend`: Python, LAMMPS, or future software integration tests.
- `large_model`: tests that use a large model such as `nep89`.

For `cpu_nep3`, parity compares the public adapter path against direct calls to
the underlying NEP CPU class. That keeps the test focused on adapter-owned
boundaries: type mapping, AoS/SoA conversion, batch offsets, energy reduction,
and virial reduction.
