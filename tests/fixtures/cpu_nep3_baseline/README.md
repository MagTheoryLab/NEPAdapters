# CPU NEP3 Baseline Fixture

This fixture is committed test data, not Python package data.

- `nep.txt` is the small Te/Pb NEP model used by the CPU baseline tests.
- `train.xyz` contains one fixed labelled structure. The `energy`, `force`,
  and `virial` fields are golden labels generated from the current trusted
  `cpu_nep3` adapter path. Future CPU/CUDA/Python/LAMMPS tests compare
  against these values instead of only comparing two live code paths.

Regenerate this fixture only when the baseline physics contract is intentionally
changed and the new values have been reviewed.
