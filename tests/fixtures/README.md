# Fixtures

Keep fixtures small and intentional.

Current committed fixtures:

- `cpu_nep3_baseline/`: small Te/Pb model plus one fixed labelled structure.
  The labels are generated from the current trusted `cpu_nep3` path and are
  treated as golden DFT labels for repository correctness tests.

Useful future fixtures:

- Tiny spin structure.
- Model files trimmed to the minimum needed for compatibility checks.

Do not commit generated benchmark outputs or large training artifacts here.
