# Fixtures

Keep fixtures small and intentional.

Current committed fixtures:

- `cpu_baseline/`: small Te/Pb model plus one fixed labelled structure.
  The labels are generated from the trusted NEP CPU implementation and are
  treated as golden model-output labels for repository correctness tests.
- `nep_cpu_reference/`: self-contained 250-atom ordinary NEP and qNEP cases
  copied from the NEP_CPU test suite, including force, raw9 virial, and
  descriptor references.

Useful future fixtures:

- Tiny spin structure.
- Model files trimmed to the minimum needed for compatibility checks.

Do not commit generated benchmark outputs or large training artifacts here.
