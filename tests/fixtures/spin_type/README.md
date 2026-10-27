# Spin type contract fixtures

Copied unchanged from MagTheoryLab/GPUMD commit
`2bdbc67e815158fd97c32305fe206b589bfc6f28`,
`tests_pytest/fixtures/nep_spin3`.

These fixed Spin3 models use `spin_type Fe`. Ge spins are zero but Ge still
contributes magnetic geometry. Expected energies, forces, magnetic forces,
virials and descriptors come from the independent TorchNEP FP64/autograd oracle.
Historical fixtures in the parent directory remain unchanged, including their
nonzero environment-only Ge spins and explicit environment selection semantics.

`noncollinear.json` and `c*_lammps_reference.txt` are extracted from the
noncollinear cases of those same oracles; total energy and virial are summed
from their per-atom values, with no NEPAdapters-generated reference values.

`noncollinear_raw_spins.json` adds finite nonzero spins on Ge to the same input.
New spin_type inference must match the unchanged zero-spin oracle while
preserving the raw caller input. Magnetic labels are not inference inputs.
