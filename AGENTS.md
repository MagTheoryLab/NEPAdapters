# Repository Agent Notes

## Python Environment

Use the `mysci` conda environment for Python-facing work:

```sh
source /Users/superbing/miniconda3/etc/profile.d/conda.sh
conda activate mysci
```

For CMake builds that enable the Python frontend, pass:

```sh
-DPython3_EXECUTABLE=/Users/superbing/miniconda3/envs/mysci/bin/python
```

## Virial Ordering Contract

Keep virial ordering explicit at public API boundaries:

- `find_force_batch` follows `NEP::compute` raw9 order:
  `xx, xy, xz, yx, yy, yz, zx, zy, zz`.
- `find_force_lammps_neighbors` follows LAMMPS order. `total_virial6` is
  `xx, yy, zz, xy, xz, yz`; per-atom raw9 is
  `xx, yy, zz, xy, xz, yz, yx, zx, zy`.
- Engine-internal storage may use a different layout, but copyback must convert
  to the relevant public API order before returning results.
