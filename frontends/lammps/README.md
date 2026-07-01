# LAMMPS Frontend

LAMMPS integration belongs here as a frontend.

Its job is to translate LAMMPS atoms, box, type mapping, neighbor lists, forces,
energies, and virials into the core runtime API. It should not be treated as a
compute engine.

The preferred distribution shape is a runtime plugin when practical, with a
source package fallback for older LAMMPS builds.

Current CPU target:

- pair style: `nep/cpu`
- backend: `cpu_nep3`
- runtime plugin: `nepadaptersplugin.so`
- compute path: LAMMPS neighbor lists are forwarded through the
  external-neighbor runtime API to `NEP::compute_for_lammps`.
- MPI status: local `mpirun -np 1/2/4` smoke has been validated for energy,
  force, per-atom energy, `stress/atom`, and `centroid/stress/atom`.

Local smoke test:

```sh
cmake -S . -B build-lammps \
  -DNEP_ADAPTERS_BUILD_TESTS=ON \
  -DNEP_ADAPTERS_ENABLE_LAMMPS=ON \
  -DNEP_ADAPTERS_LAMMPS_SOURCE_DIR=/path/to/lammps
cmake --build build-lammps -j2
ctest --test-dir build-lammps -L lammps --output-on-failure
```

Plugin usage after building:

```lammps
plugin load /path/to/nepadaptersplugin.so
pair_style nep/cpu
pair_coeff * * nep.txt Fe
```
