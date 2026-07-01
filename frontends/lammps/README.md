# LAMMPS Frontend

LAMMPS integration belongs here as a frontend.

Its job is to translate LAMMPS atoms, box, type mapping, neighbor lists, forces,
energies, and virials into the core runtime API. It should not be treated as a
compute engine.

The preferred distribution shape is a runtime plugin when practical, with a
source package fallback for older LAMMPS builds.

Current CPU target:

- pair style: `nep/adapters/cpu`
- backend: `cpu_nep3`
- runtime plugin: `nepadaptersplugin.so`
- scope: one MPI rank. Multi-rank LAMMPS needs the external-neighbor engine path
  before it can be claimed correct.

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
pair_style nep/adapters/cpu
pair_coeff * * nep.txt Fe
```
