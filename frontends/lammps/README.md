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
- baseline status: when a LAMMPS executable is provided, the runtime plugin is
  tested against `tests/fixtures/cpu_nep3_baseline/train.xyz` for total energy,
  per-atom energy sum, forces, and virial reconstructed from `stress/atom`.

Local smoke test:

```sh
cmake -S . -B build-lammps \
  -DNEP_ADAPTERS_BUILD_TESTS=ON \
  -DNEP_ADAPTERS_ENABLE_LAMMPS=ON \
  -DNEP_ADAPTERS_LAMMPS_SOURCE_DIR=/path/to/lammps \
  -DNEP_ADAPTERS_LAMMPS_EXECUTABLE=/path/to/lmp
cmake --build build-lammps -j2
ctest --test-dir build-lammps -L lammps --output-on-failure
```

The runtime baseline smoke can also be run directly:

```sh
python3 tools/run_lammps_baseline_smoke.py \
  --lmp /path/to/lmp \
  --plugin build-lammps/frontends/lammps/nepadaptersplugin.so \
  --model tests/fixtures/cpu_nep3_baseline/nep.txt \
  --fixture tests/fixtures/cpu_nep3_baseline/train.xyz
```

Plugin usage after building:

```lammps
plugin load /path/to/nepadaptersplugin.so
pair_style nep/cpu
pair_coeff * * nep.txt Fe
```
