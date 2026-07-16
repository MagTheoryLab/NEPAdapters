# LAMMPS Frontend

LAMMPS integration belongs here as a frontend.

Its job is to translate LAMMPS atoms, box, type mapping, neighbor lists, forces,
energies, and virials into the core runtime API. It should not be treated as a
compute engine.

The supported distribution shape is the runtime plugin.

Current targets:

- pair style: `nep/cpu` when the `cpu` engine is enabled
- backend: `cpu`
- pair style: `nep/gpu` when the CUDA engine and a CUDA-enabled LAMMPS Kokkos
  build are enabled; this path is device-only and has no host fallback
- backend: `cuda`
- runtime plugin: `nepadaptersplugin.so`
- compute path: `nep/cpu` forwards host neighbor lists through the
  external-neighbor runtime API; `nep/gpu` passes Kokkos device views directly
  to the CUDA engine.
- MPI status: local `mpirun -np 1/2/4` smoke has been validated for energy,
  force, per-atom energy, `stress/atom`, and `centroid/stress/atom`.
- baseline status: when a LAMMPS executable is provided, the runtime plugin is
  tested against `tests/fixtures/cpu_baseline/train.xyz` for total energy,
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
  --model tests/fixtures/cpu_baseline/nep.txt \
  --fixture tests/fixtures/cpu_baseline/train.xyz
```

Plugin usage after building:

```lammps
plugin load /path/to/nepadaptersplugin.so
pair_style nep/cpu
pair_coeff * * nep.txt Fe
```

CUDA-enabled builds can use the same coefficient shape:

```sh
cmake -S . -B .build/lammps-cuda \
  -DNEP_ADAPTERS_ENABLE_CUDA=ON \
  -DNEP_ADAPTERS_ENABLE_LAMMPS=ON \
  -DNEP_ADAPTERS_LAMMPS_SOURCE_DIR=/path/to/lammps \
  -DNEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR=/path/to/lammps-kokkos-build
```

The GPU plugin can be built with `NEP_ADAPTERS_ENABLE_CPU=OFF`; it does not
use or require a CPU compute fallback. The Kokkos package must have
`Kokkos_ENABLE_CUDA=ON`; configure or runtime use without CUDA device execution
is rejected.

```lammps
plugin load /path/to/nepadaptersplugin.so
pair_style nep/gpu
pair_coeff * * nep.txt Fe
```
