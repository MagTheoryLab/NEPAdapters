# cpu Performance Maintenance

`cpu` is the only supported CPU engine. It must keep passing committed
golden labels, the strict FP64 oracle, finite-difference gates, and CPU/CUDA
parity where CUDA is available.

## Default Path

Keep these optimizations in the default `cpu` path:

- symbol-isolated `cpu` engine built from the optimized NEP CPU source;
- OpenMP force paths with thread-local force/virial reduction;
- LAMMPS external-neighbor workspace reuse, including edge caches and persistent
  force/virial scratch;
- contracted angular force layout;
- CBLAS/Accelerate-backed batch ANN when available, with the portable scalar
  implementation retained for systems without a CBLAS provider;
- radial tables, defaulting to ON on Apple builds and available as an explicit
  build option elsewhere.

These are the current user-facing performance wins. Avoid adding new public
switches unless a benchmark shows a stable, meaningful gain across the intended
platforms.

## Removed Experiment Class

Do not keep or reintroduce these as default or user-facing fallback paths unless
new evidence clears a higher bar:

- AVX512 prefer paths;
- ANN q-block caches;
- portable ANN micro-kernels;
- shape-only L4/N5 angular dispatch;
- short edge-tiled SIMD helpers;
- table-index or interpolation micro-optimizations that move less than noise.

Small gains below roughly 3 percent are not worth a new branch, CMake option, or
platform-specific code path unless they also simplify the implementation.

## Benchmark Gates

Use `.build/<name>` build trees. A minimal correctness gate is:

```sh
cmake -S . -B .build/cpu-opt-clean \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEP_ADAPTERS_ENABLE_CPU=ON \
  -DNEP_ADAPTERS_BUILD_TESTS=ON \
  -DNEP_ADAPTERS_BUILD_BENCHMARKS=ON
cmake --build .build/cpu-opt-clean -j8
OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=1 \
  ctest --test-dir .build/cpu-opt-clean -LE bench --output-on-failure
```

On Apple builds, add the local libomp flags from `AGENTS.md` if CMake does not
find OpenMP automatically.

Then run benchmark smoke tests:

```sh
OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=1 \
  ctest --test-dir .build/cpu-opt-clean -L bench --output-on-failure
```

For performance readout, use the NEP89 benchmark in both batch and LAMMPS modes:

```sh
OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=1 \
  .build/cpu-opt-clean/benchmarks/nep_adapters_bench_cpu_nep89 \
  --engine cpu --mode lammps --replicate 4x4x2 --rank-grid 2x1x1 \
  --iterations 1000 --warmup 20 --phase-timer
```

Record durable conclusions in docs. Keep raw profiles, job scripts, temporary
plans, and machine-local reports out of git history.

## Future Work Bar

The next CPU work should be cleanup or benchmark hardening unless a profile
shows a first-tier hotspot with a clear path to a stable gain. Reasonable future
directions are:

- small-box neighbor-list redesign with strict parity coverage;
- coarser OpenMP regions after the current default path is stable;
- all-platform radial-table validation before making it globally default.

Spin, CUDA, and LAMMPS MPI work should build on the backend/frontend boundary
without adding a compatibility CPU backend or more experimental switches.
