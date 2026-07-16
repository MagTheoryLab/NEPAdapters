# Benchmarks

Benchmarks are performance probes, not correctness tests. Correctness belongs in
`tests/`; throughput and scaling belong here.

Use standard runners where they already fit:

- C/C++ engine microbenchmarks are built by CMake and exposed through CTest with
  the `bench` label.
- Python frontend benchmarks should use `pytest-benchmark` once the Python
  frontend exists.
- LAMMPS benchmarks should be real LAMMPS input cases under `benchmarks/lammps/`.

Current local run:

```sh
cmake -S . -B build-bench \
  -DNEP_ADAPTERS_BUILD_TESTS=ON \
  -DNEP_ADAPTERS_BUILD_BENCHMARKS=ON
cmake --build build-bench -j2
ctest --test-dir build-bench -L bench --output-on-failure
```

Benchmark executables finish by printing one JSON line with at least:

- benchmark name;
- engine/frontend name;
- model or case name;
- atom count and iteration count;
- whether the binary was compiled with OpenMP;
- OpenMP thread count;
- elapsed seconds;
- evaluations per second;
- atom-steps per second.

Do not commit generated benchmark result files. Put local result captures under
`benchmarks/results/`.

Generate a combined correctness and throughput report with:

```sh
source /Users/superbing/miniconda3/etc/profile.d/conda.sh
conda activate mysci
python3 tools/generate_test_report.py
```

The report is written under `reports/`, which is treated as local generated
output. The same run also writes `reports/performance_conditions.json` with the
fixed scale selected by the automatic saturation scan.

The report benchmark is intentionally single-process and OpenMP-only:

```sh
python3 tools/generate_test_report.py --openmp-threads 4
```

LAMMPS MPI correctness can be included in the same report when a local LAMMPS
binary and plugin are available:

```sh
python3 tools/generate_test_report.py \
  --lammps-source-dir /path/to/lammps \
  --lmp-executable /path/to/lmp \
  --lammps-plugin /path/to/nepadaptersplugin.so
```

When `--lmp-executable` is provided, the report also runs the real LAMMPS
baseline smoke against `tests/fixtures/cpu_baseline/` before the MPI smoke.

For a heavier local scaling sweep, pass explicit supercell factors:

```sh
python3 tools/generate_test_report.py \
  --auto-scales 1x1x1,2x2x1,2x2x2,3x3x3,4x4x4 \
  --iterations 3 \
  --openmp-threads 4 \
  --min-saturation-atoms 2000
```
