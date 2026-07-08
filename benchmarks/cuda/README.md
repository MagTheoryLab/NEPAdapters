# CUDA Benchmarks

CUDA engine benchmarks belong here once the CUDA engine is wired into the core
SPI.

Use the same case names and labels as CPU benchmarks where possible, then add
CUDA-specific profiler captures beside the command:

- CTest label: `bench;performance;engine;cuda`
- Nsight Systems for timeline and CPU/GPU overlap.
- Nsight Compute for kernel-level occupancy, memory traffic, and instruction
  mix.

Do not store profiler output in Git. Keep local captures under
`benchmarks/results/`.

`nep_adapters_bench_cuda_batch` accepts `--model radial|angular|q134` so the
same batched driver can compare radial-only, regular `L=4` angular, and a
heavier high-body force path. Use `--batch-only` for profiler runs where the
single-structure comparison loop would otherwise dominate the capture. The
reported `batched_vs_single_loop` compares one batched call against repeatedly
calling the same CUDA engine one structure at a time; optimization-over-baseline
should be computed from saved benchmark results for the baseline commit.
