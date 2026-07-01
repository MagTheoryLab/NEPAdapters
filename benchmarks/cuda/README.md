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
