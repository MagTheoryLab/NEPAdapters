# CPU 性能维护

`cpu` 是唯一受支持的 CPU engine。它必须持续通过固定 golden label、严格 FP64 oracle、有限差分，以及可用时的 CPU/CUDA parity。

## 默认保留的优化

当前 `cpu` 默认路径保留：

- 基于优化 NEP CPU 源码构建、符号隔离的 `cpu` engine；
- 使用 thread-local force/virial reduction 的 OpenMP force 路径；
- LAMMPS 外部邻居 workspace 复用，包括 edge cache 和持久 force/virial scratch；
- 收缩后的 angular force 布局；
- 有 CBLAS provider 时使用 CBLAS/Accelerate batch ANN，无 provider 时保留 portable scalar 实现；
- radial table：Apple 默认开启，其他平台可通过显式构建选项启用。

这些是当前用户可见的稳定收益。除非 benchmark 证明在目标平台上存在稳定且有意义的提升，否则不要增加新的公共开关。

## 不应恢复的实验路径

没有更强新证据时，不要重新引入：

- AVX512 prefer 路径；
- ANN q-block cache；
- portable ANN micro-kernel；
- 只按 shape 分派的 L4/N5 angular 路径；
- 短 edge-tiled SIMD helper；
- 收益小于噪声的 table index 或 interpolation 微优化。

低于约 3% 的小收益不值得新增 branch、CMake option 或平台专用实现，除非改动同时显著简化代码。

## 性能门禁

构建目录统一放在 `.build/<name>`：

```sh
cmake -S . -B .build/cpu-clean \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEP_ADAPTERS_ENABLE_CPU=ON \
  -DNEP_ADAPTERS_BUILD_TESTS=ON \
  -DNEP_ADAPTERS_BUILD_BENCHMARKS=ON
cmake --build .build/cpu-clean -j8
OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=1 \
  ctest --test-dir .build/cpu-clean -LE bench --output-on-failure
```

Apple 构建如果没有自动找到 OpenMP，再按 `AGENTS.md` 添加本地 libomp 参数。

正确性通过后运行 benchmark smoke：

```sh
OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=1 \
  ctest --test-dir .build/cpu-clean -L bench --output-on-failure
```

NEP89 性能读取应同时覆盖 batch 与 LAMMPS。LAMMPS 示例：

```sh
OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=1 \
  .build/cpu-clean/benchmarks/nep_adapters_bench_cpu_nep89 \
  --engine cpu --mode lammps --replicate 4x4x2 --rank-grid 2x1x1 \
  --iterations 1000 --warmup 20 --phase-timer
```

长期结论写入文档。raw profile、job script、临时 plan 和机器本地报告不进入 Git。

## 后续工作的门槛

如果 profiler 没有显示一线 hotspot 和清晰的稳定收益路径，下一轮 CPU 工作应以清理或强化 benchmark 为主。可能的方向：

- 带严格 parity 的 small-box 邻居表重构；
- 默认路径稳定后的更粗粒度 OpenMP region；
- 把 radial table 设为全平台默认前，先完成跨平台验证。

spin、CUDA 和 LAMMPS MPI 都应继续复用 backend/frontend 边界，不增加兼容 CPU backend 或更多实验开关。
