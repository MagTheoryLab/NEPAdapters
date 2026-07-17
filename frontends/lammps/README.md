# LAMMPS 前端

LAMMPS 集成属于 frontend。它负责转换 LAMMPS 的原子、盒子、类型映射、邻居表、能量、力和 virial 约定，不属于计算引擎。

当前发布形式为 runtime plugin：

| pair style | 后端 | 输入路径 |
|---|---|---|
| `nep/cpu` | `cpu` | host 外部邻居表 |
| `nep/gpu` | `cuda` | Kokkos CUDA device view |

插件文件名为 `nepadaptersplugin.so`。`nep/gpu` 是纯设备路径，不提供 host 或 CPU fallback。

当前 plugin 只注册上述 pair style，没有 `nep/spin/cpu` 或 `nep/spin/gpu`。pair 内部可以识别 spin 模型，并在缺少 LAMMPS `atom_style spin` 的 `sp`/`fm` 数据时报错；但真实 LAMMPS spin 端到端发布门禁尚未完成，因此当前文档不把 spin LAMMPS 列为生产支持面。

当前 MPI smoke 已检查 `mpirun -np 1/2/4` 下的能量、力、每原子能量、`stress/atom` 和 `centroid/stress/atom`。提供 LAMMPS 可执行文件时，测试还会使用 `tests/fixtures/cpu_baseline/train.xyz` 核对总能量、每原子能量和、力以及从 `stress/atom` 重建的 virial。

## 构建 CPU 插件

```sh
cmake -S . -B .build/lammps \
  -DNEP_ADAPTERS_BUILD_TESTS=ON \
  -DNEP_ADAPTERS_ENABLE_LAMMPS=ON \
  -DNEP_ADAPTERS_LAMMPS_SOURCE_DIR=/path/to/lammps \
  -DNEP_ADAPTERS_LAMMPS_EXECUTABLE=/path/to/lmp
cmake --build .build/lammps -j2
ctest --test-dir .build/lammps -L lammps --output-on-failure
```

直接运行 baseline smoke：

```sh
python3 tools/run_lammps_baseline_smoke.py \
  --lmp /path/to/lmp \
  --plugin .build/lammps/frontends/lammps/nepadaptersplugin.so \
  --model tests/fixtures/cpu_baseline/nep.txt \
  --fixture tests/fixtures/cpu_baseline/train.xyz
```

LAMMPS 输入：

```lammps
plugin load /path/to/nepadaptersplugin.so
pair_style nep/cpu
pair_coeff * * nep.txt Fe
```

## 构建 CUDA 插件

```sh
cmake -S . -B .build/lammps-cuda \
  -DNEP_ADAPTERS_ENABLE_CUDA=ON \
  -DNEP_ADAPTERS_ENABLE_LAMMPS=ON \
  -DNEP_ADAPTERS_LAMMPS_SOURCE_DIR=/path/to/lammps \
  -DNEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR=/path/to/lammps-kokkos-build
```

GPU plugin 可以使用 `NEP_ADAPTERS_ENABLE_CPU=OFF` 构建，不依赖 CPU 计算路径。Kokkos 必须设置 `Kokkos_ENABLE_CUDA=ON`；没有 CUDA device execution 时，配置或运行都会被拒绝。

```lammps
plugin load /path/to/nepadaptersplugin.so
pair_style nep/gpu
pair_coeff * * nep.txt Fe
```

跑完应检查 CTest 是否通过，并确认 LAMMPS 实际加载的是本次构建的 plugin。GPU 路径缺少 Kokkos device state 时会直接报错，不会改走 host 邻居表。
