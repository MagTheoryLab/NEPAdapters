# LAMMPS 前端指南

NEPAdapters 的 LAMMPS 集成负责转换原子、盒子、类型映射、邻居表、能量、力和 virial 约定。它是 frontend，不是新的计算后端。

## 先选安装方式

| 方式 | 当前状态 | 是否重编 LAMMPS | 适合谁 |
|---|---|---:|---|
| runtime plugin | **正式支持并持续测试，推荐** | 否 | 绝大多数用户 |
| 编进 LAMMPS `src/` | 技术上可行，但当前是自定义集成 | 是 | 需要维护自有 LAMMPS 构建的人 |

不是只有 plugin 才能工作。pair 头文件保留了 LAMMPS `PairStyle` 注册宏，源码可以参与 LAMMPS 静态构建；但只复制文件到 `lammps/src/` 不会自动完成依赖和链接。具体差异见 [源码树集成](#源码树集成不是只复制文件)。

## plugin 和 pair style

| 插件 | pair style | 后端 | 数据路径 |
|---|---|---|---|
| `nepadapterscpuplugin.so` | `nep/cpu` | `cpu` | host 外部邻居表 |
| `nepadaptersgpuplugin.so` | `nep/gpu`、`nep/gpu/kk`、`nep/gpu/kk/device` | `cuda` | Kokkos CUDA device view |

CPU 和 GPU 是两个独立插件，可单独构建，也可安装到同一目录。GPU pair 是纯设备路径，没有 host 或 CPU fallback。

spin 模型没有单独的 `nep/spin/cpu` 或 `nep/spin/gpu` 名称。仍使用 `nep/cpu` 或 `nep/gpu/kk`，pair 会读取模型 capability，并要求 `atom_style spin` 或 `atom_style spin/kk` 提供 `sp` 和 `fm` 数据。

## 已验证支持范围

| 路径 | 当前状态 |
|---|---|
| ordinary NEP / CPU plugin | 支持；普通 MPI 1/2/4 ranks smoke 已覆盖能量、力、每原子能量和 virial |
| ordinary NEP / CUDA Kokkos plugin | 支持 |
| spin NEP / CPU plugin | 单 rank 真实 LAMMPS 门禁已覆盖能量、力、磁力、virial 和 spin 读回 |
| spin NEP / CUDA Kokkos plugin | 单 rank 在常规 CTest；Sai V100 单节点 1/2/4/8 MPI ranks 已做专项正确性和 TSPIN/dynspin 验证 |
| qNEP / CUDA Kokkos plugin | **不支持**；运行时明确拒绝，需要独立的 ghost/charge 数据流 |

边界要区分清楚：CUDA spin MPI 多 rank 不是“没测试”，而是已经完成过 1/2/4/8 ranks 专项验证；当前仓库常规 CTest 仍只自动注册单 rank spin plugin case。CPU spin MPI 尚没有同等级的正式门禁，因此不要把“CUDA spin MPI 已验证”扩大成“所有 spin MPI 都支持”。

qNEP CUDA batch API 可用，不表示 qNEP LAMMPS Kokkos 可用。两者输入和跨 rank/ghost 数据契约不同。

## 构建 CPU plugin

编译插件只需要 LAMMPS 源码树，不要求已有 `lmp` 可执行文件：

```sh
cmake -S . -B .build/lammps-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEP_ADAPTERS_ENABLE_LAMMPS=ON \
  -DNEP_ADAPTERS_LAMMPS_SOURCE_DIR=/path/to/lammps
cmake --build .build/lammps-cpu --target nepadapterscpuplugin -j2
```

构建产物位于：

```text
.build/lammps-cpu/frontends/lammps/nepadapterscpuplugin.so
```

若还要执行真实 LAMMPS smoke test，再配置可执行文件并打开测试：

```sh
cmake -S . -B .build/lammps-cpu-test \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEP_ADAPTERS_BUILD_TESTS=ON \
  -DNEP_ADAPTERS_ENABLE_LAMMPS=ON \
  -DNEP_ADAPTERS_LAMMPS_SOURCE_DIR=/path/to/lammps \
  -DNEP_ADAPTERS_LAMMPS_EXECUTABLE=/path/to/lmp
cmake --build .build/lammps-cpu-test -j2
ctest --test-dir .build/lammps-cpu-test -L lammps --output-on-failure
```

`NEP_ADAPTERS_LAMMPS_EXECUTABLE` 是测试输入，不是 plugin 的编译依赖。

## 安装并加载 CPU plugin

runtime-only 安装会把 NEPAdapters engine 静态收进 plugin，安装前缀只留下运行所需的 `.so`：

```sh
cmake -S . -B .build/lammps-install \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/nepadapters \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DBUILD_SHARED_LIBS=OFF \
  -DNEP_ADAPTERS_BUILD_TESTS=OFF \
  -DNEP_ADAPTERS_INSTALL_DEVELOPMENT_FILES=OFF \
  -DNEP_ADAPTERS_ENABLE_LAMMPS=ON \
  -DNEP_ADAPTERS_LAMMPS_SOURCE_DIR=/path/to/lammps
cmake --build .build/lammps-install --target nepadapterscpuplugin -j2
cmake --install .build/lammps-install
```

加载 plugin：

```sh
export LAMMPS_PLUGIN_PATH=/path/to/nepadapters/lib
/path/to/lmp -in lmp.in
```

`LAMMPS_PLUGIN_PATH` 必须是目录。LAMMPS 会扫描其中名称以 `plugin.so` 结尾的文件。也可以在输入文件中写：

```lammps
plugin load /path/to/nepadapterscpuplugin.so
```

环境变量和 `plugin load` 二选一，不要重复加载同一插件。

普通 NEP 输入示例：

```lammps
atom_style atomic
pair_style nep/cpu
pair_coeff * * nep.txt Fe
```

## 构建 CUDA Kokkos plugin

先准备一个启用了 CUDA Kokkos 的 LAMMPS 构建。NEPAdapters 需要它的 Kokkos package metadata 和 device 接口：

```sh
cmake -S . -B .build/lammps-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/nepadapters \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DBUILD_SHARED_LIBS=OFF \
  -DNEP_ADAPTERS_BUILD_TESTS=OFF \
  -DNEP_ADAPTERS_INSTALL_DEVELOPMENT_FILES=OFF \
  -DNEP_ADAPTERS_ENABLE_CPU=OFF \
  -DNEP_ADAPTERS_ENABLE_CUDA=ON \
  -DNEP_ADAPTERS_ENABLE_LAMMPS=ON \
  -DNEP_ADAPTERS_LAMMPS_SOURCE_DIR=/path/to/lammps \
  -DNEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR=/path/to/lammps-kokkos-build
cmake --build .build/lammps-cuda --target nepadaptersgpuplugin -j2
cmake --install .build/lammps-cuda
```

Kokkos 必须满足 `Kokkos_ENABLE_CUDA=ON`。GPU plugin 可用 `NEP_ADAPTERS_ENABLE_CPU=OFF` 独立构建。

普通 GPU 输入：

```lammps
atom_style atomic
pair_style nep/gpu/kk
pair_coeff * * nep.txt Fe
```

两 MPI ranks、两张 GPU 的启动示例：

```sh
export LAMMPS_PLUGIN_PATH=/path/to/nepadapters/lib
mpirun -np 2 /path/to/lmp \
  -k on g 2 -sf kk \
  -pk kokkos cuda/aware on neigh full comm device \
  -in lmp.in
```

运行日志应出现 `Loaded 1 plugins from ...`。缺少 Kokkos device state 时会直接报错，不会切换到 host 路径。

## spin 输入

CPU 使用 `atom_style spin`，CUDA Kokkos 使用 `atom_style spin/kk`。pair style 名称不变：

```lammps
atom_style spin/kk
pair_style nep/gpu/kk
pair_coeff * * nep-spin.txt Fe
```

模型、原子类型和 spin 初始化仍需按实际体系填写。运行前确认 LAMMPS 输出中启用了对应的 spin atom style，并在结果中检查 `fm`/磁力，而不只看总能量。

## 同时安装 CPU 和 GPU plugin

一次配置同时启用 CPU 和 CUDA，并构建两个 target：

```sh
cmake -S . -B .build/lammps-all \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/nepadapters \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DBUILD_SHARED_LIBS=OFF \
  -DNEP_ADAPTERS_BUILD_TESTS=OFF \
  -DNEP_ADAPTERS_INSTALL_DEVELOPMENT_FILES=OFF \
  -DNEP_ADAPTERS_ENABLE_CPU=ON \
  -DNEP_ADAPTERS_ENABLE_CUDA=ON \
  -DNEP_ADAPTERS_ENABLE_LAMMPS=ON \
  -DNEP_ADAPTERS_LAMMPS_SOURCE_DIR=/path/to/lammps \
  -DNEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR=/path/to/lammps-kokkos-build
cmake --build .build/lammps-all \
  --target nepadapterscpuplugin nepadaptersgpuplugin -j2
cmake --install .build/lammps-all
```

安装目录包含：

```text
/path/to/nepadapters/lib/
├── nepadapterscpuplugin.so
└── nepadaptersgpuplugin.so
```

设置一次 `LAMMPS_PLUGIN_PATH` 后，通过 `pair_style nep/cpu` 或 `pair_style nep/gpu/kk` 显式选择后端。

## 源码树集成不是只复制文件

LAMMPS 的 CMake 会扫描 `src/pair_*.cpp` 和带 `PairStyle` 宏的头文件，所以 pair 源码具备被编进 LAMMPS 的基础条件。当前相关文件是：

```text
pair_nep_adapters_common.cpp/.h
pair_nep_adapters_cpu.cpp/.h
pair_nep_adapters_cuda.cpp/.h
```

但是复制这些文件后还必须处理：

- 把 NEPAdapters 公共头文件加入 LAMMPS include path；
- 把 `NEPAdapters::cpu` 或 `NEPAdapters::cuda` 以及 core 链入 `lammps` target；
- 传播 CPU 的 OpenMP/BLAS 依赖；
- CUDA 路径传播 Kokkos/CUDA 的 include、compile definitions、编译选项和链接接口；
- 保证 LAMMPS 与 NEPAdapters 使用兼容的 MPI、精度、Kokkos 和 CUDA 配置；
- 自行维护这个组合的构建与端到端测试。

仓库当前没有可复制的官方 CMake fragment、安装 target 或 source-tree smoke gate。因此结论是：

- **可以编进 `src/`，但不是“复制后正常重编”这么简单；**
- **当前用户安装请使用 runtime plugin；**
- 若以后要把 source-tree 模式列为正式支持，需要先补官方集成 helper 和真实 LAMMPS 构建门禁。

## 常见错误

### `Loaded 0 plugins`

确认 `LAMMPS_PLUGIN_PATH` 指向目录，文件名以 `plugin.so` 结尾，并且 plugin 与当前 LAMMPS 的 ABI/MPI 构建兼容。

### `qNEP is currently supported by the CUDA batch API only`

这是当前明确的支持边界，不是安装错误。请改用 Python/C++ CUDA batch API，或使用另一个已经验证的 LAMMPS 路径。

### `spin model requires atom_style spin`

CPU 改用 `atom_style spin`；CUDA Kokkos 改用 `atom_style spin/kk`，并确保每个原子都有 vector spin 数据。

### plugin 能加载，但 CUDA pair 不能运行

检查 LAMMPS 是否确实使用 CUDA Kokkos device execution，并按 GPU 数量设置 `-k on g N`、`-sf kk` 和 `-pk kokkos ... comm device`。
