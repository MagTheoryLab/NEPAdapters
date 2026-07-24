# LAMMPS 前端指南

NEPAdapters 的 LAMMPS 集成负责转换原子、盒子、类型映射、邻居表、能量、力和 virial 约定。它是 frontend，不是新的计算后端。

## 先选安装方式

| 方式 | 当前状态 | 是否重编 LAMMPS | 适合谁 |
|---|---|---:|---|
| runtime plugin | **正式支持并持续测试，推荐** | 否 | 绝大多数用户 |
| source-tree / builtin | **正式 helper；pair 和 engine 直接编进 `lmp`** | 是 | 不能或不想在运行时加载 plugin 的用户 |

两种方式使用同一套 pair 和 engine 实现，计算语义不分叉。source-tree 模式必须使用仓库 helper，不能只手工复制 `.cpp/.h`；helper 会同时安装 CMake hook，完成依赖和链接。具体命令见 [source-tree / builtin 安装](#source-tree--builtin-安装)。

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
| spin NEP / CPU plugin | CTest 始终注册单 rank；检测到 MPI launcher 时，根据可用进程数额外注册 2/4/8 ranks；Sai DSPRHBM 单节点 1/2/4/8 ranks（每 rank 2 个 OpenMP 线程）已完成逐原子正确性对照和短程 dynspin 验证 |
| spin NEP / CUDA Kokkos plugin | 单 rank 在常规 CTest；Sai V100 单节点 1/2/4/8 MPI ranks 已做专项正确性和 TSPIN/dynspin 验证 |
| qNEP / CUDA Kokkos plugin | **不支持**；运行时明确拒绝，需要独立的 ghost/charge 数据流 |
| ordinary NEP / CPU builtin | 已在干净 LAMMPS 源码副本中完成编译和无 plugin baseline smoke |
| ordinary NEP / CUDA Kokkos builtin | 已在 RTX 4090 / CUDA 12.8 上完成干净构建和无 plugin baseline smoke |

边界要区分清楚：CPU 和 CUDA spin 均已完成单节点 1/2/4/8 ranks 专项验证；CPU CTest 在检测到 MPI launcher 时会根据 `MPIEXEC_MAX_NUMPROCS` 自动注册最多 2/4/8 ranks，CUDA CTest 当前仍只自动注册单 rank。上述结果不等同于多节点 MPI 已验证。

只运行已注册的多 rank 门禁可使用 `ctest -L mpi --output-on-failure`。在 Slurm 平台上，CTest 不负责申请计算资源，应先进入满足最大 rank 数的作业分配，再在作业内执行该命令。

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

## source-tree / builtin 安装

这种方式把 pair style 和所选 engine 直接编进 `lmp`。运行时不设置 `LAMMPS_PLUGIN_PATH`，输入文件里也不写 `plugin load`。

### 1. 安装受管源码

在 NEPAdapters 仓库根目录运行：

```sh
python3 tools/install_lammps_source.py install \
  --lammps-source /path/to/lammps \
  --backend cpu
```

`--backend` 可选 `cpu`、`cuda` 或 `both`。helper 会：

- 复制 common 和所选 backend 的 pair `.cpp/.h` 到 `lammps/src/`；
- 安装 `cmake/Modules/NEPAdaptersLAMMPSSource.cmake`；
- 写入 `.nep_adapters_source_manifest.json`，记录来源和 SHA256；
- 重装或切换 backend 时只更新未被用户修改的受管文件；
- 遇到同名本地文件或已修改文件时拒绝覆盖。

### 2. 构建 CPU builtin

仍在 NEPAdapters 仓库根目录执行，`NEP_ADAPTERS_SOURCE_DIR` 指向当前仓库：

```sh
cmake -S /path/to/lammps/cmake \
  -B /path/to/lammps/.build/nep-adapters-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PROJECT_INCLUDE=/path/to/lammps/cmake/Modules/NEPAdaptersLAMMPSSource.cmake \
  -DNEP_ADAPTERS_SOURCE_DIR="$PWD" \
  -DNEP_ADAPTERS_LAMMPS_SOURCE_BACKEND=cpu
cmake --build /path/to/lammps/.build/nep-adapters-cpu --target lmp -j2
```

运行时直接使用生成的 `lmp`：

```sh
/path/to/lammps/.build/nep-adapters-cpu/lmp -in lmp.in
```

### 3. 构建 CUDA Kokkos builtin

先安装 CUDA pair：

```sh
python3 tools/install_lammps_source.py install \
  --lammps-source /path/to/lammps \
  --backend cuda
```

再使用 LAMMPS 的 Kokkos CUDA preset。下面以 RTX 4090 / Ada 89 为例；其他 GPU 必须更换 CUDA architecture 和 Kokkos architecture：

```sh
cmake -S /path/to/lammps/cmake \
  -B /path/to/lammps/.build/nep-adapters-cuda \
  -C /path/to/lammps/cmake/presets/kokkos-cuda.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DKokkos_ARCH_ADA89=ON \
  -DPKG_KOKKOS=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DCMAKE_PROJECT_INCLUDE=/path/to/lammps/cmake/Modules/NEPAdaptersLAMMPSSource.cmake \
  -DNEP_ADAPTERS_SOURCE_DIR="$PWD" \
  -DNEP_ADAPTERS_LAMMPS_SOURCE_BACKEND=cuda
cmake --build /path/to/lammps/.build/nep-adapters-cuda --target lmp -j2
```

`both` 会把 `nep/cpu` 和 `nep/gpu*` 同时编进一个 `lmp`，CUDA/Kokkos 要求与 `cuda` 相同。CMake hook 会把 NEPAdapters core/engine 作为静态内部实现构建，即使调用方选择 shared `liblammps`，也不需要额外部署 NEPAdapters runtime library。

当前 builtin helper 只支持 LAMMPS 的 CMake 构建系统，不支持传统 make package 命令。若已有自己的 `CMAKE_PROJECT_INCLUDE`，应在那个 hook 中 `include()` 本文件，而不是覆盖掉已有逻辑。

### 更新和卸载

NEPAdapters 更新后重新运行同一条 `install` 命令，再重新配置并构建 LAMMPS。安全卸载：

```sh
python3 tools/install_lammps_source.py uninstall \
  --lammps-source /path/to/lammps
```

卸载只删除 manifest 中记录且 SHA256 未改变的文件；发现本地修改会停止，不会误删用户代码。

## 常见错误

### `Loaded 0 plugins`

确认 `LAMMPS_PLUGIN_PATH` 指向目录，文件名以 `plugin.so` 结尾，并且 plugin 与当前 LAMMPS 的 ABI/MPI 构建兼容。

### `qNEP is currently supported by the CUDA batch API only`

这是当前明确的支持边界，不是安装错误。请改用 Python/C++ CUDA batch API，或使用另一个已经验证的 LAMMPS 路径。

### `spin model requires atom_style spin`

CPU 改用 `atom_style spin`；CUDA Kokkos 改用 `atom_style spin/kk`，并确保每个原子都有 vector spin 数据。

### plugin 能加载，但 CUDA pair 不能运行

检查 LAMMPS 是否确实使用 CUDA Kokkos device execution，并按 GPU 数量设置 `-k on g N`、`-sf kk` 和 `-pk kokkos ... comm device`。

### builtin 配置后没有 `nep/cpu` 或 `nep/gpu`

确认在第一次配置前已经运行 `install_lammps_source.py`，并传入了 `CMAKE_PROJECT_INCLUDE`、`NEP_ADAPTERS_SOURCE_DIR` 和一致的 `NEP_ADAPTERS_LAMMPS_SOURCE_BACKEND`。切换 backend 后必须重新配置构建目录。
