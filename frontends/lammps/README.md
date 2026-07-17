# LAMMPS 前端

LAMMPS 集成属于 frontend。它负责转换 LAMMPS 的原子、盒子、类型映射、邻居表、能量、力和 virial 约定，不属于计算引擎。

当前发布形式为 runtime plugin：

| pair style | 后端 | 输入路径 |
|---|---|---|
| `nep/cpu` | `cpu` | host 外部邻居表 |
| `nep/gpu`、`nep/gpu/kk`、`nep/gpu/kk/device` | `cuda` | Kokkos CUDA device view |

插件文件名为 `nepadaptersplugin.so`。`nep/gpu` 是纯设备路径，不提供 host 或 CPU fallback。

当前 plugin 没有注册 `nep/spin/cpu` 或 `nep/spin/gpu`。pair 内部可以识别 spin 模型，并在缺少 LAMMPS `atom_style spin` 的 `sp`/`fm` 数据时报错；但真实 LAMMPS spin 端到端发布门禁尚未完成，因此当前文档不把 spin LAMMPS 列为生产支持面。

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

推荐由运行脚本设置插件目录：

```sh
PLUGIN="$PWD/.build/lammps/frontends/lammps/nepadaptersplugin.so"
test -f "$PLUGIN"
export LAMMPS_PLUGIN_PATH="$(dirname "$PLUGIN")"
/path/to/lmp -in lmp.in
```

`LAMMPS_PLUGIN_PATH` 必须是目录而不是 `.so` 文件路径。LAMMPS 会自动加载目录中名称以 `plugin.so` 结尾的文件；`nepadaptersplugin.so` 符合该命名规则。此时输入文件只保留模型配置：

```lammps
pair_style nep/cpu
pair_coeff * * nep.txt Fe
```

一次性调试也可以不设置环境变量，改为在输入文件中显式写 `plugin load /path/to/nepadaptersplugin.so`。两种方式选择一种，不要重复加载。

## 安装插件

runtime-only 安装使用静态内部库，安装前缀中只保留插件本身：

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
cmake --build .build/lammps-install --target nepadaptersplugin -j2
cmake --install .build/lammps-install
```

固定 `CMAKE_INSTALL_LIBDIR=lib` 后，安装布局为：

```text
/path/to/nepadapters/
└── lib/
    └── nepadaptersplugin.so
```

使用安装版本：

```sh
export LAMMPS_PLUGIN_PATH=/path/to/nepadapters/lib
/path/to/lmp -in lmp.in
```

如需让其他 CMake 项目链接 NEPAdapters，可保留默认的 `NEP_ADAPTERS_INSTALL_DEVELOPMENT_FILES=ON`。安装目录会额外包含 `include/nep_adapters/`、NEPAdapters C/C++ 库和 `lib/cmake/NEPAdapters/`。如果没有固定 `CMAKE_INSTALL_LIBDIR`，GNUInstallDirs 可能根据平台选择 `lib64`，应以配置结果为准。

## 构建 CUDA 插件

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
cmake --build .build/lammps-cuda --target nepadaptersplugin -j2
cmake --install .build/lammps-cuda
```

GPU plugin 可以使用 `NEP_ADAPTERS_ENABLE_CPU=OFF` 构建，不依赖 CPU 计算路径。Kokkos 必须设置 `Kokkos_ENABLE_CUDA=ON`；没有 CUDA device execution 时，配置或运行都会被拒绝。

```lammps
pair_style nep/gpu/kk
pair_coeff * * nep.txt Fe
```

对应的启动方式示例：

```sh
export LAMMPS_PLUGIN_PATH=/path/to/nepadapters/lib
mpirun -np 2 /path/to/lmp \
  -k on g 2 -sf kk \
  -pk kokkos cuda/aware on neigh full comm device \
  -in lmp.in
```

`nep/gpu` 仍是有效的基础名称；`-sf kk` 或显式 `/kk` 名称用于 Kokkos 后缀路径。安装前应先使用启用测试的独立构建跑过 CTest；安装后应在运行日志中确认出现 `Loaded 1 plugins from ...`。GPU 路径缺少 Kokkos device state 时会直接报错，不会改走 host 邻居表。
