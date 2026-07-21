# 构建与安装

本页面向需要从源码构建 NEPAdapters 的用户。若只想在 Python 中使用，优先看 [Python 接口指南](python.md)；若要接入 LAMMPS，直接看 [LAMMPS 前端指南](../frontends/lammps/README.md)。

## 环境要求

- CMake 3.24 或更高版本；
- 支持 C++17 的编译器；
- CPU 后端默认启用 OpenMP；
- Python 前端需要 Python 3.10+、NumPy 和 pybind11；
- CUDA 后端需要 CUDA Toolkit，LAMMPS CUDA 前端还需要启用 CUDA 的 Kokkos 构建。

所有本地构建目录建议放在仓库的 `.build/` 下。

## 最常用的构建组合

| 目标 | 关键选项 |
|---|---|
| CPU core | 默认配置即可 |
| CPU + CUDA core | `-DNEP_ADAPTERS_ENABLE_CUDA=ON` |
| Python CPU | `-DNEP_ADAPTERS_ENABLE_PYTHON=ON` |
| Python CPU + CUDA | 同时打开 Python 和 CUDA |
| LAMMPS CPU plugin | 打开 LAMMPS，并提供 LAMMPS 源码目录 |
| LAMMPS CUDA plugin | 再提供 CUDA Kokkos 构建目录 |

## CPU core：构建、测试、安装

```sh
cmake -S . -B .build/cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/nepadapters \
  -DNEP_ADAPTERS_BUILD_TESTS=ON
cmake --build .build/cpu -j2
ctest --test-dir .build/cpu --output-on-failure
cmake --install .build/cpu
```

默认安装开发文件，包括：

```text
/path/to/nepadapters/
├── include/nep_adapters/
└── lib/
    ├── libnep_adapters.*
    ├── libnep_adapters_cpu.*
    └── cmake/NEPAdapters/
```

其他 CMake 项目可使用：

```cmake
find_package(NEPAdapters CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE NEPAdapters::cpu)
```

配置时通过 `-DCMAKE_PREFIX_PATH=/path/to/nepadapters` 指向安装前缀。

## CPU + CUDA core

```sh
cmake -S . -B .build/cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DNEP_ADAPTERS_ENABLE_CPU=ON \
  -DNEP_ADAPTERS_ENABLE_CUDA=ON \
  -DNEP_ADAPTERS_BUILD_TESTS=ON
cmake --build .build/cuda -j2
ctest --test-dir .build/cuda -L cuda --output-on-failure
```

`CMAKE_CUDA_ARCHITECTURES` 未指定时默认使用 `native`。跨机器分发时不要使用 `native`，应显式列出目标架构。

## Python：直接安装

当前 PyPI 没有正式发布包。最简单的 CPU 安装方式是：

```sh
python -m pip install .
```

连同 ASE 可选接口安装：

```sh
python -m pip install '.[ase]'
```

启用 CUDA：

```sh
NEP_CUDA=1 python -m pip install .
```

默认源码安装不会自动探测 CUDA。`NEP_CUDA=1` 是 `-DNEP_ADAPTERS_ENABLE_CUDA=ON` 的短开关，CUDA 加载失败也不会回退到 CPU。

## Python：CMake 开发构建

需要运行 Python CTest 时，必须构建共享库，并把 CMake 指向实际使用的 Python：

```sh
python -m pip install numpy pybind11

cmake -S . -B .build/python \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DNEP_ADAPTERS_BUILD_TESTS=ON \
  -DNEP_ADAPTERS_ENABLE_PYTHON=ON \
  -DPython3_EXECUTABLE=/path/to/python
cmake --build .build/python -j2
ctest --test-dir .build/python -L python --output-on-failure
```

若选中的 Python 环境中没有 pybind11，配置阶段会失败。`BUILD_SHARED_LIBS=ON` 不是 Python wheel 的要求，但当前 Python CTest 注册依赖它。

CPU + CUDA Python 开发构建只需追加：

```sh
-DNEP_ADAPTERS_ENABLE_CUDA=ON \
-DCMAKE_CUDA_ARCHITECTURES=native
```

## LAMMPS

推荐 runtime plugin；如果需要把 pair 直接编进 `lmp`，使用 source-tree helper。两种方式的完整命令见 [LAMMPS 前端指南](../frontends/lammps/README.md)。

### runtime plugin

CPU plugin 的最小配置：

```sh
cmake -S . -B .build/lammps \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEP_ADAPTERS_ENABLE_LAMMPS=ON \
  -DNEP_ADAPTERS_LAMMPS_SOURCE_DIR=/path/to/lammps
cmake --build .build/lammps --target nepadapterscpuplugin -j2
```

`NEP_ADAPTERS_LAMMPS_EXECUTABLE` 只用于运行时 smoke test，不是编译插件的必填项。CUDA plugin 还必须启用 CUDA，并设置 `NEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR`。完整安装和运行命令见 [LAMMPS 前端指南](../frontends/lammps/README.md)。

### source-tree / builtin

```sh
python3 tools/install_lammps_source.py install \
  --lammps-source /path/to/lammps \
  --backend cpu

cmake -S /path/to/lammps/cmake \
  -B /path/to/lammps/.build/nep-adapters-cpu \
  -DCMAKE_PROJECT_INCLUDE=/path/to/lammps/cmake/Modules/NEPAdaptersLAMMPSSource.cmake \
  -DNEP_ADAPTERS_SOURCE_DIR="$PWD" \
  -DNEP_ADAPTERS_LAMMPS_SOURCE_BACKEND=cpu
cmake --build /path/to/lammps/.build/nep-adapters-cpu --target lmp -j2
```

生成的 `lmp` 已内置 `nep/cpu`，运行时不需要 plugin 环境变量。CUDA 和 `both` 模式见 LAMMPS 前端指南。

## 主要 CMake 选项

| 选项 | 默认值 | 说明 |
|---|---:|---|
| `NEP_ADAPTERS_ENABLE_CPU` | `ON` | 构建 CPU engine |
| `NEP_ADAPTERS_ENABLE_CUDA` | `OFF` | 构建 CUDA engine；`NEP_CUDA=1` 可改变默认值 |
| `NEP_ADAPTERS_ENABLE_PYTHON` | `OFF` | 构建 Python frontend |
| `NEP_ADAPTERS_ENABLE_LAMMPS` | `OFF` | 构建 LAMMPS frontend/plugin |
| `NEP_ADAPTERS_BUILD_TESTS` | `ON` | 注册正确性测试 |
| `NEP_ADAPTERS_BUILD_BENCHMARKS` | `OFF` | 构建性能测试 |
| `NEP_ADAPTERS_INSTALL_DEVELOPMENT_FILES` | `ON` | 安装头文件、库和 CMake package metadata |
| `NEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM` | `OFF` | 编译实验性的 qNEP PPPM/cuFFT 路径 |
| `NEP_ADAPTERS_CPU_ENABLE_OPENMP` | `ON` | 启用 CPU OpenMP |
| `NEP_ADAPTERS_CPU_ENABLE_NATIVE_ARCH` | `OFF` | 使用本机 CPU 指令优化，不适合可移植二进制 |
| `NEP_ADAPTERS_CPU_USE_RADIAL_TABLE` | `ON` | 直接 CMake 构建默认启用 radial table；wheel 配置会关闭 |

LAMMPS 相关路径：

| 选项 | 是否必填 | 说明 |
|---|---:|---|
| `NEP_ADAPTERS_LAMMPS_SOURCE_DIR` | 是 | 包含 `src/pair.h` 的 LAMMPS 源码树 |
| `NEP_ADAPTERS_LAMMPS_EXECUTABLE` | 否 | 用于真实 plugin smoke test 的 `lmp` |
| `NEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR` | CUDA plugin 必填 | 启用 CUDA 的 LAMMPS Kokkos 构建树 |

## 常见问题

### 配置成功，但 Python 测试没有出现

确认同时设置了 `NEP_ADAPTERS_BUILD_TESTS=ON`、`NEP_ADAPTERS_ENABLE_PYTHON=ON` 和 `BUILD_SHARED_LIBS=ON`。

### CUDA 构建在另一台 GPU 上不能运行

`native` 只面向构建机器。重新配置 `CMAKE_CUDA_ARCHITECTURES`，列出目标 GPU 架构。

### LAMMPS plugin 编译成功，但没有端到端测试

配置时再传入 `-DNEP_ADAPTERS_LAMMPS_EXECUTABLE=/path/to/lmp`。可执行文件是测试输入，不是插件编译依赖。

### 安装目录只有 plugin，没有头文件和库

检查是否设置了 `NEP_ADAPTERS_INSTALL_DEVELOPMENT_FILES=OFF`。runtime-only 安装会有意省略开发文件。
