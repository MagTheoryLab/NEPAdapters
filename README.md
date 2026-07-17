# NEPAdapters

NEPAdapters 提供统一的 NEP 运行时。同一套模型可以通过 Python 或 LAMMPS 调用，并明确选择 CPU 或 CUDA 后端。

native engine 和 Python 接口当前支持普通 NEP、spin NEP 和 qNEP。生产 CPU 后端统一使用 `cpu`；CUDA 后端使用 `cuda`。不再提供 `cpu_nep3`，也不会在 GPU 不可用或模型不受支持时自动回退到 CPU。LAMMPS 的普通 NEP frontend 已进入发布面；spin LAMMPS 尚未作为生产 frontend 发布。

## 快速入口

| 你要做什么 | 入口 | 直接使用 |
|---|---|---|
| 构建并测试 CPU 版本 | [构建 CPU 版本](#构建-cpu-版本) | `cmake -S . -B .build/release ...` |
| 构建 Python wheel | [使用 Python 接口](#使用-python-接口) | `python -m build --wheel` |
| 构建 CUDA 版本 | [启用 CUDA](#启用-cuda) | `python3 tools/run_cuda_tests.py --cuda-arch 89` |
| 安装 LAMMPS 插件 | [使用 LAMMPS 插件](#使用-lammps-插件) | `-DNEP_ADAPTERS_ENABLE_LAMMPS=ON` |
| 查看测试覆盖 | [运行测试](#运行测试) | `ctest --test-dir ...` |
| 查看自动打包规则 | [自动构建 wheel](#自动构建-wheel) | `.github/workflows/python-package.yml` |

## 选择后端

| 场景 | 后端 | 要求 |
|---|---|---|
| 普通 CPU 计算、Python CPU wheel、LAMMPS CPU 插件 | `cpu` | C++17；OpenMP 可选 |
| NVIDIA GPU 批计算或 LAMMPS Kokkos | `cuda` | CUDA Toolkit；LAMMPS 路径还需要启用 CUDA 的 Kokkos |

两个后端通过相同的核心 API 暴露能力，但实现、打包和测试保持分离：

- Python CPU wheel 只包含 `nep_cpu`。
- Python GPU wheel 包含 `nep_cpu` 和按需加载的 `nep_gpu`。
- 必须显式使用 `backend="cuda"` 才会加载 GPU 扩展。
- CUDA 加载失败、NEP3 模型或未编译的 PPPM 请求都会直接报错，不会切换算法或后端。

## 构建 CPU 版本

```sh
cmake -S . -B .build/release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEP_ADAPTERS_BUILD_TESTS=ON
cmake --build .build/release -j2
ctest --test-dir .build/release --output-on-failure
```

CPU 实现已随仓库放在 `engines/cpu/native/`，不需要额外下载 `NEP_CPU` 源码。

跑完应检查：

- `ctest` 显示全部测试通过；
- 构建目录位于 `.build/`，没有在仓库根目录生成临时文件；
- 默认构建没有加载 CUDA、Python 或 LAMMPS。

## 使用 Python 接口

项目约定使用 `mysci` 环境进行本地 Python 开发：

```sh
source /Users/superbing/miniconda3/etc/profile.d/conda.sh
conda activate mysci
python -m build --wheel
```

CPU wheel 使用 `NEPCalculator`：

```python
from nep_adapters import NEPCalculator

calculator = NEPCalculator("nep.txt", backend="cpu")
prediction = calculator.predict_structures(structures)
energy, force_blocks, virial_blocks = calculator.calculate(structures)
```

结构默认按全周期 `(1, 1, 1)` 处理。显式传入非全周期值或 `None` 会报错；接口没有非周期或旧哨兵值 fallback。

普通模型使用 `calculate()` 和 `descriptors()`。spin 模型必须使用独立的 `calculate_spin()` 和 `descriptors_spin()`，并提供形状为 `(natoms, 3)` 的 spin 数组。高层接口还提供单结构、批结构和 descriptor 聚合方法。

NumPy 是必需的运行时依赖。ASE 是可选依赖；普通 `import nep_adapters` 不会导入 ASE。详细接口和数组布局见 [Python 前端说明](frontends/python/README.md)。

## 启用 CUDA

RTX 4090 / Ada 正确性测试：

```sh
python3 tools/run_cuda_tests.py --cuda-arch 89
```

V100 节点使用：

```sh
python3 tools/run_cuda_tests.py --cuda-arch 70
```

该脚本负责配置 CUDA、编译测试并运行 `ctest -L cuda`。它是正确性门禁，不是性能 benchmark。

本机构建 GPU wheel：

```sh
python -m build --wheel \
  -Ccmake.define.NEP_ADAPTERS_ENABLE_CUDA=ON
```

仓库默认编译 `sm_89`。发布 wheel 会嵌入 `sm_70`、`sm_75`、`sm_80`、`sm_86`、`sm_89` 的 SASS，并保留 `compute_89` PTX 用于较新架构的前向 JIT：

```sh
CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=70-real;75-real;80-real;86-real;89-real;89-virtual' \
python -m build --wheel \
  -Ccmake.define.NEP_ADAPTERS_ENABLE_CUDA=ON
```

默认 CUDA 构建支持 qNEP direct 路径，不编译实验性 PPPM，也不链接 cuFFT。如需单独测试 PPPM，可运行：

```sh
python3 tools/run_cuda_tests.py --cuda-arch 89 --qnep-pppm
```

PPPM 当前只适合单节点实验，不属于默认生产包。未启用 PPPM 的构建收到 `NEP_ADAPTERS_QNEP_KSPACE=pppm` 时会直接报错。

## 使用 LAMMPS 插件

CPU 插件：

```sh
cmake -S . -B .build/lammps \
  -DNEP_ADAPTERS_BUILD_TESTS=ON \
  -DNEP_ADAPTERS_ENABLE_LAMMPS=ON \
  -DNEP_ADAPTERS_LAMMPS_SOURCE_DIR=/path/to/lammps \
  -DNEP_ADAPTERS_LAMMPS_EXECUTABLE=/path/to/lmp
cmake --build .build/lammps -j2
ctest --test-dir .build/lammps -L lammps --output-on-failure
```

构建目录中的插件可以通过 `LAMMPS_PLUGIN_PATH` 自动加载：

```sh
PLUGIN="$PWD/.build/lammps/frontends/lammps/nepadaptersplugin.so"
export LAMMPS_PLUGIN_PATH="$(dirname "$PLUGIN")"
/path/to/lmp -in lmp.in
```

LAMMPS 会在启动时扫描该目录中名称以 `plugin.so` 结尾的文件，因此 `lmp.in` 不需要包含构建路径：

```lammps
pair_style nep/cpu
pair_coeff * * nep.txt Fe
```

如需安装到固定位置，建议使用静态内部库和 runtime-only 安装，让安装目录只包含一个 LAMMPS 插件：

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
export LAMMPS_PLUGIN_PATH=/path/to/nepadapters/lib
```

此时插件路径是 `/path/to/nepadapters/lib/nepadaptersplugin.so`。如果不显式设置 `CMAKE_INSTALL_LIBDIR=lib`，实际目录以 CMake 选择的 `lib` 或 `lib64` 为准。启用 `NEP_ADAPTERS_INSTALL_DEVELOPMENT_FILES=ON` 时，还会安装头文件、C/C++ 库和 `lib/cmake/NEPAdapters/` 包元数据。

CUDA 插件要求 LAMMPS Kokkos 已启用 CUDA，支持 `pair_style nep/gpu`；使用 Kokkos 后缀模式时也可显式写 `pair_style nep/gpu/kk`。GPU pair style 直接传递设备视图，不提供 host 或 CPU fallback。不要同时使用 `LAMMPS_PLUGIN_PATH` 和输入文件中的 `plugin load` 重复加载同一个插件。当前发布门禁只声明普通 NEP pair style；spin 模型虽然已有内部数据通路，但在真实 LAMMPS 端到端测试完成前不作为生产支持面。详细构建、安装和提交脚本示例见 [LAMMPS 前端说明](frontends/lammps/README.md)。

## 运行测试

常用 CTest 标签：

- `contract`：核心 API 与 engine SPI 契约；
- `smoke`：模型加载和有限值检查；
- `parity`：与固定 oracle 或参考数据比对；
- `frontend`：Python、LAMMPS 等前端集成；
- `cuda`、`device`、`kokkos`：CUDA 与设备输入路径；
- `force`：有限差分或力分量一致性；
- `bench`、`performance`：性能测试，不参与普通正确性门禁。

CPU 测试使用仓库内固定 fixture、严格 FP64 oracle、有限差分和 CPU/CUDA parity，不保留第二套 CPU 后端作为 fallback。完整覆盖见 [测试说明](tests/README.md)。

构建 benchmark：

```sh
cmake -S . -B .build/bench \
  -DNEP_ADAPTERS_BUILD_TESTS=ON \
  -DNEP_ADAPTERS_BUILD_BENCHMARKS=ON
cmake --build .build/bench -j2
ctest --test-dir .build/bench -L bench --output-on-failure
```

## Python wheel 的依赖边界

wheel 不会复制当前 Python 环境：

- NumPy 由包管理器单独安装；
- ASE 是可选依赖，不会嵌入 wheel；
- pybind11、CMake、scikit-build-core 和编译器只用于构建；
- Linux GPU wheel 不携带 `libcuda`、`libcudart` 或 `libcufft`；CUDA Runtime 静态链接进 `nep_gpu`，NVIDIA 驱动由目标机器提供；
- glibc、libstdc++、libm 和 libgcc 使用目标平台系统库。

发布时必须检查 `auditwheel show` 和 wheel 归档内容。发现新的动态依赖时应明确审查，不允许用 repair 失败后的原 wheel 作为 fallback。

## 自动构建 wheel

[`.github/workflows/python-package.yml`](.github/workflows/python-package.yml) 构建：

- CPython 3.10–3.14 的 Linux x86_64、macOS x86_64/arm64 和 Windows x86_64 CPU wheel；
- source distribution；
- CPython 3.10–3.14 的 Linux x86_64 CUDA wheel。

创建 GitHub Release 时，CPU wheel 和 source distribution 通过 Trusted Publishing 发布到 PyPI。CUDA wheel 使用标准 `1cuda` build tag，只作为 GitHub Release 附件，避免普通 `pip install nep-adapters` 误选 GPU 包。

每个 wheel 都会在隔离环境中安装并执行真实 CPU 计算。GPU wheel 还会检查 `nep_gpu` 导入和归档内容；完整 CUDA 数值测试仍需要有 GPU 的发布节点。

## 目录结构

- `include/nep_adapters/`：公共 C/C++ API、数据视图和能力定义；
- `src/`：核心注册、调度和错误处理；
- `engines/cpu/`：普通、spin 和 charge 模型的 CPU 实现；
- `engines/cuda/`：batch 与 LAMMPS Kokkos 共用的 CUDA 实现；
- `frontends/python/`：pybind11、NumPy 高层接口和可选 ASE 适配；
- `frontends/lammps/`：LAMMPS runtime plugin；
- `tests/`：契约、fixture、oracle 和集成测试；
- `benchmarks/`：吞吐、扩展性与 profiler 入口；
- `docs/design.md`：架构与边界说明。

核心运行时必须保持可独立构建：它不依赖 Python、LAMMPS 或 CUDA。LAMMPS 和 Python 是 frontend，不是 compute backend。
