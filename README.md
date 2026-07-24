# NEPAdapters

NEPAdapters 为 NEP 模型提供统一的推理运行时：同一个模型可以通过 Python、C/C++ 或 LAMMPS 调用，并显式选择 CPU 或 CUDA 后端。

如果选中的后端不可用，或该后端不支持当前模型，接口会明确报错，不会静默改用其他后端。普通 NEP、qNEP、spin NEP、dipole、polarizability 和 DFT-D3 使用各自明确的接口，避免输出含义或形状随模型类型悄悄变化。

## 先选你要的入口

| 使用场景 | 从这里开始 |
|---|---|
| Python / NumPy / ASE | [Python 接口指南](docs/python.md) |
| CMake 构建、测试和安装 | [构建与安装](docs/build.md) |
| LAMMPS CPU 或 CUDA | [LAMMPS 前端指南](frontends/lammps/README.md) |
| C/C++ API | [公共头文件](include/nep_adapters/api.h) 与 [架构说明](docs/design.md) |
| 测试和正确性验证 | [测试说明](tests/README.md) |

## Python 快速开始

仅使用 NumPy 接口：

```sh
python -m pip install nep-adapters
```

同时使用 ASE：

```sh
python -m pip install 'nep-adapters[ase]'
```

使用 ASE 做一次普通 NEP 计算：

```python
from ase import Atoms
from nep_adapters.ase import NepAseCalculator

atoms = Atoms(
    "Fe2",
    positions=[[0.0, 0.0, 0.0], [1.43, 1.43, 1.43]],
    cell=[2.86, 2.86, 2.86],
    pbc=True,
)
atoms.calc = NepAseCalculator("nep.txt", backend="cpu")

print(atoms.get_potential_energy())
print(atoms.get_forces())
```

预编译 wheel 支持 CPython 3.10–3.14：

| 平台 | wheel 包含的后端 | 运行要求 |
|---|---|---|
| Linux x86_64 | CPU + CUDA | CPU 可直接使用；CUDA 需要 NVIDIA 驱动和受支持的 GPU |
| Windows x86_64 | CPU + CUDA | CPU 可直接使用；CUDA 需要 NVIDIA 驱动和受支持的 GPU |
| macOS x86_64 / arm64 | CPU | 不提供 CUDA 后端 |

Linux 和 Windows 的预编译 wheel 已包含 CUDA 运行时，不要求另外安装 CUDA Toolkit。普通 `import nep_adapters` 或选择 `backend="cpu"` 不会初始化 CUDA；只有显式选择 `backend="cuda"` 或检查 CUDA 状态时才会加载 GPU 扩展。CUDA 加载或计算失败时不会改用 CPU。

CUDA wheel 内置 `sm_60` 和 `sm_89` SASS，并保留 `compute_60` PTX。没有对应 SASS 的较新架构会在驱动支持时通过 PTX 即时编译运行，因此兼容范围更广，但可能增加首次加载时间或产生一定性能损失。

可以运行下面的命令检查扩展、驱动、设备，并实际完成一次 CUDA 内存分配、核函数启动、同步和结果回传：

```sh
python -c "from nep_adapters import backend_status; s=backend_status('cuda'); print(s); assert s.installed and s.available, s"
```

需要修改源码或只为本机 GPU 编译时，从仓库安装：

```sh
git clone https://github.com/MagTheoryLab/NEPAdapters.git
cd NEPAdapters
python -m pip install '.[ase]'
```

源码安装会依次查找 `CUDACXX`、`CUDAToolkit_ROOT`、`CUDA_PATH`、
`CUDA_HOME` 和 `PATH` 中的 NVCC。检测到 CUDA Toolkit 时自动构建
CPU+CUDA，否则只构建 CPU。不指定 `CMAKE_CUDA_ARCHITECTURES` 时使用
`native`，只为构建机器上的 GPU 编译。`NEP_CUDA=1` 和 `NEP_CUDA=0`
可分别强制启用或禁用 CUDA。

更多 NumPy batch、qNEP、spin、descriptor 和错误处理示例见 [Python 接口指南](docs/python.md)。

## CMake 快速开始

只构建 CPU core 和测试：

```sh
cmake -S . -B .build/release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEP_ADAPTERS_BUILD_TESTS=ON
cmake --build .build/release -j2
ctest --test-dir .build/release --output-on-failure
```

Python 源码安装会自动查找 NVCC；独立 CMake 和 LAMMPS 构建仍按目标显式
打开 CUDA，避免有 Toolkit 的机器意外改变 CPU-only 构建。完整组合、安装命令
和选项默认值见 [构建与安装](docs/build.md)。

## LAMMPS 安装方式

LAMMPS 有两种支持的安装方式：

- **运行时插件（推荐）**：单独构建共享库，通过 `LAMMPS_PLUGIN_PATH` 加载，不需要重新编译 LAMMPS；
- **源码树内置**：运行 `tools/install_lammps_source.py`，把受管的 pair style 源文件复制到 `lammps/src/`，再通过随附的 CMake 配置把 core 和 engine 静态编入 `lmp`。运行时不需要设置插件环境变量。

不要手工只复制 `.cpp` 和 `.h` 文件；安装脚本还会写入连接依赖所需的 CMake 配置、记录文件哈希，并提供安全卸载。两种方式的完整命令见 [LAMMPS 前端指南](frontends/lammps/README.md)。

## 当前支持范围

| 前端 / 后端 | 普通 NEP | qNEP | spin NEP | 其他响应模型 |
|---|---|---|---|---|
| Python / CPU | 支持 | charge/BEC 支持 | 支持 | dipole、polarizability、DFT-D3 支持；DFT-D3 仅普通非 spin、非 charge 模型 |
| Python / CUDA | 支持 NEP4/NEP5 | 支持 direct；预编译 wheel 不包含 PPPM | 支持 | 不支持 dipole、polarizability、DFT-D3 |
| LAMMPS / CPU 插件 | 支持，已完成普通模型 MPI 冒烟测试 | 尚未列入已验证支持范围 | 单进程已纳入自动化测试；spin MPI 尚未覆盖同等范围的回归测试 | 不适用 |
| LAMMPS / CUDA Kokkos 插件 | 支持 | **不支持，运行时会明确报错** | 已验证单节点 1/2/4/8 个 MPI 进程 | 不适用 |

补充边界：

- 所有 Python batch 接口目前只接受全周期 `pbc=(1, 1, 1)`。
- LAMMPS spin 不使用单独的 `nep/spin/*` 名称；仍使用 `nep/cpu` 或 `nep/gpu/kk`，并配合 `atom_style spin` 或 `spin/kk`。
- qNEP 的 CUDA batch API 可用，但 LAMMPS Kokkos 需要独立的 ghost/charge 数据流，因此当前会明确拒绝该组合，不会静默降级。
- virial 在不同公共入口有不同顺序；调用前请查 [Python 接口指南](docs/python.md) 或 [公共 API 契约](include/nep_adapters/api.h)。

## 项目结构

- `include/nep_adapters/`：公共 C/C++ API 和数据约定；
- `src/`：core 注册、调度和错误处理；
- `engines/cpu/`、`engines/cuda/`：计算后端；
- `frontends/python/`、`frontends/lammps/`：用户前端；
- `tests/`：契约、fixture、oracle 和集成测试；
- `benchmarks/`：性能与扩展性入口。

LAMMPS 和 Python 都是前端，不是计算后端；算法实现只保留在 engine 层。

## 许可证

NEPAdapters 以 [GNU GPL v3 或更高版本](engines/cpu/native/LICENSE.NEP_CPU) 发布。
