# NEPAdapters

NEPAdapters 是统一的 NEP 推理运行时：同一个模型可以通过 Python、C/C++ 或 LAMMPS 调用，并显式选择 CPU 或 CUDA 后端。

项目不会在后端不可用或模型不受支持时静默回退。普通 NEP、qNEP、spin NEP、dipole、polarizability 和 DFT-D3 使用各自明确的接口，避免返回值随模型类型悄悄变化。

## 先选你要的入口

| 使用场景 | 从这里开始 |
|---|---|
| Python / NumPy / ASE | [Python 接口指南](docs/python.md) |
| CMake 构建、测试和安装 | [构建与安装](docs/build.md) |
| LAMMPS CPU 或 CUDA | [LAMMPS 前端指南](frontends/lammps/README.md) |
| C/C++ API | [公共头文件](include/nep_adapters/api.h) 与 [架构说明](docs/design.md) |
| 测试和正确性门禁 | [测试说明](tests/README.md) |

## Python 快速开始

PyPI 目前还没有可直接安装的正式包，请从源码安装。CPU 版本：

```sh
git clone <NEPAdapters repository URL>
cd NEPAdapters
python -m pip install '.[ase]'
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

CUDA 源码安装需要 CUDA Toolkit，并且必须显式启用：

```sh
NEP_CUDA=1 python -m pip install .
```

选择 `backend="cuda"` 才会加载 GPU 扩展；失败时不会改用 CPU。更多 NumPy batch、qNEP、spin、descriptor 和错误处理示例见 [Python 接口指南](docs/python.md)。

## CMake 快速开始

只构建 CPU core 和测试：

```sh
cmake -S . -B .build/release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEP_ADAPTERS_BUILD_TESTS=ON
cmake --build .build/release -j2
ctest --test-dir .build/release --output-on-failure
```

默认只启用 CPU。Python、CUDA 和 LAMMPS 都需要显式打开；完整组合、安装命令和选项默认值见 [构建与安装](docs/build.md)。

## LAMMPS 安装方式

LAMMPS 有两种正式安装方式：

- **runtime plugin（推荐）**：单独构建 `.so`，通过 `LAMMPS_PLUGIN_PATH` 加载，不需要重新编译 LAMMPS；
- **source-tree / builtin**：运行 `tools/install_lammps_source.py`，把受管 pair 文件复制到 `lammps/src/`，再通过随附的 CMake hook 把 core/engine 静态编进 `lmp`。运行时不需要 plugin 环境变量。

不要手工只复制 `.cpp/.h`；官方 helper 还会安装依赖连接所需的 CMake hook、记录文件哈希并提供安全卸载。两种方式的完整命令见 [LAMMPS 前端指南](frontends/lammps/README.md)。

## 当前支持范围

| 前端 / 后端 | 普通 NEP | qNEP | spin NEP | 其他响应模型 |
|---|---|---|---|---|
| Python / CPU | 支持 | charge/BEC 支持 | 支持 | dipole、polarizability、DFT-D3 支持；DFT-D3 仅普通非 spin、非 charge 模型 |
| Python / CUDA | NEP4/NEP5 支持 | direct 支持；PPPM 需单独编译 | 支持 | dipole、polarizability、DFT-D3 不支持 |
| LAMMPS / CPU plugin | 支持，含普通 MPI smoke | 尚未列入已验证支持面 | 单 rank 已进入门禁；CPU spin MPI 尚无同等级正式门禁 | 不适用 |
| LAMMPS / CUDA Kokkos plugin | 支持 | **不支持，运行时明确拒绝** | 单节点 1/2/4/8 MPI ranks 已验证 | 不适用 |

补充边界：

- 所有 Python batch 接口目前只接受全周期 `pbc=(1, 1, 1)`。
- LAMMPS spin 不使用虚构的 `nep/spin/*` 名称；仍使用 `nep/cpu` 或 `nep/gpu/kk`，并配合 `atom_style spin` / `spin/kk`。
- qNEP 的 CUDA batch API 可用，但 LAMMPS Kokkos 需要独立的 ghost/charge 数据流，因此当前 fail-closed。
- virial 在不同公共入口有不同顺序；调用前请查 [Python 接口指南](docs/python.md) 或 [公共 API 契约](include/nep_adapters/api.h)。

## 项目结构

- `include/nep_adapters/`：公共 C/C++ API 和数据约定；
- `src/`：core 注册、调度和错误处理；
- `engines/cpu/`、`engines/cuda/`：计算后端；
- `frontends/python/`、`frontends/lammps/`：用户前端；
- `tests/`：契约、fixture、oracle 和集成测试；
- `benchmarks/`：性能与扩展性入口。

LAMMPS 和 Python 都是 frontend，不是 compute backend；算法实现只保留在 engine 层。
