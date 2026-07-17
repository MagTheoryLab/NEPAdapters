# Python 前端

本目录提供 pybind11 绑定、NumPy 高层接口和可选 ASE 适配。

默认 wheel 只构建 CPU，不导入也不链接 CUDA。只有显式构建 GPU wheel 并选择 `backend="cuda"` 时，Python 才会按需加载 `nep_gpu`。

## 返回数据

底层 `Model` 直接返回 NumPy 数组，不在 Python 中复制后端算法。

普通模型的 `Model.calculate()` 返回：

| 返回值 | 形状 | 含义 |
|---|---|---|
| `potential` | `(natoms,)` | 每原子势能 |
| `forces` | `(natoms, 3)` | 每原子力 |
| `virials` | `(natoms, 9)` | 每原子 raw9 virial |

`Model.descriptors()` 返回 `(natoms, descriptor_dim)` 的 descriptor 矩阵。`Model.model_info()` 不执行计算，直接返回 cutoff、能力标志和 `descriptor_dim` 等模型信息。

spin 模型必须使用显式接口，避免普通调用遗漏 spin 输入：

- `Model.calculate_spin()` 接受 `(natoms, 3)` 的 spin 数组，返回势能、力、每原子 virial、磁力和 torque；
- `Model.descriptors_spin()` 返回 spin 模型的 descriptor 矩阵；
- torque 定义为 `spin × magnetic_force`。

## 使用 `NEPCalculator`

`NEPCalculator` 以 NumPy 为核心，不依赖 ASE。结构对象需要提供：

- `get_chemical_symbols()` 或 `symbols`；
- `positions`；
- `cell`；
- 可选的 `pbc`。

`pbc` 在底层 NumPy 方法和高层 calculator 中都默认取 `(1, 1, 1)`。没有 `pbc` 属性的结构同样按全周期处理。显式非全周期值和显式 `None` 都会报错；项目不提供非周期或旧哨兵值 fallback。

`predict_structures()` 返回 `Prediction`：

| 字段 | 形状 | 含义 |
|---|---|---|
| `energy` | `(nstructures,)` | 每个结构的总能量 |
| `potential` | `(natoms,)` | 拼接后的每原子势能 |
| `forces` | `(natoms, 3)` | 拼接后的每原子力 |
| `virials` | `(natoms, 9)` | 拼接后的每原子 virial |
| `structure_virials` | `(nstructures, 9)` | 每个结构的平均每原子 virial |

NepTrainKit 风格调用：

```python
from nep_adapters import NEPCalculator

calculator = NEPCalculator("nep.txt", backend="cpu")
energy, force_blocks, virial_blocks = calculator.calculate(structures)
```

空 batch 返回空数组和空 block 列表。

descriptor 接口：

- `get_descriptor(structure)`：返回单个结构的每原子 descriptor；
- `get_structures_descriptor(structures, mean_descriptor=True)`：每个结构返回一个平均 descriptor；
- `mean_descriptor=False`：返回拼接后的每原子 descriptor 矩阵。

spin 高层接口包括：

- `predict_spin_arrays()`；
- `predict_spin_structures()`；
- `predict_spin_descriptors_arrays()`；
- `predict_spin_descriptors()`；
- `calculate_spin()`；
- `get_spin_descriptor()`；
- `get_spin_structures_descriptor()`。

结构可以通过 `(natoms, 3)` 的 `spins` 属性或 `arrays["spins"]` 提供 spin；调用方也可以显式传入拼接后的 spin 数组。

## 可选 ASE 接口

安装 `ase` extra 后显式导入适配器：

```python
from nep_adapters.ase import NepAseCalculator

atoms.calc = NepAseCalculator("nep.txt")
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

普通 `import nep_adapters` 不会导入 ASE。

## 本地测试

```sh
cmake -S . -B .build/python \
  -DBUILD_SHARED_LIBS=ON \
  -DNEP_ADAPTERS_BUILD_TESTS=ON \
  -DNEP_ADAPTERS_ENABLE_PYTHON=ON \
  -DPython3_EXECUTABLE=/Users/superbing/miniconda3/envs/mysci/bin/python
cmake --build .build/python -j2
ctest --test-dir .build/python -L python --output-on-failure
```

## 构建 wheel

CPU wheel：

```sh
python -m build --wheel
```

wheel 配置默认关闭 CUDA、qNEP PPPM/cuFFT、OpenMP 和 C/C++ 开发文件安装。基础 wheel 包含 Python 包和 ABI 匹配的 `nep_cpu` 扩展。独立 CMake 安装仍默认保留开发库、头文件、OpenMP 和 package metadata。

GPU wheel：

```sh
python -m build --wheel \
  -Ccmake.define.NEP_ADAPTERS_ENABLE_CUDA=ON
```

GPU wheel 同时包含 `nep_cpu` 和 `nep_gpu`。导入 `nep_adapters` 或选择 `cpu` 时只加载 `nep_cpu`；只有显式选择 `backend="cuda"` 才按需导入 `nep_gpu`。CUDA 加载失败不会切换到 CPU。

CUDA 接受已实现的 NEP4/NEP5 协议。NEP3 模型传给 `backend="cuda"` 会明确报不支持。qNEP direct 模型可以通过 `nep_gpu` 执行 `calculate()` 和 `descriptors()`。

仓库默认 CUDA 架构为 `sm_89`，用于本地 RTX 4090。发布 wheel 可同时嵌入 V100、T4、A100、A10/RTX 30 和 RTX 4090 的 SASS，并保留 `compute_89` PTX：

```sh
CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=70-real;75-real;80-real;86-real;89-real;89-virtual' \
python -m build --wheel \
  -Ccmake.define.NEP_ADAPTERS_ENABLE_CUDA=ON
```

发布 wheel 必须在目标 manylinux 构建镜像中编译。`auditwheel repair` 可以补充平台 tag 或允许的动态库，但不能降低较新宿主编译器引入的 GLIBC/GLIBCXX 符号版本。

## wheel 依赖边界

wheel 不会复制当前 Python 环境：

- NumPy 是运行时依赖，由包管理器单独安装；
- ASE 是可选依赖，不嵌入 wheel；
- pybind11、CMake、scikit-build-core 和编译器只用于构建；
- 默认 Linux GPU wheel 不携带 `libcuda`、`libcudart` 或 `libcufft`；CUDA Runtime 静态链接进 `nep_gpu`，NVIDIA 驱动由目标机器提供；
- glibc、libstdc++、libm 和 libgcc 使用系统库，其最低符号版本由 manylinux 构建环境决定。

发布门禁必须执行 `auditwheel show` 和归档内容检查。未来如果增加新的非系统动态库，应显式审查，不允许把意外 vendoring 当成正常结果。
