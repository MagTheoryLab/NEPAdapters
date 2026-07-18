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

qNEP 使用 `Model.calculate_charge()`，固定返回：

| 返回值 | 形状 | 含义 |
|---|---|---|
| `potential` | `(natoms,)` | 每原子势能 |
| `forces` | `(natoms, 3)` | 每原子力 |
| `virials` | `(natoms, 9)` | 每原子 raw9 virial |
| `charges` | `(natoms,)` | 每原子电荷 |
| `becs` | `(natoms, 9)` | 每原子 Born effective charge tensor |

charge 模型调用普通 `Model.calculate()` 会报错。非 charge 模型调用
`calculate_charge()` 也会报错；接口不会根据模型类型静默改变 tuple 长度。

spin 模型必须使用显式接口，避免普通调用遗漏 spin 输入：

- `Model.calculate_spin()` 接受 `(natoms, 3)` 的 spin 数组，返回势能、力、每原子 virial 和磁力；
- `Model.descriptors_spin()` 返回 spin 模型的 descriptor 矩阵；

其他正式底层接口：

- `Model.dipoles()`：返回 `(nstructures, 3)`；
- `Model.polarizabilities()`：返回 `(nstructures, 6)`，顺序为 `xx, yy, zz, xy, yz, zx`；
- `Model.calculate_dftd3()`：只计算 DFT-D3 修正；
- `Model.calculate_with_dftd3()`：计算普通 NEP 与 DFT-D3 之和；
- `Model.cancel()` / `Model.reset_cancel()`：设置或复位线程安全的取消状态。

DFT-D3 两个接口都接受 `functional`、`cutoff` 和 `cutoff_cn`，并返回与普通 `calculate()` 相同的三个数组。当前只支持 CPU 上的普通非 spin、非 charge 势模型。dipole、polarizability 和 DFT-D3 在 CUDA 上明确 unsupported，不会切换 CPU。

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

所有生产输出为 `float64`。`virials`、`structure_virials` 和 `becs` 使用 raw9 顺序 `xx, xy, xz, yx, yy, yz, zx, zy, zz`。

NepTrainKit 风格调用：

```python
from nep_adapters import NEPCalculator

calculator = NEPCalculator("nep.txt", backend="cpu")
energy, force_blocks, virial_blocks = calculator.calculate(structures)
```

空 batch 返回空数组和空 block 列表。

qNEP 高层接口为 `predict_charge_arrays()`、`predict_charge_structures()` 和
`calculate_charge()`。它们返回 `ChargePrediction`，在 `Prediction` 字段之外增加
`charges`、`becs`、`charge_blocks()` 和 `bec_blocks()`。空 batch 的 charge/BEC
形状稳定为 `(0,)` 和 `(0, 9)`。

响应和 DFT-D3 高层接口包括：

- `predict_dipoles()` / `get_structures_dipole()`；
- `predict_polarizabilities()` / `get_structures_polarizability()`；
- `predict_dftd3_structures()` / `calculate_dftd3()`；
- `predict_with_dftd3_structures()` / `calculate_with_dftd3()`。

`NEPCalculator.cancel()` 与 `reset_cancel()` 转发到底层模型。native 计算释放 GIL，另一个 Python/UI 线程可以发出取消请求。取消在 CPU/CUDA 结构边界响应；安全的阶段边界也会检查。CUDA 不抢占正在运行的单个 kernel，但 kernel 返回后会停止后续结构或阶段。取消只抛出异常，不返回部分 `Prediction`。

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

wheel 配置默认关闭 CUDA 和 qNEP PPPM/cuFFT，但 CPU 扩展始终启用 OpenMP。基础 wheel 包含 Python 包、ABI 匹配的 `nep_cpu` 扩展和必要时由 wheel 修复工具打包的 OpenMP runtime。独立 CMake 安装仍默认保留开发库、头文件、OpenMP 和 package metadata。

GPU wheel：

```sh
NEP_CUDA=1 python -m build --wheel
```

直接从源码安装 GPU 版使用短开关：

```sh
NEP_CUDA=1 pip install .
```

不传 `CMAKE_CUDA_ARCHITECTURES` 时使用 `native`。默认 `pip install .` 只构建
启用 OpenMP 的 CPU 扩展，不会探测 CUDA。macOS 源码构建需要先执行
`brew install libomp`；CMake 会自动读取 Homebrew 安装路径。

高级用法仍可通过
`--config-settings=cmake.define.NEP_ADAPTERS_ENABLE_CUDA=ON` 显式传递 CMake
选项；一般用户不需要记忆这条长命令。

GPU wheel 同时包含 `nep_cpu` 和 `nep_gpu`。导入 `nep_adapters` 或选择 `cpu` 时只加载 `nep_cpu`；只有显式选择 `backend="cuda"` 才按需导入 `nep_gpu`。CUDA 加载失败不会切换到 CPU。

CUDA 接受已实现的 NEP4/NEP5 协议。NEP3 模型传给 `backend="cuda"` 会明确报不支持。qNEP direct 模型通过 `nep_gpu` 执行 `calculate_charge()` 和 `descriptors()`；普通 `calculate()` 对 charge 模型保持 fail-closed。

发布 wheel 不使用 `native`，而是同时嵌入 V100、T4、A100、A10/RTX 30 和
RTX 4090 的 SASS，并保留 `compute_89` PTX：

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
- CPU 和 GPU wheel 都启用 OpenMP；Linux/macOS 修复后的 wheel 可能携带 `libgomp` 或 `libomp`，Windows 使用对应的 OpenMP runtime；
- 默认 Linux GPU wheel 不携带 `libcuda`、`libcudart` 或 `libcufft`；CUDA Runtime 静态链接进 `nep_gpu`，NVIDIA 驱动由目标机器提供；
- glibc、libstdc++、libm 和 libgcc 使用系统库，其最低符号版本由 manylinux 构建环境决定。

发布门禁必须执行 `auditwheel show` 和归档内容检查。未来如果增加新的非系统动态库，应显式审查，不允许把意外 vendoring 当成正常结果。
