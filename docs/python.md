# Python 接口指南

Python frontend 提供两层接口：

- `NEPCalculator`：面向 NumPy、结构列表和批计算；
- `NepAseCalculator`：面向 ASE 的标准 calculator 工作流。

底层 `Model` 适合需要完全控制数组布局的调用方。一般用户从 `NEPCalculator` 或 ASE 开始即可。

## 安装

仅使用 NumPy 接口：

```sh
python -m pip install nep-adapters
```

同时使用 ASE：

```sh
python -m pip install 'nep-adapters[ase]'
```

预编译 wheel 支持 CPython 3.10–3.14。Linux x86_64 和 Windows x86_64
wheel 同时包含 CPU 与 CUDA 后端，不需要另外安装 CUDA Toolkit；使用 CUDA
仍需兼容的 NVIDIA 驱动和 GPU。macOS x86_64 与 arm64 wheel 只包含 CPU
后端。

从源码安装时，构建系统会自动查找 NVCC：找到时构建 CPU+CUDA，找不到时
只构建 CPU。

```sh
python -m pip install .
```

`NEP_CUDA=1` 和 `NEP_CUDA=0` 可分别强制源码构建启用或禁用 CUDA。
NumPy 是必需依赖，ASE 是可选依赖。

## 第一次计算

```python
from ase import Atoms
from nep_adapters import NEPCalculator

atoms = Atoms(
    "Fe2",
    positions=[[0.0, 0.0, 0.0], [1.43, 1.43, 1.43]],
    cell=[2.86, 2.86, 2.86],
    pbc=True,
)

with NEPCalculator("nep.txt", backend="cpu") as calculator:
    prediction = calculator.predict_structures([atoms])

print(prediction.energy)       # (nstructures,)
print(prediction.forces)       # (total_atoms, 3)
print(prediction.virials)      # (total_atoms, 9)
```

`backend` 只能显式选择 `"cpu"` 或 `"cuda"`。库没有 `auto` 后端，也不会在 CUDA 失败时回退到 CPU。

## ASE calculator

```python
from nep_adapters.ase import NepAseCalculator

atoms.calc = NepAseCalculator("nep.txt", backend="cpu")
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

普通 `import nep_adapters` 不会导入 ASE。只有使用 ASE 适配器时才需要安装 `ase` extra。

spin 模型也使用同一个 `NepAseCalculator`。结构必须提供 `(natoms, 3)` 的 vector spin；ASE 的 vector `initial_magmoms` 可直接使用，scalar `initial_magmoms` 不会被自动猜成方向。

核心计算接口保持 GPUMD 的 pressure-positive virial 约定。ASE 适配器只在边界处转换为 ASE 的能量-应变导数约定：

```text
stress_ASE = -virial_total / volume
           = -num_atoms * structure_virials / volume
```

其中 `structure_virials` 是平均每原子 virial。这个负号转换不会改变 NumPy/C API 返回的原始 virial。

## 先检查模型和后端

```python
from nep_adapters import backend_status, inspect_model

model_info = inspect_model("nep.txt")
print(model_info.model_type)
print(model_info.elements)
print(model_info.descriptor_dim)

cuda = backend_status("cuda")
print(cuda.available, cuda.reason)
```

`inspect_model()` 不执行预测，可用于在 UI 或批处理开始前确认模型类型、元素、cutoff、能力和 SHA256。`backend_status("cuda")` 会报告扩展、运行时、设备和显存状态，并通过一次内存分配、核函数启动、同步和结果回传验证 CUDA 计算链路。

## 普通 NEP 批计算

```python
from nep_adapters import NEPCalculator

calculator = NEPCalculator("nep.txt", backend="cpu")
prediction = calculator.predict_structures(structures)

energies = prediction.energy
force_blocks = prediction.force_blocks()
virial_blocks = prediction.virial_blocks(mean=True)
```

兼容已有调用方的简写：

```python
energies, force_blocks, virial_blocks = calculator.calculate(structures)
```

`Prediction` 的主要字段：

| 字段 | 形状 | 含义 |
|---|---|---|
| `energy` | `(nstructures,)` | 每个结构的总能量 |
| `potential` | `(total_atoms,)` | 拼接后的每原子势能 |
| `forces` | `(total_atoms, 3)` | 拼接后的每原子力 |
| `virials` | `(total_atoms, 9)` | 拼接后的每原子 virial |
| `structure_virials` | `(nstructures, 9)` | 每个结构的平均每原子 virial |

所有输出均为 `float64`。Python raw9 virial 顺序是：

```text
xx, xy, xz, yx, yy, yz, zx, zy, zz
```

virial 的符号与 GPUMD 原生计算保持一致；如果调用方需要 ASE stress，必须按上一节的公式转换。

## 直接传 NumPy 数组

```python
prediction = calculator.predict_arrays(
    types=types,              # int32, (total_atoms,)
    positions=positions,      # float64, (total_atoms, 3)
    boxes=boxes,              # float64, (nstructures, 9)
    atom_counts=atom_counts,  # int32, (nstructures,)
    pbc=(1, 1, 1),
)
```

`types` 是模型元素列表中的零基索引。当前 batch API 只支持全周期；显式传入非全周期或 `None` 会报错。

## qNEP

qNEP 必须使用 charge 接口，普通 `calculate()` 会拒绝 charge 模型：

```python
prediction = calculator.predict_charge_structures(structures)
print(prediction.charges)  # (total_atoms,)
print(prediction.becs)     # (total_atoms, 9)

energies, forces, virials, charges, becs = (
    calculator.calculate_charge(structures)
)
```

CPU 支持 charge/BEC。CUDA 支持 qNEP direct batch；PPPM 只有在构建时显式打开 `NEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM=ON` 才存在。qNEP CUDA batch 可用不代表 LAMMPS Kokkos 可用，后者当前明确拒绝。

## spin NEP

spin 模型必须使用显式 spin 接口：

```python
prediction = calculator.predict_spin_structures(structures, spins=spin_blocks)
print(prediction.mforces)

energies, forces, virials, mforces = calculator.calculate_spin(
    structures,
    spins=spin_blocks,
)
```

`spins` 可以是拼接后的 `(total_atoms, 3)` 数组，也可以是每个结构一个 `(natoms, 3)` block。结构自身的 `spin` / `spins` 数据和显式参数同时存在但不一致时，接口会拒绝歧义输入。

## descriptor 和响应模型

普通、qNEP、dipole 和 polarizability 模型使用：

```python
descriptors = calculator.predict_descriptors(structures)
```

spin 模型使用：

```python
descriptors = calculator.predict_spin_descriptors(structures, spins=spin_blocks)
```

其他高层入口：

| 模型 / 计算 | 接口 |
|---|---|
| dipole | `predict_dipoles()` |
| polarizability | `predict_polarizabilities()` |
| 仅 DFT-D3 修正 | `predict_dftd3_structures()` |
| NEP + DFT-D3 | `predict_with_dftd3_structures()` |

dipole、polarizability 和 DFT-D3 当前只支持 CPU。DFT-D3 只接受普通非 spin、非 charge 势模型。

## CUDA 显存预算

```python
calculator = NEPCalculator("nep.txt", backend="cuda")

estimate = calculator.estimate_workspace(
    atom_capacity=100_000,
    structure_capacity=8,
)
print(estimate.total_bytes)

safe_limit = calculator.recommend_max_atoms()
```

`atom_capacity` 表示一个 batch 中最大单结构的原子数，不会把一个超大结构拆开。`recommend_max_atoms()` 只在 CUDA 后端有意义。

## 错误和取消

所有公共入口抛出 `NepAdaptersError` 的稳定子类，包括：

- `InvalidInputError`；
- `UnsupportedModelError`；
- `BackendUnavailableError`；
- `ModelLoadError`；
- `BackendRuntimeError`；
- `OutOfMemoryError`；
- `CancelledError`。

异常带有稳定的 `code`，并在适用时带 `backend` 和 `operation`。调用方应判断这些字段，不要匹配完整 native 错误文案。

计算运行时可从另一个 Python/UI 线程调用 `calculator.cancel()`；取消不会返回部分结果。再次使用前调用 `calculator.reset_cancel()`。

## 能力边界速查

| 能力 | CPU | CUDA |
|---|---:|---:|
| 普通 NEP | 支持 | NEP4/NEP5 支持 |
| qNEP charge/BEC | 支持 | direct 支持；PPPM 需单独编译 |
| spin NEP | 支持 | 支持 |
| dipole | 支持 | 不支持 |
| polarizability | 支持 | 不支持 |
| DFT-D3 | 受限支持 | 不支持 |

需要构建开发版或运行 Python CTest 时，见 [构建与安装](build.md#python-cmake-开发构建)。
