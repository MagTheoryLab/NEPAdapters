# 测试说明

CTest 是 native 测试的统一入口。

## 标签

| 标签 | 覆盖范围 |
|---|---|
| `contract` | 核心 API 或 engine SPI 契约 |
| `smoke` | 模型加载和有限值计算 |
| `parity` | runtime 输出与可信 oracle 比较 |
| `engine` | engine 层测试 |
| `frontend` | Python、LAMMPS 等集成测试 |
| `cuda` | CUDA 后端测试 |
| `device` | 需要 CUDA device kernel 的测试 |
| `kokkos` | LAMMPS/Kokkos 形状的设备邻居表测试 |
| `force` | 有限差分或力分量一致性门禁 |
| `large_model` | 使用 `nep89` 等大模型的测试 |
| `domain_decomp` | 多 rank LAMMPS 所需的 local、ghost 和 foldback 语义 |
| `calculator` | Python calculator 高层接口 |
| `ase` | 可选 ASE adapter smoke；未安装 ASE 时正常 skip |

## CPU 参考与 fixture

默认 `cpu` 测试使用 `tests/fixtures/cpu_baseline/`。该目录包含固定模型、结构、golden `energy`、`force`、`virial` 和每原子 `descriptor.txt`。这些数据只用于仓库测试，不会打进 Python wheel。

CPU 正确性同时使用固定 golden label 和独立编译的严格 FP64 oracle，覆盖：

- 类型映射；
- AoS/SoA 转换；
- batch offset；
- 能量和 virial reduction；
- descriptor 布局；
- spin 导数；
- LAMMPS 邻居 view 转换。

项目不保留第二套 CPU backend 作为 runtime 或测试 fallback。

`tests/fixtures/nep_cpu_reference/` 保存 NEP_CPU 的 250 原子普通 NEP3 和 qNEP 固定 case。CPU 直接核对力、每原子 raw9 virial 和 descriptor，不要求配置另一个仓库。CUDA 复用 qNEP case；NEP3 测试只锁定明确的 unsupported 状态。

## Python 测试

普通模型和 spin 模型使用独立门禁。spin calculator 测试覆盖：

- native 与高层 calculation/descriptor parity；
- magnetic force 和 torque；
- 单结构与多结构 batch；
- 结构自带 spin 和显式 spin 数组；
- 空结果；
- 错误模型拒绝；
- 全周期输入契约。

Python/batch 始终使用常规 `compute` API，不继承 LAMMPS 邻居表或 ghost 原子语义。

## LAMMPS 测试

`nep_adapters_cpu_lammps_neighbors_test` 覆盖 LAMMPS 形状的 `ilist`、`numneigh`、`firstneigh`、`type_map`、`double** x`、`double** f` 和 raw9 每原子 virial，并与固定 golden label 比较。

`nep_adapters_lammps_plugin_baseline_test` 是真实 LAMMPS runtime 测试。当 `NEP_ADAPTERS_LAMMPS_EXECUTABLE` 指向 `lmp` 时，它会加载 `nepadaptersplugin.so`，使用 `pair_style nep/cpu` 运行固定 fixture，并核对总能量、每原子能量和、力以及从 `stress/atom` 重建的 virial。

`nep_adapters_domain_decomp_contract_test` 是纯 C++ 测试，不调用 LAMMPS。它比较完整体系与 2 个包含 ghost 原子的模拟 rank-local 体系，然后执行 ghost force foldback 和 virial reduction。各 engine 的外部邻居 runner 应复用这一契约形状。

`nep_adapters_virial_order_test` 固定普通 NEP `compute` 与 LAMMPS `compute_for_lammps` 之间的分量顺序。

`tools/run_lammps_mpi_smoke.py` 使用 `mpirun -np 1/2/4` 运行本地 plugin，并将多 rank 的力、每原子能量、每原子 stress、总势能和压力分量与 1 rank 结果比较。

## CUDA 测试

```sh
python3 tools/run_cuda_tests.py
```

V100 使用 `--cuda-arch 70`，RTX 4090 / Ada 使用 `--cuda-arch 89`。脚本会启用 CUDA、构建测试并执行 `ctest -L cuda`。这是正确性门禁；MD 吞吐、`ncu` 和 `nsys` 采样放在 `benchmarks/` 或外部 job 脚本。

CUDA gate 覆盖：

- engine 注册和公共 capability；
- host batch API 与 LAMMPS host-neighbor simulation；
- device model/workspace 上传和内部邻居表构建；
- radial、angular、high-body 和 ZBL 力有限差分；
- CPU/CUDA triclinic parity；
- LAMMPS/Kokkos strided device-neighbor 输入、输出布局、type map、virial 顺序和邻居容量错误；
- qNEP direct reciprocal-space 的力、virial 和 descriptor 参考值；
- 默认构建的 PPPM fail-closed，或 PPPM 构建的参考值一致性。

LAMMPS `nep/gpu` 只支持启用 CUDA 的 Kokkos。缺少 Kokkos device state 是硬错误，测试不允许依赖 GPU pair style 的 host-neighbor fallback。

提供 CUDA Kokkos LAMMPS 可执行文件时，CTest 还会把本次构建的 plugin 加载进真实 LAMMPS，检查 `nep/gpu` 的能量、每原子能量、力和 virial。
