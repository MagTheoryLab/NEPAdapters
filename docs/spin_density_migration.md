# c516 密度描述符迁移

NEPAdapters 的 spin 后端对齐 GPUMD-MagTheoryLab `c516` 的
`cd6ed6b9f46646176486dafc418a36453fb68c70`（2026-11-11 核对）。
普通 NEP、qNEP、响应模型及公共 C/Python/LAMMPS 接口保留；旧 Spin1/2/3
计算代码、模型 fixture 和测试已移除。

## 模型怎么用

直接读取 c516 导出的块模型：`nep <N> <elements...>`、`format_version 1`、
`model nep_spin`、`spin_version 1`。`spin_soc 0/1` 分别选择非 SOC/SOC。
旧 raw spin 文件、`model nep_spin3` 和旧的投影/元素选择字段明确报错。
旧模型不能通过改标签迁移，需使用新描述符重新训练并导出权重。

`spin_type` 只选择自旋自由度。未选元素的有效自旋、磁力及 spin-transfer
为零，但几何环境、原子力和能量中心保留。尤其是非磁 Bi/O 中心，spin
密度描述符也可以非零。原始输入自旋不被修改；省略该字段时所有元素可磁。

新 spin 参数接在 ANN 和普通结构系数之后，顺序为：

1. 径向混合 `radial_mix[(n_max_angular+1) * C]`；
2. 路径权重 `path_weights[path_count * C]`；
3. 中心混合 `center_mix[(primitive_count-1) * C]`。

新描述符复用结构角向系数，删除旧独立磁性径向系数和
`spin_projection_size`。文件的结构系数仍在读取时转置；新 spin 参数保持导出顺序。
`energy_reference` 仅用双精度恢复每元素能量零点。

范围与 c516 对齐：`spin_compress=1..9`、`spin_l_max=0..4`、
`spin_order=1..4`、`spin_basis_size=basis_size_angular=0..12`；
spin 维数不超过 96，总维数不超过 199。`spin_cutoff` 可以统一指定或按元素
指定，pair cutoff 取平均，且每个值不能超过结构 radial cutoff。
结构角向 cutoff 至少为 3，结构 `l_max` 至少为 2；spin 不支持 flexible ZBL。

## 后端方案与来源

CPU 使用独立 FP64 forward/解析反向实现；CUDA 移植 c516 的布局、系数表、
generic 核与 C3L2S2 编译布局，常用形状使用四线程协作。
CUDA 预收缩结构角向系数与径向混合参数，删除旧 spin workspace。
LAMMPS 只负责邻居及输入/输出适配，CPU/CUDA 的 virial 顺序在公共边界转换。
GPU 的 batch 当前仍按结构串行调度 spin，保持原 API 的多盒语义。
本轮没有移植 c516 的 edge radial cache，也没有作吞吐提升声明。

改编文件为 `engines/common/spin_density_{layout,tables}.hpp`、
`engines/cuda/spin_density_{kernels,md_layouts}.cuh`，对应上游
`src/force/nep_spin/spin_{layout,tables,kernels,md_layouts}.cuh`。
保留上游 LGPL 来源声明及 `engines/common/LICENSE.GPUMD`。

## 数值验收

冻结 fixture 见 `tests/fixtures/spin_density/README.md`。九套模型、57 个状态：
SOC 标签来自独立 e3nn FP64/autograd；非 SOC 来自独立 Legendre 标量能量与
有限差分，生成器不使用生产 spherical-harmonic/CG 表。
覆盖能量、原子力、磁力、virial、非 SOC spin 描述符；另检查坐标/自旋/应变
有限差分、三斜周期胞、零自旋、孤立原子、非磁能量中心及 excluded ghosts。
FP64 综合测试用每中心能量对邻居自旋的有限差分检查 spin-transfer/热流收缩。
Python 检查多盒 batch、ASE、原始输入保留、能量零点与 ZBL。

2026-11-11 的最终验收：

| 环境 | 结果 |
|---|---|
| macOS、mysci、Release CPU/OpenMP、Python | 非 benchmark CTest **39/39 通过** |
| Sai `16v100n02`、1 × V100-SXM2-16GB、CUDA 12.9.1、SM70、Release shared | 非 benchmark CTest **87/87 通过** |
| 真实 LAMMPS CPU | Newton off：1/2/4/8 MPI；Newton on：1 MPI，通过 |
| 真实 LAMMPS Kokkos/CUDA | Newton off：1/2/4/8 MPI，共享单卡，通过 |

Sai 作业 `2374825` 为 `COMPLETED / 0:0`。默认 `OMP_NUM_THREADS=4`，
MPI 测试显式为 1；密度 oracle 测试为 4。已从编译日志核实 CUDA 源文件重编译，
并核对 CPU/CUDA 新实现的源码 SHA256；没有沿用旧实现的目标文件。
此作业还通过了 CUDA batch 的结构/自旋 benchmark 入口小型冒烟；
本地 CPU 外部邻居 benchmark 的 27 个中心、9 个 ghost 冒烟通过。
辅助入口作业 `2374934` 也以 `COMPLETED / 0:0` 结束：27 原子、平均约
23.9 个邻居，覆盖结构/自旋模型、AoS/SoA、公共 API 与 workspace 复用路径，
启用总量、per-atom、spin-transfer 和 phase breakdown。此处补齐了复用路径的
`spin_type` 掩码传递与磁力缓冲清零；日志见
`.build/sai-density-bench-fixed-final.json`。这些仅是入口冒烟，不用于判断性能。

LAMMPS C3L2S2 独立参考的最大绝对误差如下（1/2/4/8 MPI 合并统计）：

| 后端 | 总能量 | 原子力分量 | 磁力分量 | virial6 分量 |
|---|---:|---:|---:|---:|
| CPU | 1.25e-8 | 1.82e-9 | 5.58e-9 | 6.71e-9 |
| CUDA | 2.39e-8 | 2.75e-8 | 5.65e-8 | 3.22e-8 |

独立标签按 GPUMD 的 FP32 模型参数精度计算，CPU 保留文件的 FP64 数值。
因此仅将这套新 LAMMPS CPU fixture 的能量容差设为 1e-7；
原子力/磁力容差保持 1e-8，未改普通模型的门限或参考标签。

验收中修复了 excluded species 的 spin-transfer 屏蔽、CUDA spin-transfer
重复调用前清零，以及 LAMMPS 空 rank 跳过反向通信的问题。完全空的 rank
也必须参加零长度消息交换，否则会遗留 MPI 消息并破坏后续 dump。
修复后 CPU/CUDA 的 8 MPI dump 均完整且无空字节，数值通过相同参考门限。

可追溯证据保留在本地 `.build/spin-density-local-final-tests.log`、
`.build/sai-density-cycle7-final.json`、`.build/sai-density-final-ctest.json`
与 `tmp/hpc-platform/runs/nepadapters-spin-density-migration.json`。
远端日志为独立工作区中的 `density-mpi-validation/slurm-2374825.out`。
所有远程工作在独立 `nepadapters-spin-density-migration` workspace；
不覆盖正式安装、module 或生产作业。
数值推理验收不代表材料训练精度、长期动力学稳定性或性能验收。
