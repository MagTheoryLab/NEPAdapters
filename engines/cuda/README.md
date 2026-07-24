# CUDA 引擎

本目录包含 CUDA engine。CUDA 依赖只能存在于该 engine 和显式启用 CUDA 的包中；core 与 CPU-only Python wheel 必须能在没有 CUDA 时构建。

## 支持范围

CUDA 模型协议支持：

- 普通与 spin NEP4；
- 普通 NEP5；
- 已实现的 NEP4 charge 模式；
- qNEP calculation 和结构 descriptor 输出。

普通结构描述符接受 `n_max_radial=0..12`、
`n_max_angular=0..8`、`basis_size_radial=0..16`、
`basis_size_angular=0..12`、`num_types=1..118`、单隐藏层
`ANN=1..120`。结构
`l_max_3body=0..8` 均可推理；其中 GPUMD 当前训练接口使用 `2..8`，
`0/1` 仅为旧模型和 radial-only 模型保留。结构 angular descriptor
总数不得超过 90。

`l_max` 同时兼容旧的 `420/421` 编码、当前 0/1 flags，以及 GPUMD
现阶段仍会输出的 `q222=2` 加 `q112/q123/q233/q134` 混合格式。
按元素类型变化的 radial/angular cutoff、typewise ZBL cutoff、flexible
ZBL 和双隐藏层 ANN 尚未实现；这些模型会明确返回 unsupported，不会把
cutoff 合并后静默计算。

CUDA 不接受 NEP3，返回 `NEPA_STATUS_UNSUPPORTED`，调用方不会被重定向到 CPU。

qNEP 默认使用 direct reciprocal-space。实验性单节点 PPPM 只有在 `NEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM=ON` 时才参与编译，也只有该构建链接 cuFFT。默认构建收到 `NEP_ADAPTERS_QNEP_KSPACE=pppm` 时会直接报错。

这一边界保证默认包不依赖 cuFFT，也避免在 PPPM 尚不适合多节点生产时静默切换算法。

## 代码边界

- `cuda_engine.cpp`：engine 接口和调用方特定 staging；
- `force_pipeline.*`：模型驱动的完整 force dataflow。调用方提供 topology 与输出请求；普通/spin 选择、ZBL 组合和 virial 放置都留在模块内部；
- `device_operations.hpp`：device staging、邻居构建、descriptor、ANN、force 和输出的唯一私有接口。各 `.cu` 文件是独立 CUDA 编译单元，不是公共模块；
- `spin_onsite.cu`：spin orchestration，descriptor 与 force device 实现放在 2 个私有 `.cuh` 中；
- `model_protocol.*`：模型协议解析与参数数量；
- `device_model.hpp`：device model 接口；`model_parameters.cpp` 和 `device_model.cu` 分别实现 host packing 与 device upload；
- `device_workspace.hpp`：workspace plan、allocation 与 view；`workspace_plan.cpp` 和 `device_workspace.cu` 实现规划和分配。

kernel 应依赖这些内部契约，不得重复解析模型文本或复制 buffer 尺寸计算。

普通 NEP4/NEP5 的 `nep.txt` 协议以 `torchnep/src/force/nep.cu` 为参考。descriptor dimension 根据 `n_max` 与 `l_max ... has_q_*` body-channel flag 计算，不能从 `ANN` 行反推。

参数布局为后续缓存局部性保留明确边界：

- `ann_type_major`：每个元素类型的 W0/B0/W1 连续存放；
- descriptor coefficient：radial 与 angular 区域分别连续；
- `q_scaler`：独立线性数组。

`device_model.cu` 参与每个 `NEP_ADAPTERS_ENABLE_CUDA=ON` 构建，负责上传参数并提供稳定 device pointer。`device_workspace.cu` 持有每次调用的 device memory，通过普通 `DeviceWorkspaceView` 暴露 buffer，kernel 不接触 allocation ownership。

## 调用模式

CUDA engine 有 2 种输入模式：

1. `find_force_batch` 接受坐标、盒子和结构 metadata，由 internal-neighbor workspace 构建邻居表；
2. `find_force_lammps_neighbors` 接受 LAMMPS 风格外部邻居表，stage active atom，并复用调用方 topology。

batch box 使用 NEP/GPUMD 3×3 顺序：`ax,bx,cx, ay,by,cy, az,bz,cz`。Triclinic 路径直接使用该矩阵完成 fractional-to-Cartesian 转换。

### Host 数据准备

`host_staging.*` 负责：

- 将 batch AoS position 转为 SoA position 和 per-structure metadata；
- 将 LAMMPS 的 `ilist`、`numneigh`、`firstneigh` 转为 active atom index 和 slot-major 邻居数组。

### Device 数据准备

`device_staging.cu` 负责：

- direct batch：复制 raw host array，在 GPU 上执行 AoS-to-SoA 和 `atom_to_structure`；
- LAMMPS host-neighbor：host 侧先 gather `double**`/`int**` 行，再由 CUDA kernel 完成 type map 与 dense slot-major 邻居打包。

## 邻居表

`internal_neighbor_builder.cu` 使用面向大体系的 cell-list，不是 all-pairs 原型：

1. 统计每个 cell 的原子数；
2. prefix scan 构造 CSR cell offset；
3. scatter 原子到 cell-contiguous 存储；
4. 每个中心原子遍历相邻 cell，填充 radial 和 angular 邻居表。

两种调用模式都使用 `center + atom_capacity * slot` 的 slot-major 布局，使同一 slot 的 center thread 连续。

internal-neighbor builder 支持正交和 triclinic 全周期盒子。它在 fractional coordinate 中分 bin，保持生产路径为 cell-list；pair geometry 留给 descriptor core，避免邻居阶段生成下一阶段会覆盖的数据。

正确性测试使用独立 brute-force PBC oracle，包含倾斜 triclinic 盒子。all-pairs 只能作为小测试 oracle，不能进入生产 descriptor/force dataflow。

## Descriptor、ANN 与力

`angular_descriptor.cu` 提供普通、spin 和 charge 共用的结构 descriptor core：

- 直接读取 position 和 slot-major neighbor list；
- 在 per-block scratch 中计算 radial basis sum；
- 不使用 global radial/angular basis cache，直接累加 angular tile；
- descriptor 布局为 `atom + atom_capacity * descriptor_index`；
- force backprop 只保留 minimum-image pair vector、distance 和 `sum_fxyz`。

支持的 angular channel 包括普通 3-body `L=1..8`，以及 `q222`、`q1111`、`q112`、`q123`、`q233` 和 `q134`。

`ann_energy.cu` 处理打包后的 NEP4/NEP5 单隐藏层布局，写入每原子 `potential`，并把 descriptor derivative 放入 `fp`。

`radial_force.cu` 完成 radial force。公共 `find_force_batch` 对正交和 triclinic 单结构 NEP4 执行完整 device pipeline：staging、internal cell-list、descriptor、ANN、force 和 host copyback。

`zbl_force.cu` 实现 universal non-flexible ZBL。当 `zbl_outer <= cutoff_radial` 时复用 radial 邻居表，在 ANN backprop 后累加每原子 ZBL 势能、力和 virial，并通过有限差分门禁。Flexible ZBL 与 typewise ZBL cutoff 返回 `NEPA_STATUS_UNSUPPORTED`。

`angular_force.cu` 实现 angular force。公共门禁覆盖常规 `L=1..8` 和全部已支持 high-body channel，并使用有限差分检查 energy/force。direct batch 与其他路径复用同一 batched workspace。

`device_output.cu` 在 GPU 上完成 batch 与 LAMMPS reduction/layout conversion：

- force 从 SoA 打包为公共布局；
- virial 在输出边界转换为对应公共顺序；
- total 在 copyback 前完成 reduction。

## LAMMPS Kokkos

本地 LAMMPS Kokkos 坐标使用 `X_FLOAT*[3]` view，通过 `x(i,0..2)` 访问；邻居使用 `neighbors(i,j)`。具体 layout 由 Kokkos/LAMMPS build macro 控制。

Kokkos frontend 可直接传 device view，但 host-neighbor API 仍应 stage 到 engine 自有执行布局。LAMMPS view 不能成为 core ABI。

普通与 spin 共用 `force_pipeline`，具体阶段保持私有。charge device operation 隔离在 `qnep_charge.cu`，后续 charge integration 应组合现有结构和 short-range 阶段，不修改调用方接口。

## 测试边界

测试按调用形状拆分：

- 普通 batch；
- LAMMPS host-neighbor simulation；
- LAMMPS Kokkos-style device staging。

Kokkos simulation 在公共 device-input ABI 冻结前只作为 workspace/layout 契约。未验证的普通 NEP 形状必须返回 `NEPA_STATUS_UNSUPPORTED`，不能使用隐藏 fallback。

可以从维护中的 NEP_GPU 路径复用经过测量的 kernel 思路，但 staging、batch 和 neighbor 契约必须由 NEPAdapters 自己拥有，不能重新暴露旧 LAMMPS bridge 形状。
