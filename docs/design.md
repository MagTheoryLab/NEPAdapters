# NEPAdapters 架构设计

## 项目定位

NEPAdapters 的价值不是包装某一个实现，而是让运行时语义、多个计算引擎、前端、打包策略和一致性测试保持统一。

项目分为 3 层：

- `core`：模型与运行时语义、数据 view、capability、错误处理和调度；
- `engines`：受支持的 `cpu` 与 `cuda` 实现；
- `frontends`：Python、LAMMPS 以及未来的软件集成。

## 必须保持的边界

- LAMMPS 是 frontend，不是 engine。
- Python 是 frontend，不是 runtime。
- core target 不包含或链接 Python、LAMMPS。
- engine 实现核心 engine SPI，不依赖 frontend。
- frontend 只使用公共 runtime API，不包含 engine 内部头文件。
- ASE 只能作为可选 Python adapter，不能成为核心 calculator 的依赖。

## 计算引擎策略

`cpu` 是唯一受支持的 CPU engine。正确性由固定 golden fixture、严格 FP64 oracle、有限差分和 CPU/CUDA parity 共同约束。已经删除的旧 CPU shim 不作为 runtime、测试或兼容 fallback 保留。

CUDA 基础数据流应先保持为普通 NEP 概念：模型解析、自有邻居表、host/device staging、descriptor 与 ANN、force scatter 和 virial reduction。spin 与 charge 在这一数据流上增加明确的 buffer 和 kernel，不能反过来定义所有模型的基础形状。

CUDA 实现要保留可读的内部边界：

- `model_protocol`：解析模型协议并计算参数数量；
- `workspace_plan`：命名并计算 device buffer；
- 各 `.cu` 文件：针对这些契约实现 kernel；
- `force_pipeline`：组织完整计算流程，不暴露成新的公共 API。

不要重新形成一个同时混合文本解析、LAMMPS 约定、staging、kernel 和诊断的大型 bridge 文件。

CUDA 测试必须区分调用形状：

- 普通 batch API；
- LAMMPS host-neighbor simulation；
- LAMMPS Kokkos/device input。

Kokkos 路径不能强行经过 host pointer API。只有定义并验证 device-input 契约后，才能把相应 kernel 标记为受支持。

### CUDA 的 2 种输入模式

**直接坐标模式：** 调用方传入原子类型、坐标、盒子、PBC 和结构范围。CUDA engine 自己构建邻居表，因此 internal-neighbor workspace 持有结构、盒子和 PBC metadata。

**外部邻居模式：** 调用方已经提供邻居拓扑。host 调用使用 host-neighbor API；LAMMPS GPU frontend 使用显式 Kokkos device-input API。CUDA engine 将拓扑转换或 alias 到 slot-major 执行布局，不依赖 batch box。

两种模式进入 descriptor 和 ANN kernel 前，应汇合到同一套 slot-major 邻居数组和 SoA 原子/输出 buffer。这是性能关键边界；两侧 staging 可以保留调用方特定实现。

内部 CUDA 邻居生成必须使用适合大体系的 cell-list，不能使用 all-pairs 生产实现。当前路径采用 CSR 风格：

1. 统计每个 cell 的原子数；
2. prefix scan 得到 offset；
3. 将 atom id scatter 到 cell-contiguous 存储；
4. 遍历相邻 cell，填充 radial 和 angular slot-major 邻居数组。

all-pairs 只允许作为小体系正确性 oracle。

### LAMMPS Kokkos 边界

LAMMPS Kokkos 是第 3 种 frontend 输入形状，不等同于 host-neighbor API。本地 LAMMPS 中，坐标是按 `x(i,0..2)` 访问的 `X_FLOAT*[3]` view，邻居数据是 Kokkos `int**` view，有时还包含 transpose view。

device-input 路径保持这些数据在 GPU 上，但最终 NEP kernel 的执行布局仍由 CUDA engine 决定。核心 ABI 不应直接继承 LAMMPS view 布局。

## 前端策略

Python 和 LAMMPS 的数据形状不同，不能要求它们使用相同调用方式。

- Python 需要模型加载、batch prediction、descriptor、显式 engine 选择和直接 NumPy 数组；
- LAMMPS 需要 pair/plugin，把 atom、box、type map、neighbor list、energy、force 和 virial 约定转换为核心 view。

LAMMPS pair style 放在 `frontends/lammps/`。runtime plugin 和 source integration 都必须复用同一套 core 与 engine。

CPU LAMMPS pair 通过 LAMMPS 形状的 external-neighbor 契约进入 `cpu`，底层转发到 NEP CPU 的 `compute_for_lammps`。Python 始终使用普通 batch `compute`，不能继承 LAMMPS neighbor list 或 ghost atom 语义。

用户可见的 pair style 不暴露内部 framework 名称。当前实际注册：

- `nep/cpu`：CPU NEP；
- `nep/gpu`、`nep/gpu/kk`、`nep/gpu/kk/device`：CUDA Kokkos NEP。

pair 内部能够识别 spin capability，并要求 CPU 路径的 `atom_style spin` 或 CUDA 路径的 `atom_style spin/kk` 提供 `sp` 和 `fm`。CPU 与 CUDA 单 rank、全周期 spin pair 计算已通过真实 LAMMPS 端到端门禁。CUDA/Kokkos spin 还在 Sai V100 单节点完成了 1/2/4/8 MPI ranks 的专项正确性、TSPIN/dynspin 和扩展性验证；常规 CTest 目前仍只注册单 rank spin plugin case。CPU spin MPI 尚无同等级正式门禁，不能把 CUDA 的多 rank 结论扩大到全部后端。不要虚构未注册的 `nep/spin/cpu` 或 `nep/spin/gpu` 名称。

## 公共 API 与 engine SPI

公共 API 面向 Python、NepTrainKit、LAMMPS 和 C/C++ 用户，应保持窄而稳定。v1 后按 semver 演进。当前公共词汇位于 `api.h`、`views.hpp` 和 `capability.hpp`。

模型语义在加载后由 `nepa_model_kind()` 与 capability flags 明确表达。普通、charge、spin、dipole 和 polarizability 使用独立计算入口；`nepa_find_force_batch()` 不会因为加载了 qNEP 而增加输出。无 spin 输入的普通、charge、dipole 和 polarizability 模型共享 `nepa_find_descriptors()`，它调用各模型自身的正式 native descriptor 路径。qNEP 使用 `nepa_find_charge_batch()`，dipole/polarizability 使用各自的独立 result struct。DFT-D3 使用 `NepaDftd3Parameters` 和 `NepaDftd3Result`，避免把响应量或修正项硬塞进 `NepaFindForceResult`。

DFT-D3 engine SPI 只调用 CPU native 的 `compute_dftd3` 与 `compute_with_dftd3`。它不复制色散算法，也不接受未验证的 spin、charge、dipole 或 polarizability 组合。CUDA 当前没有对应实现，返回 unsupported 且禁止 CPU fallback。

取消状态属于 model，使用原子状态位。C API 为 `nepa_cancel_model()` 和 `nepa_reset_cancel()`；batch engine 至少在每个结构边界检查状态。已启动的 CUDA kernel 不被强制抢占，返回后才停止后续工作。前端只有在整个调用成功时才发布结果；cancelled 状态不会被包装成部分成功。Python 计算持有共享 model 生命周期并释放 GIL，因此 `close()`、`cancel()` 和正在执行的调用不会形成 use-after-free。

engine SPI 面向 engine 实现者，位于 `engine.hpp`；它不是面向应用的稳定 ABI。`include/nep_adapters/api.h` 的 C API 从 v1 起按 semver 演进，raw9 virial、模型类型、capability 和 data-view 约定属于兼容边界。

## 打包策略

Python 发布设计只有一个 `nep-adapters` distribution 和一个版本序列；当前 PyPI 尚无正式包。计划中的 Linux x86_64 wheel 同时包含独立的 `nep_cpu` 与 `nep_gpu` 扩展；macOS 和 Windows 没有受支持的 CUDA 运行面，只包含 `nep_cpu`。所有平台都不包含 LAMMPS。

导入包或选择 `backend="cpu"` 不加载 CUDA；只有显式选择 `backend="cuda"` 才加载 GPU 模块。CUDA 加载或模型能力失败时直接报错，不允许切换到 CPU。`auto` 属于 NepTrainKit 等调用方策略，不进入 backend registry。

LAMMPS integration 由源码仓库单独构建：

- runtime plugin 是当前正式支持并持续测试的安装方式；
- pair 源码保留 `PairStyle` 宏，技术上可以编进 LAMMPS，但只复制到 `src/` 不足以完成 NEPAdapters、OpenMP/BLAS 或 Kokkos/CUDA 的 include 和链接集成；
- source-tree 模式当前没有官方 CMake helper 和端到端门禁，属于自定义集成，不能在用户文档中写成已支持的一键安装；
- LAMMPS 只链接 runtime/engine，不依赖 Python。

Linux combined wheel 默认关闭 qNEP PPPM/cuFFT。实验性 PPPM 只有显式编译才存在；未启用时请求 PPPM 必须 fail-closed。CUDA Runtime 静态链接，wheel 不打包动态 `libcudart`、`libcuda` 或 `libcufft`。

## CPU 基准

1. core 可在没有 CUDA、Python 和 LAMMPS 时独立构建；
2. `cpu` 是唯一 CPU 生产边界；
3. pybind11 直接交换 NumPy 数组，不在 Python 侧做 native 返回后的形状重算；
4. CPU LAMMPS pair 通过 `compute_for_lammps` 和外部邻居表工作；
5. 生成报告记录 correctness、LAMMPS MPI smoke 和 OpenMP 原子扩展性；
6. 小型 golden fixture 放在 `tests/fixtures/`，绝不打进 Python wheel。

## 测试与性能策略

测试分为 2 类：

- **正确性：** API/SPI 契约、golden label、严格 FP64、有限差分和 frontend integration。默认 CPU、Python 和 LAMMPS 测试都与 `tests/fixtures/cpu_baseline/` 的固定标签比较；
- **性能：** 吞吐、扩展性和 profiler 证据。

CTest 是 native test 和 benchmark 的统一调度器。使用 `contract`、`smoke`、`parity`、`frontend`、`engine`、`python`、`lammps`、`bench`、`performance` 等标签选择测试范围，不另外维护重复 runner。

Python 开发使用 `mysci` 环境。默认 calculator 不导入 ASE，`nep_adapters.ase` 单独负责 ASE `Calculator` 和 `SinglePointCalculator` 集成。

LAMMPS 性能只使用 `benchmarks/lammps/` 中的真实输入，不在仓库中增加重复 MD driver。当前 CPU 报告包含 LAMMPS compile/plugin smoke 和本地 `mpirun -np 1/2/4` 正确性检查。

多 rank LAMMPS 正确性从 backend-neutral C++ domain decomposition 契约开始：完整体系参考、rank-local owned atom、ghost atom、紧凑外部邻居表、ghost force foldback 和 virial reduction。不同 engine 可以为同一契约提供 runner。

## 红线

- 不允许 LAMMPS neighbor-list 形状定义 Python batch API。
- 不允许基础 Python wheel 依赖 CUDA 动态库。
- 不允许 core 包含 Python 或 LAMMPS 头文件。
- 不增加第 2 个 CPU backend 或兼容 fallback，除非出现明确且受支持的真实用例。
- 不通过 fallback 掩盖不支持的模型、缺失 GPU、未启用 PPPM 或动态库问题。
- 不通过公共 API 暴露 engine SPI 类型。
- 不把机器相关 benchmark baseline 混入 correctness pass/fail 阈值。先记录吞吐；只有存在明确的逐机器 baseline 时才增加性能回归门禁。
