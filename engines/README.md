# 计算引擎

计算引擎实现核心 engine SPI，负责数值计算，不依赖 Python 或 LAMMPS 前端。

当前支持：

- `cpu`：普通、spin 和 charge NEP 模型的 CPU 引擎；
- `cuda`：CUDA 引擎；LAMMPS 路径要求启用 CUDA 的 Kokkos。

生产代码通过核心能力和 registry 选择引擎。只有测试代码应直接比较不同引擎。
