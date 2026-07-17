# CPU baseline 测试数据

本目录是仓库测试数据，不会打进 Python 包。

- `nep.txt`：CPU baseline 使用的小型 Te/Pb NEP 模型；
- `train.xyz`：一个固定带标签结构，其中 `energy`、`force` 和 `virial` 是由可信 NEP CPU 实现生成的 golden label。CPU、Python 和 LAMMPS 测试直接与这些值比较，不只比较两条实时实现；
- `descriptor.txt`：同一结构的每原子 descriptor 矩阵，按 `(natoms, descriptor_dim)` row-major 保存。

只有在物理契约有意变化、且新参考值已经审查时，才能重新生成本 fixture。
