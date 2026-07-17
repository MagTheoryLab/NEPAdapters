# 测试数据

fixture 应保持小而明确。

当前固定数据：

- `cpu_baseline/`：小型 Te/Pb 模型、一个固定结构和 golden label。标签来自可信 CPU 实现，用于仓库正确性门禁；
- `nep_cpu_reference/`：从 NEP_CPU 测试套件复制的 250 原子普通 NEP 与 qNEP case，包含力、每原子 raw9 virial 和 descriptor 参考值。

未来 fixture 只有在覆盖真实缺口时才添加，例如最小 spin 结构或用于协议兼容检查的裁剪模型。

不要在这里提交 benchmark 输出、大型训练数据或临时计算产物。
