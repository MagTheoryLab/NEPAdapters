# NEP_CPU 数值参考数据

本目录保存从相邻 NEP_CPU 仓库 commit `9977dab39ae7ec23d64a73a1788815002824ace4` 复制的最小普通 NEP 和 qNEP 数值测试。复制时源 fixture 目录保持干净。

每个 case 只保留模型、周期性 `xyz.in` 结构，以及固定的力、每原子 raw9 virial 和 descriptor 参考值。生成输出、有限差分数据、GPU 输出、源码和构建文件均未复制。

`nep` 是 250 原子的普通 NEP3 模型，`qnep` 是 250 原子的 `nep4_charge1` 模型。两者都通过公共 `cpu` 后端测试；启用 CUDA 后，qNEP 测试复用相同 fixture。CUDA 明确不支持 NEP3，不会回退到 CPU。
