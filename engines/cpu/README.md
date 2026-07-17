# CPU 引擎

本目录包含唯一受支持的 CPU 引擎。

实现可以使用 OpenMP、SIMD、缓存友好布局和算法重构，但每项生产功能都必须通过固定 golden fixture、严格 FP64 oracle、有限差分和适用的 CPU/CUDA 一致性测试。

项目不保留旧 CPU 后端或兼容 fallback。
