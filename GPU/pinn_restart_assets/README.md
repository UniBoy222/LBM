# PINN Poisson Restart Assets

这个目录用于从干净的 `GPU/` 地基重启 PINN 降低 Poisson 迭代次数方向。

## 目录

- `tools/`：保留给新方向复用的 benchmark、Poisson gate、residual diagnostics 和物理验证脚本。
- `reference_data/`：保留旧主线中最有用的参考 CSV/MD，用于确认新 PINN 方案是否真的减少迭代且不破坏参考场。
- `configs/`：保留小网格和基准网格参数样例。
- `docs/`：保留旧 GPU 实现说明、Poisson 可行性判断和 onepass 小结。
- `legacy_archive/`：旧论文封版材料、图、表和大输出的本地归档；该目录被 `.gitignore` 忽略，不建议提交到 GitHub。

## 根目录保留内容

`GPU/` 根目录只保留可编译地基：

- `Inamuro.*`
- `InamuroCUDA.*`
- `InamuroSolver.*`
- `LBMBase.hpp`
- `common.*`
- `main.cpp`
- `compare_test.cpp`
- `Makefile`
- `params_small.in`

## 新方向边界

新方向不要写成“PINN 直接替代 Poisson”。建议主线是：

`PINN pressure initializer + residual-controlled Poisson correction`

基本 gate：

1. PINN 给出压力初值或压力修正。
2. 继续执行 Poisson correction。
3. residual 达标才接受。
4. 不达标自动回退 full Poisson。
5. 同时比较迭代次数、总耗时、压力误差、质量误差和液滴形态。

## 快速验证

在 `GPU/` 根目录运行：

```bash
make compare
make test-onepass
```

这两个命令用于确认地基求解器和 onepass 参考路径仍然可用。
