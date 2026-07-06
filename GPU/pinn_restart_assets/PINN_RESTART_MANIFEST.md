# PINN Restart Manifest

## 必须使用的地基

这些文件保留在 `GPU/` 根目录，用作新 PINN 项目的 baseline solver：

| 文件 | 作用 |
| --- | --- |
| `main.cpp` | GPU/CPU runner，支持 Poisson 模式和 diagnostics 参数 |
| `compare_test.cpp` | CPU/GPU/reference gate 基础程序 |
| `InamuroCUDA.cu/.hpp` | GPU onepass Poisson 和原始 split/fused 路径 |
| `Inamuro.cpp/.hpp` | CPU reference solver |
| `InamuroSolver.cpp/.hpp` | solver 包装 |
| `common.cpp/.hpp` | 公共工具 |
| `params_small.in` | 快速 smoke 参数 |

## 推荐复用工具

| 文件 | 位置 | 用途 |
| --- | --- | --- |
| `poisson_reference_gate.py` | `tools/` | 判断候选压力策略是否接近 onepass/full Poisson 参考场 |
| `poisson_diagnostics.py` | `tools/` | 汇总 residual diagnostics |
| `poisson_sweep.py` | `tools/` | 快速扫描 Poisson 策略和误差 |
| `benchmark_repeat.py` | `tools/` | 重复性能测试，输出均值/标准差 |
| `benchmark_sweep.py` | `tools/` | baseline/fused/onepass 性能对比 |
| `validation_metrics.py` | `tools/` | 双液滴和场指标解析 |

## 不建议提交的旧材料

`legacy_archive/` 包含旧论文包、图表、大型 `.plt` 输出和中间 CSV。它用于本地追溯，不进入 GitHub。

## PINN 第一阶段建议

1. 先生成训练/验证数据，不训练模型。
2. 数据至少包含：`p`、`rho`、`u/v/w`、`fei`、Poisson residual 曲线、迭代次数。
3. 先做小网格 `params_small.in`。
4. 评价只看四个 gate：
   - pressure relative L2 / max abs
   - Poisson iteration reduction
   - phase mass error
   - final regime / component count

## GitHub 提交策略

建议只提交：

- 根目录地基源码；
- `pinn_restart_assets/README.md`；
- `pinn_restart_assets/PINN_RESTART_MANIFEST.md`；
- `pinn_restart_assets/tools/` 中的轻量脚本；
- `pinn_restart_assets/reference_data/` 中的小型 CSV/MD。

不要提交：

- `legacy_archive/`
- `advisor_package*.tar.gz`
- `.plt` 大输出
- 编译产物 `lbm_gpu` / `lbm_compare`
