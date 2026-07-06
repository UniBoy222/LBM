# Poisson Solver Feasibility

## Decision

当前三条路线的优先级：

1. **Multigrid / preconditioned fixed-point route**：保留，作为下一阶段主线。
2. **MRT pressure collision**：暂不作为主线，只能作为高风险备选。
3. **FFT / DCT direct solver**：不作为主线。

关键理由：当前压力求解不是普通标量 Poisson，而是 D3Q15 pressure distribution `hh` 的固定点迭代。直接标量化已经在 `scalar_stencil_scale2` 中失败，`p_rel_l2=1`。因此后续算法必须尽量保持现有 `hh -> p=sum(hh)` 固定点，而不是替换成未经等价证明的标量方程。

## Current Solver Facts

- 状态变量：`hh` 为 `N_cells * 15`，`p` 为带 z ghost 的宏观场。
- 每个 Poisson iter：pressure collision、stream、x 方向 slip/bounce-back、压力求和。
- 当前最佳实现：`onepass` 将 collision + stream + boundary + pressure sum 融合。
- 收敛控制：最多 1000 次，每 `poisson_check_interval` 检查 `p` 相对变化。
- 边界：x 为 slip/mirror，y/z 为 periodic。
- 正式性能：`graph-onepass` 总加速约 `1.298x`，但 Poisson 仍占约 `99.4%`。

## Route Assessment

| route | 可行性 | 审稿认可度 | 风险 | 判断 |
| --- | --- | --- | --- | --- |
| MRT pressure collision | 中 | 中 | 高 | 会改变非守恒 pressure moments；可能减少迭代，但 strict CPU/GPU gate 很可能失败 |
| Multigrid / preconditioner | 中高 | 高 | 中高 | 最像算法贡献；适合解释低频误差收敛，但必须围绕现有 `hh` 固定点构造 |
| FFT / DCT direct solver | 低 | 中 | 高 | 只适合常系数、可分离标量问题；当前两相 `rho` 变系数且 x/y/z 边界混合 |

## Why Not Direct Scalar Multigrid

已有负结果说明不能直接把 pressure iteration 改写成简单 `lap(p)=source`：

- `scalar_stencil_scale2` 速度很快，但 `p_rel_l2=1`，速度场也明显偏离。
- `source_aware_hh_init` 没有降低迭代数，说明只给 `hh` 一个解析源项初值不足以改变收敛。
- `pressure_relax_scale` 能减少迭代，但压力误差巨大或 NaN，说明简单改松弛谱不稳。

因此 multigrid 不能从 scalar Jacobi 版本开始。它必须先证明与当前 D3Q15 `hh` 固定点一致，至少要作为 exact fixed-point preconditioner，而不是替代物理路径。

## Recommended Implementation Gate

已实现一个低风险入口：**matrix-free fixed-point residual diagnostics**。

目标不是立刻写完整 multigrid，而是先把当前 onepass Poisson 迭代抽象成可观测的线性固定点过程：

```text
h_next = F(h; rho, div_u, boundary)
residual_h = h_next - h
p = sum_q h_q
```

验收：

- 在 `params_small.in` 上，`--poisson-diagnostics` 不改变现有 onepass 结果。
- 1/5/20 步 CPU/GPU gate 继续通过。
- 能输出每个 Poisson check segment 的 residual 历史、迭代数、压力误差。

只有这个 gate 通过后，才进入 two-grid 或 Anderson/GMRES-like acceleration。否则直接写 multigrid 会变成不可验证的大改。

## Implemented Diagnostic Target

已实现：

- `--poisson-diagnostics`：输出每个 timestep 的 Poisson residual history。
- CSV 字段：`step, iteration, pressure_l1_delta, pressure_l1_norm, relative_error, converged`。
- 小网格 gate：`onepass` 诊断模式不改变 `fei/rho/u/v/w/p`。

Smoke 结果：

- `make gpu compare` 通过。
- `make poisson-diagnostics-smoke` 通过。
- 诊断 CSV 显示 step 1 在 iter 800 收敛，step 2 到 iter 1000 未达到 `1e-3`，这解释了为什么后续 accelerator 必须减少低频残差，而不是继续调 check interval。

正式 48x96x128、20 步结果：

- 命令：`make poisson-diagnostics-48-20`。
- 输出：`poisson_diagnostics_48x96x128_20.csv`、`poisson_diagnostics_48x96x128_20_summary.csv`、`poisson_diagnostics_48x96x128_20_summary.md`。
- 20 步中 2 步收敛，18 步打满 1000 次。
- 平均 final iteration 为 960，平均 final relative error 为 0.0184。
- median late ratio 为 0.9524，说明后期 residual 明显慢尾。

判断：当前瓶颈不是 launch overhead，而是固定点迭代低频/慢模态收敛。下一候选应优先保持固定点不变，例如 full-step fixed-point relaxation、Anderson/GMRES-like acceleration 或 coarse-grid correction。

Full-step fixed-point relaxation 筛选结果：

- 新增参数：`--poisson-fixed-point-relax omega`。
- 候选：`omega=0.9, 1.05, 1.1, 1.2`。
- 5 步 gate 中 4 个候选全部失败。
- `omega=0.9` 未加速且 `p_rel_l2=9.446e-02`。
- `omega>1` 虽可能减少 wall time 或迭代数，但出现巨大误差或 NaN。

判断：单参数 fixed-point relaxation 不可作为论文主线。下一步若继续做算法级优化，应跳过所有单参数 relaxation，只保留更完整的 Anderson/GMRES-like acceleration 或严格 coarse-grid preconditioner。

Anderson m1 筛选结果：

- 新增参数：`--poisson-anderson-m1`、`--poisson-anderson-beta-max`。
- 候选：`beta_max=0, 0.5, 1.0`。
- 5 步 gate 中 3 个候选全部失败。
- `beta_max=0` 最稳，但 `p_rel_l2=2.492e-03`，仍未过 `1e-4` accuracy gate，且 wall time speedup 只有 0.530。
- `beta_max=0.5/1.0` 仅把平均迭代从 960 降到 940，但压力误差分别到 `4.613e-02`、`7.197e-02`，且全局 reduction 使运行更慢。

判断：朴素 Anderson/Aitken m1 不可作为论文主线。若继续做 Anderson，应改成更完整的 m>1/Krylov 设计并减少每迭代 host reduction；若继续 two-grid，则必须做等价 coarse operator，而不是启发式压力外推。

Spatial residual 诊断结果：

- 新增参数：`--poisson-spatial-diagnostics`。
- 指标：`block_low_frequency_fraction = sum_blocks |sum(delta_p)| / sum_cells |delta_p|`，block size 为 4。
- 命令：`make poisson-spatial-diagnostics-48-5`。
- 结果文件：`poisson_spatial_diagnostics_48x96x128_5_summary.md`。
- 5 步全部打满 1000 次，median late ratio 为 0.9075。
- median final block low-frequency fraction 为 0.9936，median mean block low-frequency fraction 为 0.9779。

判断：慢尾 residual 主要是粗尺度同号成分，two-grid/coarse-grid correction 具备进入候选实现的必要条件。但这只说明粗尺度方向值得做，不等于任意 block correction 都可接受。

启发式 block two-grid correction 筛选结果：

- 新增参数：`--poisson-two-grid-correction`、`--poisson-two-grid-strength`。
- 实现方式：每次 residual check 后统计 4^3 block 的 `delta_p` block sum，并将 block 平均修正写回 `p/hh/p_prev`。
- 候选：`strength=1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1`。
- 5 步 gate 中 7 个 two-grid 候选全部 strict 失败。
- `strength=1e-4` 压力误差最小，`p_rel_l2=9.427e-05`，只过 `1e-4` accuracy gate，但 Poisson 迭代仍为 960，没有算法收益。
- `strength>=5e-3` 才把迭代从 960 降到 940，但 `p_rel_l2>=4.716e-03`，不可作为论文主贡献。

判断：block-sum 外推不是等价 preconditioner，只能作为负结果。若继续 coarse-grid 路线，必须构造保持当前 `hh -> p=sum(hh)` 固定点的 coarse operator / residual correction，而不是直接平均 `delta_p`。

这一步有三个作用：

1. 给 multigrid/预条件设计提供真实收敛曲线。
2. 证明瓶颈来自低频残差还是局部高频误差。
3. 为后续 Anderson、two-grid、MRT 任一路线提供统一评估基线。

## Stop Conditions

以下情况应停止该路线：

- 诊断模式改变数值结果。
- 后续 accelerator 不能在 strict gate 下减少至少 30% Poisson iterations。
- 20 步 gate 中 `p_rel_l2` 超过 `1e-8` 量级，或质量误差明显放大。

## Paper Positioning

如果后续 fixed-point accelerator 成功，论文贡献可以写成：

> A memory-aware D3Q15 pressure fixed-point solver with exact CPU/GPU validation and physics-gated acceleration for 3D two-phase LBM.

如果失败，则当前 onepass fusion + negative algorithm boundary 仍可支撑较完整工程论文，但不建议强冲 SCI 一区。
