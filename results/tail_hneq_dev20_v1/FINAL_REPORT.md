# 完整 p+hneq 动力学尾态严格 warm-start Oracle：最终报告

## 结论

- **严格正确性与迭代数：Go。** development 20 步中，baseline 总 Poisson 内迭代 `130600`，replay 为 `2000`，下降 `98.468606%`；20/20 均在恢复尾态后的首次 100 次检查退出，`fallback=0`。
- **正式 E2E：Go。** 预热后独立重复 5 次，baseline 中位数 `157387.828253 ms`（IQR `301.063551 ms`），replay 中位数 `5843.321279 ms`（IQR `12.158853 ms`），下降 `96.287311%`，加速 `26.934653x`；20/20 步的 replay 墙钟中位数均优于 baseline。
- **完整场门禁：Go。** 最大压力残差 `9.993306985713183e-4`，最大 D3Q15 固定点残差 `1.072553628858834e-5`，最大恢复误差 `2.220446049250313e-15`，最大完整场相对误差 `6.929244850797704e-15`；全程有限。
- 本结论仅证明严格收敛 `p+hneq` 尾态 warm-start 的 Oracle 上界，不代表学习模型已经能预测这些量，也不是可部署模型结论。

## 干净主分支

- GitHub `main` 与 `codex/clean-unbounded-main` 均固定在根提交 `dac9d562fdacdf275a7a822ed0ac0c0f3f3dac45`。
- 主分支只有 25 个源码、参数、Makefile、`.gitignore` 和说明文件；不含 Oracle、审计、结果或训练文件。
- `GPU/Inamuro.cpp` 与 `CPU/Inamuro.cpp` 的旧 `for (... <= 1000)` 已改为带溢出及 NaN/Inf 硬失败的严格 `while (true)`。
- GPU 入口固定 `check_interval=100`、`tolerance=1e-3`、`poisson_iteration_limit=0`，非零上限会在启动时拒绝。
- 干净主分支工作树：`/home/jzh/whileLBM_warmstart_v1`。
- 本 Oracle、审计与证据仅位于实验分支 `codex/warmstart-p-hneq-tail-v1`。

## 冻结算法

1. baseline 使用 GPU split collision、split boundary、固定顺序归约、双残差、`check_interval=100`、`tol=1e-3`、`iteration_limit=0`。
2. 若严格收敛发生在 `K`，保存完成 gauge、固定点残差计算和 `p_prev` 更新后的 `J=K-100` 状态：`p_tail`、cell-major 15 通道 `hneq_tail`、`p_prev_tail`、活动相位及协议元数据。
3. replay 在同一 Poisson 入口完成 source 零模投影和 gauge 取得后，校验 header/payload，再重建 `h_i=E_i p_tail+hneq_i`，同步 `p` 与 `p_prev`。
4. 随后执行原始 `collision→stream→x-slip→p=sum(h)` 100 次；首次检查真实计算压力变化与离散固定点双残差。若失败，代码会记录 fallback 并继续无上限求解；本次 runner 对任何 fallback 立即判失败。
5. reference 最终场仅在 candidate 完成及下载后读取用于比较，从不注回求解状态。

## 20 步正确性结果

| 指标 | 结果 |
|---|---:|
| baseline 总内迭代 | 130600 |
| replay 总内迭代 | 2000 |
| 首次检查退出 | 20/20 |
| fallback | 0 |
| 最大压力残差 | 9.993306985713183e-4 |
| 最大固定点残差 | 1.072553628858834e-5 |
| 最大恢复误差 | 2.220446049250313e-15 |
| 最大原始压力相对误差 | 1.306627487248048e-15 |
| 最大去均值压力相对误差 | 1.306312912279359e-15 |
| 最大压力梯度相对误差 | 4.384659269492467e-15 |
| 最大生产压力修正相对误差 | 6.929244850797704e-15 |
| 最大速度相对误差 | 2.337280031492501e-15 |
| 最大 u/v/w 相对误差 | 3.051671416368132e-15 / 3.032074563065387e-15 / 1.777777700942689e-15 |
| 最大 rho/fei 相对误差 | 1.224184033695511e-16 / 1.778948102077841e-16 |
| 最大 h 矩相对误差 | 1.308850425970995e-15 |
| 质量/形态 | 最大质量误差 4.105065893553650e-18；形态误差 0 |

## 算子复核

- 合成 D3Q15 CPU/GPU/Torch float64 逐组件最坏绝对误差：`3.3306690738754696e-16`。
- development step1–2 真实状态 Torch CPU/CUDA 最坏门禁误差：`6.661338147750939e-16`；生产 C++ 固定点分量最坏相对差：`1.445401113742535e-13`。
- 均小于 `1e-12`，gauge、JVP/VJP、有限非零梯度门禁通过。

## 正式计时协议

- baseline/replay 各预热 1 次，不计入；随后各 5 个新进程，按 baseline→replay 交错运行。
- baseline benchmark 禁止 tail capture、reference 写出和 Poisson diagnostics I/O。
- replay 完整计入 artifact 读取、校验、`h=Ep+hneq` 重建及同步；最终 reference 验证 I/O 不计入 solver 计时。
- 每步计时边界为 `performTimeStepGPU()` 调用前至 `gpu.synchronize()` 后，包含原始 Poisson 尾段、`correct_uvw` 和 `update_hh`；20 步 E2E 为 20 个生产步计时之和。
- 四分位数采用 Hyndman–Fan Type 7；所有原始样本均保留，包括 replay 的高 I/O 样本 `7621.808063 ms`。
- baseline 五次：`157524.019963, 157552.523296, 157387.828253, 157095.933607, 157222.956412 ms`。
- replay 五次：`5843.321279, 7621.808063, 5797.290709, 5848.315778, 5836.156925 ms`。
- replay warm-start I/O 中位数 `3141.681359 ms`（20 步合计），已包含在 replay E2E 中。

## 证据

- 最终机器可读指标：`/home/jzh/whileLBM_tail_hneq_oracle_v1/results/tail_hneq_dev20_v1/FINAL_METRICS.json`
- 逐步计时统计：`/home/jzh/whileLBM_tail_hneq_oracle_v1/results/tail_hneq_dev20_v1/timing_per_step.csv`
- 正确性 capture/replay：`/home/jzh/whileLBM_tail_hneq_oracle_v1/results/tail_hneq_dev20_v1/capture`、`/home/jzh/whileLBM_tail_hneq_oracle_v1/results/tail_hneq_dev20_v1/replay`
- 5 次正式计时：`/home/jzh/whileLBM_tail_hneq_oracle_v1/results/tail_hneq_dev20_v1/timing`
- 合成与真实状态跨实现审计：`/home/jzh/whileLBM_tail_hneq_oracle_v1/results/tail_hneq_stage0_v1`
- SHA256 清单：`/home/jzh/whileLBM_tail_hneq_oracle_v1/results/tail_hneq_dev20_v1/FINAL_SHA256SUMS`

## 边界

- 未读取封存数据或禁止目录，未运行 1000/8000 个外层时间步，未修改干净主分支 solver。
- 固定顺序归约只用于消除 CUDA 原子归约的末位非确定性，不作为加速机制。
- 下一阶段应学习可压缩尾态表示并设置严格回退；在学习模型通过独立整轨迹验证前，不得把本 Oracle Go 表述为学习 warm-start Go。
