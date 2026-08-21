# 纯净单 GPU CUDA 基线

## 范围与来源

- 基础提交：`1a6e3a9f24a5d9d04d7b2f3d077e412faae824f7`。
- 本目录只包含 LBM 数值求解代码，不包含神经网络、模型权重、PINN 或训练管线。
- 当前只实现单 GPU；没有域分解、跨 GPU 通信或多 GPU 加速声明。

## 数值合同

压力 Poisson 求解沿用原始版本的合同：

- 每个外层时间步最多执行 1000 次 Poisson 迭代；
- 每 100 次检查一次压力相对变化；
- 相对变化严格小于 `1e-3` 时提前退出；
- 未收敛时在第 1000 次停止，并明确报告 `converged=no`；
- 每100次检查压力残差的有限性，完整求解步结束后检查全部物理域 `ff/gg/hh/rho/fei/u/v/w/p`；非有限状态报错。

这里的“1000”是 Poisson 内层迭代上限，不是外层时间步数。该历史合同只有压力变化残差，不等同于后续研究使用的严格双残差合同。

## 已修复的问题

- 生产入口现在实际调用 `InamuroCUDA`，不再误走 CPU 时间步。
- `ff`、`gg`、`hh` 分别使用独立的 ping-pong 缓冲，避免物理状态互相覆盖。
- 删除固定 101 次的占位循环，恢复旧版 `1e-3` 单残差、1000 上限退出语义。
- 压力历史、流式临时量和首步标志改为实例状态，避免多个实例互相污染。
- CUDA 裸指针对象禁止复制和移动，避免双重释放。
- 参数文件缺失、字段不完整、非有限、基本网格或关键分母参数非法时立即失败，不再静默使用默认工况。
- CUDA构造与计时资源具备异常清理，网格尺寸在进入分配前检查int索引上限。
- 性能输出同时区分阶段计时小计与包含边界、修正、有限性检查的完整求解步墙钟。
- 构建只使用 `GPU/` 下的纯数值源码；CUDA 架构可配置。

## 构建与运行

RTX 5080（当前环境）：

```bash
cd GPU
make CUDA_ARCH=sm_120 all
./lbm_gpu --params params.in --steps 2
```

其他 GPU 请显式设置对应的 `CUDA_ARCH`。程序只接受：

```text
--params <file> --steps <positive integer> --output-every <nonnegative integer>
```

## CPU/GPU 对照验证

```bash
cd GPU
make CUDA_ARCH=sm_120 test
./lbm_compare --params params.in --steps 2
```

对照程序逐步检查：

- `rho`、`fei`、`u`、`v`、`w`、`p`；
- `ff`、`gg`、`hh` 的全部 45 个方向场；
- time0的6个宏观量完整存储（含ghost）与45个分布方向H2D/D2H逐值往返；
- CPU/GPU Poisson 退出迭代数及每100次的完整残差轨迹；
- `fei=sum(ff)`、`sum(hh)=p`、`hh=Ei*p` 和 x 壁面配对；
- 全场 finite，并用仅存在于测试二进制的钩子注入NaN验证拒绝路径；生产二进制不含该钩子。

提交前实测结果：

- 三轴不等的 `12x16x20` time0精确往返及第1步等价通过；Poisson按旧版 `1e-3` 压力变化单残差在400次收敛。
- `16x32x32` 连续 10 步等价通过；最坏逐场 relative-L2 为 `7.200362898809e-15`，仅第1步收敛，其余9步达到上限。
- `48x96x128, DD=32` 连续 2 步等价通过；CPU/GPU 每步均在 1000 次上限停止且未收敛，最坏逐场 relative-L2 为 `1.310519272287e-15`。
- 小网格连续 2 步重复 5 次，迭代数、残差与摘要逐字节一致。
- 实际GPU入口及自动创建输出目录/写文件路径均通过冒烟测试；缺失参数文件以非零状态退出。

最终编译输入、二进制哈希、命令、退出码和摘要见 [`SINGLE_GPU_VALIDATION.json`](SINGLE_GPU_VALIDATION.json)。

完整生产尺寸的两步测试验证的是 CPU/GPU 数值一致性；由于原始 `1e-3` 单残差合同在 1000 次内未收敛，它不构成严格 Poisson 收敛证明。

## 已知边界

- 当前验证证明该 CUDA 路径与本仓库 CPU 参考实现高度一致，并检查若干末端代数一致性；尚未逐阶段比较全部导数、stream和boundary内部张量，也不是对全部物理模型的独立验证。
- 当前没有严格 D3Q15 fixed-point 双残差、无限迭代或多 GPU 实现。
- `Complete solver-step wall` 不含构造、初始H2D、结果下载和写盘，不能称为应用E2E或加速结果。
- 当前 WSL/RTX 5080 环境中的 `compute-sanitizer` 报告 `WDDM debugger interface` 初始化失败和 `Device not supported`，因此不能声称 sanitizer 通过；普通 CUDA 运行与对照测试已通过。
- 进入多 GPU 开发前，应先冻结本提交和上述测试结果，再单独实现域分解与通信。
