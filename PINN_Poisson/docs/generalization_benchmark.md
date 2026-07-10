# Generalization Benchmark

当前模型 `pressure_initializer_paired_abs_residual32.pt` 只在 `params_small.in`
的窄分布上训练。泛化验证使用 `PINN_Poisson/configs/generalization/`
下的 held-out 参数，不写入训练 manifest。

运行：

```bash
PINN_Poisson/.venv/bin/python PINN_Poisson/scripts/run_generalization_benchmark.py
```

输出：

```bash
PINN_Poisson/results/benchmarks/generalization_pressure_init.csv
```

`poisson_check_interval=100`、`steps=1` 当前结果：

| case | baseline iter | PINN iter | PINN fallback | PINN pressure rel-L2 | oracle iter |
| --- | ---: | ---: | ---: | ---: | ---: |
| velocity_low | 800 | 800 | 0 | 4.04e-2 | 400 |
| velocity_high | 800 | 700 | 0 | 1.74e-3 | 400 |
| single_droplet | 800 | 700 | 0 | 2.84e-1 | 400 |
| tanh_interface | 900 | 1900 | 1 | 0 | 500 |

结论：

- 当前模型不是已验证泛化模型。
- `velocity_high` 有小收益，`velocity_low` 无迭代收益。
- `single_droplet` 是危险 case：residual gate 接受，但 pressure error 很大。
- `tanh_interface` 能触发 fallback，但失败 attempt 吃满 1000 iter，总代价高。

新增 `--pressure-init-max-iterations N` 可限制 PINN 初次 correction 的预算，
未收敛会提前 fallback；fallback 仍跑完整 Poisson。该参数不是默认策略：
设太小会误伤本来需要 400-700 iter 才收敛的可接受 case。

已测 `tanh_interface`：

| cap | PINN total iter | fallback |
| ---: | ---: | ---: |
| unset | 1900 | 1 |
| 400 | 1300 | 1 |
| 100 | 1000 | 1 |

下一步应训练带 held-out 多形态数据的模型，并增加运行时 OOD/quality gate，
不能只依赖 residual 达标。

## Augmented Model

新增脚本：

```bash
PINN_Poisson/scripts/build_augmented_manifest.py
PINN_Poisson/scripts/input_quality_gate.py
```

训练 manifest：

```bash
PINN_Poisson/data/augmented_manifest.csv
```

它包含原 `paired_manifest.csv` 的 12 条样本，加上：

- `velocity_low`
- `velocity_high`
- `single_droplet`
- `tanh_interface`

当前增强模型：

```bash
PINN_Poisson/models/pressure_initializer_augmented_abs_residual32.pt
```

训练内泛化 case 结果：

| case | baseline iter | augmented PINN iter | pressure rel-L2 | fallback | oracle iter |
| --- | ---: | ---: | ---: | ---: | ---: |
| velocity_low | 800 | 500 | 6.35e-3 | 0 | 400 |
| velocity_high | 800 | 600 | 2.53e-3 | 0 | 400 |
| single_droplet | 800 | 500 | 1.64e-2 | 0 | 400 |
| tanh_interface | 900 | 600 | 4.30e-4 | 0 | 500 |

这说明增强训练修复了已加入训练集的失败 case，但不能单独证明外推泛化。

## Validation Cases

新增未参与增强训练的 validation 参数：

```bash
PINN_Poisson/configs/validation/
```

使用增强模型 + augmented input quality gate：

| case | baseline iter | augmented PINN iter | input gate | pressure rel-L2 | oracle iter |
| --- | ---: | ---: | ---: | ---: | ---: |
| velocity_extreme | 800 | 600 | accept | 5.32e-3 | 400 |
| offset_pair | 800 | 500 | accept | 1.48e-2 | 400 |
| single_droplet_shifted | 800 | 500 | accept | 1.35e-2 | 400 |
| tanh_interface_wide | 1000 | 1000 | reject | 0 | 400 |

若不启用 input quality gate，`tanh_interface_wide` 会被 residual gate 接受，
但 pressure rel-L2 为 `2.61e-1`。这说明 residual-only gate 仍不够。

当前推荐实验策略：

1. 对模型使用对应训练 manifest 的 `input_quality_gate.py` 做预测前筛查。
2. 通过筛查才运行 PINN initializer。
3. PINN correction 仍必须 residual 达标。
4. 未通过筛查或 residual 未达标，走 full Poisson fallback。

## Safe Pipeline Entry

正式封装脚本：

```bash
PINN_Poisson/scripts/run_safe_pressure_initializer.py
```

该脚本接收一个 pre-Poisson 场文件，按如下顺序执行：

1. 根据 `--quality-manifest` 计算 input quality gate。
2. 通过时调用 `predict_pressure_initializer.py` 导出 pressure-init 文件。
3. 调用 `run_pressure_init_gate.py` 跑 residual-controlled correction。
4. input gate 拒绝时跳过 PINN，直接跑 no-init baseline/full Poisson。
5. residual 未达标时由 CUDA runner 自动 fallback。

Validation smoke：

| case | input gate | used PINN | iter | pressure rel-L2 |
| --- | ---: | ---: | ---: | ---: |
| velocity_extreme | accept | yes | 600 | 5.32e-3 |
| tanh_interface_wide | reject | no | 1000 | 0 |

结果记录在：

```bash
PINN_Poisson/results/benchmarks/safe_pressure_initializer_validation.csv
```

## Batch Benchmark

标准批量入口：

```bash
PINN_Poisson/scripts/run_safe_pipeline_benchmark.py
```

默认执行 validation case set，并为每个 case 自动运行：

1. 采集 `pre_poisson/post_poisson` pair。
2. 跑 no-init baseline 作为 reference。
3. 跑 safe pipeline。
4. 跑 oracle pressure init 对照。
5. 输出统一 CSV。

示例：

```bash
PINN_Poisson/.venv/bin/python PINN_Poisson/scripts/run_safe_pipeline_benchmark.py \
  --case-set validation \
  --csv PINN_Poisson/results/benchmarks/safe_pipeline_validation_benchmark.csv
```

当前 validation safe-pipeline 结果：

| case | quality | used PINN | iter | pressure rel-L2 |
| --- | ---: | ---: | ---: | ---: |
| velocity_extreme | accept | yes | 600 | 5.32e-3 |
| offset_pair | accept | yes | 500 | 1.48e-2 |
| single_droplet_shifted | accept | yes | 500 | 1.35e-2 |
| tanh_interface_wide | reject | no | 1000 | 0 |
