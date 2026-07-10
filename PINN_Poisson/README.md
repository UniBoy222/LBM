# PINN Poisson

主线是 `PINN pressure initializer + residual-controlled Poisson correction`。

当前入口：

1. 采集 reference 字段和压力初始化文件：
   ```bash
   python3 PINN_Poisson/scripts/collect_reference_data.py \
     --build --steps 5 --sample-steps 1,2,3,4,5 --poisson onepass
   ```
2. 采集推荐的训练配对数据：Poisson 前输入场 -> Poisson 后压力目标：
   ```bash
   python3 PINN_Poisson/scripts/collect_paired_data.py \
     --build --steps 5 --sample-steps 1,2,3,4,5 --poisson onepass
   ```
3. 训练压力初始化器，需要 PyTorch：
   ```bash
   python3 PINN_Poisson/scripts/train_pressure_initializer.py \
     --manifest PINN_Poisson/data/paired_manifest.csv --device cuda
   ```
   配对数据也支持训练压力修正：
   ```bash
   python3 PINN_Poisson/scripts/train_pressure_initializer.py \
     --manifest PINN_Poisson/data/paired_manifest.csv \
     --target-mode delta --device cuda
   ```
4. 用模型导出 `p_pred` 或 `delta_p`：
   ```bash
   python3 PINN_Poisson/scripts/predict_pressure_initializer.py FIELD.plt --out p_pred.bin
   python3 PINN_Poisson/scripts/predict_pressure_initializer.py FIELD.plt \
     --model PINN_Poisson/models/pressure_initializer_paired_delta.pt \
     --out delta_p.bin --mode delta --scale 0.5
   ```
5. 跑 residual-controlled gate：
   ```bash
   python3 PINN_Poisson/scripts/run_pressure_init_gate.py \
     --pressure-init-file p_pred.bin \
     --pressure-init-mode absolute \
     --poisson-check-interval 100
   ```
   不传 `--pressure-init-file` 时可用同一套指标跑 no-init baseline。
   可用 `--pressure-init-max-iterations N` 限制 PINN 初次 correction 的预算；未收敛会提前 fallback 到 full Poisson。
   可选测试源项一致的 `hh` 初始化：
   ```bash
   python3 PINN_Poisson/scripts/run_pressure_init_gate.py \
     --pressure-init-file p_pred.bin \
     --pressure-init-mode absolute \
     --source-aware-hh-init --source-aware-hh-scale 1.0
   ```
6. 无 PyTorch 的端到端 oracle benchmark：
   ```bash
   python3 PINN_Poisson/scripts/run_oracle_pressure_init_benchmark.py \
     --steps 1 --poisson onepass --pressure-boundary fused
   ```
7. 离线评估模型 pressure error：
   ```bash
   python3 PINN_Poisson/scripts/evaluate_pressure_initializer.py \
     --manifest PINN_Poisson/data/paired_manifest.csv \
     --model PINN_Poisson/models/pressure_initializer.pt
   ```
8. sweep source-aware `hh` 初始化：
   ```bash
   python3 PINN_Poisson/scripts/sweep_source_aware_hh_init.py
   ```

当前 step1 对比汇总见：

```bash
PINN_Poisson/results/benchmarks/pressure_init_step1_comparison.csv
```

在 `params_small.in`、`onepass`、`fused`、`poisson_check_interval=100` 下：

| 方法 | Poisson iter | total ms/step | pressure rel-L2 | fallback |
| --- | ---: | ---: | ---: | ---: |
| no-init baseline | 800 | 97.862 | 0 | 0 |
| oracle pressure init | 400 | 49.034 | 5.20e-4 | 0 |
| old default PINN | 400 | 49.113 | 2.05e-2 | 0 |
| paired absolute residual32 PINN | 400 | 49.069 | 4.42e-3 | 0 |
| residual delta scale 0.25 | 700 | 85.394 | 9.57e-3 | 0 |

当前实际最佳模型：

```bash
PINN_Poisson/models/pressure_initializer_paired_abs_residual32.pt
PINN_Poisson/data/pressure_init/pinn_paired_abs_residual32_step1.bin
```

泛化验证见：

```bash
PINN_Poisson/docs/generalization_benchmark.md
PINN_Poisson/scripts/run_generalization_benchmark.py
PINN_Poisson/configs/generalization/
```

当前结论：模型仍是窄分布 initializer，不应声明已泛化；`single_droplet`
会被 residual gate 接受但 pressure error 高，`tanh_interface` 会 fallback。

增强模型与 input quality gate 已加入：

```bash
PINN_Poisson/scripts/build_augmented_manifest.py
PINN_Poisson/scripts/input_quality_gate.py
PINN_Poisson/models/pressure_initializer_augmented_abs_residual32.pt
```

增强模型在 validation 中可处理 `velocity_extreme`、`offset_pair`、
`single_droplet_shifted`；`tanh_interface_wide` 仍需 input quality gate 拒绝后回退。

正式安全入口：

```bash
PINN_Poisson/.venv/bin/python PINN_Poisson/scripts/run_safe_pressure_initializer.py PRE_POISSON.plt \
  --params PARAMS.in \
  --model PINN_Poisson/models/pressure_initializer_augmented_abs_residual32.pt \
  --quality-manifest PINN_Poisson/data/augmented_manifest.csv
```

流程为：input quality gate 通过才预测 pressure init；随后运行 residual-controlled
Poisson correction；quality 拒绝或 residual 不达标时走 no-init/full-Poisson fallback。

批量标准评测入口：

```bash
PINN_Poisson/.venv/bin/python PINN_Poisson/scripts/run_safe_pipeline_benchmark.py \
  --case-set validation
```

输出：

```bash
PINN_Poisson/results/benchmarks/safe_pipeline_benchmark.csv
```

GPU runner 新增参数：

```bash
--pressure-init-file FILE
--pressure-init-dir DIR
--pressure-init-mode absolute|delta
--pressure-init-max-iterations N
--pressure-init-check-interval N
--write-poisson-pairs
--poisson-pair-dir DIR
--poisson-pair-phase pre|post|both
--poisson-pair-format tecplot|features
--poisson-pair-steps A,B,C
--poisson-pair-start-step N
--poisson-pair-interval N
--source-aware-hh-init
--source-aware-hh-scale X
```

启用后流程为：读取 PINN 压力初值或修正，初始化 GPU Poisson 的 `p/hh`，继续执行现有 Poisson correction；若 residual 未达标，恢复初始化前状态并自动回退同一 Poisson 配置。

多步 replay benchmark：

```bash
PINN_Poisson/.venv/bin/python PINN_Poisson/scripts/run_multistep_safe_replay.py \
  --steps 10
```

该脚本先采集 no-init baseline 的每步 `pre_poisson/post_poisson`，再从 baseline
pre-Poisson 轨迹生成 `--pressure-init-dir` schedule，随后连续跑 10 步 safe replay，
逐步比较 `rho/fei/u/v/w/p`、相质量和形态指标。它不是最终 closed-loop
PINN rollout，因为 PINN 输入来自 baseline 轨迹；用途是先筛掉多步 replay 下会累积误差或不省迭代的方案。

当前 10-step 结论见：

```bash
PINN_Poisson/docs/multistep_replay_benchmark.md
```

多步 closed-loop benchmark：

```bash
PINN_Poisson/.venv/bin/python PINN_Poisson/scripts/run_multistep_closed_loop.py \
  --steps 10
```

该脚本启动 GPU solver 后，每步先等待当前 safe trajectory 的 `pre_poisson`
输出，再现场运行 input quality gate 和 PINN 推理，然后 GPU solver 继续
residual-controlled Poisson correction。GPU runner 使用：

```bash
--pressure-init-wait-dir DIR
--pressure-init-wait-timeout-ms N
```

每步等待 `3D#########.bin`，若 Python 写出 `3D#########.skip` 则本步直接 no-init/full Poisson。

当前已在 `PINN_Poisson/.venv` 安装 PyTorch CUDA 版；训练/推理使用该 venv，CUDA gate、reference 采集和 oracle benchmark 本身不依赖 PyTorch。

48x96x128 / 8000 配置当前入口：

```bash
PINN_Poisson/.venv/bin/python PINN_Poisson/scripts/run_multistep_closed_loop.py \
  --params GPU/pinn_restart_assets/configs/params_baseline_8000_stride1.in \
  --steps 100 \
  --output-steps 20,50,75,100 \
  --model PINN_Poisson/models/pressure_initializer_48x96x128_patch_delta_residual32.pt \
  --quality-manifest PINN_Poisson/data/paired_manifest_48x96x128.csv \
  --pressure-init-max-iterations 300 \
  --pinn-steps 20,50,75,100 \
  --adaptive-residual-gate
```

当前扩展结果：

| case | policy | baseline iter/step | safe iter/step | accepts | fallbacks |
| --- | --- | ---: | ---: | ---: | ---: |
| 100-step | accepted-only 20/100 | 789 | 786 | 2 | 0 |
| 100-step | pre-only pair I/O + single Tecplot read per candidate | 789 | 784 | 2 | 0 |
| 200-step | 20/100/150/200, check interval 50 | 453.5 | 452.5 | 4 | 0 |
| 1000-step | 20/100/150/200/500/1000, check interval 50 | 188.0 | 184.7 | 6 | 0 |
| 2000-step | sparse 20/100/150/200/400/500/1000/1500/2000 | 144.0 | 142.275 | 9 | 0 |
| 4000-step | sparse500 through 4000 | 122.0 | 121.088 | 13 | 0 |
| 8000-step | sparse500 through 8000 | 111.0 | 110.494 | 21 | 0 |
| 8000-step | pre-only pair I/O + single Tecplot read per candidate | 111.0 | 110.494 | 21 | 0 |

注意：2000-step dense100 虽然 `21/21` 接受且 `0` fallback，但平均迭代从 baseline
`144.0` 退化到 `144.7`，说明 residual gate 保证安全不等于 dense probing 一定省迭代。
当前可声明的是：48x96x128 sparse500 schedule 已泛化到 full 8000-step，`21/21`
PINN candidates 全部通过 residual gate，`0` fallback，并保持 `field_health_pass=1`。
迭代数有小幅收益：`111.0 -> 110.494` iter/step；kernel 计时小幅收益：
`321.034 -> 319.589 ms/step`。但端到端 wall time 仍未加速：
`322.165 -> 328.184 ms/step`，主要受 Python 推理、Tecplot I/O 和 wait-dir 同步影响。

Runtime 优化入口：

- `run_multistep_closed_loop.py` 默认向 GPU runner 传 `--poisson-pair-phase pre`，
  closed-loop 只写 PINN 推理需要的 `pre_poisson` pair，不再写未使用的
  `post_poisson` pair。
- Python gate 现在每个候选只读取一次 Tecplot，并复用同一份数据做 finite check、
  input quality gate 和 PINN 推理。
- GPU runner 支持 `--poisson-pair-format features`，写出紧凑 feature binary
  (`PINNF1`)；closed-loop 默认使用该格式。字段顺序为 `u/v/w/rho/fei/press`，
  与 Tecplot reader 返回结构一致，input quality gate 和 PINN 推理可复用。
- 100-step smoke：`runtime_opt_preonly_step100.csv`，`2/2` accepts，`0` fallback，
  `789 -> 784` iter/step，`field_health_pass=1`。
- 8000-step optimized：`runtime_opt_preonly_step8000_48delta_check50_sparse500.csv`，
  `21/21` accepts，`0` fallback，`field_health_pass=1`。wall time 从原 safe
  `328.184` 降到 `325.193 ms/step`，但仍慢于 baseline `322.165 ms/step`。
- 8000-step feature-pack：`runtime_opt_featurebin_pack_step8000_48delta_check50_sparse500.csv`，
  `21/21` accepts，`0` fallback，`field_health_pass=1`，数值误差与 pre-only
  Tecplot 版本一致；wall time 为 `325.591 ms/step`，没有继续改善。当前证据说明
  Tecplot pair 解析/写出不是剩余 wall-time 瓶颈，下一步应转向提高迭代收益或减少
  Python/wait-dir 同步和质量门开销。

High-coverage candidate scan:

- `run_multistep_closed_loop.py` 现在会输出 `*_candidates.csv`，每个 candidate
  包含 timing、`baseline_iter`、`safe_iter`、accept/fallback、本步迭代收益、
  到下个 candidate 前的后续迭代变化，以及可用 checkpoint 的场误差。
- Timing 证据：feature read 约 `5 ms/candidate`，finite check 约 `100 ms`，
  input quality gate 约 `370 ms`，PINN infer+write 约 `313-393 ms`。这不是当前
  主线瓶颈；主线瓶颈是 coverage 和 accepted 后续扰动。
- 8000-step high-coverage 结果：

| schedule | candidates | accepts | fallbacks | iter/step | max pressure rel-L2 at checkpoints |
| --- | ---: | ---: | ---: | ---: | ---: |
| sparse500 | 21 | 21 | 0 | 110.494 | 1.19e-2 |
| every100 + step20 | 81 | 81 | 0 | 110.800 | 2.57e-2 |
| every50 + step20 | 161 | 161 | 0 | 109.881 | 5.98e-2 |
| every20 | 400 | 400 | 0 | 108.969 | 1.32e-1 |

- every20 暴露出 13 个 accepted 但净负收益的候选：
  `120,140,160,180,320,340,580,640,680,700,740,780,820`。
- `select_candidate_schedule.py` 可从 scan CSV 生成 selector schedule 和 hard cases：
  `candidate_selector_every20_positive_steps.txt` 保留 `387/400` 个正收益候选；
  `candidate_selector_every20_hard_cases.csv` 导出 `22` 个强正/负样本，作为训练数据扩展种子。
