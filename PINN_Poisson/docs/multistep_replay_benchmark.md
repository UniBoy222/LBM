# Multi-Step Safe Replay Benchmark

目标：验证 `PINN pressure initializer + residual-controlled Poisson correction`
在连续多步中是否仍能减少 Poisson 迭代，并保持 `rho/fei/u/v/w/p` 接近 no-init baseline。

当前脚本：

```bash
PINN_Poisson/.venv/bin/python PINN_Poisson/scripts/run_multistep_safe_replay.py \
  --steps 10 \
  --label multistep_safe_replay_step10 \
  --csv PINN_Poisson/results/benchmarks/multistep_safe_replay_step10.csv
```

流程：

1. 跑 no-init baseline，采集每步 `pre_poisson/post_poisson` pair 和输出场。
2. 用 baseline 的每步 `pre_poisson` 通过 input quality gate。
3. gate 通过才调用 PINN 生成 `3D#########.bin` pressure-init schedule。
4. GPU runner 用 `--pressure-init-dir` 连续 replay。
5. 每步用 residual gate 决定接受或 fallback。
6. 对每步输出比较 baseline 的 `rho/fei/u/v/w/p`、相质量和形态指标。

边界：这不是最终 closed-loop rollout。PINN 输入来自 baseline pre-Poisson 轨迹，
不是 safe trajectory 运行时现场生成的 pre-Poisson 状态。

## 当前 10-Step 结果

配置：`params_small.in`、`onepass`、`pressure-boundary=fused`、
`poisson_check_interval=100`、`poisson_tolerance=1e-3`。

| method | avg iter/step | total iter est | total ms est | fallback steps | pressure rel-L2 final |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 980 | 9800 | 968.67 | 0 | 0 |
| safe replay | 1850 | 18500 | 1748.08 | 9/10 | 1.59e-4 |
| oracle replay | 1840 | 18400 | 1728.69 | 9/10 | 6.77e-6 |

safe replay 字段最大误差：

| field | max rel-L2 |
| --- | ---: |
| rho | 5.63e-7 |
| fei | 7.71e-7 |
| u | 2.88e-5 |
| v | 3.16e-5 |
| w | 3.96e-5 |
| pressure | 2.56e-3 |
| phase mass | 9.85e-9 |

结论：

- 当前模型不能声明多步泛化。
- 数值字段在 replay 中仍接近 baseline，但没有降迭代；反而因为 9 步 fallback 变慢。
- oracle pressure schedule 也 fallback 9 步，说明问题不只是 PINN 压力误差，`p` 单独初始化可能不足以改善多步 Poisson 固定点状态。
- 下一步应研究 `p + hh/source` 一致初始化，或增加更早的 runtime reject 机制，避免 bad initializer 吃满 1000 次再 fallback。

## Closed-Loop 结果

新增入口：

```bash
PINN_Poisson/.venv/bin/python PINN_Poisson/scripts/run_multistep_closed_loop.py \
  --steps 10 --device cpu
```

closed-loop 与 replay 的区别：每步 PINN 输入来自当前 safe trajectory 的
`pre_poisson`，不是 baseline 轨迹。

全步启用 PINN：

| method | avg iter/step | wall ms/step | fallback steps | final pressure rel-L2 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 980 | 96.91 | 0 | 0 |
| safe closed-loop | 1850 | 526.78 | 9/10 | 1.52e-4 |

保守 runtime policy，仅 step1 启用 PINN：

```bash
PINN_Poisson/.venv/bin/python PINN_Poisson/scripts/run_multistep_closed_loop.py \
  --steps 10 --pinn-max-step 1 --device cpu
```

| method | avg iter/step | wall ms/step | fallback steps | final pressure rel-L2 | final rho rel-L2 | final fei rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 980 | 98.08 | 0 | 0 | 0 | 0 |
| safe closed-loop, step1 only | 950 | 117.87 | 0 | 1.52e-4 | 5.65e-7 | 6.05e-7 |

该策略首次实现多步迭代数下降，但 wall time 仍未胜出：当前 Python/C++ 文件同步和
CPU 推理下，safe wall speedup 约为 `0.83x`。这说明：

- 数值准确性可以保持：`rho/fei` 为 `1e-6` 量级，压力最终约 `1.5e-4`。
- 迭代数可小幅下降：`980 -> 950`。
- 总耗时还不能宣称加速；需要嵌入式/常驻推理，或让 step2+ 也通过 residual gate。

`source-aware hh` scale sweep（0.25、0.5、1.0、2.0）在 2-step closed-loop 上都仍然
step2 fallback，说明当前简单 source-aware `hh` 初始化不足以修复多步问题。

## 48x96x128 / 8000-Config Direction

`params_small.in` 不是 8000 步验收配置：它在短程内会出现场发散。后续主线切到：

```bash
GPU/pinn_restart_assets/configs/params_baseline_8000_stride1.in
```

短程诊断确认该配置前 20 步与 reference residual summary 一致：step1-18 打满
1000 次，step19 用 800 次收敛，step20 用 400 次收敛。因此不能在启动区全步启用
PINN；必须使用“跳过启动区 + 周期探测 + residual gate + fallback”。

新增 48 网格资产：

```bash
PINN_Poisson/data/paired_manifest_48x96x128.csv
PINN_Poisson/models/pressure_initializer_48x96x128_patch_delta_residual32.pt
```

训练方式为 `delta_p = post - pre`，使用 patch 训练，避免整批 48x96x128 体数据一次性占满 GPU。

当前关键结果：

| case | policy | avg iter/step | attempts | accepts | fallbacks | note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 30-step baseline | none | 940 | 0 | 0 | 0 | 48x96x128 |
| 30-step safe | adaptive, candidates 20/25/26/27/30 | 930 | 1 | 1 | 0 | step20: 400 -> 300 |
| 30-step safe | no adaptive, same candidates | 970 | 5 | 1 | 4 | unsafe to probe densely |
| 100-step baseline | none | 789 | 0 | 0 | 0 | 48x96x128 |
| 100-step safe | adaptive, candidates 20/50/75/100 | 789 | 3 | 2 | 1 | step20 and 100 accepted; step75 fallback |
| 100-step safe | accepted-only candidates 20/100 | 786 | 2 | 2 | 0 | step20: 400 -> 300; step100: 300 -> 200 |
| 100-step safe runtime-opt | pre-only pair I/O, single Tecplot read per candidate | 784 | 2 | 2 | 0 | step20: 200; step100: 100 |
| 200-step baseline | none | 453.5 | 0 | 0 | 0 | 48x96x128 |
| 200-step safe | candidates 20/100/150/200, PINN check interval 50 | 452.5 | 4 | 4 | 0 | step20: 400 -> 200; 100: 300 -> 100; 150/200: 100 -> 50 |
| 1000-step baseline | none | 188.0 | 0 | 0 | 0 | 48x96x128 |
| 1000-step safe | candidates 20/100/150/200/500/1000, PINN check interval 50 | 184.7 | 6 | 6 | 0 | all candidates accepted |
| 2000-step safe dense100 | candidates 20,100,200,...,2000 | 144.7 | 21 | 21 | 0 | numerically safe but worse than baseline 144.0 |
| 2000-step safe sparse | candidates 20/100/150/200/400/500/1000/1500/2000 | 142.275 | 9 | 9 | 0 | net iteration win vs baseline 144.0 |
| 4000-step baseline | none | 122.0 | 0 | 0 | 0 | 48x96x128 |
| 4000-step safe sparse500 | early sparse + 500-step probes from 2500 to 4000 | 121.088 | 13 | 13 | 0 | net iteration win, no fallback |
| 8000-step baseline | none | 111.0 | 0 | 0 | 0 | 48x96x128 |
| 8000-step safe sparse500 | early sparse + 500-step probes from 2500 to 8000 | 110.494 | 21 | 21 | 0 | full 8000 accepted, no fallback |
| 8000-step safe runtime-opt | pre-only pair I/O, single Tecplot read per candidate | 110.494 | 21 | 21 | 0 | same numerics, lower wall overhead |
| 8000-step safe feature-pack | compact binary feature pair, device-side packing | 110.494 | 21 | 21 | 0 | same numerics, no wall-time win |

Numerical health in the 100-step runs remains good: `field_health_pass=1`,
`component_count=2` matches baseline at output checkpoints, `rho/fei/u/v/w/p` differences are
small after fallback/accepted correction. Wall time is not yet improved because Python inference,
Tecplot pair I/O, and wait-dir synchronization dominate the sparse-probe runs.

Current conclusion:

- This is now a real 48x96x128 closed-loop initializer, not the old 16x32x32 model.
- It now proves full 8000-step sparse500 generalization under residual gate, but not wall-time speedup.
- Dense probing is harmful; residual fallback protects correctness but costs iterations.
- Accepted-only probing has the first 100-step net iteration win on the 8000-step config:
  `789 -> 786` iter/step with `0` fallback.
- A separate PINN-attempt residual check interval is now available. With
  `--pressure-init-check-interval 50`, late accepted steps can go below the global
  `poisson_check_interval=100` floor while fallback/full Poisson remains on the baseline interval.
- Dense probing is still unsafe for performance. In the 2000-step dense100 run all 21
  candidates were accepted and no fallback occurred, but the 500-999 interval became more
  expensive and the run regressed from `144.0` to `144.7` iter/step.
- Sparse probing currently generalizes through 4000 steps on the 8000-step config. The
  4000-step sparse500 run accepted all 13 candidates with `0` fallback, reduced average
  Poisson iterations from `122.0` to `121.088`, and kept `field_health_pass=1`.
- Sparse probing now generalizes through full 8000 steps on the same config. The 8000-step
  sparse500 run accepted all 21 candidates with `0` fallback, reduced average Poisson
  iterations from `111.0` to `110.494`, and kept `field_health_pass=1`.
- Numerical differences versus baseline remain small over the 8000-step run:
  max `rho_rel_l2=3.51e-4`, max `fei_rel_l2=2.03e-4`, max `pressure_rel_l2=1.19e-2`
  at step 20, and max phase mass relative error `3.95e-7`.
- Kernel timing improves slightly (`321.034 -> 319.589 ms/step`), but end-to-end wall
  time regresses (`322.165 -> 328.184 ms/step`). The current evidence supports guarded
  sparse candidate schedules, not dense PINN probing and not yet wall-time acceleration.

Runtime optimization status:

- `run_multistep_closed_loop.py` now defaults to `--poisson-pair-phase pre`; the GPU
  runner keeps `both` as its standalone default for compatibility.
- The Python loop now reads each candidate Tecplot once and reuses the parsed fields for
  finite checks, input quality metrics, and PINN inference.
- 100-step optimized smoke (`runtime_opt_preonly_step100.csv`) accepted `2/2` candidates
  with `0` fallback and reduced average iterations from `789` to `784`; wall speed is
  still roughly flat on this short run.
- Full 8000 optimized run (`runtime_opt_preonly_step8000_48delta_check50_sparse500.csv`)
  accepted `21/21` candidates with `0` fallback and identical field-error maxima to the
  earlier sparse500 run. It reduced safe wall time from `328.184` to `325.193 ms/step`,
  but this remains slower than the baseline `322.165 ms/step`.
- Compact feature binary exchange is implemented via `--poisson-pair-format features`.
  The full 8000 feature-pack run
  (`runtime_opt_featurebin_pack_step8000_48delta_check50_sparse500.csv`) accepted `21/21`
  candidates with `0` fallback and kept the same field-error maxima, but wall time was
  `325.591 ms/step`. This rules out Tecplot serialization/parsing as the dominant
  remaining bottleneck for sparse500.

## High-Coverage Candidate Scan

The main bottleneck is now candidate coverage and per-candidate iteration benefit, not
feature I/O. `run_multistep_closed_loop.py` writes a per-candidate CSV next to the main
benchmark CSV. Each row records:

- Python timing: pre-snapshot wait, feature read, finite check, input quality gate,
  PINN infer+initializer write.
- Immediate effect: `baseline_iter`, `safe_iter`, and `iter_delta_step`.
- Follow-on effect: iteration delta over the next 1/5/20/50/100 steps and until the
  next candidate.
- Gate result: input quality accept, pressure-init accept/fallback.
- Field error when the candidate step is also an output checkpoint.

8000-step scan results:

| schedule | candidates | accepts | fallbacks | avg iter/step | step delta sum | until-next delta sum | max pressure rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse500 | 21 | 21 | 0 | 110.494 | n/a | n/a | 1.19e-2 |
| every100 + step20 | 81 | 81 | 0 | 110.800 | 4600 | -3000 | 2.57e-2 |
| every50 + step20 | 161 | 161 | 0 | 109.881 | 9450 | -500 | 5.98e-2 |
| every20 | 400 | 400 | 0 | 108.969 | 22550 | -6300 | 1.32e-1 |

Interpretation:

- Higher coverage clearly increases immediate iteration savings.
- Residual gate accepted every candidate in all dense scans, so residual acceptance is
  not sufficient as a performance selector.
- Some accepted candidates perturb later steps. In the every20 scan, the negative net
  candidates are:
  `120,140,160,180,320,340,580,640,680,700,740,780,820`.
- Dense scans increase field error at sparse checkpoints while keeping
  `field_health_pass=1`. This is acceptable for exploration, not yet an accuracy policy.
- Average per-candidate Python timing is roughly: feature read `5 ms`, finite check
  `100 ms`, input quality gate `368-371 ms`, PINN infer+write `313-393 ms`. This is
  measurable overhead but not the blocker for iteration reduction research.

Selector seed:

```bash
PINN_Poisson/.venv/bin/python PINN_Poisson/scripts/select_candidate_schedule.py \
  PINN_Poisson/results/benchmarks/candidate_scan_every20_step8000_48delta_candidates.csv \
  --out-steps PINN_Poisson/results/benchmarks/candidate_selector_every20_positive_steps.txt \
  --out-csv PINN_Poisson/results/benchmarks/candidate_selector_every20_positive.csv \
  --hard-cases-csv PINN_Poisson/results/benchmarks/candidate_selector_every20_hard_cases.csv
```

This keeps `387/400` positive-net candidates and exports `22` hard cases for training
data expansion.
