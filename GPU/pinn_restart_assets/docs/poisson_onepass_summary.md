# Poisson One-Pass 优化小结

## 改动

- 新增 `--poisson onepass`。
- 将 Poisson 迭代内的 `collision+stream` 与 `boundary+pressure sum` 合并为一个 kernel。
- 新增 `d_p_tmp`，保证 one-pass kernel 只读旧压力场、只写新压力场，避免同 kernel 内读写竞争。
- one-pass kernel 使用 `16x4x4` block，是当前短测中最优配置。

## 正确性

`params_small.in`，20 步 CPU/GPU gate：

| field | relative L2 |
| --- | ---: |
| fei | 4.238308e-11 |
| rho | 7.042929e-12 |
| u | 3.807026e-11 |
| v | 4.616758e-11 |
| w | 4.646077e-11 |
| p | 4.490609e-13 |

phase mass relative difference: `1.965062e-11`。

同样参数下，`--poisson-graph` 的 graph-onepass 20 步 gate 也全部 PASS，误差与非 graph onepass 一致。

结论：one-pass 路径保持 strict CPU/GPU 等价。

## 正式性能

`48x96x128`，预热 10 步，计时 50 步，三次重复：

| variant | ms/step mean | ms/step std | MLUPS mean | speedup vs split |
| --- | ---: | ---: | ---: | ---: |
| split | 3223.948 | 13.869 | 0.183 | 1.000 |
| all-fused | 2823.127 | 13.314 | 0.209 | 1.142 +/- 0.001 |
| graph all-fused | 2824.288 | 8.605 | 0.209 | 1.142 +/- 0.001 |
| one-pass | 2483.368 | 4.360 | 0.237 | 1.298 +/- 0.003 |
| graph one-pass | 2483.965 | 5.686 | 0.237 | 1.298 +/- 0.004 |

相对 all-fused，one-pass 约 `1.137x`。

## 判断

这是一条有效的 Poisson 专项优化：收益来自减少 Poisson 迭代内一次全局内存读写和一次 kernel launch。

但它还不是一区级最终贡献：Poisson 仍占约 `99.4%`，总加速约 `1.30x`。下一步需要继续研究 Poisson 迭代算法本身，不能只做普通 fusion。
