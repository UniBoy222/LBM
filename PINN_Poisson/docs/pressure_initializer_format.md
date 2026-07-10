# Pressure Initializer Format

二进制格式用于 `--pressure-init-file`：

- 8 bytes magic: `PINNP1\0\0`
- `int32 lx, ly, lz`
- `double[lx * ly * lz]`，小端序
- 数据顺序：`z -> y -> x`，只包含物理域，不含 z ghost layer

`--pressure-init-mode absolute` 表示文件值是 `p_pred`。

`--pressure-init-mode delta` 表示文件值是 `delta_p`，运行时会加到当前压力场上。

CUDA 侧会把初始化后的物理域压力写入 `p`，用 `h_i = p * Ei` 重建 `hh`，并把 residual 的上一压力缓存种子设为初始化后的压力；随后照常运行 Poisson correction。

若同时启用 `--source-aware-hh-init --source-aware-hh-scale X`，Poisson correction 开始前会用当前速度散度源项重建 `hh`，用于测试压力初值与固定点 residual 的一致性。
